# ============================================
# image_agent.py
# ============================================
import os
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from api.nas_engine   import run_quick_nas
from api.auto_trainer import run_yolo_detection

BASE_DIR    = Path(__file__).parent.parent.parent
TRAINED_DIR = BASE_DIR / "models" / "trained"
TRAINED_DIR.mkdir(parents=True, exist_ok=True)


# ── DINOv2 linear head ────────────────────────────────────────────────────────

class DINOv2LinearClassifier(nn.Module):
    """Tiny linear head on top of DINOv2 frozen embeddings (384-dim)."""
    def __init__(self, num_classes, embedding_dim=384):
        super().__init__()
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, embeddings):
        return self.classifier(embeddings)


# ── Agent ─────────────────────────────────────────────────────────────────────

class ImageAgent:
    NAME = "Image Agent"

    def __init__(self):
        self.trained_model         = None
        self.trained_classes       = []
        self.model_arch            = None   # "efficientnet_v2_s" | "resnet18_fallback" | "dinov2_linear"
        self._dinov2_processor     = None
        self._dinov2_backbone      = None   # frozen DINOv2-small
        self._dinov2_linear        = None   # trained linear head
        print("  ImageAgent loaded")

    # ── DINOv2 backbone (lazy, cached) ────────────────────────────────────────

    def _load_dinov2(self):
        """Load DINOv2-small once; subsequent calls are no-ops."""
        if self._dinov2_backbone is not None:
            return
        print("[DINOv2] Loading foundation model (one-time)...")
        from transformers import AutoImageProcessor, AutoModel
        self._dinov2_processor = AutoImageProcessor.from_pretrained(
            "facebook/dinov2-small")
        self._dinov2_backbone = AutoModel.from_pretrained(
            "facebook/dinov2-small")
        self._dinov2_backbone.eval()
        for p in self._dinov2_backbone.parameters():
            p.requires_grad = False
        print("[DINOv2] Ready.")

    def _extract_embeddings(self, images):
        """
        Extract CLS-token embeddings from PIL images.
        images: list[PIL.Image]  →  tensor (N, 384)
        """
        self._load_dinov2()
        inputs = self._dinov2_processor(images=images, return_tensors="pt")
        with torch.no_grad():
            out = self._dinov2_backbone(**inputs)
        return out.last_hidden_state[:, 0, :]  # CLS token

    def _extract_embeddings_from_tensors(self, tensor_batch):
        """
        Extract embeddings from pre-normalized image tensors (B, C, H, W).
        DINOv2-small uses the same ImageNet mean/std as torchvision loaders,
        so tensors can be passed directly after resizing to 224x224.
        """
        self._load_dinov2()
        if tensor_batch.shape[-1] != 224 or tensor_batch.shape[-2] != 224:
            tensor_batch = F.interpolate(
                tensor_batch, size=(224, 224),
                mode='bilinear', align_corners=False)
        if tensor_batch.shape[1] == 1:
            tensor_batch = tensor_batch.repeat(1, 3, 1, 1)
        with torch.no_grad():
            out = self._dinov2_backbone(pixel_values=tensor_batch)
        return out.last_hidden_state[:, 0, :]

    # ── DINOv2 training ───────────────────────────────────────────────────────

    def train_with_dinov2(self, train_loader, test_loader,
                          num_classes, hash_id, classes=None):
        """
        Foundation model training pipeline:
          1. Extract DINOv2 embeddings for all images (frozen backbone)
          2. Train tiny linear head on embeddings (20 epochs, ~1-2s)
          3. Save linear weights only (~50 KB)
        Returns metrics dict compatible with train_transfer output format.
        """
        start     = time.time()
        EMBED_DIM = 384
        device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # --- Extract all embeddings ------------------------------------------
        def _embed_loader(loader, label):
            all_emb, all_lbl = [], []
            print(f"[DINOv2] Extracting {label} embeddings...", flush=True)
            for images, labels in loader:
                emb = self._extract_embeddings_from_tensors(images)
                all_emb.append(emb.cpu())
                all_lbl.append(labels.cpu())
            return torch.cat(all_emb), torch.cat(all_lbl)

        train_emb, train_lbl = _embed_loader(train_loader, "train")
        test_emb,  test_lbl  = _embed_loader(test_loader,  "test")
        print(f"[DINOv2] Embeddings: train={train_emb.shape}, "
              f"test={test_emb.shape}")

        # --- Train linear head -----------------------------------------------
        print(f"[DINOv2] Training linear head ({num_classes} classes)...")
        linear    = DINOv2LinearClassifier(num_classes, EMBED_DIM).to(device)
        optimizer = torch.optim.Adam(linear.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        N, BATCH = len(train_emb), 64
        last_train_acc = 0.0
        for epoch in range(20):
            perm    = torch.randperm(N)
            correct = total = 0
            for i in range(0, N, BATCH):
                idx   = perm[i:i+BATCH]
                emb_b = train_emb[idx].to(device)
                lbl_b = train_lbl[idx].to(device)
                optimizer.zero_grad()
                out   = linear(emb_b)
                loss  = criterion(out, lbl_b)
                loss.backward()
                optimizer.step()
                preds    = out.argmax(1)
                correct += (preds == lbl_b).sum().item()
                total   += len(lbl_b)
            last_train_acc = round(100 * correct / total, 2)
            if (epoch + 1) % 5 == 0:
                print(f"  [DINOv2] Epoch {epoch+1}/20 -> {last_train_acc}%")

        # --- Evaluate --------------------------------------------------------
        linear.eval()
        with torch.no_grad():
            out      = linear(test_emb.to(device))
            preds    = out.argmax(1)
            test_acc = round(
                100 * (preds == test_lbl.to(device)).sum().item()
                / len(test_lbl), 2)

        training_time = round(time.time() - start, 1)

        # --- Save ------------------------------------------------------------
        weights_path = str(TRAINED_DIR / f"{hash_id}_image_linear.pth")
        meta_path    = str(TRAINED_DIR / f"{hash_id}_image_linear_meta.json")

        torch.save(linear.state_dict(), weights_path)
        model_size_kb = os.path.getsize(weights_path) / 1024

        class_list = classes or [str(i) for i in range(num_classes)]
        with open(meta_path, 'w') as f:
            json.dump({
                "model_type":       "dinov2_linear",
                "method":           "dinov2_linear",
                "num_classes":      num_classes,
                "embedding_dim":    EMBED_DIM,
                "training_time":    training_time,
                "foundation_model": "facebook/dinov2-small",
                "train_accuracy":   last_train_acc,
                "test_accuracy":    test_acc,
                "hash_id":          hash_id,
                "classes":          class_list,
            }, f, indent=2)

        # Cache for predict_image
        self._dinov2_linear  = linear.cpu()
        self.model_arch      = "dinov2_linear"
        self.trained_classes = class_list

        print(f"[DINOv2] Done! Test: {test_acc}%  Time: {training_time}s  "
              f"Size: {model_size_kb:.0f} KB")

        return {
            "train_accuracy":  last_train_acc,
            "test_accuracy":   test_acc,
            "accuracy":        test_acc / 100,
            "model_type":      "dinov2_linear",
            "method":          "dinov2_linear",
            "training_time":   training_time,
            "model_size_kb":   round(model_size_kb, 1),
            "model_path":      weights_path,
            "epoch_history":   [],
        }

    # ── Load model (DINOv2 + legacy) ──────────────────────────────────────────

    def load_trained_model(self, model_path: str,
                            classes: list, num_classes: int):
        """
        Load weights.
        Detection order:
          1. DINOv2 linear  — filename ends with _image_linear.pth
                            OR companion _classes.json has method=dinov2_linear
          2. EfficientNetV2-S
          3. ResNet18 fallback
        """
        model_path_str = str(model_path)

        # -- 0. Check companion _classes.json for dinov2_linear tag -----------
        stem      = os.path.splitext(model_path_str)[0]
        cls_json  = stem + "_classes.json"
        if os.path.exists(cls_json):
            try:
                with open(cls_json) as f:
                    saved = json.load(f)
                if saved.get("method") == "dinov2_linear":
                    return self._load_dinov2_linear(
                        model_path_str, saved.get("classes", classes))
            except Exception:
                pass

        # -- 1. DINOv2 by filename suffix -------------------------------------
        if model_path_str.endswith("_image_linear.pth"):
            meta_p = model_path_str.replace(".pth", "_meta.json")
            saved  = {}
            if os.path.exists(meta_p):
                with open(meta_p) as f:
                    saved = json.load(f)
            return self._load_dinov2_linear(
                model_path_str, saved.get("classes", classes))

        # -- 2. Full image model (EfficientNetV2-S / ResNet18) ----------------
        state = None
        try:
            state = torch.load(model_path_str, map_location="cpu",
                               weights_only=True)
        except Exception as e:
            print(f"  ImageAgent could not read checkpoint: {e}")
            return

        # EfficientNetV2-S
        try:
            from torchvision.models import (efficientnet_v2_s,
                                             EfficientNet_V2_S_Weights)
            weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1
            model   = efficientnet_v2_s(weights=weights)
            if "classifier.1.weight" in state:
                num_classes = state["classifier.1.weight"].shape[0]
            model.classifier[1] = nn.Linear(
                model.classifier[1].in_features, num_classes)
            model.load_state_dict(state)
            model.eval()
            self.trained_model   = model
            self.trained_classes = classes
            self.model_arch      = "efficientnet_v2_s"
            print(f"  ImageAgent loaded EfficientNetV2-S — {num_classes} classes")
            return
        except Exception as e:
            print(f"  ImageAgent EfficientNetV2-S failed ({e}), "
                  f"falling back to ResNet18")

        # ResNet18
        try:
            import torchvision.models as tvm
            if "fc.weight" in state:
                num_classes = state["fc.weight"].shape[0]
            model    = tvm.resnet18(weights=None)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            model.load_state_dict(state)
            model.eval()
            self.trained_model   = model
            self.trained_classes = classes
            self.model_arch      = "resnet18_fallback"
            print(f"  ImageAgent loaded ResNet18 fallback — {num_classes} classes")
        except Exception as e:
            print(f"  ImageAgent model load failed: {e}")

    def _load_dinov2_linear(self, weights_path, classes):
        """Internal: load a saved DINOv2 linear head."""
        try:
            self._load_dinov2()
            state     = torch.load(weights_path, map_location="cpu",
                                   weights_only=True)
            embed_dim = state["classifier.weight"].shape[1]
            nc        = state["classifier.weight"].shape[0]
            linear    = DINOv2LinearClassifier(nc, embed_dim)
            linear.load_state_dict(state)
            linear.eval()
            self._dinov2_linear  = linear
            self.model_arch      = "dinov2_linear"
            self.trained_classes = list(classes) if classes else [str(i) for i in range(nc)]
            print(f"  ImageAgent loaded DINOv2 linear — {nc} classes")
        except Exception as e:
            print(f"  ImageAgent DINOv2 linear load failed: {e}")

    # ── Predict ───────────────────────────────────────────────────────────────

    def predict_image(self, image_path: str) -> dict:
        """Real inference — DINOv2 linear, EfficientNetV2-S, or ResNet18."""

        # DINOv2 path
        if self.model_arch == "dinov2_linear" and self._dinov2_linear is not None:
            try:
                from PIL import Image as PILImage
                img  = PILImage.open(image_path).convert("RGB")
                emb  = self._extract_embeddings([img])   # (1, 384)
                with torch.no_grad():
                    out   = self._dinov2_linear(emb)
                    probs = torch.softmax(out, dim=1)
                    conf  = float(probs.max())
                    idx   = int(probs.argmax())
                label = (self.trained_classes[idx]
                         if idx < len(self.trained_classes) else str(idx))
                return {
                    "label":      label,
                    "confidence": round(conf, 3),
                    "model_arch": self.model_arch,
                }
            except Exception as e:
                return {"label": "error", "confidence": 0.0,
                        "error": str(e), "fake": False}

        # Legacy full-model path (EfficientNetV2-S / ResNet18)
        if self.trained_model is None:
            return {
                "label":      "error",
                "confidence": 0.0,
                "error":      "No trained model. Run training first.",
                "fake":       False,
            }
        try:
            import torchvision.transforms as T
            from PIL import Image as PILImage
            tfm = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225]),
            ])
            img    = PILImage.open(image_path).convert("RGB")
            tensor = tfm(img).unsqueeze(0)
            with torch.no_grad():
                out   = self.trained_model(tensor)
                probs = torch.softmax(out, dim=1)
                conf  = float(probs.max())
                idx   = int(probs.argmax())
            label = (self.trained_classes[idx]
                     if idx < len(self.trained_classes) else str(idx))
            return {
                "label":      label,
                "confidence": round(conf, 3),
                "model_arch": self.model_arch,
            }
        except Exception as e:
            return {"label": "error", "confidence": 0.0,
                    "error": str(e), "fake": False}

    # ── Orchestrator run ──────────────────────────────────────────────────────

    def run(self, problem: str, image_data: str = "") -> dict:
        start = time.time()
        print(f"  ImageAgent running NAS for: {problem[:40]}")
        nas = run_quick_nas(num_classes=10)
        result = {
            "status":       "success",
            "agent":        self.NAME,
            "type":         "image_detection",
            "architecture": nas["architecture"],
            "parameters":   nas["parameters"],
            "search_time":  nas["search_time"],
            "elapsed":      round(time.time() - start, 2),
        }
        if image_data:
            try:
                import tempfile, base64, io
                from PIL import Image
                img_bytes = base64.b64decode(image_data.split(',')[1])
                img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
                tmp = tempfile.mktemp(suffix='.jpg')
                img.save(tmp)
                detections         = run_yolo_detection(tmp)
                result["boxes"]    = detections.get("boxes", [])
                result["yolo_ran"] = True
                if os.path.exists(tmp):
                    os.remove(tmp)
            except Exception as e:
                result["yolo_error"] = str(e)
        return result
