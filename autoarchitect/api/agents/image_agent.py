# ============================================
# image_agent.py
# ============================================
import time
import torch
import torch.nn as nn
from api.nas_engine   import run_quick_nas
from api.auto_trainer import run_yolo_detection


class ImageAgent:
    NAME = "Image Agent"

    def __init__(self):
        self.trained_model   = None
        self.trained_classes = []
        self.model_arch      = None   # "efficientnet_v2_s" or "resnet18_fallback"
        print("  ImageAgent loaded")

    def load_trained_model(self, model_path: str,
                            classes: list, num_classes: int):
        """Load weights — tries EfficientNetV2-S first, falls back to ResNet18."""
        state = None
        try:
            state = torch.load(model_path, map_location="cpu",
                               weights_only=True)
        except Exception as e:
            print(f"  ImageAgent could not read checkpoint: {e}")
            return

        # -- Primary: EfficientNetV2-S ----------------------------------------
        try:
            from torchvision.models import (efficientnet_v2_s,
                                             EfficientNet_V2_S_Weights)
            weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1
            model   = efficientnet_v2_s(weights=weights)
            # Derive actual num_classes from saved weights when available
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
            print(f"  ImageAgent EfficientNetV2-S failed ({e}), falling back to ResNet18")

        # -- Fallback: ResNet18 -----------------------------------------------
        try:
            import torchvision.models as models
            # Derive actual num_classes from saved weights
            if "fc.weight" in state:
                num_classes = state["fc.weight"].shape[0]
            model    = models.resnet18(weights=None)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            model.load_state_dict(state)
            model.eval()
            self.trained_model   = model
            self.trained_classes = classes
            self.model_arch      = "resnet18_fallback"
            print(f"  ImageAgent loaded ResNet18 fallback — {num_classes} classes")
        except Exception as e:
            print(f"  ImageAgent model load failed: {e}")

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
                import tempfile, base64, os, io
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

    def predict_image(self, image_path: str) -> dict:
        """Real inference using trained model (EfficientNetV2-S or ResNet18 fallback)."""
        if self.trained_model is None:
            return {
                "label":      "error",
                "confidence": 0.0,
                "error":      "No trained model. Run training first.",
                "fake":       False,
            }
        try:
            import torchvision.transforms as T
            from PIL import Image
            tfm = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225]),
            ])
            img    = Image.open(image_path).convert("RGB")
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
