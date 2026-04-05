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
        self.trained_model  = None
        self.trained_classes = []
        print("  🖼️  ImageAgent loaded")

    def load_trained_model(self, model_path: str,
                            classes: list, num_classes: int):
        """Called after self_trainer finishes."""
        try:
            import torchvision.models as models
            model    = models.resnet18(weights=None)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            state    = torch.load(model_path, map_location="cpu",
                                  weights_only=True)
            model.load_state_dict(state)
            model.eval()
            self.trained_model   = model
            self.trained_classes = classes
            print(f"  🖼️  ImageAgent model loaded — {num_classes} classes")
        except Exception as e:
            print(f"  ⚠️  ImageAgent model load failed: {e}")

    def run(self, problem: str, image_data: str = "") -> dict:
        start = time.time()
        print(f"  🖼️  Running image NAS for: {problem[:40]}")
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
                if os.path.exists(tmp): os.remove(tmp)
            except Exception as e:
                result["yolo_error"] = str(e)
        return result

    def predict_image(self, image_path: str) -> dict:
        """Real inference using trained model."""
        if self.trained_model is None:
            return {"label": "unknown", "confidence": 0.0}
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
            return {"label": label, "confidence": round(conf, 3)}
        except Exception as e:
            return {"label": "error", "confidence": 0.0, "error": str(e)}
