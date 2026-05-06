import os
import json
import time
import torch
from api.nas_engine import run_quick_nas, DARTSNet

VOCAB_SIZE = 1000


class TextAgent:
    NAME = "Text Agent"
    CLASSES = [
        "Positive", "Negative", "Neutral", "Spam",
        "Urgent", "Important", "Low priority",
        "High risk", "Flagged", "Safe"
    ]

    def __init__(self):
        self.trained_model   = None
        self.trained_classes = self.CLASSES
        self.vocab           = {}
        self._vocab_path     = None
        print("  TextAgent loaded")

    def load_trained_model(self, model_path: str,
                            classes: list, num_classes: int):
        """Called after self_trainer finishes."""
        try:
            model = DARTSNet(C=16, num_cells=3, num_classes=num_classes)
            state = torch.load(model_path, map_location="cpu",
                               weights_only=True)
            model.load_state_dict(state)
            model.eval()
            self.trained_model   = model
            self.trained_classes = classes

            # Load matching vocab: {hash}_text_vocab.json
            h                = os.path.splitext(os.path.basename(model_path))[0].split('_')[0]
            self._vocab_path = os.path.join(os.path.dirname(os.path.abspath(model_path)),
                                            f"{h}_text_vocab.json")
            if os.path.exists(self._vocab_path):
                with open(self._vocab_path) as vf:
                    self.vocab = json.load(vf)
                print(f"  TextAgent vocab loaded — {len(self.vocab)} words")
            else:
                self.vocab = {}
                print(f"  TextAgent vocab not found: {self._vocab_path}")

            print(f"  TextAgent model loaded — {num_classes} classes")
        except Exception as e:
            print(f"  TextAgent model load failed: {e}")

    def run(self, problem: str, image_data: str = "") -> dict:
        start = time.time()
        print(f"  TextAgent running NAS for: {problem[:40]}")
        nas = run_quick_nas(num_classes=10)
        return {
            "status":       "success",
            "agent":        self.NAME,
            "type":         "text_classification",
            "architecture": nas["architecture"],
            "parameters":   nas["parameters"],
            "search_time":  nas["search_time"],
            "dataset":      "IMDB / custom",
            "classes":      self.trained_classes,
            "elapsed":      round(time.time() - start, 2),
        }

    def predict(self, text: str) -> dict:
        """Classify text using the trained model."""
        if self.trained_model is None:
            return {
                "label":      "error",
                "confidence": 0.0,
                "error":      "No trained model. Run training first.",
                "fake":       False,
            }

        # Retry vocab load if empty but path is known
        if len(self.vocab) == 0 and self._vocab_path and os.path.exists(self._vocab_path):
            with open(self._vocab_path) as vf:
                self.vocab = json.load(vf)

        if len(self.vocab) == 0:
            return {
                "label":      "error",
                "confidence": 0.0,
                "error":      "Vocabulary not loaded. Run training first.",
                "fake":       False,
            }

        try:
            vec = torch.zeros(VOCAB_SIZE)
            for w in str(text).lower().split():
                if w in self.vocab:
                    idx = self.vocab[w]
                    if idx < VOCAB_SIZE:
                        vec[idx] += 1
            if vec.sum() > 0:
                vec = vec / vec.sum()
            pad = torch.zeros(3 * 32 * 32)
            pad[:VOCAB_SIZE] = vec[:3 * 32 * 32]
            tensor = pad.reshape(1, 3, 32, 32)

            with torch.no_grad():
                out   = self.trained_model(tensor)
                probs = torch.softmax(out, dim=1)
                k     = min(3, len(self.trained_classes))
                top_v, top_i = torch.topk(probs[0], k)

            best_idx = int(probs.argmax())
            label    = (self.trained_classes[best_idx]
                        if best_idx < len(self.trained_classes) else str(best_idx))
            conf     = round(float(probs[0, best_idx]), 3)

            top3_predictions = [
                {
                    "label":      (self.trained_classes[int(i)]
                                   if int(i) < len(self.trained_classes) else str(int(i))),
                    "confidence": round(float(v), 3),
                }
                for i, v in zip(top_i.tolist(), top_v.tolist())
            ]

            return {
                "label":            label,
                "confidence":       conf,
                "top3_predictions": top3_predictions,
            }
        except Exception as e:
            return {"label": "error", "confidence": 0.0,
                    "error": str(e), "fake": False}
