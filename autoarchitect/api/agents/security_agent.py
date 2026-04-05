
# ============================================
# security_agent.py
# ============================================
import time
import torch
import torch.nn as nn
from api.nas_engine import run_quick_nas


class SecurityAgent:
    NAME = "Security Agent"
    THREAT_LEVELS = {
        0: ("Safe",              "OK",    "#00e676"),
        1: ("Low risk",          "LOW",   "#ffab00"),
        2: ("Suspicious",        "WARN",  "#ff9100"),
        3: ("High risk",         "HIGH",  "#f50057"),
        4: ("Attack detected",   "ATCK",  "#d50000"),
        5: ("Fraud detected",    "FRAUD", "#d50000"),
        6: ("Malware detected",  "MAL",   "#d50000"),
        7: ("Intrusion attempt", "INTR",  "#aa00ff"),
        8: ("Phishing detected", "PHSH",  "#f50057"),
        9: ("Critical threat",   "CRIT",  "#d50000"),
    }

    def __init__(self):
        self.trained_model   = None
        self.trained_classes = [v[0] for v in self.THREAT_LEVELS.values()]
        self.vocab           = {}
        print("  SecurityAgent loaded")

    def load_trained_model(self, model_path: str,
                            classes: list, num_classes: int):
        """Called after self_trainer finishes."""
        try:
            from api.nas_engine import DARTSNet
            model = DARTSNet(C=16, num_cells=3,
                             num_classes=num_classes)
            state = torch.load(model_path, map_location="cpu",
                               weights_only=True)
            model.load_state_dict(state)
            model.eval()
            self.trained_model   = model
            self.trained_classes = classes
            print(f"  SecurityAgent model loaded — {num_classes} classes")
        except Exception as e:
            print(f"  ⚠️  SecurityAgent model load failed: {e}")

    def run(self, problem: str, image_data: str = "") -> dict:
        start = time.time()
        print(f"  Running security NAS for: {problem[:40]}")
        nas = run_quick_nas(num_classes=10)
        return {
            "status":        "success",
            "agent":         self.NAME,
            "type":          "security_analysis",
            "architecture":  nas["architecture"],
            "parameters":    nas["parameters"],
            "search_time":   nas["search_time"],
            "threat_levels": {
                str(k): {"label": v[0], "code": v[1], "color": v[2]}
                for k, v in self.THREAT_LEVELS.items()
            },
            "elapsed": round(time.time() - start, 2),
        }

    def predict_threat(self, text: str) -> dict:
        """Real threat detection using trained DARTS model."""
        if self.trained_model is None:
            return {"label": "unknown", "confidence": 0.0}
        try:
            VOCAB_SIZE = 1000
            vec = torch.zeros(VOCAB_SIZE)
            for w in str(text).lower().split():
                if w in self.vocab:
                    vec[self.vocab[w]] += 1
            if vec.sum() > 0:
                vec = vec / vec.sum()
            pad = torch.zeros(3 * 32 * 32)
            pad[:VOCAB_SIZE] = vec[:3 * 32 * 32]
            tensor = pad.reshape(1, 3, 32, 32)
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
