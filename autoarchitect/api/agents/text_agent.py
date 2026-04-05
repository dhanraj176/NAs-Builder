import time
from api.nas_engine import run_quick_nas

class TextAgent:
    NAME = "Text Agent"
    CLASSES = [
        "Positive", "Negative", "Neutral", "Spam",
        "Urgent", "Important", "Low priority",
        "High risk", "Flagged", "Safe"
    ]

    def __init__(self):
        print("  📝  TextAgent loaded")

    def run(self, problem: str, image_data: str = "") -> dict:
        start = time.time()
        print(f"  📝  Running text NAS for: {problem[:40]}")
        nas = run_quick_nas(num_classes=10)
        return {
            "status":       "success",
            "agent":        self.NAME,
            "type":         "text_classification",
            "architecture": nas["architecture"],
            "parameters":   nas["parameters"],
            "search_time":  nas["search_time"],
            "dataset":      "IMDB / custom",
            "classes":      self.CLASSES,
            "elapsed":      round(time.time() - start, 2),
        }
