"""
multimodal_agent.py -- AutoArchitect MultimodalAgent
CLIP-based zero-shot image+text classification and visual QA.
No training required -- CLIP generalizes across open-vocabulary domains.
"""

import numpy as np
from pathlib import Path


class MultimodalAgent:
    NAME = "Multimodal Agent"

    def __init__(self, name="MultimodalAgent"):
        self.name           = name
        self.clip_model     = None
        self.clip_processor = None
        self.device         = "cpu"
        print(f"  [MultimodalAgent] {name} loaded")

    # -- CLIP LOADER -----------------------------------------------------------

    def _lazy_load_clip(self):
        """Lazy-load CLIP once; returns True on success, False on any failure."""
        if self.clip_model is not None:
            return True
        try:
            from transformers import CLIPProcessor, CLIPModel
            print("  [MultimodalAgent] Loading CLIP (openai/clip-vit-base-patch32)...")
            self.clip_processor = CLIPProcessor.from_pretrained(
                "openai/clip-vit-base-patch32")
            self.clip_model = CLIPModel.from_pretrained(
                "openai/clip-vit-base-patch32")
            self.clip_model.eval()
            print("  [MultimodalAgent] CLIP loaded")
            return True
        except Exception as e:
            print(f"  [MultimodalAgent] CLIP load failed: {e}")
            return False

    # -- PREDICT ---------------------------------------------------------------

    def predict(self, image_path, text_query=None):
        """
        Encode image with CLIP; optionally compute cosine similarity to text_query.
        Returns {similarity, image_features_norm, text_features_norm} or error dict.
        """
        if not self._lazy_load_clip():
            return {
                "label":      "error",
                "confidence": 0.0,
                "error":      "CLIP unavailable (no internet or transformers not installed)",
                "fake":       False,
            }
        try:
            import torch
            from PIL import Image

            image = Image.open(image_path).convert("RGB")

            if text_query:
                inputs = self.clip_processor(
                    text=[text_query], images=image,
                    return_tensors="pt", padding=True)
                with torch.no_grad():
                    outputs  = self.clip_model(**inputs)
                    img_feat = outputs.image_embeds
                    txt_feat = outputs.text_embeds
                    sim      = torch.nn.functional.cosine_similarity(
                        img_feat, txt_feat).item()
                return {
                    "similarity":          round(float(sim), 4),
                    "image_features_norm": round(float(img_feat.norm().item()), 4),
                    "text_features_norm":  round(float(txt_feat.norm().item()), 4),
                    "agent_used":          "MultimodalAgent",
                }
            else:
                inputs = self.clip_processor(images=image, return_tensors="pt")
                with torch.no_grad():
                    img_feat = self.clip_model.get_image_features(**inputs)
                return {
                    "image_features_norm": round(float(img_feat.norm().item()), 4),
                    "agent_used":          "MultimodalAgent",
                }
        except Exception as e:
            return {"label": "error", "confidence": 0.0,
                    "error": str(e), "fake": False}

    # -- ZERO-SHOT CLASSIFY ----------------------------------------------------

    def classify_with_labels(self, image_path, candidate_labels):
        """
        Zero-shot CLIP classification over candidate_labels.
        candidate_labels: e.g. ["cat", "dog", "bird"]
        Returns {label, confidence, all_scores, agent_used} or error dict.
        """
        if not self._lazy_load_clip():
            return {
                "label":      "error",
                "confidence": 0.0,
                "error":      "CLIP unavailable (no internet or transformers not installed)",
                "fake":       False,
            }
        try:
            import torch
            from PIL import Image

            image  = Image.open(image_path).convert("RGB")
            inputs = self.clip_processor(
                text=candidate_labels, images=image,
                return_tensors="pt", padding=True)

            with torch.no_grad():
                outputs          = self.clip_model(**inputs)
                logits_per_image = outputs.logits_per_image  # (1, num_labels)
                probs            = logits_per_image.softmax(dim=1)[0].cpu().numpy()

            idx   = int(np.argmax(probs))
            label = candidate_labels[idx]
            conf  = float(probs[idx])

            all_scores = {lbl: round(float(p), 4)
                          for lbl, p in zip(candidate_labels, probs)}

            return {
                "label":      label,
                "confidence": round(conf, 4),
                "all_scores": all_scores,
                "agent_used": "MultimodalAgent",
            }
        except Exception as e:
            return {"label": "error", "confidence": 0.0,
                    "error": str(e), "fake": False}

    # -- VQA -------------------------------------------------------------------

    def visual_question_answer(self, image_path, question):
        """
        Answer a yes/no or short-answer question about an image using CLIP.
        Yes/no questions are detected automatically; otherwise key words become candidates.
        """
        question_lower = question.lower().rstrip("?").strip()

        yn_starters = ("is ", "does ", "are ", "do ", "was ", "were ",
                        "has ", "have ", "can ", "will ", "should ")
        if any(question_lower.startswith(s) for s in yn_starters):
            candidates = ["yes", "no"]
        else:
            stop = {"what", "which", "how", "the", "this", "that", "there",
                    "when", "where", "why", "who", "show", "tell", "about"}
            words = [w.strip(".,?!") for w in question.split()
                     if w.lower() not in stop and len(w) > 3]
            candidates = list(dict.fromkeys(words))[:4]
            if len(candidates) < 2:
                candidates = ["yes", "no"]

        result = self.classify_with_labels(image_path, candidates)
        result["question"] = question
        return result

    # -- ORCHESTRATOR STUB -----------------------------------------------------

    def run(self, problem, image_data=""):
        return {
            "status":       "success",
            "agent":        self.NAME,
            "type":         "multimodal_classification",
            "model_loaded": self.clip_model is not None,
            "message": (
                f"MultimodalAgent ready. "
                f"Call classify_with_labels(image_path, labels). "
                f"Problem: {problem[:60]}"
            ),
        }
