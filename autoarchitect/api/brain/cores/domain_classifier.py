"""
api/brain/cores/domain_classifier.py
Brain Core 2 — Domain Classifier (distilled from DeepSeek V3)

Qwen2.5-1.5B-Instruct + LoRA adapter fine-tuned on 672 examples.
Routes ML problems to the best AutoArchitect agent.
"""

import os
import json
import torch

ADAPTER_DEFAULT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(__file__)))),
    "models", "brain_cores", "core2_domain_classifier_adapter"
)

SYSTEM_PROMPT = (
    "You are AutoArchitect's Domain Classifier. Given an ML "
    "problem, recommend the best agent from: ImageAgent, "
    "TextAgent, TabularAgent, AudioAgent, MultimodalAgent, "
    "MedicalAgent, SecurityAgent. Output JSON with fields: "
    "primary_agent, secondary_agents, confidence, reasoning."
)

REQUIRED_FIELDS = [
    "primary_agent", "secondary_agents", "confidence", "reasoning",
]

VALID_AGENTS = {
    "ImageAgent", "TextAgent", "TabularAgent", "AudioAgent",
    "MultimodalAgent", "MedicalAgent", "SecurityAgent",
}


class DomainClassifier:
    """
    Brain Core 2 — distilled from DeepSeek V3.
    Routes ML problems to the appropriate AutoArchitect agent.

    First call to classify() triggers a one-time model load
    (~30s — downloads Qwen2.5-1.5B base if not cached).
    Subsequent calls are fast (<1s on CPU).
    """

    def __init__(self, adapter_path=None):
        self.adapter_path = adapter_path or ADAPTER_DEFAULT
        self.model        = None
        self.tokenizer    = None
        self._loaded      = False

    def _lazy_load(self):
        if self._loaded:
            return

        if not os.path.isdir(self.adapter_path):
            raise FileNotFoundError(
                f"Core 2 adapter not found: {self.adapter_path}\n"
                "Run Colab fine-tuning and place the adapter folder "
                "at models/brain_cores/core2_domain_classifier_adapter/"
            )

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel

            print("[Core2] Loading Qwen2.5-1.5B base model (one-time)...")
            base_model = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen2.5-1.5B-Instruct",
                torch_dtype=torch.float16,
                device_map="cpu",
            )

            print("[Core2] Loading LoRA adapter...")
            self.model = PeftModel.from_pretrained(base_model,
                                                    self.adapter_path)
            self.model.eval()

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.adapter_path)

            self._loaded = True
            print("[Core2] Ready.")

        except Exception as e:
            print(f"[Core2] Failed to load: {e}")
            raise

    def classify(self, problem_text: str) -> dict:
        """
        Classify an ML problem and recommend the best agent.

        Returns a dict with keys:
            primary_agent, secondary_agents, confidence, reasoning

        On failure returns:
            {"error": "<reason>", "raw_output": "<raw text>"}
        """
        self._lazy_load()

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"Problem: {problem_text}"},
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)

        inputs = self.tokenizer([prompt], return_tensors="pt")

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        decoded = self.tokenizer.decode(
            output[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        try:
            start = decoded.find('{')
            end   = decoded.rfind('}') + 1
            if start >= 0 and end > start:
                result = json.loads(decoded[start:end])
                return result
        except Exception:
            pass

        return {
            "error":      "Failed to parse model output",
            "raw_output": decoded,
        }

    def is_available(self) -> bool:
        """Return True if the adapter directory exists on disk."""
        return os.path.isdir(self.adapter_path)

    def validate_output(self, result: dict) -> bool:
        """Return True if result passes schema and agent checks."""
        if "error" in result:
            return False
        if not all(f in result for f in REQUIRED_FIELDS):
            return False
        if result.get("primary_agent") not in VALID_AGENTS:
            return False
        return True


# ── Module-level singleton ────────────────────────────────────────────────────

_instance: DomainClassifier = None


def get_domain_classifier() -> DomainClassifier:
    global _instance
    if _instance is None:
        _instance = DomainClassifier()
    return _instance
