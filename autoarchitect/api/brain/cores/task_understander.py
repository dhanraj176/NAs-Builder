"""
api/brain/cores/task_understander.py
Brain Core 1 — Task Understander (distilled from DeepSeek V3)

Qwen2.5-1.5B-Instruct + LoRA adapter fine-tuned on 672 examples.
Parses ML problem descriptions into structured JSON analysis.
"""

import os
import json
import torch

ADAPTER_DEFAULT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(__file__)))),
    "models", "brain_cores", "core1_task_understander_adapter"
)

SYSTEM_PROMPT = (
    "You are AutoArchitect's Task Understander. "
    "Given an ML problem description, output structured "
    "JSON analysis. The domain field MUST be exactly one "
    "of these 7 values: image, text, tabular, audio, "
    "multimodal, medical, security. Required fields: "
    "primary_intent, domain, complexity, "
    "real_time_required, multi_modal, key_entities."
)

REQUIRED_FIELDS = [
    "primary_intent", "domain", "complexity",
    "real_time_required", "multi_modal", "key_entities",
]

VALID_DOMAINS = {
    "image", "text", "tabular", "audio",
    "multimodal", "medical", "security",
}


class TaskUnderstander:
    """
    Brain Core 1 — distilled from DeepSeek V3.
    Understands ML problem intent and outputs structured analysis.

    First call to understand() triggers a one-time model load
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
                f"Core 1 adapter not found: {self.adapter_path}\n"
                "Run Colab fine-tuning and place the adapter folder "
                "at models/brain_cores/core1_task_understander_adapter/"
            )

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel

            print("[Core1] Loading Qwen2.5-1.5B base model (one-time)...")
            base_model = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen2.5-1.5B-Instruct",
                torch_dtype=torch.float16,
                device_map="cpu",
            )

            print("[Core1] Loading LoRA adapter...")
            self.model = PeftModel.from_pretrained(base_model,
                                                    self.adapter_path)
            self.model.eval()

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.adapter_path)

            self._loaded = True
            print("[Core1] Ready.")

        except Exception as e:
            print(f"[Core1] Failed to load: {e}")
            raise

    def understand(self, problem_text: str) -> dict:
        """
        Analyse an ML problem description.

        Returns a dict with keys:
            primary_intent, domain, complexity,
            real_time_required, multi_modal, key_entities

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
                # Normalise domain to lowercase
                if "domain" in result:
                    result["domain"] = str(result["domain"]).lower()
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
        """Return True if result passes schema and domain checks."""
        if "error" in result:
            return False
        if not all(f in result for f in REQUIRED_FIELDS):
            return False
        if result.get("domain", "").lower() not in VALID_DOMAINS:
            return False
        return True


# ── Module-level singleton ────────────────────────────────────────────────────

_instance: TaskUnderstander = None


def get_task_understander() -> TaskUnderstander:
    global _instance
    if _instance is None:
        _instance = TaskUnderstander()
    return _instance
