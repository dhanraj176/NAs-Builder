"""
api/brain/distilled_brain.py
Unified brain that runs all 3 distilled cores.
Replaces MetaNet with frontier-distilled intelligence (DeepSeek V3 → Qwen2.5-1.5B).
"""


class DistilledBrain:
    """
    Unified brain that runs all 3 distilled cores.
    Replaces MetaNet with frontier-distilled intelligence.
    """

    def __init__(self):
        self.core1    = None  # lazy load
        self.core2    = None
        self.core3    = None
        self._loaded  = False

    def _lazy_load_all(self):
        """Load all 3 cores on first use."""
        if self._loaded:
            return

        from api.brain.cores.task_understander import TaskUnderstander
        from api.brain.cores.domain_classifier import DomainClassifier
        from api.brain.cores.architecture_advisor import ArchitectureAdvisor

        print("[DistilledBrain] Loading 3 cores...")
        self.core1 = TaskUnderstander()
        self.core2 = DomainClassifier()
        self.core3 = ArchitectureAdvisor()
        self._loaded = True
        print("[DistilledBrain] Ready.")

    def think(self, problem_text: str) -> dict:
        """
        Run all 3 cores in sequence and return unified analysis.

        Returns a dict with keys:
            understanding  -- Core 1 output (primary_intent, domain, complexity, ...)
            classification -- Core 2 output (primary_agent, secondary_agents, ...)
            architecture   -- Core 3 output (execution_mode, agent_topology, ...)
            source         -- "distilled_brain_v1"
            confidence     -- min confidence across all 3 cores
        """
        self._lazy_load_all()

        understanding  = self.core1.understand(problem_text)
        classification = self.core2.classify(problem_text)
        architecture   = self.core3.advise(problem_text)

        return {
            "understanding":  understanding,
            "classification": classification,
            "architecture":   architecture,
            "source":         "distilled_brain_v1",
            "confidence":     min(
                understanding.get("confidence",  0.9),
                classification.get("confidence", 1.0),
                architecture.get("confidence",   0.7),
            ),
        }

    def think_with_fallback(self, problem_text: str) -> dict:
        """
        Try distilled brain first, fall back to MetaNet if anything fails.
        """
        try:
            return self.think(problem_text)
        except Exception as e:
            print(f"[DistilledBrain] Failed: {e}")
            print("[DistilledBrain] Falling back to MetaNet")
            from api.brain.meta_learner import get_meta_learner
            metanet = get_meta_learner()
            return {
                "metanet_result":  metanet.predict(problem_text),
                "source":          "metanet_fallback",
                "fallback_reason": str(e),
            }

    def is_available(self) -> bool:
        """Return True if all 3 adapter directories exist on disk."""
        import os
        from api.brain.cores.task_understander import ADAPTER_DEFAULT as CORE1
        from api.brain.cores.domain_classifier  import ADAPTER_DEFAULT as CORE2
        from api.brain.cores.architecture_advisor import ADAPTER_DEFAULT as CORE3
        return all(os.path.isdir(p) for p in [CORE1, CORE2, CORE3])

    def validate_output(self, result: dict) -> bool:
        """Return True if result has all 3 brain sections."""
        if result.get("source") == "metanet_fallback":
            return "metanet_result" in result
        return all(k in result for k in
                   ["understanding", "classification", "architecture", "source"])


# ── Module-level singleton ────────────────────────────────────────────────────

_instance: DistilledBrain = None


def get_distilled_brain() -> DistilledBrain:
    global _instance
    if _instance is None:
        _instance = DistilledBrain()
    return _instance
