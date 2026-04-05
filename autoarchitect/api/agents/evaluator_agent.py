# ============================================
# AutoArchitect — Evaluator Agent
# Uses REAL accuracy from self_trainer
# Falls back to architecture scoring only
# if no real accuracy available
# ============================================

import time

class EvaluatorAgent:
    NAME      = "Evaluator Agent"
    EXCELLENT = 85
    GOOD      = 70

    def __init__(self):
        print("  EvaluatorAgent loaded")

    def evaluate(self, fusion_result: dict, problem: str) -> dict:
        start = time.time()
        print(f"  Evaluating fused model...")

        arch    = fusion_result.get("fused_architecture",
                  fusion_result.get("architecture", []))
        domains = fusion_result.get("domains_combined", ["unknown"])
        params  = fusion_result.get("total_parameters",
                  fusion_result.get("parameters", 0))

        # ── Use REAL accuracy if available ────────────────────────────
        real_accuracy = (
            fusion_result.get("test_accuracy") or
            fusion_result.get("avg_accuracy")  or
            fusion_result.get("train_accuracy")
        )

        scores = self._score_architecture(arch, domains, params, real_accuracy)

        weaknesses = self._find_weaknesses(scores)
        feedback   = self._generate_feedback(weaknesses, domains)
        avg_score  = round(sum(scores.values()) / len(scores), 1)

        # If real accuracy exists, weight it heavily in final score
        if real_accuracy and real_accuracy > 0:
            avg_score = round((avg_score * 0.3) + (real_accuracy * 0.7), 1)
            print(f"  Real accuracy: {real_accuracy}% — weighted into score")

        avg_score = min(100, avg_score)
        verdict   = (
            "excellent"         if avg_score >= self.EXCELLENT else
            "good"              if avg_score >= self.GOOD      else
            "needs_improvement"
        )

        elapsed = round(time.time() - start, 2)
        print(f"  Score: {avg_score}% ({verdict})")

        return {
            "status":         "success",
            "agent":          self.NAME,
            "type":           "evaluation",
            "scores":         scores,
            "avg_score":      avg_score,
            "verdict":        verdict,
            "weaknesses":     weaknesses,
            "feedback":       feedback,
            "real_accuracy":  real_accuracy,
            "ready_to_cache": avg_score >= self.GOOD,
            "elapsed":        elapsed,
            "message": (
                f"Score: {avg_score}% — {verdict}. "
                f"{'Ready to cache.' if avg_score >= self.GOOD else 'Needs refinement.'}"
            )
        }

    def _score_architecture(self, arch, domains, params,
                             real_accuracy=None) -> dict:
        # If real accuracy available use it as base for all scores
        if real_accuracy and real_accuracy > 0:
            base = real_accuracy
            return {
                "accuracy":   min(100, base),
                "coverage":   min(100, 70 + len(domains) * 15),
                "depth":      min(100, 60 + len(arch) * 8),
                "diversity":  min(100, base * 0.95),
                "efficiency": min(100, base * 0.9 if params < 500000 else base * 0.8),
            }

        # Fallback — architecture-only scoring
        complexity  = (
            95 if params < 150000 else
            80 if params < 500000 else
            65 if params < 1000000 else 50
        )
        fusion_bonus = 10 if len(domains) > 1 else 0
        ops = set()
        for cell in arch:
            for op in cell.get("operations", []):
                ops.add(op.get("operation", ""))
        return {
            "complexity": min(100, complexity + fusion_bonus),
            "coverage":   min(100, 70 + len(domains) * 15),
            "depth":      min(100, 60 + len(arch) * 8),
            "diversity":  min(100, 50 + len(ops) * 12),
            "innovation": min(100, 75 + fusion_bonus + len(domains) * 5),
        }

    def _find_weaknesses(self, scores: dict) -> list:
        return [
            {"metric": m, "score": s, "description": self._desc(m)}
            for m, s in scores.items() if s < self.GOOD
        ]

    def _desc(self, metric: str) -> str:
        return {
            "accuracy":   "Model accuracy below target — more training data needed",
            "complexity": "Model too large — NAS should find smaller ops",
            "coverage":   "Limited domain coverage — add more agents",
            "depth":      "Too shallow — needs more cells",
            "diversity":  "Low op diversity — NAS in local optimum",
            "innovation": "Follows known patterns too closely",
            "efficiency": "Model size vs accuracy tradeoff could improve",
        }.get(metric, "Needs improvement")

    def _generate_feedback(self, weaknesses, domains) -> list:
        feedback = [
            {
                "accuracy":   "Upload more labeled data for this domain",
                "complexity": "Agents: prioritize skip and avgpool ops",
                "coverage":   f"Consider adding agents for: {domains}",
                "depth":      "All agents: increase num_cells to 5",
                "diversity":  "NAS agents: increase arch_weight lr",
                "innovation": "Try deeper search space exploration",
                "efficiency": "Consider pruning or quantization",
            }.get(w["metric"], "Review architecture")
            for w in weaknesses
        ]
        return feedback if feedback else ["Architecture optimal — ready for deployment"]
