# ============================================
# evaluator_agent.py
# ============================================
import time


class EvaluatorAgent:
    NAME      = "Evaluator Agent"
    EXCELLENT = 85
    GOOD      = 70

    def __init__(self):
        print("  EvaluatorAgent loaded")

    # ── PUBLIC dispatch — detect call style ───────────────────────────────────

    def evaluate(self, first_arg, second_arg=None):
        """
        Two calling conventions:
          evaluate(predictions: list, ground_truth: list)
              -> real ML metrics via sklearn
          evaluate(fusion_result: dict, problem: str)
              -> architecture quality scoring (legacy, used by orchestrator)
        """
        if isinstance(second_arg, list):
            return self._ml_evaluate(first_arg, second_arg)
        return self._evaluate_architecture(first_arg, second_arg)

    # ── NEW: real ML metrics ──────────────────────────────────────────────────

    def _ml_evaluate(self, predictions, ground_truth: list) -> dict:
        """Compute sklearn metrics on a batch of predictions vs ground truth."""
        from sklearn.metrics import (precision_score, recall_score,
                                     f1_score, accuracy_score)

        # Accept predictions as labels OR as dicts {label, confidence}
        if predictions and isinstance(predictions[0], dict):
            pred_labels   = [p.get("label", str(p)) for p in predictions]
            confidences   = [float(p.get("confidence", 1.0)) for p in predictions]
        else:
            pred_labels   = list(predictions)
            confidences   = [1.0] * len(pred_labels)

        uncertain_count = sum(1 for c in confidences if c < 0.6)

        acc  = accuracy_score(ground_truth, pred_labels)
        prec = precision_score(ground_truth, pred_labels,
                               average="weighted", zero_division=0)
        rec  = recall_score(ground_truth, pred_labels,
                            average="weighted", zero_division=0)
        f1   = f1_score(ground_truth, pred_labels,
                        average="weighted", zero_division=0)

        avg_conf         = sum(confidences) / len(confidences) if confidences else 0
        calibration_err  = abs(avg_conf - acc)
        quality_score    = round(f1 * 0.7 + (1.0 - calibration_err) * 0.3, 3)

        verdict = (
            "excellent"         if f1 >= 0.85 else
            "good"              if f1 >= 0.70 else
            "needs_improvement"
        )

        print(f"  ML evaluate: acc={acc:.3f} f1={f1:.3f} quality={quality_score:.3f}")

        return {
            "accuracy":              round(acc,  3),
            "precision":             round(prec, 3),
            "recall":                round(rec,  3),
            "f1":                    round(f1,   3),
            "uncertain_predictions": uncertain_count,
            "quality_score":         quality_score,
            "verdict":               verdict,
            "n_samples":             len(pred_labels),
        }

    def validate_single(self, prediction: dict) -> dict:
        """Runtime quality check on one prediction — no ground truth needed."""
        conf  = float(prediction.get("confidence", 0.0))
        label = prediction.get("label", "unknown")
        fake  = prediction.get("fake", None)

        quality      = ("high" if conf >= 0.8 else
                        "medium" if conf >= 0.6 else "low")
        flag         = conf < 0.6 or fake is True

        return {
            "label":          label,
            "confidence":     conf,
            "quality":        quality,
            "flag_for_review": flag,
            "reason": (
                "Fake/error prediction" if fake is True else
                "Low confidence — human review recommended" if conf < 0.6 else
                "Prediction within acceptable range"
            ),
        }

    # ── LEGACY: architecture quality scoring (used by orchestrator) ───────────

    def _evaluate_architecture(self, fusion_result: dict,
                                problem: str = None) -> dict:
        start = time.time()
        print(f"  Evaluating fused model...")

        arch    = fusion_result.get("fused_architecture",
                  fusion_result.get("architecture", []))
        domains = fusion_result.get("domains_combined", ["unknown"])
        params  = fusion_result.get("total_parameters",
                  fusion_result.get("parameters", 0))

        real_accuracy = (
            fusion_result.get("test_accuracy") or
            fusion_result.get("avg_accuracy")  or
            fusion_result.get("train_accuracy")
        )

        scores = self._score_architecture(arch, domains, params, real_accuracy)

        weaknesses = self._find_weaknesses(scores)
        feedback   = self._generate_feedback(weaknesses, domains)
        avg_score  = round(sum(scores.values()) / len(scores), 1)

        if real_accuracy and real_accuracy > 0:
            avg_score = round((avg_score * 0.3) + (real_accuracy * 0.7), 1)
            print(f"  Real accuracy: {real_accuracy}% -- weighted into score")

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
                f"Score: {avg_score}% -- {verdict}. "
                f"{'Ready to cache.' if avg_score >= self.GOOD else 'Needs refinement.'}"
            ),
        }

    def _score_architecture(self, arch, domains, params, real_accuracy=None) -> dict:
        if real_accuracy and real_accuracy > 0:
            base = real_accuracy
            return {
                "accuracy":   min(100, base),
                "coverage":   min(100, 70 + len(domains) * 15),
                "depth":      min(100, 60 + len(arch) * 8),
                "diversity":  min(100, base * 0.95),
                "efficiency": min(100, base * 0.9
                               if params < 500000 else base * 0.8),
            }
        complexity   = (95 if params < 150000 else
                        80 if params < 500000 else
                        65 if params < 1000000 else 50)
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
            "accuracy":   "Model accuracy below target -- more training data needed",
            "complexity": "Model too large -- NAS should find smaller ops",
            "coverage":   "Limited domain coverage -- add more agents",
            "depth":      "Too shallow -- needs more cells",
            "diversity":  "Low op diversity -- NAS in local optimum",
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
        return feedback if feedback else ["Architecture optimal -- ready for deployment"]
