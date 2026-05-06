# ============================================
# fusion_agent.py
# ============================================
import time


class FusionAgent:
    NAME = "Fusion Agent"

    def __init__(self):
        # Per-agent weight cache; updated externally when accuracy feedback arrives
        self._weight_cache = {}
        print("  FusionAgent loaded")

    # ── NEW: prediction-level weighted confidence fusion ──────────────────────

    def fuse(self, agent_results: list, weights: dict = None) -> dict:
        """
        Fuse classification predictions from multiple agents.

        agent_results: [{label, confidence, agent_name}, ...]
        weights:       optional {agent_name: float} — uses cache then equal fallback
        """
        if not agent_results:
            return {"error": "No agent results to fuse"}

        if len(agent_results) == 1:
            r = dict(agent_results[0])
            r["fusion_method"] = "passthrough"
            r.setdefault("contributing_agents", [r.get("agent_name", "unknown")])
            r.setdefault("weights_used", {})
            return r

        agent_names = [r.get("agent_name", f"agent_{i}")
                       for i, r in enumerate(agent_results)]

        # Resolve weights: explicit > cache > equal
        if weights is None:
            equal = 1.0 / len(agent_results)
            weights = {n: self._weight_cache.get(n, equal) for n in agent_names}

        total_w = sum(weights.get(n, 1.0) for n in agent_names)
        if total_w == 0:
            total_w = len(agent_results)

        # Accumulate weighted confidence per label
        label_scores = {}
        label_agents = {}
        for r in agent_results:
            label = r.get("label", "unknown")
            name  = r.get("agent_name", "unknown")
            conf  = float(r.get("confidence", 0.0))
            w     = weights.get(name, 1.0) / total_w
            label_scores[label] = label_scores.get(label, 0.0) + conf * w
            label_agents.setdefault(label, []).append(name)

        # Determine winner
        winner       = max(label_scores, key=label_scores.get)
        winner_score = round(label_scores[winner], 3)

        # Flag uncertain when top-2 scores are within 5 pp
        sorted_scores = sorted(label_scores.values(), reverse=True)
        uncertain     = (len(sorted_scores) >= 2 and
                         (sorted_scores[0] - sorted_scores[1]) < 0.05)

        print(f"  Fusion: winner='{winner}' conf={winner_score}"
              f"{' [UNCERTAIN]' if uncertain else ''}")

        return {
            "label":               winner,
            "confidence":          winner_score,
            "contributing_agents": label_agents.get(winner, []),
            "weights_used":        weights,
            "fusion_method":       "weighted_confidence",
            "uncertain":           uncertain,
            "all_label_scores":    {k: round(v, 3) for k, v in label_scores.items()},
        }

    def update_weights(self, agent_name: str, weight: float):
        """Feed accuracy feedback back into the weight cache."""
        self._weight_cache[agent_name] = weight

    # ── LEGACY: architecture-level fusion (called by orchestrator) ────────────

    def fuse_architectures(self, agent_results: list, problem: str) -> dict:
        """Merge NAS architecture dicts from multiple domain agents."""
        start = time.time()
        print(f"  Fusing {len(agent_results)} NAS architectures...")

        if not agent_results:
            return {"error": "No agent results to fuse"}

        if len(agent_results) == 1:
            return agent_results[0]

        fused_arch     = []
        total_params   = 0
        domains        = []
        all_accuracies = {}

        for i, result in enumerate(agent_results):
            domain = result.get("domain", f"agent_{i}")
            arch   = result.get("architecture", [])
            params = result.get("parameters", 0)
            domains.append(domain)
            total_params += params

            acc = (result.get("test_accuracy") or
                   result.get("avg_accuracy")  or
                   result.get("accuracy") or 0)
            if acc:
                all_accuracies[domain] = acc

            for cell in arch:
                fused_arch.append({
                    "cell":       cell["cell"],
                    "source":     domain,
                    "branch":     i + 1,
                    "operations": cell["operations"],
                })

        best_op      = self._find_best_ops(agent_results)
        real_weights = self._compute_weights(agent_results)

        if all_accuracies:
            real_confidence = round(
                sum(all_accuracies.values()) / len(all_accuracies), 1)
        else:
            real_confidence = 0.0

        fused_arch.append({
            "cell":       len(fused_arch) + 1,
            "source":     "fusion",
            "branch":     0,
            "operations": [{
                "operation":  best_op,
                "confidence": real_confidence,
                "fusion":     True,
                "combines":   domains,
                "weights":    real_weights,
            }],
        })

        avg_accuracy = (
            round(sum(all_accuracies.values()) / len(all_accuracies), 1)
            if all_accuracies else 0
        )

        elapsed = round(time.time() - start, 2)
        print(f"  Fusion complete -- {len(domains)} architectures merged")
        if avg_accuracy:
            print(f"  Average real accuracy: {avg_accuracy}%")

        return {
            "status":             "success",
            "agent":              self.NAME,
            "type":               "multi_agent_fusion",
            "architecture":       fused_arch,
            "fused_architecture": fused_arch,
            "domains_combined":   domains,
            "parameters":         total_params,
            "total_parameters":   total_params,
            "fusion_strategy":    "parallel_branch_fusion",
            "all_accuracies":     all_accuracies,
            "avg_accuracy":       avg_accuracy,
            "search_time":        elapsed,
            "elapsed":            elapsed,
            "message": (
                f"Fused {len(domains)} NAS architectures. "
                f"Average accuracy: {avg_accuracy}%"
                if avg_accuracy else
                f"Fused {len(domains)} NAS architectures."
            ),
        }

    # ── helpers (used by fuse_architectures) ─────────────────────────────────

    def _find_best_ops(self, results: list) -> str:
        op_counts = {}
        for result in results:
            for cell in result.get("architecture", []):
                for op in cell.get("operations", []):
                    name = op.get("operation", "conv5x5")
                    op_counts[name] = op_counts.get(name, 0) + 1
        return max(op_counts, key=op_counts.get) if op_counts else "conv5x5"

    def _compute_weights(self, results: list) -> dict:
        op_totals = {}
        op_counts = {}
        for result in results:
            for cell in result.get("architecture", []):
                for op in cell.get("operations", []):
                    name    = op.get("operation", "unknown")
                    weights = op.get("weights", {})
                    if weights:
                        for op_name, w in weights.items():
                            op_totals[op_name] = op_totals.get(op_name, 0) + w
                            op_counts[op_name] = op_counts.get(op_name, 0) + 1
                    else:
                        op_totals[name] = op_totals.get(name, 0) + 1
                        op_counts[name] = op_counts.get(name, 0) + 1

        if not op_totals:
            return {"conv5x5": 0.6, "conv3x3": 0.2, "skip": 0.1, "avgpool": 0.1}

        total = sum(op_totals.values())
        return {k: round(v / total, 3) for k, v in op_totals.items()}
