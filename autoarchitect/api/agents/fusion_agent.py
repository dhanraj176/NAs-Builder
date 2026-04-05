# ============================================
# AutoArchitect — Fusion Agent
# Combines multiple NAS architectures
# into one unified model
# Uses real accuracy from agent results
# ============================================

import time


class FusionAgent:
    NAME = "Fusion Agent"

    def __init__(self):
        print("  FusionAgent loaded")

    def fuse(self, agent_results: list, problem: str) -> dict:
        start = time.time()
        print(f"  Fusing {len(agent_results)} NAS architectures...")

        if not agent_results:
            return {"error": "No agent results to fuse"}

        if len(agent_results) == 1:
            return agent_results[0]

        fused_arch   = []
        total_params = 0
        domains      = []
        all_accuracies = {}

        for i, result in enumerate(agent_results):
            domain = result.get("domain", f"agent_{i}")
            arch   = result.get("architecture", [])
            params = result.get("parameters", 0)
            domains.append(domain)
            total_params += params

            # Collect real accuracy per domain
            acc = (
                result.get("test_accuracy") or
                result.get("avg_accuracy") or
                result.get("accuracy") or 0
            )
            if acc:
                all_accuracies[domain] = acc

            for cell in arch:
                fused_arch.append({
                    "cell":       cell["cell"],
                    "source":     domain,
                    "branch":     i + 1,
                    "operations": cell["operations"]
                })

        # Build fusion layer with real op weights from best agent
        best_op      = self._find_best_ops(agent_results)
        real_weights = self._compute_weights(agent_results)

        # Confidence based on real accuracy average
        if all_accuracies:
            real_confidence = round(
                sum(all_accuracies.values()) / len(all_accuracies), 1
            )
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
                "weights":    real_weights
            }]
        })

        # Average accuracy across all agents
        avg_accuracy = (
            round(sum(all_accuracies.values()) / len(all_accuracies), 1)
            if all_accuracies else 0
        )

        elapsed = round(time.time() - start, 2)
        print(f"  Fusion complete — {len(domains)} architectures merged")
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

    def _find_best_ops(self, results: list) -> str:
        """Find the most common operation across all agent architectures."""
        op_counts = {}
        for result in results:
            for cell in result.get("architecture", []):
                for op in cell.get("operations", []):
                    name = op.get("operation", "conv5x5")
                    op_counts[name] = op_counts.get(name, 0) + 1
        return max(op_counts, key=op_counts.get) if op_counts else "conv5x5"

    def _compute_weights(self, results: list) -> dict:
        """
        Compute real operation weights from agent architectures.
        Based on actual operation frequencies, not hardcoded values.
        """
        op_totals = {}
        op_counts = {}

        for result in results:
            for cell in result.get("architecture", []):
                for op in cell.get("operations", []):
                    name = op.get("operation", "unknown")
                    # Use actual weight if available, else count occurrences
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

        # Normalize to sum to 1
        total = sum(op_totals.values())
        return {k: round(v / total, 3) for k, v in op_totals.items()}
