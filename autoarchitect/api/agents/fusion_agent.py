# ============================================
# fusion_agent.py
# ============================================
import os
import json
import re
import time

BRAIN_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "brain_data",
)
_TOPOLOGY_HISTORY_FILE = os.path.join(BRAIN_DIR, "topology_history.json")
_FUSION_WEIGHTS_FILE   = os.path.join(BRAIN_DIR, "fusion_weights.json")

# Utility/terminal agent types — excluded when determining domain or assigning
# per-agent reliability credit (they don't classify, they transform or output).
_TERMINAL_AGENTS = {"report", "severity", "optimizer"}


def _class_to_weight_key(class_name: str) -> str:
    """Convert CamelCase class name to snake_case weight key.

    'ImageAgent'     -> 'image_agent'
    'MultimodalAgent'-> 'multimodal_agent'
    'DynamicAgent'   -> 'dynamic_agent'
    """
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", class_name)
    return s.lower()


class FusionAgent:
    NAME = "Fusion Agent"

    def __init__(self):
        # Per-agent weight cache; updated externally when accuracy feedback arrives
        self._weight_cache  = {}
        # Learned per-domain weights loaded from fusion_weights.json
        self._weights_cache: dict = {}
        print("  FusionAgent loaded")
        self.learn_weights_from_cache()

    # ── Learned weight management ─────────────────────────────────────────────

    def learn_weights_from_cache(
        self,
        history_path: str = None,
        weights_path: str = None,
    ) -> dict:
        """
        Scan brain_data/topology_history.json and compute per-domain,
        per-agent reliability weights from every entry whose accuracy > 0.7.

        For each qualifying entry:
          - domain  = first non-terminal agent in topology.agents
          - credit  = every non-terminal agent in that topology receives the
                      entry's accuracy score, accumulated as a running mean

        Result is written to brain_data/fusion_weights.json and cached
        in self._weights_cache.

        Parameters
        ----------
        history_path : override path to topology_history.json (for tests)
        weights_path : override path to fusion_weights.json (for tests)
        """
        h_path = history_path or _TOPOLOGY_HISTORY_FILE
        w_path = weights_path or _FUSION_WEIGHTS_FILE

        if not os.path.exists(h_path):
            return {}

        try:
            with open(h_path, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception:
            return {}

        # Accumulate: {domain: {agent_weight_key: [acc, ...]}}
        acc_by_domain: dict = {}

        for entry in history:
            accuracy = entry.get("accuracy")
            if accuracy is None or float(accuracy) < 0.7:
                continue

            topology = entry.get("topology", {})
            agents   = topology.get("agents", [])
            if not agents:
                continue

            primary = [a for a in agents if a not in _TERMINAL_AGENTS]
            if not primary:
                continue

            domain = primary[0]
            acc_val = float(accuracy)

            for agent_short in primary:
                key = f"{agent_short}_agent"
                acc_by_domain.setdefault(domain, {}).setdefault(key, [])
                acc_by_domain[domain][key].append(acc_val)

        weights: dict = {}
        for domain, agent_accs in acc_by_domain.items():
            weights[domain] = {
                key: round(sum(accs) / len(accs), 4)
                for key, accs in agent_accs.items()
            }

        os.makedirs(os.path.dirname(w_path), exist_ok=True)
        try:
            with open(w_path, "w", encoding="utf-8") as f:
                json.dump(weights, f, indent=2)
        except Exception:
            pass

        self._weights_cache = weights
        n_agents = sum(len(v) for v in weights.values())
        print(f"  [FusionAgent] Learned weights: {n_agents} agent weights "
              f"across {len(weights)} domains "
              f"(from {len(history)} history entries)")
        return weights

    def get_weights_for_domain(
        self,
        domain: str,
        agent_names: list,
        weights_path: str = None,
    ) -> tuple:
        """
        Return normalized fusion weights for a list of agent class names,
        looked up in the learned weights for domain.

        Agents not found in the learned weights receive a neutral prior of 0.5.

        Parameters
        ----------
        domain      : primary domain (e.g. 'image', 'text', 'tabular')
        agent_names : list of class names (e.g. ['ImageAgent', 'TextAgent'])
        weights_path: override path (for tests)

        Returns
        -------
        (weights_dict, weights_source) where weights_source is
        'learned' if at least one agent had a learned weight, else 'default'.
        """
        weights_map = self._weights_cache
        if not weights_map:
            w_path = weights_path or _FUSION_WEIGHTS_FILE
            if os.path.exists(w_path):
                try:
                    with open(w_path, "r", encoding="utf-8") as f:
                        weights_map = json.load(f)
                except Exception:
                    weights_map = {}

        domain_weights = weights_map.get(domain, {})
        result: dict = {}
        has_learned   = False

        for name in agent_names:
            wk = _class_to_weight_key(name)
            if wk in domain_weights:
                result[name] = domain_weights[wk]
                has_learned  = True
            else:
                result[name] = 0.5

        total = sum(result.values())
        if total > 0:
            result = {k: round(v / total, 6) for k, v in result.items()}

        return result, ("learned" if has_learned else "default")

    # ── NEW: prediction-level weighted confidence fusion ──────────────────────

    def fuse(self, agent_results: list, weights: dict = None,
             domain: str = None) -> dict:
        """
        Fuse classification predictions from multiple agents.

        agent_results : [{label, confidence, agent_name}, ...]
        weights       : explicit {agent_name: float} — skips weight lookup
        domain        : if provided and weights is None, look up learned
                        weights for this domain from fusion_weights.json

        Output includes 'weights_source': 'provided' | 'learned' | 'default'
        """
        if not agent_results:
            return {"error": "No agent results to fuse"}

        if len(agent_results) == 1:
            r = dict(agent_results[0])
            r["fusion_method"]  = "passthrough"
            r["weights_source"] = "passthrough"
            r.setdefault("contributing_agents", [r.get("agent_name", "unknown")])
            r.setdefault("weights_used", {})
            return r

        agent_names = [r.get("agent_name", f"agent_{i}")
                       for i, r in enumerate(agent_results)]

        # Resolve weights: explicit > domain-learned > cache > equal
        if weights is not None:
            weights_source = "provided"
        elif domain:
            weights, weights_source = self.get_weights_for_domain(
                domain, agent_names)
        else:
            equal  = 1.0 / len(agent_results)
            raw    = {n: self._weight_cache.get(n, equal) for n in agent_names}
            total  = sum(raw.values()) or 1.0
            weights = {k: v / total for k, v in raw.items()}
            weights_source = "default"

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
              f"{' [UNCERTAIN]' if uncertain else ''}"
              f" [{weights_source} weights]")

        return {
            "label":               winner,
            "confidence":          winner_score,
            "contributing_agents": label_agents.get(winner, []),
            "weights_used":        weights,
            "weights_source":      weights_source,
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


# ── Module-level convenience ──────────────────────────────────────────────────

def refresh_fusion_weights(
    history_path: str = None,
    weights_path: str = None,
) -> dict:
    """Rebuild fusion_weights.json from topology history. Safe to call anytime."""
    return FusionAgent().learn_weights_from_cache(
        history_path=history_path,
        weights_path=weights_path,
    )
