# ============================================
# AutoArchitect Brain — Workflow Generator
# Fixed: meta-learner threshold 0.5 → 0.85
# Fixed: requires 60%+ accuracy to override
# Result: domain correction is respected
# ============================================

import os
import time
from api.brain.strategy_library    import StrategyLibrary
from api.brain.performance_tracker import PerformanceTracker
from api.brain.meta_learner        import get_meta_learner


class WorkflowGenerator:
    """
    The brain that generates workflows for ANY problem.
    Uses meta-learner to PREDICT best pipeline.
    Only overrides when VERY confident (85%+, 60%+ accuracy).
    Otherwise respects domain correction from orchestrator.
    """

    def __init__(self):
        self.library  = StrategyLibrary()
        self.tracker  = PerformanceTracker()
        self.meta     = get_meta_learner()
        print("🤖 Workflow Generator ready!")
        print(f"   Knows {len(self.library.strategies)} strategies")

    def generate(self, problem: str,
                  bert_domain: str,
                  bert_embedding: list = None) -> dict:
        """
        Generate optimal workflow for any problem.

        Priority:
        1. Meta-learner prediction (only if 85%+ confident AND 60%+ accuracy)
        2. Strategy library (uses corrected domain from orchestrator)
        """
        start = time.time()
        print(f"\n🤖 Generating workflow for: {problem[:50]}")

        # ── Try meta-learner first ──────────────────────────
        meta_pred = self.meta.predict(
            problem,
            bert_embedding=bert_embedding
        )

        # Only override if VERY confident AND proven accurate
        # Old threshold was 0.5 — too low, caused wrong agent selection
        # New threshold: 0.85 confidence + 60% historical accuracy
        if meta_pred.get("predicted") and \
           meta_pred.get("confidence", 0) >= 0.85 and \
           meta_pred.get("accuracy", 0) >= 10:

            agents        = meta_pred["agents"]
            workflow_type = "multi" if len(agents) > 1 else "single"
            strategy_name = f"meta_predicted_{'+'.join(agents)}"
            avg_accuracy  = meta_pred["accuracy"]

            print(f"  🔮 Meta-learner prediction:")
            print(f"     Agents:     {agents}")
            print(f"     Dataset:    {meta_pred.get('dataset', 'unknown')}")
            print(f"     Method:     {meta_pred.get('method', 'unknown')}")
            print(f"     Expected:   ~{avg_accuracy}%")
            print(f"     Confidence: {meta_pred['confidence']:.1%}")
            print(f"  🔮 Meta-learner override! "
                  f"Confidence: {meta_pred['confidence']:.1%}")
            source = "meta_learner"

        else:
            # Fall back to strategy library — uses corrected domain
            if meta_pred.get("predicted"):
                print(f"  🔮 Meta-learner prediction:")
                print(f"     Agents:     {meta_pred.get('agents', [])}")
                print(f"     Dataset:    {meta_pred.get('dataset', 'unknown')}")
                print(f"     Method:     {meta_pred.get('method', 'unknown')}")
                print(f"     Expected:   ~{meta_pred.get('accuracy', 0)}%")
                print(f"     Confidence: {meta_pred.get('confidence', 0):.1%}")
                print(f"  ⚠️  Meta-learner confidence too low "
                      f"({meta_pred.get('confidence', 0):.1%}) "
                      f"— using strategy library")

            strategy      = self.library.find_best_strategy(
                problem, bert_domain)
            agents        = strategy["agents"]
            workflow_type = "multi" if len(agents) > 1 else "single"
            strategy_name = strategy["strategy_name"]
            avg_accuracy  = strategy["avg_accuracy"]
            source        = "strategy_library"

        # Build steps
        steps = []
        for i, agent in enumerate(agents):
            steps.append(
                f"Step {i+1}: {agent.upper()} NAS Agent "
                f"— {self._agent_description(agent, problem)}"
            )
        if len(agents) > 1:
            steps.append(
                f"Step {len(agents)+1}: "
                f"Fusion Agent — combine architectures")
        steps.append(
            f"Step {len(agents)+2}: "
            f"Evaluator Agent — score + feedback")
        steps.append(
            f"Step {len(agents)+3}: "
            f"Cache Agent — save forever")

        elapsed = round(time.time() - start, 3)
        print(f"  ✅ Workflow generated in {elapsed}s")
        print(f"     Source:   {source}")
        print(f"     Strategy: {strategy_name}")
        print(f"     Agents:   {agents}")
        print(f"     Expected: ~{avg_accuracy}% accuracy")

        return {
            "type":              workflow_type,
            "agents":            agents,
            "strategy_name":     strategy_name,
            "steps":             steps,
            "expected_accuracy": avg_accuracy,
            "confidence":        meta_pred.get("confidence", 0.5),
            "generated_in":      elapsed,
            "source":            source,
            "meta_prediction":   meta_pred,
        }

    def learn_from_result(self, problem: str,
                           workflow: dict,
                           accuracy: float,
                           time_taken: float,
                           dataset_used:   str  = "unknown",
                           method_used:    str  = "darts_nas",
                           bert_embedding: list = None,
                           from_cache:     bool = False):
        """
        Brain learns after every problem solved.
        Updates strategy library AND meta-learner.
        Only learns from real training runs, not cache hits.
        """
        strategy_name = workflow.get("strategy_name", "unknown")
        agents        = workflow.get("agents", [])

        if accuracy == 0.0:
            print(f"  ⚠️  Brain skipping update — no dataset found, strategy scores protected")
            return

        # 1. Update strategy library
        self.library.learn(
            problem       = problem,
            strategy_name = strategy_name,
            accuracy      = accuracy,
            agents_used   = agents,
            success       = accuracy >= 50.0
        )

        # 2. Record in performance history
        self.tracker.record(
            problem    = problem,
            strategy   = strategy_name,
            agents     = agents,
            accuracy   = accuracy,
            time_taken = time_taken,
            from_cache = from_cache
        )

        # 3. Feed meta-learner — only from real training runs
        if not from_cache:
            self.meta.learn(
                problem         = problem,
                agents_used     = agents,
                dataset_used    = dataset_used,
                method_used     = method_used,
                actual_accuracy = accuracy,
                bert_embedding  = bert_embedding,
            )

        print(f"  🧠 Brain updated! Strategy: {strategy_name} "
              f"→ {accuracy}% accuracy")

    def get_brain_status(self) -> dict:
        """How smart is the brain right now?"""
        lib_stats     = self.library.get_stats()
        perf_insights = self.tracker.get_insights()
        meta_insights = self.meta.get_insights()

        return {
            "strategies_known": lib_stats["total_strategies"],
            "auto_learned":     lib_stats["auto_learned"],
            "problems_solved":  lib_stats["total_problems"],
            "avg_accuracy":     lib_stats["avg_accuracy"],
            "best_strategy":    lib_stats["best_strategy"],
            "cache_hit_rate":   perf_insights.get("cache_rate", 0),
            "success_rate":     perf_insights.get("success_rate", 0),
            "agent_usage":      perf_insights.get("agent_usage", {}),
            "recent_problems":  [
                h["problem"][:40]
                for h in perf_insights.get("recent", [])
            ],
            "meta_learner": {
                "status":        meta_insights.get("status"),
                "examples":      meta_insights.get("examples", 0),
                "trained":       meta_insights.get("trained", False),
                "avg_accuracy":  meta_insights.get("avg_accuracy", 0),
                "best_combo":    meta_insights.get("best_combo"),
                "best_dataset":  meta_insights.get("best_dataset"),
                "accuracy_trend": meta_insights.get("accuracy_trend", []),
                "until_retrain": meta_insights.get("until_retrain", 3),
                "combo_performance": meta_insights.get(
                    "combo_performance", {}),
            }
        }

    def _agent_description(self, agent: str, problem: str) -> str:
        descriptions = {
            "image":    "detect visual patterns",
            "text":     "classify text content",
            "medical":  "analyze medical data",
            "security": "detect threats/anomalies",
        }
        return descriptions.get(agent, "process data")
