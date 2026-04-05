# ============================================
# AutoArchitect Brain — Performance Tracker
# Tracks what works, feeds back to brain
# ============================================

import os
import json
from datetime import datetime

BRAIN_DIR = os.path.join(
    os.path.dirname(os.path.dirname(
        os.path.dirname(__file__))),
    'brain_data'
)
HISTORY_FILE = os.path.join(BRAIN_DIR, 'history.json')


class PerformanceTracker:
    """
    Tracks every problem solved.
    Identifies patterns.
    Feeds insights back to strategy library.
    """

    def __init__(self):
        os.makedirs(BRAIN_DIR, exist_ok=True)
        self._load()
        print(f"📊 Performance Tracker: "
              f"{len(self.history)} problems tracked")

    def record(self, problem: str, strategy: str,
               agents: list, accuracy: float,
               time_taken: float, from_cache: bool = False):
        """Record a solved problem."""
        entry = {
            "problem":     problem,
            "strategy":    strategy,
            "agents":      agents,
            "accuracy":    accuracy,
            "time_taken":  time_taken,
            "from_cache":  from_cache,
            "solved_at":   datetime.now().isoformat(),
            "success":     accuracy >= 50.0
        }
        self.history.append(entry)
        self._save()

        print(f"  📊 Recorded: {problem[:30]} "
              f"→ {accuracy}% in {time_taken}s")

    def get_insights(self) -> dict:
        """What has the brain learned?"""
        if not self.history:
            return {"message": "No history yet"}

        total     = len(self.history)
        cached    = sum(1 for h in self.history
                       if h.get("from_cache"))
        successes = sum(1 for h in self.history
                       if h.get("success"))
        avg_acc   = round(
            sum(h.get("accuracy", 0) for h in self.history)
            / total, 1)
        avg_time  = round(
            sum(h.get("time_taken", 0) for h in self.history)
            / total, 1)

        # Agent frequency
        agent_counts = {}
        for h in self.history:
            for a in h.get("agents", []):
                agent_counts[a] = agent_counts.get(a, 0) + 1

        # Best performing strategy
        strategy_acc = {}
        strategy_cnt = {}
        for h in self.history:
            s = h.get("strategy", "unknown")
            strategy_acc[s] = strategy_acc.get(s, 0) + h.get("accuracy", 0)
            strategy_cnt[s] = strategy_cnt.get(s, 0) + 1

        best_strategy = max(
            strategy_acc.items(),
            key=lambda x: x[1] / max(strategy_cnt[x[0]], 1)
        )[0] if strategy_acc else "none"

        return {
            "total_problems":   total,
            "cache_hits":       cached,
            "cache_rate":       round(cached/total*100, 1),
            "success_rate":     round(successes/total*100, 1),
            "avg_accuracy":     avg_acc,
            "avg_time":         avg_time,
            "agent_usage":      agent_counts,
            "best_strategy":    best_strategy,
            "recent":           self.history[-5:]
        }

    def _load(self):
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, 'r') as f:
                self.history = json.load(f)
        else:
            self.history = []

    def _save(self):
        with open(HISTORY_FILE, 'w') as f:
            json.dump(self.history, f, indent=2)
