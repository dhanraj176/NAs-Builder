# ============================================
# AutoArchitect Brain - Strategy Library
# Learns which strategies work best
# Gets smarter with every problem solved
# ============================================

import os
import json
import time
from datetime import datetime

BRAIN_DIR = os.path.join(
    os.path.dirname(os.path.dirname(
        os.path.dirname(__file__))),
    'brain_data'
)
STRATEGY_FILE = os.path.join(BRAIN_DIR, 'strategies.json')


SEED_STRATEGIES = {
    "visual_detection": {
        "description":  "Detect/find objects in images",
        "keywords":     ["detect", "find", "spot", "identify",
                         "image", "photo", "camera", "visual",
                         "object", "recognition", "classification"],
        "agents":       ["image"],
        "model":        "ResNet18",
        "dataset":      "CIFAR-10",
        "avg_accuracy": 74.89,
        "uses":         1,
        "success_rate": 0.95,
        "learned_at":   datetime.now().isoformat(),
        "examples": [
            "detect potholes in road images",
            "identify defects in products",
            "spot wildfires in satellite images"
        ]
    },
    "medical_analysis": {
        "description":  "Analyze medical scans and symptoms",
        "keywords":     ["medical", "xray", "mri", "scan",
                         "diagnos", "tumor", "disease", "patient",
                         "health", "clinical", "symptom", "chest"],
        "agents":       ["medical"],
        "model":        "ResNet18",
        "dataset":      "MedMNIST",
        "avg_accuracy": 41.0,
        "uses":         1,
        "success_rate": 0.80,
        "learned_at":   datetime.now().isoformat(),
        "examples": [
            "detect tumor in brain MRI",
            "classify chest xray diseases",
            "identify diabetic retinopathy"
        ]
    },
    "text_classification": {
        "description":  "Classify and analyze text content",
        "keywords":     ["text", "classify", "sentiment",
                         "spam", "email", "review", "comment",
                         "language", "nlp", "words", "message"],
        "agents":       ["text"],
        "model":        "TF-IDF + Neural",
        "dataset":      "IMDB",
        "avg_accuracy": 43.2,
        "uses":         1,
        "success_rate": 0.85,
        "learned_at":   datetime.now().isoformat(),
        "examples": [
            "classify spam emails",
            "detect toxic comments",
            "analyze customer sentiment"
        ]
    },
    "security_threat": {
        "description":  "Detect fraud, threats, anomalies",
        "keywords":     ["fraud", "threat", "security",
                         "attack", "anomaly", "intrusion",
                         "malware", "suspicious", "phishing",
                         "hack", "breach", "vulnerability"],
        "agents":       ["security"],
        "model":        "Isolation Forest",
        "dataset":      "KDD Cup",
        "avg_accuracy": 70.0,
        "uses":         1,
        "success_rate": 0.88,
        "learned_at":   datetime.now().isoformat(),
        "examples": [
            "detect fraud in transactions",
            "identify network intrusions",
            "classify security threats"
        ]
    },
    "content_analysis": {
        "description":  "Analyze visual content for engagement and virality",
        "keywords":     ["youtube", "thumbnail", "viral", "content",
                         "social", "media", "post", "video",
                         "engagement", "analyze", "title", "caption",
                         "creator", "channel", "views", "clicks",
                         "marketing", "brand", "audience", "reach"],
        "agents":       ["image", "text"],
        "model":        "ResNet18 + TF-IDF",
        "dataset":      "user_data",
        "avg_accuracy": 75.0,
        "uses":         1,
        "success_rate": 0.90,
        "learned_at":   datetime.now().isoformat(),
        "examples": [
            "analyze YouTube thumbnail for viral potential",
            "analyze social media post engagement",
            "analyze content for viral potential",
            "check if my thumbnail will get clicks"
        ]
    },
    "visual_text_analysis": {
        "description":  "Analyze both images and text together",
        "keywords":     ["and", "with", "both",
                         "image", "text", "visual", "content",
                         "detect", "classify", "combined"],
        "agents":       ["image", "text"],
        "model":        "ResNet18 + TF-IDF",
        "dataset":      "user_data",
        "avg_accuracy": 22.6,
        "uses":         1,
        "success_rate": 0.90,
        "learned_at":   datetime.now().isoformat(),
        "examples": [
            "detect fraud and classify suspicious messages",
            "analyze content and sentiment",
            "detect objects and describe them"
        ]
    },
    "medical_text_analysis": {
        "description":  "Combine medical imaging with text analysis",
        "keywords":     ["medical", "report", "scan",
                         "diagnos", "symptom", "patient",
                         "classify", "severity", "stage"],
        "agents":       ["medical", "text"],
        "model":        "ResNet18 + TF-IDF",
        "dataset":      "MedMNIST + IMDB",
        "avg_accuracy": 42.1,
        "uses":         1,
        "success_rate": 0.85,
        "learned_at":   datetime.now().isoformat(),
        "examples": [
            "detect tumor and classify cancer stage",
            "analyze scan and generate report",
            "diagnose from image and symptoms"
        ]
    },
    "security_text_analysis": {
        "description":  "Detect threats in both data and text",
        "keywords":     ["fraud", "suspicious", "classify",
                         "threat", "text", "message", "email",
                         "detect", "and", "filter"],
        "agents":       ["security", "text"],
        "model":        "IsoForest + TF-IDF",
        "dataset":      "KDD Cup + IMDB",
        "avg_accuracy": 60.0,
        "uses":         1,
        "success_rate": 0.88,
        "learned_at":   datetime.now().isoformat(),
        "examples": [
            "detect fraud and classify suspicious text messages",
            "identify phishing emails and classify threat level",
            "detect spam and analyze message content"
        ]
    }
}


class StrategyLibrary:
    """
    The brain's memory.
    Stores what works, learns from every problem.
    Gets smarter over time automatically.
    """

    def __init__(self):
        os.makedirs(BRAIN_DIR, exist_ok=True)
        self._load_or_seed()
        print(f"Strategy Library loaded: "
              f"{len(self.strategies)} strategies")

    def find_best_strategy(self, problem: str,
                            bert_domain: str) -> dict:
        problem_lower = problem.lower()
        scored        = []

        for name, strategy in self.strategies.items():
            score    = 0.0
            keywords = strategy.get("keywords", [])
            matches  = sum(1 for kw in keywords
                          if kw in problem_lower)
            score   += matches * 0.3

            agents = strategy.get("agents", [])
            if bert_domain in agents:
                score += 0.4

            acc    = strategy.get("avg_accuracy", 0)
            score += (acc / 100) * 0.2

            uses   = strategy.get("uses", 0)
            score += min(uses / 100, 0.1)

            if score > 0:
                scored.append({
                    "name":     name,
                    "strategy": strategy,
                    "score":    round(score, 3)
                })

        if not scored:
            return self._default_strategy(bert_domain)

        scored.sort(key=lambda x: x["score"], reverse=True)
        best = scored[0]

        print(f"  Best strategy: {best['name']} "
              f"(score: {best['score']})")
        if len(scored) > 1:
            print(f"  Runner-up: {scored[1]['name']} "
                  f"(score: {scored[1]['score']})")

        return {
            "strategy_name":  best["name"],
            "agents":         best["strategy"]["agents"],
            "model":          best["strategy"]["model"],
            "dataset":        best["strategy"]["dataset"],
            "confidence":     best["score"],
            "avg_accuracy":   best["strategy"]["avg_accuracy"],
            "all_candidates": [
                {"name": s["name"], "score": s["score"]}
                for s in scored[:3]
            ]
        }

    def learn(self, problem: str, strategy_name: str,
              accuracy: float, agents_used: list,
              success: bool = True):
        print(f"  Brain learning from: {problem[:40]}")
        print(f"     Strategy: {strategy_name}")
        print(f"     Accuracy: {accuracy}%")

        if strategy_name in self.strategies:
            s        = self.strategies[strategy_name]
            old_acc  = s["avg_accuracy"]
            old_uses = s["uses"]
            s["avg_accuracy"] = round(
                (old_acc * old_uses + accuracy) /
                (old_uses + 1), 2)
            s["uses"]         += 1
            s["success_rate"]  = round(
                (s["success_rate"] * old_uses +
                 (1.0 if success else 0.0)) /
                (old_uses + 1), 3)
            s["last_used"]     = datetime.now().isoformat()
            print(f"  Updated: {strategy_name} "
                  f"accuracy {old_acc}% -> {s['avg_accuracy']}%")
        else:
            new_name = self._generate_strategy_name(
                problem, agents_used)
            keywords = self._extract_keywords(problem)
            self.strategies[new_name] = {
                "description":  f"Learned from: {problem[:50]}",
                "keywords":     keywords,
                "agents":       agents_used,
                "model":        "AutoArchitect NAS",
                "dataset":      "auto-selected",
                "avg_accuracy": accuracy,
                "uses":         1,
                "success_rate": 1.0 if success else 0.0,
                "learned_at":   datetime.now().isoformat(),
                "examples":     [problem],
                "auto_learned": True
            }
            print(f"  NEW strategy discovered: {new_name}")

        self._save()

    def get_stats(self) -> dict:
        total_uses = sum(
            s.get("uses", 0)
            for s in self.strategies.values())
        avg_acc = round(
            sum(s.get("avg_accuracy", 0)
                for s in self.strategies.values()) /
            max(len(self.strategies), 1), 1)
        auto_learned = sum(
            1 for s in self.strategies.values()
            if s.get("auto_learned", False))

        return {
            "total_strategies": len(self.strategies),
            "auto_learned":     auto_learned,
            "seed_strategies":  len(self.strategies) - auto_learned,
            "total_problems":   total_uses,
            "avg_accuracy":     avg_acc,
            "best_strategy":    max(
                self.strategies.items(),
                key=lambda x: x[1].get("avg_accuracy", 0)
            )[0] if self.strategies else "none",
            "strategies": {
                name: {
                    "agents":       s["agents"],
                    "avg_accuracy": s["avg_accuracy"],
                    "uses":         s["uses"],
                    "auto_learned": s.get("auto_learned", False)
                }
                for name, s in self.strategies.items()
            }
        }

    def _load_or_seed(self):
        if os.path.exists(STRATEGY_FILE):
            with open(STRATEGY_FILE, 'r') as f:
                self.strategies = json.load(f)
            print(f"  Loaded {len(self.strategies)} strategies")
        else:
            self.strategies = SEED_STRATEGIES.copy()
            self._save()
            print(f"  Seeded with {len(self.strategies)} strategies")

    def _save(self):
        with open(STRATEGY_FILE, 'w') as f:
            json.dump(self.strategies, f, indent=2)

    def _default_strategy(self, domain: str) -> dict:
        return {
            "strategy_name":  f"{domain}_default",
            "agents":         [domain],
            "model":          "ResNet18",
            "dataset":        "CIFAR-10",
            "confidence":     0.5,
            "avg_accuracy":   50.0,
            "all_candidates": []
        }

    def _generate_strategy_name(self, problem: str,
                                  agents: list) -> str:
        words  = problem.lower().split()[:3]
        clean  = [w for w in words if len(w) > 3][:2]
        suffix = "_".join(agents)
        base   = "_".join(clean) if clean else "custom"
        return f"{base}_{suffix}"

    def _extract_keywords(self, problem: str) -> list:
        stopwords = {
            "the", "a", "an", "in", "on", "at",
            "for", "and", "or", "to", "from",
            "with", "using", "via", "my", "our"
        }
        words = problem.lower().split()
        return [w for w in words
                if len(w) > 3 and w not in stopwords][:8]
