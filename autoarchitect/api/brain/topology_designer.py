"""
topology_designer.py — Network Topology Designer
The brain that decides HOW agents connect for any problem.

Given a plain English problem → decides:
  - How many agents needed
  - Which agents
  - How they connect (sequential / parallel / conditional)
  - What each agent does with its output

This is the core of ANAS — Agent Network Architecture Search.
Every topology built → stored → brain learns → next time better.
"""

import json
from pathlib import Path
from datetime import datetime


# ── Topology types ─────────────────────────────────────────────────────────

SEQUENTIAL   = "sequential"
PARALLEL     = "parallel"
CONDITIONAL  = "conditional"
PIPELINE     = "pipeline"
HIERARCHICAL = "hierarchical"


# ── Agent catalog ──────────────────────────────────────────────────────────

AGENT_CATALOG = {
    "image": {
        "description": "Detects and classifies objects in images/video",
        "input":  "image file or camera feed",
        "output": "label + confidence + bounding box",
        "keywords": ["image", "photo", "camera", "detect", "visual", "video",
                     "picture", "scan", "see", "watch", "monitor visually",
                     "dumping", "pothole", "face", "object"],
    },
    "text": {
        "description": "Classifies and analyzes text content",
        "input":  "text string or document",
        "output": "label + confidence + category",
        "keywords": ["text", "message", "email", "spam", "classify",
                     "document", "content", "language", "word", "sentence",
                     "fake news", "review", "post", "comment",
                     "log", "configuration", "config", "docker", "container"],
    },
    "sentiment": {
        "description": "Analyzes emotional tone and sentiment",
        "input":  "text string",
        "output": "positive/negative/neutral + score",
        "keywords": ["sentiment", "emotion", "feeling", "opinion", "attitude",
                     "tone", "happy", "angry", "negative", "positive",
                     "review", "feedback", "reaction", "brand"],
    },
    "severity": {
        "description": "Classifies severity or priority level",
        "input":  "any prediction result",
        "output": "HIGH / MEDIUM / LOW + urgency score",
        "keywords": ["severity", "priority", "urgent", "critical", "level",
                     "rank", "grade", "serious", "minor", "classify severity",
                     "how bad", "danger", "risk"],
    },
    "report": {
        "description": "Generates structured reports and files alerts",
        "input":  "any prediction result",
        "output": "formatted report + alert sent",
        "keywords": ["report", "alert", "notify", "file", "log", "record",
                     "send", "email", "dashboard", "automatically file",
                     "notification", "track"],
    },
    "medical": {
        "description": "Analyzes medical images and health data",
        "input":  "xray / scan / health data",
        "output": "diagnosis label + confidence",
        "keywords": ["medical", "health", "xray", "scan", "diagnosis",
                     "disease", "patient", "clinical", "symptom", "hospital",
                     "pneumonia", "tumor", "abnormal"],
    },
    "security": {
        "description": "Detects threats, fraud, vulnerabilities and anomalies",
        "input":  "transaction / log / network / config data",
        "output": "threat label + risk score",
        "keywords": ["security", "fraud", "threat", "anomaly", "intrusion",
                     "attack", "suspicious", "malicious", "risk", "hack",
                     "unauthorized", "breach", "phishing",
                     "vulnerability", "vulnerabilities", "docker",
                     "container", "exploit", "malware", "ransomware",
                     "firewall", "penetration", "cve"],
    },
    "audience": {
        "description": "Scores and segments leads and customers",
        "input":  "customer profile or message",
        "output": "score 0-100 + segment label",
        "keywords": ["audience", "lead", "customer", "score", "segment",
                     "target", "prospect", "marketing", "sales", "crm",
                     "convert", "network marketing", "mlm"],
    },
    "optimizer": {
        "description": "Analyzes performance and recommends improvements",
        "input":  "results from other agents",
        "output": "recommendations + strategy adjustments",
        "keywords": ["optimize", "improve", "performance", "strategy",
                     "recommend", "adjust", "better", "enhance", "tune",
                     "best", "maximize", "boost"],
    },
}


# ── Topology templates — proven patterns ───────────────────────────────────

TOPOLOGY_TEMPLATES = {
    "detect_and_report": {
        "description": "Detect something → report it",
        "agents":   ["image", "report"],
        "topology": SEQUENTIAL,
        "keywords": ["detect", "monitor", "watch", "alert"],
    },
    "detect_classify_report": {
        "description": "Detect → classify severity → report",
        "agents":   ["image", "severity", "report"],
        "topology": SEQUENTIAL,
        "keywords": ["detect", "classify", "severity", "illegal", "dumping", "pothole"],
    },
    "text_analyze_report": {
        "description": "Analyze text → classify → report",
        "agents":   ["text", "report"],
        "topology": SEQUENTIAL,
        "keywords": ["spam", "fake news", "classify text", "email"],
    },
    "sentiment_audience_report": {
        "description": "Sentiment + audience scoring for marketing",
        "agents":   ["sentiment", "audience", "report"],
        "topology": HIERARCHICAL,
        "keywords": ["marketing", "brand", "customer", "network marketing", "sales"],
    },
    "multimodal_fusion": {
        "description": "Image + text together → fused decision",
        "agents":   ["image", "text", "report"],
        "topology": PARALLEL,
        "keywords": ["image and text", "visual and language", "multimodal"],
    },
    "medical_pipeline": {
        "description": "Medical scan → diagnosis → severity → alert doctor",
        "agents":   ["medical", "severity", "report"],
        "topology": SEQUENTIAL,
        "keywords": ["medical", "xray", "diagnosis", "patient", "hospital"],
    },
    # ── FIXED: security now matches docker/vulnerability problems ──────────
    "security_detect_report": {
        "description": "Detect security threats → classify → alert team",
        "agents":   ["security", "text", "report"],
        "topology": SEQUENTIAL,
        "keywords": ["security", "vulnerability", "vulnerabilities", "threat",
                     "docker", "container", "intrusion", "attack", "malware",
                     "breach", "unauthorized", "exploit", "firewall",
                     "hack", "cve", "penetration"],
    },
    "fraud_pipeline": {
        "description": "Detect fraud → score risk → alert",
        "agents":   ["security", "severity", "report"],
        "topology": SEQUENTIAL,
        "keywords": ["fraud", "transaction", "bank", "financial", "anomaly"],
    },
    "full_marketing_network": {
        "description": "Complete marketing automation network",
        "agents":   ["text", "sentiment", "audience", "optimizer", "report"],
        "topology": HIERARCHICAL,
        "keywords": ["grow business", "network marketing", "automate marketing",
                     "leads", "sales automation"],
    },
}


class TopologyDesigner:
    """
    Designs multi-agent network topologies from plain English problems.
    Learns from every topology it builds — gets smarter every time.
    """

    def __init__(self):
        self.data_dir     = Path("brain_data")
        self.data_dir.mkdir(exist_ok=True)
        self.topology_log = self.data_dir / "topology_history.json"
        self.history      = self._load_history()
        print(f"🏗️  TopologyDesigner ready — {len(self.history)} topologies learned")

    # ── Main entry point ───────────────────────────────────────────────────

    def design(self, problem: str, domain: str = None,
               meta_suggestion: dict = None) -> dict:
        problem_lower = problem.lower()

        # 1. Meta-learner suggestion first
        if meta_suggestion and meta_suggestion.get("agents"):
            topology = self._from_meta_suggestion(problem, meta_suggestion)
            topology["source"] = "meta_learner"
            self._store(problem, topology)
            return topology

        # 2. Cache check — stricter threshold to avoid wrong matches
        cached = self._check_cache(problem_lower)
        if cached:
            cached["source"] = "cache"
            print(f"  ⚡ Topology cache hit!")
            return cached

        # 3. Template match
        template_match = self._match_template(problem_lower)
        if template_match:
            topology = self._from_template(problem, template_match)
            topology["source"] = "template"
            self._store(problem, topology)
            return topology

        # 4. Rule-based fallback
        topology = self._rule_based_design(problem, problem_lower, domain)
        topology["source"] = "rule_based"
        self._store(problem, topology)
        return topology

    # ── Design strategies ──────────────────────────────────────────────────

    def _from_meta_suggestion(self, problem: str, suggestion: dict) -> dict:
        agents = suggestion.get("agents", [])
        return self._build_topology_dict(
            problem, agents,
            topology_type=self._infer_topology_type(agents),
            confidence=suggestion.get("confidence", 0.85),
        )

    def _match_template(self, problem_lower: str):
        best_match = None
        best_score = 0
        for name, template in TOPOLOGY_TEMPLATES.items():
            score = sum(1 for kw in template["keywords"] if kw in problem_lower)
            if score > best_score:
                best_score = score
                best_match = (name, template)
        if best_score >= 1:
            print(f"  📐 Template matched: {best_match[0]} (score {best_score})")
            return best_match
        return None

    def _from_template(self, problem: str, match: tuple) -> dict:
        name, template = match
        return self._build_topology_dict(
            problem,
            agents        = template["agents"],
            topology_type = template["topology"],
            confidence    = 0.88,
            template_name = name,
        )

    def _rule_based_design(self, problem: str, problem_lower: str,
                           domain: str = None) -> dict:
        scores = {}
        for agent_name, agent_info in AGENT_CATALOG.items():
            score = sum(1 for kw in agent_info["keywords"] if kw in problem_lower)
            if score > 0:
                scores[agent_name] = score

        if not scores:
            if domain == "image":
                scores["image"] = 1
            elif domain == "medical":
                scores["medical"] = 1
            elif domain == "security":
                scores["security"] = 1
            else:
                scores["text"] = 1

        sorted_agents = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        selected = [a for a, _ in sorted_agents[:4]]

        action_words = ["report", "alert", "notify", "file", "log", "send",
                        "automatically", "monitor", "watch"]
        if any(w in problem_lower for w in action_words):
            if "report" not in selected:
                selected.append("report")

        topology_type = self._infer_topology_type(selected)
        return self._build_topology_dict(
            problem, selected, topology_type, confidence=0.75
        )

    # ── Topology builder ───────────────────────────────────────────────────

    def _build_topology_dict(self, problem: str, agents: list,
                             topology_type: str, confidence: float = 0.80,
                             template_name: str = None) -> dict:
        connections = self._build_connections(agents, topology_type)
        agent_roles = self._assign_roles(problem, agents)
        return {
            "agents":      agents,
            "topology":    topology_type,
            "connections": connections,
            "agent_roles": agent_roles,
            "confidence":  round(confidence, 3),
            "template":    template_name,
            "problem":     problem,
            "designed_at": datetime.now().isoformat(),
        }

    def _build_connections(self, agents: list, topology_type: str) -> list:
        connections = []
        if topology_type in (SEQUENTIAL, PIPELINE):
            for i in range(len(agents) - 1):
                connections.append({
                    "from": agents[i], "to": agents[i+1], "type": "output"
                })
        elif topology_type == PARALLEL:
            if len(agents) > 1:
                merger = agents[-1]
                for a in agents[:-1]:
                    connections.append({
                        "from": a, "to": merger, "type": "parallel_output"
                    })
        elif topology_type == HIERARCHICAL:
            if len(agents) >= 3:
                source = agents[0]
                merger = agents[-1]
                middle = agents[1:-1]
                for m in middle:
                    connections.append({"from": source, "to": m, "type": "fanout"})
                for m in middle:
                    connections.append({"from": m, "to": merger, "type": "merge"})
            else:
                for i in range(len(agents) - 1):
                    connections.append({
                        "from": agents[i], "to": agents[i+1], "type": "output"
                    })
        elif topology_type == CONDITIONAL:
            if len(agents) >= 2:
                connections.append({
                    "from": agents[0], "to": agents[1],
                    "type": "conditional", "condition": "confidence > 0.8"
                })
        return connections

    def _assign_roles(self, problem: str, agents: list) -> dict:
        roles = {}
        for agent in agents:
            info = AGENT_CATALOG.get(agent, {})
            if agent == "image":
                task   = f"Detect and classify visual patterns in: {problem[:50]}"
                action = "flag_detection"
            elif agent == "text":
                task   = f"Analyze text/log content for: {problem[:50]}"
                action = "classify_text"
            elif agent == "sentiment":
                task   = "Analyze emotional tone and sentiment"
                action = "score_sentiment"
            elif agent == "severity":
                task   = "Classify severity as HIGH / MEDIUM / LOW"
                action = "assign_priority"
            elif agent == "report":
                task   = "Generate report and send alerts automatically"
                action = "file_alert"
            elif agent == "medical":
                task   = "Analyze medical data and flag abnormalities"
                action = "flag_urgent"
            elif agent == "security":
                task   = f"Detect threats and vulnerabilities in: {problem[:50]}"
                action = "raise_alert"
            elif agent == "audience":
                task   = "Score leads and segment audience 0-100"
                action = "score_lead"
            elif agent == "optimizer":
                task   = "Analyze results and recommend strategy improvements"
                action = "optimize_strategy"
            else:
                task   = info.get("description", agent)
                action = "process"

            roles[agent] = {
                "task":   task,
                "action": action,
                "input":  info.get("input",  "any input"),
                "output": info.get("output", "result"),
            }
        return roles

    def _infer_topology_type(self, agents: list) -> str:
        if len(agents) == 1:
            return SEQUENTIAL
        if "optimizer" in agents or "report" in agents:
            if len(agents) >= 4:
                return HIERARCHICAL
        if "image" in agents and "text" in agents and len(agents) == 3:
            return PARALLEL
        return SEQUENTIAL

    # ── Cache + History ────────────────────────────────────────────────────

    def _check_cache(self, problem_lower: str):
        for entry in self.history:
            stored = entry.get("problem", "").lower()
            if self._similarity(problem_lower, stored) > 0.85:  # stricter
                return entry.get("topology")
        return None

    def _similarity(self, a: str, b: str) -> float:
        wa = set(a.split())
        wb = set(b.split())
        if not wa or not wb:
            return 0.0
        return len(wa & wb) / len(wa | wb)

    def _store(self, problem: str, topology: dict):
        entry = {
            "problem":   problem,
            "topology":  topology,
            "stored_at": datetime.now().isoformat(),
            "accuracy":  None,
        }
        self.history.append(entry)
        self._save_history()
        print(f"  💾 Topology stored — brain now knows {len(self.history)} topologies")

    def update_accuracy(self, problem: str, accuracy: float):
        for entry in reversed(self.history):
            if entry["problem"] == problem:
                entry["accuracy"] = accuracy
                self._save_history()
                print(f"  🧠 Brain updated: topology accuracy = {accuracy:.1%}")
                return

    def _load_history(self) -> list:
        if self.topology_log.exists():
            try:
                with open(self.topology_log) as f:
                    return json.load(f)
            except Exception:
                pass
        return []

    def _save_history(self):
        try:
            with open(self.topology_log, 'w') as f:
                json.dump(self.history, f, indent=2)
        except Exception as e:
            print(f"  Warning: could not save topology history: {e}")

    def stats(self) -> dict:
        total    = len(self.history)
        with_acc = [e for e in self.history if e.get("accuracy")]
        avg_acc  = sum(e["accuracy"] for e in with_acc) / len(with_acc) if with_acc else 0
        return {
            "topologies_designed":      total,
            "topologies_with_accuracy": len(with_acc),
            "average_accuracy":         round(avg_acc, 3),
            "templates_available":      len(TOPOLOGY_TEMPLATES),
            "agents_available":         len(AGENT_CATALOG),
        }
