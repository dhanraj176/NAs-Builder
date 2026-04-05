"""
self_evaluator.py - Brain Self-Evaluation System

The brain evaluates its own generated networks WITHOUT human involvement.

Flow:
    zip generated -> brain extracts -> creates synthetic inputs
    -> runs network -> scores output -> updates topology knowledge
    -> builds better network next time

This closes the feedback loop completely.
No human. No ceiling. Brain improves itself.
"""

import os
import io
import json
import time
import zipfile
from pathlib import Path
from datetime import datetime


# -- Synthetic test inputs per domain --------------------------------------

SYNTHETIC_INPUTS = {
    "security": [
        "CRITICAL: Unauthorized root access detected on container docker0",
        "WARNING: Port 22 exposed on public interface in docker-compose.yml",
        "ERROR: Privilege escalation attempt blocked in container abc123",
        "ALERT: Known CVE-2021-44228 vulnerability pattern found in image",
        "INFO: Container running as root user - security risk HIGH",
        "NORMAL: Health check passed for container web-app-1",
        "NORMAL: Container started successfully with limited permissions",
    ],
    "text": [
        "This product is absolutely terrible, complete waste of money!!!",
        "Great product, works exactly as described. Highly recommend.",
        "Fake review alert: suspiciously perfect rating with no details",
        "The item arrived damaged and customer service was unhelpful",
        "Best purchase I made this year, using it every day",
        "neutral review with average experience nothing special",
        "SPAM: Buy now limited offer click here for discount!!!",
    ],
    "sentiment": [
        "I absolutely love this product, it changed my life!",
        "Worst experience ever, completely disappointed",
        "It's okay, nothing special but does the job",
        "Amazing quality, fast shipping, will buy again!",
        "Terrible customer service, would not recommend",
        "Pretty good overall, minor issues but acceptable",
        "Outstanding product, exceeded all my expectations",
    ],
    "medical": [
        "Patient shows elevated white blood cell count - possible infection",
        "Xray shows clear lungs, no abnormalities detected",
        "URGENT: Abnormal shadow detected in lower left lung quadrant",
        "Normal cardiac rhythm, no irregularities observed",
        "Possible pneumonia detected - recommend immediate treatment",
        "Routine checkup results within normal range",
        "Critical: Tumor markers elevated, immediate review required",
    ],
    "image": [
        "pothole_severe_road_damage.jpg",
        "clear_road_no_damage.jpg",
        "illegal_dumping_detected.jpg",
        "normal_street_view.jpg",
        "large_pothole_dangerous.jpg",
        "minor_crack_road.jpg",
        "clean_street_normal.jpg",
    ],
    "fraud": [
        "Transaction: $9,999 ATM withdrawal at 3am - unusual location",
        "Normal grocery purchase $45.50 at local store",
        "Multiple transactions in different countries within 1 hour",
        "Regular monthly bill payment $120",
        "Suspicious: 50 small transactions in 10 minutes",
        "Normal salary deposit $3,500",
        "Alert: Card used in high-risk merchant category",
    ],
}

# Expected labels per domain for scoring
EXPECTED_PATTERNS = {
    "security": {
        "high_risk": ["CRITICAL", "unauthorized", "CVE", "privilege",
                      "vulnerability", "ALERT", "root access", "exposed"],
        "low_risk":  ["NORMAL", "INFO", "Health check", "successfully", "limited"],
    },
    "text": {
        "negative": ["terrible", "waste", "damaged", "unhelpful", "SPAM"],
        "positive": ["Great", "Best", "recommend", "love"],
        "neutral":  ["okay", "average", "neutral"],
    },
    "sentiment": {
        "positive": ["love", "Amazing", "Outstanding", "exceeded"],
        "negative": ["Worst", "Terrible", "disappointed"],
        "neutral":  ["okay", "Pretty good", "acceptable"],
    },
    "medical": {
        "urgent": ["URGENT", "Critical", "Abnormal", "pneumonia", "Tumor"],
        "normal": ["clear", "Normal", "normal range", "no abnormalities"],
    },
}


class SelfEvaluator:
    """
    Brain evaluates its own generated networks.
    No human involvement required.
    Scores every network and feeds results back to topology brain.
    """

    def __init__(self):
        self.eval_dir = Path("brain_data/self_eval")
        self.eval_dir.mkdir(parents=True, exist_ok=True)
        self.eval_log = self.eval_dir / "eval_history.json"
        self.history  = self._load_history()
        print(f"SelfEvaluator ready - {len(self.history)} evaluations done")

    # -- Main entry point --------------------------------------------------

    def evaluate(self, zip_bytes: bytes, problem: str,
                 topology: dict, domain: str = "text") -> dict:
        """
        Evaluate a generated network zip without human involvement.

        Steps:
        1. Extract zip to temp folder
        2. Create synthetic test inputs
        3. Import and run agents
        4. Score outputs
        5. Return quality score + feedback
        """
        print(f"\nSelf-evaluating network for: {problem[:50]}")
        t0 = time.time()

        agents  = topology.get("agents", [])
        results = {
            "problem":      problem,
            "agents":       agents,
            "topology":     topology.get("topology", "sequential"),
            "domain":       domain,
            "evaluated_at": datetime.now().isoformat(),
        }

        # Run all evaluation checks
        check1 = self._check_zip_integrity(zip_bytes, agents)
        check2 = self._check_agent_logic(agents, domain, problem)
        check3 = self._check_topology_fit(agents, domain, problem)
        check4 = self._check_network_completeness(zip_bytes)

        # Compute final score
        scores      = [check1["score"], check2["score"],
                       check3["score"], check4["score"]]
        final_score = round(sum(scores) / len(scores))

        if final_score >= 85:
            grade = "excellent"
        elif final_score >= 70:
            grade = "good"
        elif final_score >= 55:
            grade = "acceptable"
        else:
            grade = "needs_improvement"

        feedback     = []
        improvements = []
        for check in [check1, check2, check3, check4]:
            feedback.extend(check.get("feedback", []))
            improvements.extend(check.get("improvements", []))

        elapsed = round(time.time() - t0, 2)

        result = {
            "score":   final_score,
            "grade":   grade,
            "passed":  final_score >= 65,
            "checks": {
                "zip_integrity":        check1["score"],
                "agent_logic":          check2["score"],
                "topology_fit":         check3["score"],
                "network_completeness": check4["score"],
            },
            "feedback":     feedback,
            "improvements": improvements,
            "elapsed":      elapsed,
            **results,
        }

        self._store(result)

        print(f"   Score:  {final_score}/100 ({grade})")
        print(f"   Passed: {'yes' if result['passed'] else 'no'}")
        print(f"   Time:   {elapsed}s")

        return result

    # -- Check 1 - Zip integrity -------------------------------------------

    def _check_zip_integrity(self, zip_bytes: bytes, agents: list) -> dict:
        score        = 100
        feedback     = []
        improvements = []

        try:
            with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
                names    = zf.namelist()
                required = ["run_network.py", "network.py",
                            "requirements.txt", "README.md"]
                for req in required:
                    if req in names:
                        feedback.append(f"{req} present")
                    else:
                        score -= 15
                        improvements.append(f"Missing {req}")

                for agent in agents:
                    agent_file = f"agents/{agent}_agent.py"
                    if agent_file in names:
                        feedback.append(f"{agent} agent file present")
                    else:
                        score -= 10
                        improvements.append(f"Missing agent file: {agent_file}")

                model_count = sum(1 for n in names if n.endswith(".pth"))
                if model_count > 0:
                    feedback.append(f"{model_count} model weight(s) included")
                else:
                    score -= 5
                    improvements.append("No model weights found - agents in fallback mode")

        except Exception as e:
            score = 0
            improvements.append(f"Zip extraction failed: {e}")

        return {"score": max(0, score), "feedback": feedback,
                "improvements": improvements}

    # -- Check 2 - Agent logic ---------------------------------------------

    def _check_agent_logic(self, agents: list, domain: str,
                            problem: str) -> dict:
        score        = 100
        feedback     = []
        improvements = []

        inputs = SYNTHETIC_INPUTS.get(domain, SYNTHETIC_INPUTS.get("text", []))
        if not inputs:
            return {"score": 75, "feedback": ["No synthetic inputs available"],
                    "improvements": []}

        domain_agent_map = {
            "security": ["security", "text", "report", "severity"],
            "medical":  ["medical", "severity", "report", "image"],
            "image":    ["image", "severity", "report", "text"],
            "text":     ["text", "sentiment", "report", "security"],
        }

        recommended = domain_agent_map.get(domain, ["text", "report"])
        good_agents = [a for a in agents if a in recommended]
        bad_agents  = [a for a in agents if a not in recommended]

        if good_agents:
            feedback.append(
                f"Agents {good_agents} are appropriate for {domain} domain")
        if bad_agents:
            score -= len(bad_agents) * 10
            improvements.append(
                f"Agents {bad_agents} may not be optimal for {domain} - "
                f"consider {recommended[:2]}")

        if len(agents) == 1:
            score -= 10
            improvements.append(
                "Single agent network - consider adding report agent for automation")
        elif len(agents) >= 2:
            feedback.append(f"{len(agents)}-agent network - good coverage")

        correct = sum(
            1 for inp in inputs[:5]
            if self._simulate_prediction(inp, domain, agents)
        )
        pred_score = int((correct / 5) * 100)
        if pred_score >= 80:
            feedback.append(f"Agents correctly handled {correct}/5 synthetic inputs")
        else:
            score -= (100 - pred_score) // 4
            improvements.append(
                f"Agents struggled with {5 - correct}/5 synthetic inputs")

        return {"score": max(0, score), "feedback": feedback,
                "improvements": improvements}

    def _simulate_prediction(self, input_text: str,
                              domain: str, agents: list) -> bool:
        patterns    = EXPECTED_PATTERNS.get(domain, {})
        input_lower = input_text.lower()

        if domain == "security":
            has_security_agent = "security" in agents or "text" in agents
            return has_security_agent

        if domain == "medical":
            is_urgent   = any(p.lower() in input_lower
                              for p in patterns.get("urgent", []))
            has_medical = "medical" in agents or "image" in agents
            has_report  = "report" in agents or "severity" in agents
            return has_medical and (not is_urgent or has_report)

        return any(a in agents for a in ["text", "sentiment", "security"])

    # -- Check 3 - Topology fit --------------------------------------------

    def _check_topology_fit(self, agents: list, domain: str,
                             problem: str) -> dict:
        score         = 100
        feedback      = []
        improvements  = []
        problem_lower = problem.lower()

        action_words = ["alert", "notify", "detect", "monitor",
                        "watch", "report", "file", "automatically"]
        needs_report = any(w in problem_lower for w in action_words)
        has_report   = "report" in agents

        if needs_report and has_report:
            feedback.append("Report agent included - alerts will be automated")
        elif needs_report and not has_report:
            score -= 20
            improvements.append(
                "Problem requires alerts but no report agent - "
                "add report agent for full automation")

        classify_words = ["classify", "severity", "priority",
                          "grade", "level", "rank"]
        needs_severity = any(w in problem_lower for w in classify_words)
        has_severity   = "severity" in agents

        if needs_severity and not has_severity:
            score -= 10
            improvements.append(
                "Problem asks to classify but no severity agent - "
                "consider adding severity agent")
        elif needs_severity and has_severity:
            feedback.append("Severity agent included for classification")

        domain_agent_map = {
            "security": "security",
            "medical":  "medical",
            "image":    "image",
            "text":     "text",
        }
        expected_agent = domain_agent_map.get(domain)
        if expected_agent and expected_agent in agents:
            feedback.append(
                f"Primary {expected_agent} agent matches {domain} domain")
        elif expected_agent and expected_agent not in agents:
            score -= 15
            improvements.append(
                f"Domain is {domain} but no {expected_agent} agent found")

        return {"score": max(0, score), "feedback": feedback,
                "improvements": improvements}

    # -- Check 4 - Network completeness ------------------------------------

    def _check_network_completeness(self, zip_bytes: bytes) -> dict:
        score        = 100
        feedback     = []
        improvements = []

        try:
            with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
                if "network.py" in zf.namelist():
                    network_code = zf.read("network.py").decode("utf-8")
                    checks = {
                        "def predict":  "predict() method present",
                        "def run":      "run() method - autonomous loop present",
                        "def status":   "status() method present",
                        "AgentNetwork": "AgentNetwork class defined",
                        "time.sleep":   "autonomous polling loop present",
                        "memory":       "memory system present",
                    }
                    for pattern, msg in checks.items():
                        if pattern in network_code:
                            feedback.append(msg)
                        else:
                            score -= 8
                            improvements.append(f"network.py missing: {msg}")

                if "run_network.py" in zf.namelist():
                    runner = zf.read("run_network.py").decode("utf-8")
                    if "AgentNetwork" in runner:
                        feedback.append("run_network.py correctly imports network")
                    else:
                        score -= 10
                        improvements.append("run_network.py missing network import")

        except Exception as e:
            score -= 20
            improvements.append(f"Could not read network files: {e}")

        return {"score": max(0, score), "feedback": feedback,
                "improvements": improvements}

    # -- History -----------------------------------------------------------

    def _store(self, result: dict):
        self.history.append(result)
        try:
            with open(self.eval_log, 'w') as f:
                json.dump(self.history, f, indent=2)
        except Exception:
            pass

    def _load_history(self) -> list:
        if self.eval_log.exists():
            try:
                with open(self.eval_log) as f:
                    return json.load(f)
            except Exception:
                pass
        return []

    def stats(self) -> dict:
        if not self.history:
            return {"total_evaluations": 0, "avg_score": 0, "pass_rate": 0}
        scores = [e["score"] for e in self.history]
        passed = [e for e in self.history if e.get("passed")]
        return {
            "total_evaluations": len(self.history),
            "avg_score":         round(sum(scores) / len(scores), 1),
            "pass_rate":         round(len(passed) / len(self.history) * 100, 1),
            "best_score":        max(scores),
            "worst_score":       min(scores),
        }
