# ============================================
# AutoArchitect — Workflow Engine
# Like n8n but for NAS agents
# ============================================

import time

MULTI_DOMAIN_PATTERNS = [
    (["image", "text"],     ["detect", "classify", "and"]),
    (["image", "text"],     ["visual", "description"]),
    (["image", "text"],     ["photo", "report"]),
    (["image", "text"],     ["scan", "analyze", "report"]),
    (["image", "medical"],  ["xray", "report"]),
    (["image", "medical"],  ["scan", "diagnos"]),
    (["image", "medical"],  ["mri", "detect"]),
    (["text", "security"],  ["email", "fraud"]),
    (["text", "security"],  ["message", "threat"]),
    (["text", "security"],  ["phishing", "detect"]),
    (["image", "security"], ["face", "fraud"]),
    (["image", "security"], ["camera", "intrusion"]),
]


class WorkflowEngine:
    def __init__(self):
        print("⚙️  WorkflowEngine ready!")

    def build_workflow(self, problem: str, primary_domain: str) -> dict:
        lower   = problem.lower()
        domains = self._detect_domains(lower, primary_domain)

        if len(domains) > 1:
            print(f"  🔀 Multi-agent workflow: {domains}")
            return {
                "type":   "multi",
                "agents": domains,
                "steps":  [
                    f"Step {i+1}: Run {d.upper()} NAS Agent"
                    for i, d in enumerate(domains)
                ] + [
                    f"Step {len(domains)+1}: Fusion Agent combines architectures",
                    f"Step {len(domains)+2}: Evaluator Agent tests combined model",
                    f"Step {len(domains)+3}: Cache complete pipeline forever",
                ]
            }
        else:
            print(f"  ➡️  Single-agent workflow: {domains[0]}")
            return {
                "type":   "single",
                "agents": domains,
                "steps":  [
                    f"Step 1: Run {domains[0].upper()} NAS Agent",
                    "Step 2: Evaluator Agent tests result",
                    "Step 3: Cache solution forever",
                ]
            }

    def _detect_domains(self, problem: str, primary: str) -> list:
        for domains, keywords in MULTI_DOMAIN_PATTERNS:
            if all(kw in problem for kw in keywords):
                return domains

        if " and " in problem or " with " in problem:
            detected = []
            if any(w in problem for w in ["image","photo","picture","detect","visual","camera"]):
                detected.append("image")
            if any(w in problem for w in ["text","classify","sentiment","spam","review"]):
                detected.append("text")
            if any(w in problem for w in ["medical","xray","mri","scan","diagnos","tumor"]):
                detected.append("medical")
            if any(w in problem for w in ["fraud","security","threat","attack","intrusion","anomaly"]):
                detected.append("security")
            if len(detected) > 1:
                return detected

        return [primary]
