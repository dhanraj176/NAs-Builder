# ============================================
# AutoArchitect — Output Generator
# Turns technical JSON into human readable
# actionable results using Groq LLM
# Works for ANY problem automatically
# ============================================

import os
import requests
import json

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL   = "llama-3.1-8b-instant"


def generate_output(problem: str,
                    result: dict,
                    groq_key: str = "") -> dict:
    """
    Convert technical NAS result into
    human readable actionable output.

    Works for ANY problem:
    → "analyze YouTube thumbnail" → viral score + tips
    → "screen resumes"           → candidate match + reasons
    → "verify news article"      → credibility score + flags
    → "monitor cameras"          → threat level + actions
    → "diagnose symptoms"        → risk assessment + next steps
    """
    if not groq_key:
        groq_key = os.getenv("GROQ_API_KEY", "")

    # Build context from result
    agents_used   = result.get("agents_used", [])
    avg_accuracy  = result.get("avg_accuracy",
                    result.get("test_accuracy", 0))
    eval_score    = 0
    eval_feedback = []

    evaluation = result.get("evaluation", {})
    if evaluation:
        eval_score    = evaluation.get("avg_score", 0)
        eval_feedback = evaluation.get("feedback", [])

    all_accuracies = result.get("all_accuracies", {})
    from_cache     = result.get("from_cache", False)

    # Build scores summary for LLM
    scores_text = ""
    if all_accuracies:
        scores_text = "\n".join([
            f"- {domain.upper()} Agent: {acc}% accuracy"
            for domain, acc in all_accuracies.items()
        ])
    elif avg_accuracy:
        scores_text = f"- Overall accuracy: {avg_accuracy}%"

    if eval_score:
        scores_text += f"\n- Architecture quality: {eval_score}%"

    prompt = f"""You are AutoArchitect AI, an expert system that analyzes problems and provides actionable insights.

Problem: "{problem}"

Technical Analysis Results:
{scores_text if scores_text else "Analysis complete"}
Agents deployed: {', '.join(agents_used) if agents_used else 'auto-selected'}
From cache: {from_cache}

Generate a concise, actionable report for a NON-TECHNICAL user.
Format your response as JSON with these exact fields:
{{
    "overall_score": <number 0-100>,
    "verdict": "<one word: Excellent/Good/Fair/Poor>",
    "summary": "<2 sentence summary of what was found>",
    "findings": [
        "<finding 1 with emoji>",
        "<finding 2 with emoji>",
        "<finding 3 with emoji>"
    ],
    "recommendations": [
        "<specific action 1>",
        "<specific action 2>",
        "<specific action 3>"
    ],
    "next_steps": "<one clear next step the user should take>",
    "confidence": "<High/Medium/Low>"
}}

Rules:
- Be specific to the problem "{problem}"
- Use emojis in findings (✅ ⚠️ ❌ 🔍 📊)
- Make recommendations actionable and specific
- Keep each item under 15 words
- Overall score should reflect real usefulness
- Do NOT mention neural networks or technical terms
- Write as if talking to a business owner
- Return ONLY valid JSON, nothing else"""

    try:
        if not groq_key:
            return _fallback_output(problem, eval_score,
                                     avg_accuracy, agents_used)

        headers = {
            "Authorization": f"Bearer {groq_key}",
            "Content-Type":  "application/json",
        }
        payload = {
            "model":       GROQ_MODEL,
            "messages":    [{"role": "user", "content": prompt}],
            "max_tokens":  500,
            "temperature": 0.3,
        }
        resp = requests.post(
            GROQ_API_URL, json=payload,
            headers=headers, timeout=30
        )
        resp.raise_for_status()
        data    = resp.json()
        content = data["choices"][0]["message"]["content"]

        # Clean and parse JSON
        content = content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        content = content.strip()

        output = json.loads(content)

        # Ensure all fields exist
        output.setdefault("overall_score",   eval_score or 75)
        output.setdefault("verdict",         "Good")
        output.setdefault("summary",         f"Analysis complete for: {problem}")
        output.setdefault("findings",        ["✅ Analysis completed successfully"])
        output.setdefault("recommendations", ["Review the results and take action"])
        output.setdefault("next_steps",      "Apply the recommendations above")
        output.setdefault("confidence",      "Medium")

        output["generated_by"] = "Llama 3 via Groq"
        output["problem"]      = problem
        return output

    except json.JSONDecodeError as e:
        print(f"⚠️ JSON parse error: {e}")
        return _fallback_output(problem, eval_score,
                                 avg_accuracy, agents_used)
    except Exception as e:
        print(f"⚠️ Output generator error: {e}")
        return _fallback_output(problem, eval_score,
                                 avg_accuracy, agents_used)


def _fallback_output(problem: str, eval_score: float,
                      accuracy: float,
                      agents: list) -> dict:
    """
    Fallback when Groq unavailable.
    Still useful, rule-based output.
    """
    score = eval_score or accuracy or 70

    if score >= 85:
        verdict  = "Excellent"
        emoji    = "🏆"
        summary  = (f"AutoArchitect successfully analyzed your problem "
                    f"and built a high-quality AI solution.")
    elif score >= 70:
        verdict  = "Good"
        emoji    = "✅"
        summary  = (f"AutoArchitect built a solid AI solution "
                    f"for your problem.")
    elif score >= 55:
        verdict  = "Fair"
        emoji    = "⚠️"
        summary  = (f"AutoArchitect built a working solution. "
                    f"More data would improve results.")
    else:
        verdict  = "Needs Work"
        emoji    = "🔧"
        summary  = (f"AutoArchitect built an initial solution. "
                    f"Upload more labeled data for better accuracy.")

    agent_str = ", ".join(agents) if agents else "AI"

    return {
        "overall_score":   round(score),
        "verdict":         verdict,
        "summary":         summary,
        "findings": [
            f"{emoji} {agent_str.upper()} agent analyzed your problem",
            f"📊 Architecture quality score: {eval_score}%",
            f"🧠 Solution cached — instant next time",
        ],
        "recommendations": [
            "Upload your own labeled data for higher accuracy",
            "Test with real examples from your use case",
            "Run the same problem again to see cache speedup",
        ],
        "next_steps":  "Upload 50+ labeled examples for 85%+ accuracy",
        "confidence":  "High" if score >= 70 else "Medium",
        "generated_by": "AutoArchitect (offline mode)",
        "problem":      problem
    }


def generate_workflow_output(problem: str,
                              workflow_results: list,
                              groq_key: str = "") -> dict:
    """
    Generate output for complete workflow pipelines.
    Used when multiple agents process different inputs.

    workflow_results = [
        {"agent": "image", "score": 87, "input": "thumbnail"},
        {"agent": "text",  "score": 72, "input": "title"},
    ]
    """
    if not groq_key:
        groq_key = os.getenv("GROQ_API_KEY", "")

    scores_text = "\n".join([
        f"- {r['agent'].upper()} analysis of {r.get('input','data')}: "
        f"{r.get('score', 0)}%"
        for r in workflow_results
    ])

    overall = round(
        sum(r.get("score", 0) for r in workflow_results) /
        max(len(workflow_results), 1)
    )

    prompt = f"""You are AutoArchitect AI analyzing a multi-component problem.

Problem: "{problem}"

Analysis Results:
{scores_text}
Overall Score: {overall}/100

Generate a business-friendly actionable report as JSON:
{{
    "overall_score": {overall},
    "verdict": "<Excellent/Good/Fair/Poor>",
    "summary": "<what was analyzed and key finding>",
    "findings": [
        "<specific finding about component 1 with emoji>",
        "<specific finding about component 2 with emoji>",
        "<overall pattern finding with emoji>"
    ],
    "recommendations": [
        "<specific improvement for component 1>",
        "<specific improvement for component 2>",
        "<strategic recommendation>"
    ],
    "next_steps": "<single most important action>",
    "confidence": "<High/Medium/Low>"
}}

Be specific to "{problem}". No technical jargon.
Return ONLY valid JSON."""

    try:
        if not groq_key:
            return _fallback_output(
                problem, overall, overall, [])

        headers = {
            "Authorization": f"Bearer {groq_key}",
            "Content-Type":  "application/json",
        }
        payload = {
            "model":       GROQ_MODEL,
            "messages":    [{"role": "user", "content": prompt}],
            "max_tokens":  500,
            "temperature": 0.3,
        }
        resp = requests.post(
            GROQ_API_URL, json=payload,
            headers=headers, timeout=30
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        content = content.strip()
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]

        output = json.loads(content.strip())
        output["generated_by"] = "Llama 3 via Groq"
        output["problem"]      = problem
        return output

    except Exception as e:
        print(f"⚠️ Workflow output error: {e}")
        return _fallback_output(
            problem, overall, overall, [])
