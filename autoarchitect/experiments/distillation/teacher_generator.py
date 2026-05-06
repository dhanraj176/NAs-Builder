"""
teacher_generator.py
====================
Distillation micro-batch generator — Day 12 of 21.

Uses DeepSeek V3 (deepseek-chat) as the teacher model to generate
structured training data for three brain-core distillation targets:

  1. TaskUnderstanding  — intent / domain / complexity parsing
  2. DomainClassification — agent routing decisions
  3. ArchitectureAdvice   — topology / execution-mode recommendation

50 seed problems × 3 cores = 150 API calls.
All responses are validated against Pydantic schemas before being saved.
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

# ── Paths ─────────────────────────────────────────────────────────────────────

_HERE      = Path(__file__).resolve().parent
_ROOT      = _HERE.parent.parent
_ENV_PATH  = _ROOT / ".env"
_OUT_DIR   = _HERE

load_dotenv(_ENV_PATH)

# ── Pydantic schemas ──────────────────────────────────────────────────────────


class TaskUnderstanding(BaseModel):
    primary_intent: str
    domain: Literal[
        "image", "text", "tabular", "audio",
        "multimodal", "medical", "security"
    ]
    complexity: Literal["simple", "medium", "complex"]
    real_time_required: bool
    multi_modal: bool
    key_entities: list[str]


class DomainClassification(BaseModel):
    primary_agent: str
    secondary_agents: list[str]
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str


class ArchitectureAdvice(BaseModel):
    execution_mode: Literal["sequential", "parallel", "hybrid"]
    agent_topology: list[str]
    expected_accuracy: float = Field(ge=0.0, le=1.0)
    rationale: str


# ── System prompts ────────────────────────────────────────────────────────────

SYSTEM_PROMPT_TASK = """You are an ML expert.
Given a plain English ML problem, output JSON matching this schema:
{
  "primary_intent": str,
  "domain": "image|text|tabular|audio|multimodal|medical|security",
  "complexity": "simple|medium|complex",
  "real_time_required": bool,
  "multi_modal": bool,
  "key_entities": [str]
}
Be concise and accurate. Output JSON only, no preamble."""

SYSTEM_PROMPT_DOMAIN = """You are an ML architect.
Given a problem, recommend the best agent from:
ImageAgent, TextAgent, TabularAgent, AudioAgent,
MultimodalAgent, MedicalAgent, SecurityAgent.

Output JSON:
{
  "primary_agent": str (one of above),
  "secondary_agents": [str] (supporting agents),
  "confidence": float 0.0-1.0,
  "reasoning": str (one sentence)
}
Output JSON only, no preamble."""

SYSTEM_PROMPT_ARCH = """You are an architecture advisor.
Given a problem, recommend execution_mode and topology.

Output JSON:
{
  "execution_mode": "sequential|parallel|hybrid",
  "agent_topology": [str] (ordered agents),
  "expected_accuracy": float 0.0-1.0,
  "rationale": str (one sentence)
}
Output JSON only, no preamble."""


# ── Generation function ───────────────────────────────────────────────────────

def generate_for_core(
    client: OpenAI,
    problems: list[dict],
    schema_class,
    system_prompt: str,
    core_name: str,
) -> tuple[list, list, int, int, float]:
    """
    Call DeepSeek for each problem with JSON mode.
    Validate every response with Pydantic.

    Returns
    -------
    results, failures, total_input_tokens, total_output_tokens, elapsed_s
    """
    results: list           = []
    failures: list          = []
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    t0 = time.time()

    for i, entry in enumerate(problems):
        problem = entry["problem"]
        call_start = time.time()
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": problem},
                ],
                response_format={"type": "json_object"},
                temperature=0.3,
                max_tokens=500,
            )
            raw       = response.choices[0].message.content
            validated = schema_class.model_validate_json(raw)
            call_ms   = round((time.time() - call_start) * 1000)

            results.append({
                "id":       entry.get("id", i + 1),
                "category": entry.get("category", "unknown"),
                "problem":  problem,
                "response": validated.model_dump(),
                "raw":      raw,
                "latency_ms": call_ms,
            })
            total_input_tokens  += response.usage.prompt_tokens
            total_output_tokens += response.usage.completion_tokens

            status = "OK"
        except Exception as e:
            call_ms = round((time.time() - call_start) * 1000)
            failures.append({
                "id":       entry.get("id", i + 1),
                "problem":  problem,
                "error":    str(e),
                "latency_ms": call_ms,
            })
            status = f"FAIL: {e}"

        print(f"  [{core_name}] {i+1:2d}/50  {status[:60]}")

    elapsed = round(time.time() - t0, 2)
    return results, failures, total_input_tokens, total_output_tokens, elapsed


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    if not api_key:
        print("[ERROR] DEEPSEEK_API_KEY not found in environment / .env")
        sys.exit(1)

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    # Load seed problems
    seed_path = _OUT_DIR / "seed_problems.json"
    with open(seed_path, "r", encoding="utf-8") as f:
        problems = json.load(f)

    print(f"\nLoaded {len(problems)} seed problems.")
    print("Starting micro batch: 50 problems x 3 cores = 150 API calls\n")

    cores = [
        ("task",     TaskUnderstanding,    SYSTEM_PROMPT_TASK,   "Task Understander"),
        ("domain",   DomainClassification, SYSTEM_PROMPT_DOMAIN, "Domain Classifier"),
        ("arch",     ArchitectureAdvice,   SYSTEM_PROMPT_ARCH,   "Architecture Advisor"),
    ]

    grand_success   = 0
    grand_fail      = 0
    grand_in_tokens = 0
    grand_out_tokens = 0
    grand_elapsed   = 0.0
    per_core_stats: list = []
    all_latencies: list  = []

    batch_start = time.time()

    for core_key, schema_cls, sys_prompt, friendly_name in cores:
        print(f"\n{'='*55}")
        print(f"  Core: {friendly_name}")
        print(f"{'='*55}")

        results, failures, in_tok, out_tok, elapsed = generate_for_core(
            client, problems, schema_cls, sys_prompt, friendly_name,
        )

        # Persist results
        res_path = _OUT_DIR / f"micro_batch_{core_key}.json"
        fail_path = _OUT_DIR / f"failures_{core_key}.json"

        with open(res_path,  "w", encoding="utf-8") as f:
            json.dump(results,   f, indent=2, ensure_ascii=False)
        with open(fail_path, "w", encoding="utf-8") as f:
            json.dump(failures,  f, indent=2, ensure_ascii=False)

        grand_success    += len(results)
        grand_fail       += len(failures)
        grand_in_tokens  += in_tok
        grand_out_tokens += out_tok
        grand_elapsed    += elapsed
        all_latencies    += [r["latency_ms"] for r in results]
        all_latencies    += [f["latency_ms"] for f in failures]

        per_core_stats.append({
            "core":          core_key,
            "friendly_name": friendly_name,
            "calls":         len(problems),
            "passed":        len(results),
            "failed":        len(failures),
            "input_tokens":  in_tok,
            "output_tokens": out_tok,
            "elapsed_s":     elapsed,
        })

        print(f"  => {len(results)} passed / {len(failures)} failed  "
              f"({in_tok + out_tok} tokens, {elapsed}s)")

    total_calls   = 150
    pass_rate     = round(grand_success / total_calls * 100, 1)
    avg_lat_ms    = round(sum(all_latencies) / max(len(all_latencies), 1))
    total_minutes = round((time.time() - batch_start) / 60, 2)

    # DeepSeek V3 pricing (as of 2024-12)
    # Input: $0.27 / 1M tokens (cache miss), output: $1.10 / 1M tokens
    cost_in   = grand_in_tokens  / 1_000_000 * 0.27
    cost_out  = grand_out_tokens / 1_000_000 * 1.10
    total_cost = round(cost_in + cost_out, 4)

    # Scale to 1000-problem full run (3 cores)
    scale_factor    = 1000 / 50
    proj_cost_1000  = round(total_cost * scale_factor, 2)
    proj_cost_ok    = proj_cost_1000 < 1.0
    pass_rate_ok    = pass_rate > 95.0
    avg_resp_ok     = avg_lat_ms < 10_000   # < 10 seconds

    overall_ok = pass_rate_ok and proj_cost_ok and avg_resp_ok

    # ── Report ────────────────────────────────────────────────────────────────
    sep = "=" * 55

    report_lines = [
        sep,
        "DISTILLATION MICRO BATCH REPORT",
        sep,
        f"Teacher model: deepseek-chat (V3)",
        f"Total API calls: {total_calls}",
        f"Successful: {grand_success} / {total_calls}",
        f"Failed: {grand_fail} / {total_calls}",
        f"Schema pass rate: {pass_rate}%",
        "",
        "Per-core breakdown:",
    ]
    for s in per_core_stats:
        report_lines.append(
            f"  {s['friendly_name']}: {s['calls']} calls, "
            f"{s['passed']} passed, {s['failed']} failed"
        )
    report_lines += [
        "",
        "Token usage:",
        f"  Input tokens:  {grand_in_tokens:,}",
        f"  Output tokens: {grand_out_tokens:,}",
        f"  Cost (50 probs): ${total_cost:.4f}",
        "",
        "Performance:",
        f"  Average response time: {avg_lat_ms / 1000:.1f}s",
        f"  Total elapsed time: {total_minutes} minutes",
        "",
        f"Projected cost for 1000 prompts (3 cores each):",
        f"  Estimated: ${proj_cost_1000:.2f}",
        "",
        "ACCEPTANCE GATE:",
        f"  Pass rate > 95%:           {'YES' if pass_rate_ok else 'NO'}  ({pass_rate}%)",
        f"  Cost < $1 for 1000 prompts:{'YES' if proj_cost_ok else 'NO'}  (${proj_cost_1000:.2f})",
        f"  Avg response < 10s:        {'YES' if avg_resp_ok else 'NO'}  ({avg_lat_ms/1000:.1f}s)",
        "",
        f"OVERALL: {'PROCEED TO FULL DATASET' if overall_ok else 'FIX ISSUES BEFORE PROCEEDING'}",
        sep,
    ]

    print("\n\n" + "\n".join(report_lines))

    if not overall_ok:
        print("\nFailure reasons:")
        if not pass_rate_ok:
            print(f"  - Pass rate {pass_rate}% below 95% threshold")
            for s in per_core_stats:
                if s["failed"] > 0:
                    print(f"    {s['friendly_name']}: {s['failed']} failures — "
                          f"check failures_{s['core']}.json")
        if not proj_cost_ok:
            print(f"  - Projected cost ${proj_cost_1000:.2f} exceeds $1.00 budget")
        if not avg_resp_ok:
            print(f"  - Average response {avg_lat_ms/1000:.1f}s exceeds 10s limit")
        print("\nRecommendations:")
        print("  1. Review failure JSON files for schema validation errors")
        print("  2. Tighten system prompts to enforce JSON structure")
        print("  3. Consider reducing max_tokens if cost is too high")

    # Save JSON report
    report_dict = {
        "generated_at":          datetime.now().isoformat(),
        "teacher_model":         "deepseek-chat",
        "total_calls":           total_calls,
        "successful":            grand_success,
        "failed":                grand_fail,
        "pass_rate_pct":         pass_rate,
        "per_core":              per_core_stats,
        "token_usage": {
            "input_tokens":      grand_in_tokens,
            "output_tokens":     grand_out_tokens,
            "total_tokens":      grand_in_tokens + grand_out_tokens,
            "cost_usd":          total_cost,
        },
        "performance": {
            "avg_latency_ms":    avg_lat_ms,
            "total_elapsed_min": total_minutes,
        },
        "projection_1000": {
            "estimated_cost_usd": proj_cost_1000,
        },
        "acceptance_gate": {
            "pass_rate_ok":   pass_rate_ok,
            "cost_ok":        proj_cost_ok,
            "latency_ok":     avg_resp_ok,
            "overall":        overall_ok,
        },
    }

    report_path = _OUT_DIR / "micro_batch_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)

    # Save markdown report
    md_path = _OUT_DIR / "MICRO_BATCH_REPORT.md"
    md_lines = ["# Distillation Micro Batch Report\n"] + [
        f"```\n{chr(10).join(report_lines)}\n```"
    ]
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"\nReports saved:")
    print(f"  {report_path}")
    print(f"  {md_path}")

    sys.exit(0 if overall_ok else 1)


if __name__ == "__main__":
    main()
