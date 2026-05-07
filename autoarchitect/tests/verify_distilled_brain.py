# -*- coding: utf-8 -*-
"""
tests/verify_distilled_brain.py -- Day 20: DistilledBrain integration tests

Validates DistilledBrain API surface and end-to-end routing via
think_with_fallback() (works offline; falls back to MetaNet if Qwen
base model not yet downloaded).

Usage:
    python tests/verify_distilled_brain.py          # offline + integration
    python tests/verify_distilled_brain.py --infer  # full inference (needs Qwen)
"""

import sys
import os
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# -- Test problems -------------------------------------------------------------

TEST_CASES = [
    {
        "problem":          "Detect potholes in road images",
        "expected_domain":  "image",
        "expected_agent":   "ImageAgent",
        "expected_modes":   {"sequential", "parallel", "hybrid"},
    },
    {
        "problem":          "Classify customer reviews by sentiment",
        "expected_domain":  "text",
        "expected_agent":   "TextAgent",
        "expected_modes":   {"sequential", "parallel", "hybrid"},
    },
    {
        "problem":          "Predict credit card fraud from transactions",
        "expected_domain":  "tabular",
        "expected_agent":   "TabularAgent",
        "expected_modes":   {"sequential", "parallel", "hybrid"},
    },
    {
        "problem":          "Transcribe customer service calls",
        "expected_domain":  "audio",
        "expected_agent":   "AudioAgent",
        "expected_modes":   {"sequential", "parallel", "hybrid"},
    },
    {
        "problem":          "Identify skin lesions from photos",
        "expected_domain":  "medical",
        "expected_agent":   "MedicalAgent",
        "expected_modes":   {"sequential", "parallel", "hybrid"},
    },
]

VALID_AGENTS = {
    "ImageAgent", "TextAgent", "TabularAgent", "AudioAgent",
    "MultimodalAgent", "MedicalAgent", "SecurityAgent",
}
VALID_MODES = {"sequential", "parallel", "hybrid"}


# -- Offline API tests (no model load) ----------------------------------------

def run_api_tests() -> bool:
    print("\n-- Offline API Tests -------------------------------------------")
    from api.brain.distilled_brain import DistilledBrain, get_distilled_brain

    checks = [
        ("DistilledBrain class exists",         True),
        ("get_distilled_brain() callable",       callable(get_distilled_brain)),
        ("singleton returns same instance",
             get_distilled_brain() is get_distilled_brain()),
        ("think() callable",                    callable(DistilledBrain().think)),
        ("think_with_fallback() callable",      callable(DistilledBrain().think_with_fallback)),
        ("is_available() callable",             callable(DistilledBrain().is_available)),
        ("validate_output() callable",          callable(DistilledBrain().validate_output)),
        ("all 3 adapters on disk",              get_distilled_brain().is_available()),
    ]

    all_pass = True
    for name, result in checks:
        status = "PASS" if result else "FAIL"
        print(f"   [{status}] {name}")
        if not result:
            all_pass = False

    return all_pass


def run_schema_tests() -> bool:
    print("\n-- Schema / validate_output Tests -----------------------------")
    from api.brain.distilled_brain import get_distilled_brain
    brain = get_distilled_brain()

    # Full valid brain output
    mock_valid = {
        "understanding":  {"primary_intent": "classify", "domain": "image",
                           "complexity": "medium", "confidence": 0.9},
        "classification": {"primary_agent": "ImageAgent",
                           "secondary_agents": [], "confidence": 0.95,
                           "reasoning": "visual task"},
        "architecture":   {"execution_mode": "sequential",
                           "agent_topology": ["ImageAgent"],
                           "expected_accuracy": 0.85,
                           "rationale": "single agent sufficient"},
        "source":         "distilled_brain_v1",
        "confidence":     0.85,
    }
    mock_fallback = {
        "metanet_result":  {"predicted": False},
        "source":          "metanet_fallback",
        "fallback_reason": "model not loaded",
    }
    mock_missing = {"source": "distilled_brain_v1"}
    mock_empty   = {}

    schema_checks = [
        ("validate_output: full brain result",  brain.validate_output(mock_valid)),
        ("validate_output: metanet fallback",   brain.validate_output(mock_fallback)),
        ("validate_output: missing sections",   not brain.validate_output(mock_missing)),
        ("validate_output: empty dict",         not brain.validate_output(mock_empty)),
    ]

    all_pass = True
    for name, result in schema_checks:
        status = "PASS" if result else "FAIL"
        print(f"   [{status}] {name}")
        if not result:
            all_pass = False

    return all_pass


# -- Integration tests (think_with_fallback — always works) --------------------

def run_integration_tests() -> bool:
    """
    Tests think_with_fallback() for all 5 problems.
    Works offline: falls back to MetaNet if Qwen not downloaded.
    PASS criteria: no exception, result has expected structure.
    """
    print("\n-- Integration Tests (think_with_fallback) ---------------------")
    from api.brain.distilled_brain import get_distilled_brain
    brain = get_distilled_brain()

    all_pass = True

    for tc in TEST_CASES:
        problem = tc["problem"]
        t0 = time.time()
        try:
            result  = brain.think_with_fallback(problem)
            elapsed = round(time.time() - t0, 3)
            source  = result.get("source", "unknown")
            valid   = brain.validate_output(result)

            if source == "distilled_brain_v1":
                agent = result.get("classification", {}).get("primary_agent", "")
                mode  = result.get("architecture",   {}).get("execution_mode", "")
                agent_ok = agent in VALID_AGENTS
                mode_ok  = mode in VALID_MODES
                status   = "PASS" if (valid and agent_ok and mode_ok) else "FAIL"
                tag      = f"[{source}] agent={agent} mode={mode}"
            else:
                # MetaNet fallback — model not downloaded yet
                status = "PASS" if valid else "FAIL"
                tag    = f"[{source}] (Qwen not downloaded — fallback active)"

            print(f"   [{status}] {problem[:45]:<45}  {elapsed:.2f}s  {tag}")
            if status == "FAIL":
                all_pass = False

        except Exception as e:
            elapsed = round(time.time() - t0, 3)
            print(f"   [FAIL] {problem[:45]:<45}  {elapsed:.2f}s  ERROR: {e}")
            traceback.print_exc()
            all_pass = False

    return all_pass


# -- Full inference tests (requires --infer + Qwen download) ------------------

def run_inference_tests() -> bool:
    """
    Tests think() directly — requires Qwen2.5-1.5B downloaded (~3GB, one-time).
    Checks all 3 sections, agent validity, and execution mode.
    """
    print("\n-- Full Inference Tests (think) --------------------------------")
    print("   NOTE: First run downloads Qwen2.5-1.5B (~3GB). Subsequent runs use cache.")
    from api.brain.distilled_brain import get_distilled_brain
    brain = get_distilled_brain()

    all_pass   = True
    pass_count = 0

    for tc in TEST_CASES:
        problem = tc["problem"]
        t0 = time.time()
        try:
            result  = brain.think(problem)
            elapsed = round(time.time() - t0, 3)

            has_understanding  = "understanding" in result
            has_classification = "classification" in result
            has_architecture   = "architecture" in result
            agent = result.get("classification", {}).get("primary_agent", "")
            mode  = result.get("architecture",   {}).get("execution_mode", "")
            agent_ok = agent in VALID_AGENTS
            mode_ok  = mode  in VALID_MODES

            ok = all([has_understanding, has_classification,
                      has_architecture, agent_ok, mode_ok])
            status = "PASS" if ok else "FAIL"

            if ok:
                pass_count += 1
            else:
                all_pass = False

            print(f"   [{status}] {problem[:40]:<40}  {elapsed:.2f}s")
            print(f"          agent={agent}  mode={mode}")
            if not agent_ok:
                print(f"          WARN: invalid agent: {agent!r}")
            if not mode_ok:
                print(f"          WARN: invalid mode:  {mode!r}")

        except Exception as e:
            elapsed = round(time.time() - t0, 3)
            print(f"   [FAIL] {problem[:40]:<40}  {elapsed:.2f}s  ERROR: {e}")
            all_pass = False

    print(f"\n   Inference result: {pass_count}/{len(TEST_CASES)} PASSED")
    return all_pass


# -- Main ----------------------------------------------------------------------

if __name__ == "__main__":
    full_infer = "--infer" in sys.argv

    print("=" * 65)
    print("  DistilledBrain Verification — Day 20")
    print("=" * 65)

    api_ok    = run_api_tests()
    schema_ok = run_schema_tests()
    integ_ok  = run_integration_tests()

    if full_infer:
        infer_ok = run_inference_tests()
    else:
        print("\n-- Full Inference -----------------------------------------------")
        print("   Skipped (pass --infer to run; requires Qwen2.5-1.5B download)")
        infer_ok = True

    all_ok = api_ok and schema_ok and integ_ok and infer_ok

    print("\n" + "=" * 65)
    print(f"  API tests:         {'PASS' if api_ok    else 'FAIL'}")
    print(f"  Schema tests:      {'PASS' if schema_ok else 'FAIL'}")
    print(f"  Integration tests: {'PASS' if integ_ok  else 'FAIL'}")
    if full_infer:
        print(f"  Inference tests:   {'PASS' if infer_ok  else 'FAIL'}")
    print(f"  OVERALL:           {'PASS' if all_ok    else 'FAIL'}")
    print("=" * 65)

    sys.exit(0 if all_ok else 1)
