"""
tests/verify_core3.py
Offline verification for Brain Core 3 — Architecture Advisor.

Validates API surface and schema logic without loading the model.
Add --infer to run full inference (loads Qwen ~3GB).

Usage:
    python tests/verify_core3.py          # schema-only (no model load)
    python tests/verify_core3.py --infer  # full inference
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from api.brain.cores.architecture_advisor import (
    ArchitectureAdvisor,
    REQUIRED_FIELDS,
    VALID_MODES,
)

TEST_CASES = [
    {
        "problem": (
            "Train a single image classifier on a labelled dataset of cats vs dogs. "
            "One model, one output — no pipeline needed."
        ),
        "expected_mode": "sequential",
    },
    {
        "problem": (
            "Analyze social media posts that contain both text captions and images. "
            "Run text sentiment and image classification simultaneously to save time."
        ),
        "expected_mode": "parallel",
    },
    {
        "problem": (
            "Route incoming support tickets: first classify the domain with TextAgent, "
            "then call the appropriate specialist agent based on the routing decision."
        ),
        "expected_mode": "sequential",
    },
    {
        "problem": (
            "Process a rich-media document: run OCR, audio transcription, and "
            "image captioning in parallel, then merge results with a TabularAgent."
        ),
        "expected_mode": "hybrid",
    },
    {
        "problem": (
            "Real-time network intrusion detection: simultaneously monitor packet "
            "headers, log files, and process metrics across multiple sensors."
        ),
        "expected_mode": "parallel",
    },
]


def run_schema_tests():
    core3 = ArchitectureAdvisor()

    print("=" * 60)
    print("Brain Core 3 — Architecture Advisor Verification")
    print("=" * 60)
    print()

    api_checks = [
        ("is_available() callable",      callable(core3.is_available)),
        ("advise() callable",             callable(core3.advise)),
        ("validate_output() callable",    callable(core3.validate_output)),
        ("adapter_path attribute",        hasattr(core3, "adapter_path")),
        ("adapter_path is string",        isinstance(core3.adapter_path, str)),
        ("is_available() returns bool",   isinstance(core3.is_available(), bool)),
    ]

    print("API surface checks:")
    all_api_pass = True
    for name, result in api_checks:
        status = "PASS" if result else "FAIL"
        if not result:
            all_api_pass = False
        print(f"  [{status}] {name}")

    print()

    mock_valid = {
        "execution_mode":   "parallel",
        "agent_topology":   ["ImageAgent", "TextAgent"],
        "expected_accuracy": 0.88,
        "rationale":        "Independent modalities can run simultaneously.",
    }
    mock_bad_mode   = dict(mock_valid, execution_mode="streaming")
    mock_missing    = {k: v for k, v in mock_valid.items()
                       if k != "expected_accuracy"}
    mock_upper_mode = dict(mock_valid, execution_mode="Parallel")

    schema_checks = [
        ("validate_output: valid mock",         core3.validate_output(mock_valid)),
        ("validate_output: invalid mode",       not core3.validate_output(mock_bad_mode)),
        ("validate_output: missing field",      not core3.validate_output(mock_missing)),
        ("validate_output: error dict",         not core3.validate_output({"error": "fail"})),
        ("validate_output: mode normalised",    core3.validate_output(mock_upper_mode)),
        ("REQUIRED_FIELDS has 4 items",         len(REQUIRED_FIELDS) == 4),
        ("VALID_MODES has 3 items",             len(VALID_MODES) == 3),
        ("'sequential' in VALID_MODES",         "sequential" in VALID_MODES),
        ("'parallel' in VALID_MODES",           "parallel" in VALID_MODES),
        ("'hybrid' in VALID_MODES",             "hybrid" in VALID_MODES),
    ]

    print("Schema validation checks:")
    all_schema_pass = True
    for name, result in schema_checks:
        status = "PASS" if result else "FAIL"
        if not result:
            all_schema_pass = False
        print(f"  [{status}] {name}")

    print()

    adapter_present = core3.is_available()
    print(f"Adapter on disk: {'YES — ' + core3.adapter_path if adapter_present else 'NOT FOUND (expected — skip inference)'}")
    print()

    passed = sum([all_api_pass, all_schema_pass])
    print(f"Offline checks: {passed}/2 groups passed")
    print()
    return all_api_pass and all_schema_pass


def run_inference_tests():
    core3 = ArchitectureAdvisor()

    if not core3.is_available():
        print("ERROR: Adapter not found. Place adapter in:")
        print(f"  {core3.adapter_path}")
        return False

    print("Running inference on 5 test cases...")
    print("=" * 60)

    passed = 0
    for i, tc in enumerate(TEST_CASES):
        result  = core3.advise(tc["problem"])
        valid   = core3.validate_output(result)
        mode_ok = (
            result.get("execution_mode", "").lower() == tc["expected_mode"]
            if valid else False
        )

        status = "PASS" if (valid and mode_ok) else "FAIL"
        passed += int(valid and mode_ok)

        print(f"[{i+1:02d}] [{status}] {tc['problem'][:55]}...")
        if valid:
            print(f"      mode: {result.get('execution_mode')}  "
                  f"expected: {tc['expected_mode']}  "
                  f"match: {mode_ok}")
            print(f"      topology: {result.get('agent_topology')}")
        else:
            print(f"      ERROR: {result.get('error', 'schema invalid')}")
        print()

    print("=" * 60)
    print(f"Inference results: {passed}/5 passed")

    overall = passed >= 4
    print()
    print(f"OVERALL: {'PASS' if overall else 'FAIL'}")
    return overall


if __name__ == "__main__":
    infer = "--infer" in sys.argv

    offline_ok = run_schema_tests()

    if infer:
        inference_ok = run_inference_tests()
        sys.exit(0 if (offline_ok and inference_ok) else 1)
    else:
        print("(Run with --infer to test actual model inference)")
        print()
        print(f"OVERALL: {'PASS' if offline_ok else 'FAIL'}")
        sys.exit(0 if offline_ok else 1)
