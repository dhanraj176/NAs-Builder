"""
tests/verify_core2.py
Offline verification for Brain Core 2 — Domain Classifier.

Validates API surface and schema logic without loading the model.
Add --infer to run full inference (loads Qwen ~3GB).

Usage:
    python tests/verify_core2.py          # schema-only (no model load)
    python tests/verify_core2.py --infer  # full inference
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from api.brain.cores.domain_classifier import (
    DomainClassifier,
    REQUIRED_FIELDS,
    VALID_AGENTS,
)

TEST_CASES = [
    {
        "problem": (
            "Classify customer emails as spam or not spam. "
            "We receive 100,000 emails per day in plain text format."
        ),
        "expected_agent": "TextAgent",
    },
    {
        "problem": (
            "Detect manufacturing defects in PCB board images "
            "captured by high-resolution cameras on the assembly line."
        ),
        "expected_agent": "ImageAgent",
    },
    {
        "problem": (
            "Predict house prices from square footage, number of rooms, "
            "zip code, school district rating, and recent sale comparables."
        ),
        "expected_agent": "TabularAgent",
    },
    {
        "problem": (
            "Identify music genres (rock, jazz, classical, hip-hop) "
            "from raw audio files uploaded by streaming platform users."
        ),
        "expected_agent": "AudioAgent",
    },
    {
        "problem": (
            "Analyze dermoscopy skin lesion images from hospital PACS "
            "to classify benign vs malignant melanoma for dermatologists."
        ),
        "expected_agent": "MedicalAgent",
    },
]


def run_schema_tests():
    core2 = DomainClassifier()

    print("=" * 60)
    print("Brain Core 2 — Domain Classifier Verification")
    print("=" * 60)
    print()

    api_checks = [
        ("is_available() callable",      callable(core2.is_available)),
        ("classify() callable",           callable(core2.classify)),
        ("validate_output() callable",    callable(core2.validate_output)),
        ("adapter_path attribute",        hasattr(core2, "adapter_path")),
        ("adapter_path is string",        isinstance(core2.adapter_path, str)),
        ("is_available() returns bool",   isinstance(core2.is_available(), bool)),
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
        "primary_agent":    "TextAgent",
        "secondary_agents": ["TabularAgent"],
        "confidence":       0.92,
        "reasoning":        "Plain text input, classification task.",
    }
    mock_bad_agent   = dict(mock_valid, primary_agent="VideoAgent")
    mock_missing     = {k: v for k, v in mock_valid.items()
                        if k != "confidence"}

    schema_checks = [
        ("validate_output: valid mock",        core2.validate_output(mock_valid)),
        ("validate_output: invalid agent",     not core2.validate_output(mock_bad_agent)),
        ("validate_output: missing field",     not core2.validate_output(mock_missing)),
        ("validate_output: error dict",        not core2.validate_output({"error": "fail"})),
        ("REQUIRED_FIELDS has 4 items",        len(REQUIRED_FIELDS) == 4),
        ("VALID_AGENTS has 7 items",           len(VALID_AGENTS) == 7),
        ("'MedicalAgent' in VALID_AGENTS",     "MedicalAgent" in VALID_AGENTS),
        ("'SecurityAgent' in VALID_AGENTS",    "SecurityAgent" in VALID_AGENTS),
        ("'MultimodalAgent' in VALID_AGENTS",  "MultimodalAgent" in VALID_AGENTS),
    ]

    print("Schema validation checks:")
    all_schema_pass = True
    for name, result in schema_checks:
        status = "PASS" if result else "FAIL"
        if not result:
            all_schema_pass = False
        print(f"  [{status}] {name}")

    print()

    adapter_present = core2.is_available()
    print(f"Adapter on disk: {'YES — ' + core2.adapter_path if adapter_present else 'NOT FOUND (expected — skip inference)'}")
    print()

    passed = sum([all_api_pass, all_schema_pass])
    print(f"Offline checks: {passed}/2 groups passed")
    print()
    return all_api_pass and all_schema_pass


def run_inference_tests():
    core2 = DomainClassifier()

    if not core2.is_available():
        print("ERROR: Adapter not found. Place adapter in:")
        print(f"  {core2.adapter_path}")
        return False

    print("Running inference on 5 test cases...")
    print("=" * 60)

    passed = 0
    for i, tc in enumerate(TEST_CASES):
        result   = core2.classify(tc["problem"])
        valid    = core2.validate_output(result)
        agent_ok = (
            result.get("primary_agent") == tc["expected_agent"]
            if valid else False
        )

        status = "PASS" if (valid and agent_ok) else "FAIL"
        passed += int(valid and agent_ok)

        print(f"[{i+1:02d}] [{status}] {tc['problem'][:55]}...")
        if valid:
            print(f"      agent: {result.get('primary_agent')}  "
                  f"expected: {tc['expected_agent']}  "
                  f"match: {agent_ok}")
            print(f"      confidence: {result.get('confidence')}")
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
