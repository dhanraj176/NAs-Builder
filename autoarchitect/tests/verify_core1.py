"""
tests/verify_core1.py
Offline verification for Brain Core 1 — Task Understander.

Runs 5 structured test cases without loading the model.
Validates schema + domain of whatever understand() returns,
OR validates against expected values when the adapter is available.

Usage:
    python tests/verify_core1.py          # schema-only (no model load)
    python tests/verify_core1.py --infer  # full inference (loads Qwen ~3GB)
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from api.brain.cores.task_understander import (
    TaskUnderstander,
    REQUIRED_FIELDS,
    VALID_DOMAINS,
)

TEST_CASES = [
    {
        "problem": (
            "I have dashcam footage from delivery trucks and need to detect "
            "potholes in real time so the fleet management system can reroute drivers."
        ),
        "expected_domain": "image",
    },
    {
        "problem": (
            "Classify customer product reviews as positive, negative, or neutral. "
            "We receive 50,000 reviews per day and need batch scoring within 1 hour."
        ),
        "expected_domain": "text",
    },
    {
        "problem": (
            "Detect fraudulent credit card transactions using 30 numerical features "
            "including transaction amount, merchant category, and historical spending patterns."
        ),
        "expected_domain": "tabular",
    },
    {
        "problem": (
            "Transcribe and classify customer service calls by issue type — "
            "billing, technical support, or general inquiry — from raw audio recordings."
        ),
        "expected_domain": "audio",
    },
    {
        "problem": (
            "Analyze dermoscopy skin lesion images to classify as benign or malignant melanoma. "
            "Dataset contains 10,000 labeled images from hospital PACS systems."
        ),
        "expected_domain": "medical",
    },
]


def run_schema_tests():
    """Validate the TaskUnderstander class API without loading the model."""
    core1 = TaskUnderstander()

    print("=" * 60)
    print("Brain Core 1 — Task Understander Verification")
    print("=" * 60)
    print()

    # API surface checks
    api_checks = [
        ("is_available() callable",      callable(core1.is_available)),
        ("understand() callable",         callable(core1.understand)),
        ("validate_output() callable",    callable(core1.validate_output)),
        ("adapter_path attribute",        hasattr(core1, "adapter_path")),
        ("adapter_path is string",        isinstance(core1.adapter_path, str)),
        ("is_available() returns bool",   isinstance(core1.is_available(), bool)),
    ]

    print("API surface checks:")
    all_api_pass = True
    for name, result in api_checks:
        status = "PASS" if result else "FAIL"
        if not result:
            all_api_pass = False
        print(f"  [{status}] {name}")

    print()

    # Validate schema on mock output (tests validate_output logic)
    mock_valid = {
        "primary_intent": "object_detection",
        "domain": "image",
        "complexity": "high",
        "real_time_required": True,
        "multi_modal": False,
        "key_entities": ["pothole", "road", "dashcam"],
    }
    mock_invalid_domain = dict(mock_valid, domain="video")
    mock_missing_field  = {k: v for k, v in mock_valid.items()
                           if k != "real_time_required"}

    schema_checks = [
        ("validate_output: valid mock",        core1.validate_output(mock_valid)),
        ("validate_output: invalid domain",    not core1.validate_output(mock_invalid_domain)),
        ("validate_output: missing field",     not core1.validate_output(mock_missing_field)),
        ("validate_output: error dict",        not core1.validate_output({"error": "fail"})),
        ("REQUIRED_FIELDS has 6 items",        len(REQUIRED_FIELDS) == 6),
        ("VALID_DOMAINS has 7 items",          len(VALID_DOMAINS) == 7),
        ("'medical' in VALID_DOMAINS",         "medical" in VALID_DOMAINS),
        ("'security' in VALID_DOMAINS",        "security" in VALID_DOMAINS),
    ]

    print("Schema validation checks:")
    all_schema_pass = True
    for name, result in schema_checks:
        status = "PASS" if result else "FAIL"
        if not result:
            all_schema_pass = False
        print(f"  [{status}] {name}")

    print()

    # Adapter availability
    adapter_present = core1.is_available()
    print(f"Adapter on disk: {'YES — ' + core1.adapter_path if adapter_present else 'NOT FOUND (expected — skip inference)'}")
    print()

    passed = sum([all_api_pass, all_schema_pass])
    total  = 2
    print(f"Offline checks: {passed}/{total} groups passed")
    print()
    return all_api_pass and all_schema_pass


def run_inference_tests():
    """Run full inference on 5 test cases (requires adapter + Qwen download)."""
    core1 = TaskUnderstander()

    if not core1.is_available():
        print("ERROR: Adapter not found. Place adapter in:")
        print(f"  {core1.adapter_path}")
        return False

    print("Running inference on 5 test cases...")
    print("=" * 60)

    passed = 0
    for i, tc in enumerate(TEST_CASES):
        result = core1.understand(tc["problem"])
        valid  = core1.validate_output(result)

        domain_ok = (
            result.get("domain", "").lower() == tc["expected_domain"]
            if valid else False
        )

        status = "PASS" if (valid and domain_ok) else "FAIL"
        passed += int(valid and domain_ok)

        print(f"[{i+1:02d}] [{status}] {tc['problem'][:55]}...")
        if valid:
            print(f"      domain: {result.get('domain')}  "
                  f"expected: {tc['expected_domain']}  "
                  f"match: {domain_ok}")
            print(f"      intent: {result.get('primary_intent')}")
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
