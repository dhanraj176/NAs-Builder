# -*- coding: utf-8 -*-
"""
Day 11 verification -- Learned fusion weights from topology history.

Tests:
  1. learn_weights_from_cache() creates fusion_weights.json
  2. image domain has image_agent key
  3. image domain has multimodal_agent key
  4. image_agent weight > multimodal_agent weight (more historical credit)
  5. weights are in [0, 1] range
  6. get_weights_for_domain() returns normalized weights (sum to 1.0)
  7. get_weights_for_domain() uses 0.5 default for unknown agents
  8. get_weights_for_domain() returns 'learned' source when found
  9. get_weights_for_domain() returns 'default' source when not found
 10. fuse() uses learned weights when domain provided
 11. image_agent wins fusion over multimodal_agent due to higher weight
 12. fuse() output includes weights_source field
 13. weights_source = 'learned' when domain weights exist
 14. weights_source = 'provided' when explicit weights given
 15. weights update after new successful entry in history
"""

import sys
import os
import json
import tempfile
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from api.agents.fusion_agent import FusionAgent, _class_to_weight_key, refresh_fusion_weights


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_history(entries: list, tmp_dir: str) -> str:
    path = os.path.join(tmp_dir, "topology_history.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)
    return path


def _make_weights_path(tmp_dir: str) -> str:
    return os.path.join(tmp_dir, "fusion_weights.json")


def _image_entry(acc: float, agents: list = None) -> dict:
    agents = agents or ["image", "report"]
    return {
        "problem": f"detect potholes in road images (acc={acc})",
        "topology": {"agents": agents},
        "accuracy": acc,
    }


def _five_mock_entries() -> list:
    """
    3 image-only entries at 0.85, 2 multi-agent entries (image+multimodal) at 0.75.

    Expected accumulated weights (domain='image'):
      image_agent:     (0.85*3 + 0.75*2) / 5 = 0.81
      multimodal_agent:(0.75*2)           / 2 = 0.75
    """
    return [
        _image_entry(0.85, ["image", "report"]),
        _image_entry(0.85, ["image", "report"]),
        _image_entry(0.85, ["image", "report"]),
        _image_entry(0.75, ["image", "multimodal", "report"]),
        _image_entry(0.75, ["image", "multimodal", "report"]),
    ]


# ---------------------------------------------------------------------------
# Tests 1-5: learn_weights_from_cache()
# ---------------------------------------------------------------------------

def test_learn_weights():
    print("\n-- Tests 1-5: learn_weights_from_cache() --")
    tmp = tempfile.mkdtemp(prefix="d11_")
    try:
        h_path = _make_history(_five_mock_entries(), tmp)
        w_path = _make_weights_path(tmp)

        agent = FusionAgent.__new__(FusionAgent)
        agent._weight_cache  = {}
        agent._weights_cache = {}

        weights = agent.learn_weights_from_cache(
            history_path=h_path,
            weights_path=w_path,
        )

        r = {}

        # 1. File created
        r["fusion_weights.json created"] = os.path.exists(w_path)
        print(f"  [{'PASS' if r['fusion_weights.json created'] else 'FAIL'}] "
              f"fusion_weights.json exists")

        img_domain = weights.get("image", {})
        print(f"  image domain weights: {img_domain}")

        # 2. image_agent key present
        r["image domain has image_agent"] = "image_agent" in img_domain
        print(f"  [{'PASS' if r['image domain has image_agent'] else 'FAIL'}] "
              f"image_agent in image domain")

        # 3. multimodal_agent key present
        r["image domain has multimodal_agent"] = "multimodal_agent" in img_domain
        print(f"  [{'PASS' if r['image domain has multimodal_agent'] else 'FAIL'}] "
              f"multimodal_agent in image domain")

        # 4. image_agent > multimodal_agent
        img_w   = img_domain.get("image_agent", 0)
        multi_w = img_domain.get("multimodal_agent", 0)
        r["image_agent weight > multimodal_agent"] = img_w > multi_w
        print(f"  [{'PASS' if r['image_agent weight > multimodal_agent'] else 'FAIL'}] "
              f"image_agent ({img_w:.4f}) > multimodal_agent ({multi_w:.4f})")

        # 5. weights in [0, 1]
        all_vals = [v for d in weights.values() for v in d.values()]
        in_range = all(0.0 <= v <= 1.0 for v in all_vals)
        r["weights in [0,1] range"] = in_range
        print(f"  [{'PASS' if in_range else 'FAIL'}] all weights in [0, 1]  "
              f"values={[round(v,3) for v in all_vals]}")

        return r
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Tests 6-9: get_weights_for_domain()
# ---------------------------------------------------------------------------

def test_get_weights():
    print("\n-- Tests 6-9: get_weights_for_domain() --")
    tmp = tempfile.mkdtemp(prefix="d11_")
    try:
        h_path = _make_history(_five_mock_entries(), tmp)
        w_path = _make_weights_path(tmp)

        agent = FusionAgent.__new__(FusionAgent)
        agent._weight_cache  = {}
        agent._weights_cache = {}
        agent.learn_weights_from_cache(history_path=h_path, weights_path=w_path)

        r = {}

        # 6. Normalized weights sum to 1.0
        weights, source = agent.get_weights_for_domain(
            "image", ["ImageAgent", "MultimodalAgent"],
            weights_path=w_path,
        )
        total = round(sum(weights.values()), 5)
        r["normalized weights sum to 1.0"] = abs(total - 1.0) < 1e-4
        print(f"  [{'PASS' if r['normalized weights sum to 1.0'] else 'FAIL'}] "
              f"sum={total}  weights={weights}")

        # 7. Unknown agent gets 0.5 default (before normalization)
        weights2, source2 = agent.get_weights_for_domain(
            "image", ["ImageAgent", "UnknownAgent99"],
            weights_path=w_path,
        )
        # ImageAgent has learned weight ~0.81, UnknownAgent99 defaults to 0.5.
        # After normalization ImageAgent fraction > 0.5.
        img_frac = weights2.get("ImageAgent", 0)
        r["unknown agent gets 0.5 default"] = 0.49 < img_frac < 1.0
        print(f"  [{'PASS' if r['unknown agent gets 0.5 default'] else 'FAIL'}] "
              f"ImageAgent frac={img_frac:.4f} (should be > 0.5 since it has learned weight)")

        # 8. 'learned' source when domain has weights
        r["learned source when weights found"] = source == "learned"
        print(f"  [{'PASS' if r['learned source when weights found'] else 'FAIL'}] "
              f"source={source!r}")

        # 9. 'default' source when domain not found
        _, source3 = agent.get_weights_for_domain(
            "nonexistent_domain", ["SomeAgent"],
            weights_path=w_path,
        )
        r["default source when domain not found"] = source3 == "default"
        print(f"  [{'PASS' if r['default source when domain not found'] else 'FAIL'}] "
              f"source={source3!r}")

        return r
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Tests 10-14: fuse() with learned weights
# ---------------------------------------------------------------------------

def test_fuse_with_domain():
    print("\n-- Tests 10-14: fuse() with learned domain weights --")
    tmp = tempfile.mkdtemp(prefix="d11_")
    try:
        h_path = _make_history(_five_mock_entries(), tmp)
        w_path = _make_weights_path(tmp)

        agent = FusionAgent.__new__(FusionAgent)
        agent._weight_cache  = {}
        agent._weights_cache = {}
        agent.learn_weights_from_cache(history_path=h_path, weights_path=w_path)

        r = {}

        # Mock results: image_agent says 'real', multimodal_agent says 'fake'.
        # image_agent has higher learned weight → 'real' should win.
        mock_results = [
            {"agent_name": "ImageAgent",      "label": "real", "confidence": 0.80},
            {"agent_name": "MultimodalAgent", "label": "fake", "confidence": 0.80},
        ]

        result = agent.fuse(mock_results, domain="image")

        # 10. fuse() uses learned weights (weights_source should be 'learned')
        r["fuse uses learned weights"] = result.get("weights_source") == "learned"
        print(f"  [{'PASS' if r['fuse uses learned weights'] else 'FAIL'}] "
              f"weights_source={result.get('weights_source')!r}")

        # 11. image_agent wins because of higher weight
        r["image_agent wins due to higher weight"] = result.get("label") == "real"
        print(f"  [{'PASS' if r['image_agent wins due to higher weight'] else 'FAIL'}] "
              f"winner={result.get('label')!r}  "
              f"scores={result.get('all_label_scores')}")

        # 12. weights_source key present
        r["weights_source key present"] = "weights_source" in result
        print(f"  [{'PASS' if r['weights_source key present'] else 'FAIL'}] "
              f"'weights_source' in fuse() output")

        # 13. weights_source = 'learned'
        r["weights_source = learned"] = result.get("weights_source") == "learned"
        print(f"  [{'PASS' if r['weights_source = learned'] else 'FAIL'}] "
              f"weights_source == 'learned'")

        # 14. weights_source = 'provided' when explicit weights given
        explicit_result = agent.fuse(
            mock_results,
            weights={"ImageAgent": 0.6, "MultimodalAgent": 0.4},
        )
        r["weights_source = provided"] = explicit_result.get("weights_source") == "provided"
        print(f"  [{'PASS' if r['weights_source = provided'] else 'FAIL'}] "
              f"weights_source='provided' when explicit weights passed")

        return r
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test 15: weights update after new history entry
# ---------------------------------------------------------------------------

def test_weights_update():
    print("\n-- Test 15: weights update after new successful entry --")
    tmp = tempfile.mkdtemp(prefix="d11_")
    try:
        h_path = _make_history(_five_mock_entries(), tmp)
        w_path = _make_weights_path(tmp)

        agent = FusionAgent.__new__(FusionAgent)
        agent._weight_cache  = {}
        agent._weights_cache = {}

        w_before = agent.learn_weights_from_cache(
            history_path=h_path, weights_path=w_path)
        img_before = w_before.get("image", {}).get("image_agent", 0)

        # Append a new high-accuracy entry
        entries = _five_mock_entries() + [_image_entry(0.99, ["image", "report"])]
        with open(h_path, "w", encoding="utf-8") as f:
            json.dump(entries, f)

        w_after = agent.learn_weights_from_cache(
            history_path=h_path, weights_path=w_path)
        img_after = w_after.get("image", {}).get("image_agent", 0)

        ok = img_after > img_before
        print(f"  image_agent before={img_before:.4f}  after={img_after:.4f}")
        print(f"  [{'PASS' if ok else 'FAIL'}] weight increased after new high-accuracy entry")
        return {"weights update after new entry": ok}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Bonus: _class_to_weight_key helper
# ---------------------------------------------------------------------------

def test_class_to_key():
    print("\n-- Bonus: _class_to_weight_key() conversion --")
    cases = [
        ("ImageAgent",      "image_agent"),
        ("TextAgent",       "text_agent"),
        ("MultimodalAgent", "multimodal_agent"),
        ("TabularAgent",    "tabular_agent"),
        ("AudioAgent",      "audio_agent"),
        ("DynamicAgent",    "dynamic_agent"),
    ]
    r = {}
    all_ok = True
    for cls, expected in cases:
        got = _class_to_weight_key(cls)
        ok  = got == expected
        if not ok:
            all_ok = False
        print(f"  [{'PASS' if ok else 'FAIL'}] {cls} -> {got!r} (expected {expected!r})")
    r["_class_to_weight_key conversions"] = all_ok
    return r


# ---------------------------------------------------------------------------
# Run all
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    results = {}

    results.update(test_learn_weights())
    results.update(test_get_weights())
    results.update(test_fuse_with_domain())
    results.update(test_weights_update())
    results.update(test_class_to_key())

    print("\n==============================")
    print("  DAY 11 VERIFICATION SUMMARY")
    print("==============================")
    all_pass = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print("  {:<50} {}".format(name, status))
        if not passed:
            all_pass = False
    print("==============================")
    sys.exit(0 if all_pass else 1)
