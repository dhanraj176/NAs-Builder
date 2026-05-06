# -*- coding: utf-8 -*-
"""
Day 10 verification -- ANAS multi-agent parallel topology support.

Tests:
  1. _ensemble_candidates() returns PARALLEL arches for image keyword
  2. _ensemble_candidates() returns PARALLEL arches for text keyword
  3. _ensemble_candidates() returns PARALLEL arches for medical keyword
  4. proxy_score() awards +0.05 bonus to complementary parallel arches
  5. to_topology_dict() includes execution_mode key
  6. execution_mode = 'parallel' for PARALLEL topology
  7. execution_mode = 'sequential' for SEQUENTIAL topology
  8. execution_mode = 'hybrid' for HIERARCHICAL topology
  9. search() includes PARALLEL ensemble in candidates for image problem
 10. search() includes PARALLEL ensemble in candidates for text problem
 11. search() includes PARALLEL ensemble in candidates for medical problem
 12. sequential single-agent paths still score and complete (no regression)
"""

import sys
import os
import tempfile
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_engine(tmp_dir):
    from api.brain.anas_search_engine import ANASSearchEngine
    from api.brain.anas_immune_system import ToxicVault, SuccessVault, ImmuneSystem

    tv     = ToxicVault(os.path.join(tmp_dir,  "toxic.json"))
    sv     = SuccessVault(os.path.join(tmp_dir, "success.json"))
    immune = ImmuneSystem(toxic_vault=tv, success_vault=sv, stats_path=None)
    return ANASSearchEngine(
        immune_system = immune,
        stats_path    = os.path.join(tmp_dir, "stats.json"),
    )


def _make_space():
    from api.brain.anas_search_space import ANASSearchSpace, SearchConstraints
    c = SearchConstraints(require_output_agent=False)
    return ANASSearchSpace(c)


# ---------------------------------------------------------------------------
# Test 1-3: _ensemble_candidates() domain coverage
# ---------------------------------------------------------------------------

def test_ensemble_candidates():
    print("\n-- Tests 1-3: _ensemble_candidates() domain coverage --")
    tmp = tempfile.mkdtemp(prefix="anas_day10_")
    try:
        engine = _make_engine(tmp)
        space  = _make_space()

        cases = [
            ("detect potholes in road image",         "image",   ["image", "multimodal"]),
            ("classify spam text messages",            "text",    ["text", "sentiment"]),
            ("diagnose medical conditions from xray",  "medical", ["medical", "image"]),
        ]

        results = {}
        for problem, kw, expected_agents in cases:
            cands = engine._ensemble_candidates(problem, [kw], space)
            found = any(
                set(c.agents) == set(expected_agents)
                for c in cands
            )
            results[f"ensemble_{kw}"] = found
            status = "PASS" if found else "FAIL"
            print(f"  [{status}] kw={kw!r}: found {expected_agents} in "
                  f"{[c.agents for c in cands]}")

        return results
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test 4: proxy_score() +0.05 bonus for complementary parallel
# ---------------------------------------------------------------------------

def test_parallel_bonus():
    print("\n-- Test 4: proxy_score() +0.05 bonus for complementary parallel --")
    tmp = tempfile.mkdtemp(prefix="anas_day10_")
    try:
        from api.brain.anas_search_space import NetworkArchitecture, PARALLEL, SEQUENTIAL

        engine = _make_engine(tmp)
        problem = "detect potholes in road image"

        arch_parallel = NetworkArchitecture(
            agents=["image", "multimodal"], topology=PARALLEL,
            metadata={"source": "ensemble_parallel", "execution_mode": "parallel"},
        )
        arch_seq = NetworkArchitecture(
            agents=["image", "multimodal"], topology=SEQUENTIAL,
            metadata={"source": "test"},
        )

        score_parallel = engine.proxy_score(arch_parallel, problem, ["image"])
        score_seq      = engine.proxy_score(arch_seq,      problem, ["image"])

        bonus = score_parallel - score_seq
        ok = bonus > 0.0
        print(f"  parallel_score={score_parallel:.4f}  seq_score={score_seq:.4f}  "
              f"bonus={bonus:+.4f}")
        print(f"  [{'PASS' if ok else 'FAIL'}] parallel score > sequential score "
              f"(bonus > 0)")
        return {"parallel_proxy_bonus": ok}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Tests 5-8: execution_mode in to_topology_dict()
# ---------------------------------------------------------------------------

def test_execution_mode_tag():
    print("\n-- Tests 5-8: execution_mode in to_topology_dict() --")
    tmp = tempfile.mkdtemp(prefix="anas_day10_")
    try:
        from api.brain.anas_search_space import (
            NetworkArchitecture, PARALLEL, SEQUENTIAL, HIERARCHICAL
        )

        engine = _make_engine(tmp)

        cases = [
            (["image", "multimodal"], PARALLEL,     "parallel",   "PARALLEL topology"),
            (["image", "report"],     SEQUENTIAL,   "sequential", "SEQUENTIAL topology"),
            (["image", "severity", "report"], HIERARCHICAL, "hybrid", "HIERARCHICAL topology"),
        ]

        results = {}
        for agents, topology, expected_mode, label in cases:
            arch = NetworkArchitecture(agents=agents, topology=topology,
                                       metadata={"source": "test"})
            td = engine.to_topology_dict(arch, "test problem", proxy_score=0.80)
            has_key   = "execution_mode" in td
            mode_ok   = td.get("execution_mode") == expected_mode
            ok        = has_key and mode_ok
            key       = f"exec_mode_{topology.lower()}"
            results[key] = ok
            status = "PASS" if ok else "FAIL"
            print(f"  [{status}] {label}: execution_mode={td.get('execution_mode')!r} "
                  f"(expected {expected_mode!r})")

        return results
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Tests 9-11: search() includes PARALLEL candidates in all_scored
# ---------------------------------------------------------------------------

def test_search_parallel_candidates():
    print("\n-- Tests 9-11: search() includes PARALLEL candidates for 3 domains --")
    tmp = tempfile.mkdtemp(prefix="anas_day10_")
    try:
        from api.brain.anas_search_space import PARALLEL

        engine = _make_engine(tmp)

        problems = [
            ("detect potholes in road image",          ["image"],   "image"),
            ("classify spam text messages",             ["text"],    "text"),
            ("diagnose medical conditions from xray",  ["medical"], "medical"),
        ]

        results = {}
        for problem, hints, label in problems:
            result = engine.search(problem=problem, domain_hints=hints, budget=30)
            all_scored = result.get("all_scored", [])

            parallel_scored = [
                (s, a) for s, a in all_scored
                if a.topology == PARALLEL
            ]
            ok = len(parallel_scored) > 0
            results[f"search_parallel_{label}"] = ok
            status = "PASS" if ok else "FAIL"
            top_parallel = parallel_scored[0] if parallel_scored else None
            print(f"  [{status}] {label}: {len(parallel_scored)} PARALLEL candidates "
                  f"in all_scored"
                  + (f", top={top_parallel[1].agents} (score={top_parallel[0]:.4f})"
                     if top_parallel else ""))

        return results
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test 12: sequential paths still work (no regression)
# ---------------------------------------------------------------------------

def test_sequential_regression():
    print("\n-- Test 12: sequential single-agent paths unaffected --")
    tmp = tempfile.mkdtemp(prefix="anas_day10_")
    try:
        from api.brain.anas_search_space import SEQUENTIAL

        engine = _make_engine(tmp)
        result = engine.search(
            problem      = "detect illegal dumping and file severity report",
            domain_hints = ["detect", "image", "severity", "report"],
            budget       = 20,
        )
        best = result["architecture"]
        ok   = (isinstance(result, dict) and
                "architecture" in result and
                result["proxy_score"] > 0.0 and
                result["evaluated"] > 0)
        print(f"  winner: {best.agents} / {best.topology}  "
              f"score={result['proxy_score']}")
        print(f"  [{'PASS' if ok else 'FAIL'}] search returned valid result "
              f"(evaluated={result['evaluated']})")
        return {"sequential_no_regression": ok}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Run all
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    results = {}

    r1 = test_ensemble_candidates()
    results.update(r1)

    r2 = test_parallel_bonus()
    results.update(r2)

    r3 = test_execution_mode_tag()
    results.update(r3)

    r4 = test_search_parallel_candidates()
    results.update(r4)

    r5 = test_sequential_regression()
    results.update(r5)

    print("\n==============================")
    print("  DAY 10 VERIFICATION SUMMARY")
    print("==============================")
    all_pass = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print("  {:<45} {}".format(name, status))
        if not passed:
            all_pass = False
    print("==============================")
    sys.exit(0 if all_pass else 1)
