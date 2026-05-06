# -*- coding: utf-8 -*-
"""
Day 8 verification -- agent_factory routing and cache-poisoning fix.

Checks:
  1. factory.create("...", "tabular")    -> TabularAgent
  2. factory.create("...", "audio")      -> AudioAgent
  3. factory.create("...", "multimodal") -> MultimodalAgent
  4. factory.create("...", "image")      -> DynamicAgent (unchanged)
  5. factory.create("...", "text")       -> DynamicAgent (unchanged)
  6. topology routes "classify spam text messages" to text agents (not image)
  7. topology domain-consistency guard rejects text-problem/image-agent combo
  8. build_network_from_problem includes tabular/audio/multimodal
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# ---------------------------------------------------------------------------
# Test 1-5: factory.create() domain dispatch
# ---------------------------------------------------------------------------
def test_factory_dispatch():
    print("\n-- Tests 1-5: AgentFactory domain dispatch --")
    from api.agents.agent_factory      import AgentFactory
    from api.agents.tabular_agent      import TabularAgent
    from api.agents.audio_agent        import AudioAgent
    from api.agents.multimodal_agent   import MultimodalAgent
    from api.agents.dynamic_agent      import DynamicAgent

    factory = AgentFactory()
    results = {}

    cases = [
        ("tabular",    "predict customer churn from transaction data",  TabularAgent),
        ("audio",      "transcribe and classify customer service calls", AudioAgent),
        ("multimodal", "zero shot image text classification with clip",  MultimodalAgent),
        ("image",      "detect potholes in road images",                DynamicAgent),
        ("text",       "classify spam text messages",                   DynamicAgent),
    ]

    all_ok = True
    for domain, problem, expected_cls in cases:
        agent = factory.create(problem=problem, domain=domain)
        ok = isinstance(agent, expected_cls)
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_ok = False
        print(f"  [{status}] domain={domain:10s} -> {type(agent).__name__:25s} "
              f"(expected {expected_cls.__name__})")
        results[domain] = ok

    return all_ok, results


# ---------------------------------------------------------------------------
# Test 6: "classify spam text messages" routes to text, not image
# ---------------------------------------------------------------------------
def test_spam_routing():
    print("\n-- Test 6: spam problem routes to text agents (cache-poison fix) --")
    from api.brain.topology_designer import TopologyDesigner

    td = TopologyDesigner()
    td.use_anas = False
    topo   = td.design("classify spam text messages")
    agents = topo.get("agents", [])
    source = topo.get("source", "?")

    ok = ("text" in agents and "image" not in agents)
    print(f"  agents={agents}  source={source}")
    print(f"  {'PASS' if ok else 'FAIL'} -- expected text in agents, image absent")
    return ok


# ---------------------------------------------------------------------------
# Test 7: domain-consistency guard rejects image-only for text problem
# ---------------------------------------------------------------------------
def test_domain_consistency_guard():
    print("\n-- Test 7: _domain_consistent() guard --")
    from api.brain.topology_designer import TopologyDesigner

    td = TopologyDesigner()

    cases = [
        # (problem, agents, should_pass)
        ("classify spam emails",          ["image", "severity", "report"], False),
        ("transcribe customer service calls", ["image", "report"],         False),
        ("detect potholes in road images", ["image", "severity", "report"], True),
        ("classify spam emails",           ["text", "report"],             True),
        ("predict churn from csv data",    ["tabular", "report"],          True),
    ]

    all_ok = True
    for problem, agents, expected in cases:
        result = td._domain_consistent(problem.lower(), agents)
        ok = (result == expected)
        if not ok:
            all_ok = False
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] consistent={result} (expected {expected}) | "
              f"{problem[:35]} -> {agents}")
    return all_ok


# ---------------------------------------------------------------------------
# Test 8: build_network_from_problem includes tabular/audio/multimodal
# ---------------------------------------------------------------------------
def test_network_builder():
    print("\n-- Test 8: build_network_from_problem has tabular/audio/multimodal --")
    import inspect
    from api.agents import agent_network
    src = inspect.getsource(agent_network.build_network_from_problem)

    ok = all(k in src for k in ["TabularAgent", "AudioAgent", "MultimodalAgent"])
    print(f"  TabularAgent in source:    {'YES' if 'TabularAgent' in src else 'NO'}")
    print(f"  AudioAgent in source:      {'YES' if 'AudioAgent' in src else 'NO'}")
    print(f"  MultimodalAgent in source: {'YES' if 'MultimodalAgent' in src else 'NO'}")
    print(f"  {'PASS' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------------------
# Run all
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    t1_ok, dispatch_results = test_factory_dispatch()
    t6_ok  = test_spam_routing()
    t7_ok  = test_domain_consistency_guard()
    t8_ok  = test_network_builder()

    results = {
        "factory tabular -> TabularAgent":       dispatch_results.get("tabular", False),
        "factory audio   -> AudioAgent":         dispatch_results.get("audio",   False),
        "factory multimodal -> MultimodalAgent": dispatch_results.get("multimodal", False),
        "factory image   -> DynamicAgent":       dispatch_results.get("image",   False),
        "factory text    -> DynamicAgent":       dispatch_results.get("text",    False),
        "spam routes to text (not image)":       t6_ok,
        "domain-consistency guard":              t7_ok,
        "network builder has new agents":        t8_ok,
    }

    print("\n==============================")
    print("  DAY 8 VERIFICATION SUMMARY")
    print("==============================")
    all_pass = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print("  {:<45} {}".format(name, status))
        if not passed:
            all_pass = False
    print("==============================")
    sys.exit(0 if all_pass else 1)
