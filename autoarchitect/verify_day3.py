# -*- coding: utf-8 -*-
"""
Day 3 verification -- FusionAgent weighted fusion, EvaluatorAgent ML metrics,
AgentNetwork.collaborate(), and error-handling in collaborate().
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))


# ---------------------------------------------------------------------------
# Test 1: FusionAgent.fuse() picks the weighted winner correctly
# ---------------------------------------------------------------------------
def test_fusion_weighted():
    print("\n-- Test 1: FusionAgent.fuse() weighted confidence fusion --")
    from api.agents.fusion_agent import FusionAgent

    agent = FusionAgent()

    # AgentA and AgentB both say "cat"; AgentC says "dog" with high confidence
    # If we weight AgentA/B low and AgentC high, "dog" should win
    results = [
        {"label": "cat",  "confidence": 0.8, "agent_name": "AgentA"},
        {"label": "cat",  "confidence": 0.7, "agent_name": "AgentB"},
        {"label": "dog",  "confidence": 0.9, "agent_name": "AgentC"},
    ]
    weights = {"AgentA": 0.1, "AgentB": 0.1, "AgentC": 0.8}

    result = agent.fuse(results, weights=weights)
    ok = (result.get("label") == "dog"
          and "weighted_confidence" in result.get("fusion_method", "")
          and "contributing_agents" in result
          and "weights_used" in result)
    print(f"  result : {result}")
    print("  PASS" if ok else "  FAIL -- expected label='dog' via weighted fusion")
    return ok


# ---------------------------------------------------------------------------
# Test 2: FusionAgent.fuse() equal-weight (majority) fallback
# ---------------------------------------------------------------------------
def test_fusion_equal_weight_majority():
    print("\n-- Test 2: FusionAgent.fuse() equal-weight majority --")
    from api.agents.fusion_agent import FusionAgent

    agent = FusionAgent()
    results = [
        {"label": "spam",   "confidence": 0.9, "agent_name": "A"},
        {"label": "spam",   "confidence": 0.8, "agent_name": "B"},
        {"label": "normal", "confidence": 0.95, "agent_name": "C"},
    ]
    # Equal weights -> spam wins (sum of confidence 0.9+0.8 > 0.95)
    result = agent.fuse(results)
    ok = result.get("label") == "spam"
    print(f"  result : {result}")
    print("  PASS" if ok else "  FAIL -- expected label='spam' with equal weights")
    return ok


# ---------------------------------------------------------------------------
# Test 3: FusionAgent uncertain flag fires when top-2 within 5 pp
# ---------------------------------------------------------------------------
def test_fusion_uncertain():
    print("\n-- Test 3: FusionAgent flags uncertain when top-2 within 5% --")
    from api.agents.fusion_agent import FusionAgent

    agent = FusionAgent()
    results = [
        {"label": "A", "confidence": 0.51, "agent_name": "X"},
        {"label": "B", "confidence": 0.49, "agent_name": "Y"},
    ]
    result = agent.fuse(results)
    ok = result.get("uncertain") is True
    print(f"  uncertain={result.get('uncertain')}  scores={result.get('all_label_scores')}")
    print("  PASS" if ok else "  FAIL -- expected uncertain=True")
    return ok


# ---------------------------------------------------------------------------
# Test 4: EvaluatorAgent.evaluate() computes real sklearn F1
# ---------------------------------------------------------------------------
def test_evaluator_ml_metrics():
    print("\n-- Test 4: EvaluatorAgent.evaluate() real sklearn metrics --")
    from api.agents.evaluator_agent import EvaluatorAgent

    ev = EvaluatorAgent()

    # Prediction dicts with confidence
    predictions = [
        {"label": "cat", "confidence": 0.9},
        {"label": "dog", "confidence": 0.85},
        {"label": "cat", "confidence": 0.3},   # uncertain
        {"label": "dog", "confidence": 0.7},
        {"label": "cat", "confidence": 0.95},
    ]
    ground_truth = ["cat", "dog", "cat", "cat", "cat"]

    result = ev.evaluate(predictions, ground_truth)
    ok = (
        "accuracy"  in result and
        "precision" in result and
        "recall"    in result and
        "f1"        in result and
        "quality_score" in result and
        isinstance(result["f1"], float) and
        result["uncertain_predictions"] == 1   # only the 0.3 conf one
    )
    print(f"  result : {result}")
    print("  PASS" if ok else "  FAIL -- expected full sklearn metrics dict")
    return ok


# ---------------------------------------------------------------------------
# Test 5: EvaluatorAgent.validate_single() flags low-confidence prediction
# ---------------------------------------------------------------------------
def test_evaluator_validate_single():
    print("\n-- Test 5: EvaluatorAgent.validate_single() quality check --")
    from api.agents.evaluator_agent import EvaluatorAgent

    ev = EvaluatorAgent()

    low  = ev.validate_single({"label": "spam", "confidence": 0.4})
    high = ev.validate_single({"label": "ham",  "confidence": 0.92})

    ok = (low.get("flag_for_review") is True and
          high.get("flag_for_review") is False and
          low.get("quality") == "low" and
          high.get("quality") == "high")
    print(f"  low  : {low}")
    print(f"  high : {high}")
    print("  PASS" if ok else "  FAIL -- unexpected quality/flag values")
    return ok


# ---------------------------------------------------------------------------
# Test 6: AgentNetwork.collaborate() combines 2 working mock agents
# ---------------------------------------------------------------------------
def test_collaborate_success():
    print("\n-- Test 6: AgentNetwork.collaborate() with 2 mock agents --")
    from api.agents.agent_network import AgentNetwork

    class MockAgentA:
        def predict(self, data):
            return {"label": "positive", "confidence": 0.8}

    class MockAgentB:
        def predict(self, data):
            return {"label": "positive", "confidence": 0.75}

    network = AgentNetwork("test_collab")
    result  = network.collaborate([MockAgentA(), MockAgentB()],
                                  task="sentiment", data="great product!")

    ok = (result.get("label") == "positive"
          and result.get("valid_count") == 2
          and result.get("task") == "sentiment")
    print(f"  result : {result}")
    print("  PASS" if ok else "  FAIL -- expected combined positive result")
    return ok


# ---------------------------------------------------------------------------
# Test 7: AgentNetwork.collaborate() — one agent fails, other still contributes
# ---------------------------------------------------------------------------
def test_collaborate_error_handling():
    print("\n-- Test 7: AgentNetwork.collaborate() error handling --")
    from api.agents.agent_network import AgentNetwork

    class GoodAgent:
        def predict(self, data):
            return {"label": "fraud", "confidence": 0.88}

    class BrokenAgent:
        def predict(self, data):
            raise RuntimeError("model not loaded")

    network = AgentNetwork("test_error")
    result  = network.collaborate([GoodAgent(), BrokenAgent()],
                                  task="fraud_detection", data="suspicious tx")

    ok = (result.get("label") == "fraud"
          and result.get("valid_count") == 1
          and result.get("agent_count") == 2
          and "error" not in result)
    print(f"  result : {result}")
    print("  PASS" if ok else "  FAIL -- expected GoodAgent result to survive")
    return ok


# ---------------------------------------------------------------------------
# Run all
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    results = {
        "FusionAgent weighted winner":     test_fusion_weighted(),
        "FusionAgent equal-weight majority": test_fusion_equal_weight_majority(),
        "FusionAgent uncertain flag":      test_fusion_uncertain(),
        "EvaluatorAgent ML metrics":       test_evaluator_ml_metrics(),
        "EvaluatorAgent validate_single":  test_evaluator_validate_single(),
        "AgentNetwork collaborate OK":     test_collaborate_success(),
        "AgentNetwork error handling":     test_collaborate_error_handling(),
    }

    print("\n==============================")
    print("  DAY 3 VERIFICATION SUMMARY")
    print("==============================")
    all_pass = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print("  {:<35} {}".format(name, status))
        if not passed:
            all_pass = False
    print("==============================")
    sys.exit(0 if all_pass else 1)
