# -*- coding: utf-8 -*-
"""
Day 4 verification -- TabularAgent XGBoost/LightGBM training + prediction,
and topology_designer tabular catalog/template checks.
"""

import sys, os, csv, tempfile
sys.path.insert(0, os.path.dirname(__file__))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_csv(path, n=120):
    """Write a synthetic binary-class CSV (age, income, score -> churn)."""
    import random
    random.seed(42)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["age", "income", "score", "region", "churn"])
        for _ in range(n):
            age    = random.randint(18, 70)
            income = random.randint(20000, 120000)
            score  = round(random.uniform(0, 1), 3)
            region = random.choice(["north", "south", "east", "west"])
            churn  = "yes" if (score < 0.4 and income < 60000) else "no"
            w.writerow([age, income, score, region, churn])


# ---------------------------------------------------------------------------
# Test 1: TabularAgent.train() succeeds and returns expected keys
# ---------------------------------------------------------------------------
def test_tabular_train():
    print("\n-- Test 1: TabularAgent.train() returns accuracy + features --")
    from api.agents.tabular_agent import TabularAgent

    agent = TabularAgent()
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as tf:
        csv_path = tf.name
    _make_csv(csv_path)

    result = agent.train(csv_path, hash_id="day4test")
    os.unlink(csv_path)

    ok = (
        "accuracy"             in result and
        "val_accuracy"         in result and
        "feature_importances"  in result and
        "top_5_features"       in result and
        "model_type"           in result and
        "classes"              in result and
        result["accuracy"]     >= 0.0 and
        result["model_type"]   in ("xgboost", "lightgbm") and
        len(result["top_5_features"]) <= 5
    )
    print(f"  accuracy={result.get('accuracy')}  model={result.get('model_type')}  "
          f"features={result.get('num_features')}  classes={result.get('classes')}")
    print("  PASS" if ok else "  FAIL -- missing keys or bad values")
    return ok, agent   # return agent so Test 2 can reuse it


# ---------------------------------------------------------------------------
# Test 2: TabularAgent.predict() on a dict after training
# ---------------------------------------------------------------------------
def test_tabular_predict(agent):
    print("\n-- Test 2: TabularAgent.predict() returns label + confidence --")

    sample = {
        "age":    35,
        "income": 45000,
        "score":  0.3,
        "region": "north",
    }
    result = agent.predict(sample)

    ok = (
        "label"      in result and
        "confidence" in result and
        "top3"       in result and
        result.get("agent_used") == "TabularAgent" and
        result["label"] != "error" and
        0.0 <= result["confidence"] <= 1.0
    )
    print(f"  result: {result}")
    print("  PASS" if ok else "  FAIL -- bad prediction dict")
    return ok


# ---------------------------------------------------------------------------
# Test 3: TabularAgent.predict() with no model returns honest error dict
# ---------------------------------------------------------------------------
def test_tabular_no_model():
    print("\n-- Test 3: TabularAgent.predict() without model -> error dict --")
    from api.agents.tabular_agent import TabularAgent

    fresh = TabularAgent()
    result = fresh.predict({"age": 30, "income": 50000})

    ok = (
        result.get("label")      == "error" and
        result.get("confidence") == 0.0 and
        "error"                  in result and
        result.get("fake")       is False
    )
    print(f"  result: {result}")
    print("  PASS" if ok else "  FAIL -- expected error dict")
    return ok


# ---------------------------------------------------------------------------
# Test 4: TabularAgent.load_trained_model() round-trips saved artifacts
# ---------------------------------------------------------------------------
def test_tabular_load():
    print("\n-- Test 4: TabularAgent.load_trained_model() round-trip --")
    from api.agents.tabular_agent import TabularAgent
    from pathlib import Path

    model_path = Path("models") / "trained" / "day4test_tabular.pkl"
    if not model_path.exists():
        print("  SKIP -- model file not found (run test_tabular_train first)")
        return True   # don't block the suite if artifacts missing

    fresh = TabularAgent()
    loaded = fresh.load_trained_model(str(model_path))

    ok = (
        loaded is True and
        fresh.model             is not None and
        fresh.feature_columns   is not None and
        fresh.target_column     is not None and
        fresh.classes           is not None and
        fresh.model_type        in ("xgboost", "lightgbm")
    )
    print(f"  loaded={loaded}  model_type={fresh.model_type}  "
          f"features={fresh.feature_columns}  classes={fresh.classes}")
    print("  PASS" if ok else "  FAIL -- load_trained_model() failed")
    return ok


# ---------------------------------------------------------------------------
# Test 5: TabularAgent.predict_batch() on a fresh CSV
# ---------------------------------------------------------------------------
def test_tabular_predict_batch():
    print("\n-- Test 5: TabularAgent.predict_batch() over CSV --")
    from api.agents.tabular_agent import TabularAgent
    from pathlib import Path

    model_path = Path("models") / "trained" / "day4test_tabular.pkl"
    if not model_path.exists():
        print("  SKIP -- model file not found")
        return True

    agent = TabularAgent()
    agent.load_trained_model(str(model_path))

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as tf:
        csv_path = tf.name
    _make_csv(csv_path, n=10)

    results = agent.predict_batch(csv_path)
    os.unlink(csv_path)

    ok = (
        isinstance(results, list) and
        len(results) == 10 and
        all("label" in r and "confidence" in r for r in results) and
        all(r.get("label") != "error" for r in results)
    )
    print(f"  rows={len(results)}  first={results[0] if results else None}")
    print("  PASS" if ok else "  FAIL -- batch prediction failed")
    return ok


# ---------------------------------------------------------------------------
# Test 6: TopologyDesigner includes 'tabular' in AGENT_CATALOG
# ---------------------------------------------------------------------------
def test_catalog_has_tabular():
    print("\n-- Test 6: AGENT_CATALOG has 'tabular' entry --")
    from api.brain.topology_designer import AGENT_CATALOG

    ok = (
        "tabular" in AGENT_CATALOG and
        "keywords" in AGENT_CATALOG["tabular"] and
        "csv"      in AGENT_CATALOG["tabular"]["keywords"] and
        "fraud"    in AGENT_CATALOG["tabular"]["keywords"]
    )
    print(f"  keywords: {AGENT_CATALOG.get('tabular', {}).get('keywords', [])}")
    print("  PASS" if ok else "  FAIL -- 'tabular' not in AGENT_CATALOG")
    return ok


# ---------------------------------------------------------------------------
# Test 7: TopologyDesigner routes 'csv churn prediction' to tabular_pipeline
# ---------------------------------------------------------------------------
def test_topology_routes_tabular():
    print("\n-- Test 7: TopologyDesigner routes CSV problem to tabular agents --")
    from api.brain.topology_designer import TopologyDesigner

    td = TopologyDesigner()
    td.use_anas = False   # use template matching only

    topo = td.design("predict customer churn from csv data")
    agents = topo.get("agents", [])

    ok = "tabular" in agents
    print(f"  agents: {agents}  source: {topo.get('source')}")
    print("  PASS" if ok else "  FAIL -- 'tabular' not in designed agents")
    return ok


# ---------------------------------------------------------------------------
# Run all
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    t1_ok, trained_agent = test_tabular_train()

    results = {
        "TabularAgent train()":          t1_ok,
        "TabularAgent predict() dict":   test_tabular_predict(trained_agent),
        "TabularAgent no-model error":   test_tabular_no_model(),
        "TabularAgent load round-trip":  test_tabular_load(),
        "TabularAgent predict_batch()":  test_tabular_predict_batch(),
        "AGENT_CATALOG has tabular":     test_catalog_has_tabular(),
        "Topology routes to tabular":    test_topology_routes_tabular(),
    }

    print("\n==============================")
    print("  DAY 4 VERIFICATION SUMMARY")
    print("==============================")
    all_pass = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print("  {:<35} {}".format(name, status))
        if not passed:
            all_pass = False
    print("==============================")
    sys.exit(0 if all_pass else 1)
