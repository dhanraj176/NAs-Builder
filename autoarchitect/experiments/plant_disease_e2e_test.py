"""
End-to-end plant disease test.

Verifies the full pipeline:
  cache purge → DataDiscoveryEngine selects plant-disease dataset
  → SelfTrainer downloads it and trains

Run: python experiments/plant_disease_e2e_test.py
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

PROBLEM = "identify plant diseases from leaf images"
DOMAIN  = "image"


def purge_caches():
    print("[1/3] Purging stale caches...")

    from api.brain.dataset_intelligence import DatasetIntelligence
    di = DatasetIntelligence()
    n_chroma = di.purge_problem_cache(["plant", "disease"])
    print(f"      ChromaDB:    {n_chroma} entries purged")

    from api.brain.data_discovery_engine import DataDiscoveryEngine
    engine = DataDiscoveryEngine()
    n_local = engine.purge_local_cache(["plant", "disease"])
    print(f"      Local cache: {n_local} entries purged")

    return engine


def verify_discovery(engine):
    print(f"\n[2/3] DataDiscoveryEngine.find() — {PROBLEM}")
    result = engine.find(PROBLEM, DOMAIN, subset_size=500)

    name       = result.get("name", "NONE") if result else "NONE"
    has_loader = bool(result and result.get("train_loader") is not None)
    train_size = result.get("train_size", 0) if result else 0

    print(f"      Dataset:     {name}")
    print(f"      Has loaders: {has_loader}")
    print(f"      Train size:  {train_size}")

    ok_name = ("plant" in name.lower() or "disease" in name.lower()
               or "leaf" in name.lower())
    assert ok_name, f"Wrong dataset selected: {name}"
    assert has_loader,   "train_loader missing — dataset did not load"
    assert train_size >= 100, f"Too few training samples: {train_size}"
    print("      PASS: correct dataset, loaders present")
    return name


def run_training():
    print(f"\n[3/3] self_train (1 epoch) — {PROBLEM}")
    from api.self_trainer import self_train
    results = self_train(PROBLEM, DOMAIN, epochs=1)

    status   = results.get("status", "?")
    dataset  = results.get("dataset", "?")
    train_ac = results.get("train_accuracy", 0)
    test_ac  = results.get("test_accuracy",  0)

    print(f"\n      Status:       {status}")
    print(f"      Dataset:      {dataset}")
    print(f"      Train acc:    {train_ac}%")
    print(f"      Test acc:     {test_ac}%")
    print(f"      Model path:   {results.get('model_path', 'none')}")

    assert status == "complete", f"Training did not complete: {status}"
    ok_dataset = ("plant" in str(dataset).lower()
                  or "disease" in str(dataset).lower()
                  or "leaf"    in str(dataset).lower())
    assert ok_dataset, f"Wrong dataset used for training: {dataset}"
    print("      PASS: training completed on plant disease dataset")
    return results


if __name__ == "__main__":
    engine  = purge_caches()
    name    = verify_discovery(engine)
    results = run_training()

    print(f"\n{'='*60}")
    print(f"ALL CHECKS PASSED")
    print(f"  Dataset:   {results.get('dataset')}")
    print(f"  Train acc: {results.get('train_accuracy')}%")
    print(f"  Test acc:  {results.get('test_accuracy')}%")
    print(f"{'='*60}")
