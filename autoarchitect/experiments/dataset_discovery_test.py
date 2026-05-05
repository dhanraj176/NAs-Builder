"""
Dataset discovery test — exercises all 4 sources across 3 scenarios.
Results saved to experiments/results/dataset_discovery_test.json
"""

import json
import sys
import time
from pathlib import Path

BASE_DIR    = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "experiments" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = RESULTS_DIR / "dataset_discovery_test.json"

sys.path.insert(0, str(BASE_DIR))

TEST_CASES = [
    {"problem": "detect weapons in airport security scans",       "domain": "image"},
    {"problem": "predict customer churn from transaction data",   "domain": "security"},
    {"problem": "identify plant diseases from leaf images",       "domain": "image"},
]


def _clean(obj):
    """Strip non-JSON-serialisable items (DataLoaders etc.)."""
    if isinstance(obj, dict):
        return {
            k: _clean(v) for k, v in obj.items()
            if k not in ("train_loader", "val_loader", "test_loader")
            and not callable(v)
        }
    if isinstance(obj, list):
        return [_clean(i) for i in obj]
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return str(obj)


def run_test():
    from api.brain.dataset_intelligence import DatasetIntelligence

    di      = DatasetIntelligence()
    results = []

    for tc in TEST_CASES:
        print(f"\n{'='*60}")
        print(f"Problem : {tc['problem']}")
        print(f"Domain  : {tc['domain']}")
        print(f"{'='*60}")

        # Purge any poisoned ChromaDB entries before the plant disease case
        if "plant" in tc["problem"].lower():
            purged = di.purge_problem_cache(["plant", "disease"])
            if purged:
                print(f"   [Cache] Purged {purged} stale entries for plant/disease")

        t0 = time.time()
        try:
            result = di.discover(tc["problem"], tc["domain"])
            error  = None
        except Exception as e:
            result = {}
            error  = str(e)
            print(f"   [ERROR] {e}")

        elapsed = round(time.time() - t0, 2)

        entry = {
            "problem":        tc["problem"],
            "domain":         tc["domain"],
            "elapsed_s":      elapsed,
            "source_counts":  result.get("source_counts", {}),
            "best":           _clean(result.get("best")),
            "top_candidates": _clean((result.get("candidates") or [])[:5]),
            "split":          "70/15/15 train/val/test",
            "error":          error,
        }
        results.append(entry)

        print(f"\nBest     : {entry['best']}")
        print(f"Sources  : {entry['source_counts']}")
        print(f"Elapsed  : {elapsed}s")

    output = {
        "test_run_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "split_policy": "70/15/15 — held-out test set never seen during training",
        "sources": ["papers_with_code", "openml", "roboflow", "huggingface"],
        "min_samples": 500,
        "ranking": "sample_count → open_license → keyword_overlap → sota_benchmark",
        "results": results,
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"Saved to : {OUTPUT_PATH}")
    return output


if __name__ == "__main__":
    run_test()
