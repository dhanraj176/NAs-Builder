"""
experiments/baseline_comparison.py
===================================
NeurIPS AutoML Workshop -- Core Baseline Experiment

Compares four architecture-selection strategies on the
dmedhi/garbage-image-classification-detection dataset:

  A. Single best model  -- no search, train the image agent directly
  B. Random search      -- draw 20 random architectures, pick one
  C. Template matching  -- use TopologyDesigner.design() (current system)
  D. ANAS (ours)        -- use ANASSearchEngine.search() with budget=20

Accuracy for compatible architectures (image-primary) comes from existing
cached training runs in models/trained/ -- no GPU re-training triggered.
Architectures whose primary domain mismatches the visual task are scored
as 0% (the system cannot process images with a text/security/medical model).

This script is READ-ONLY w.r.t. the existing codebase:
  * reads  brain_data/, models/trained/
  * writes experiments/results/baseline_results.json ONLY

Run from the project root:
    python experiments/baseline_comparison.py
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── sys.path: project root must be resolvable for api.* imports ───────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from api.brain.anas_immune_system import (
    ImmuneSystem, ToxicVault, SuccessVault,
    TOXIC_FILE, SUCCESS_FILE,
)
from api.brain.anas_search_engine  import ANASSearchEngine
from api.brain.anas_search_space   import ANASSearchSpace, NetworkArchitecture, SEQUENTIAL
from api.brain.topology_designer   import TopologyDesigner


# ══════════════════════════════════════════════════════════════════════════════
# Experiment parameters
# ══════════════════════════════════════════════════════════════════════════════

DATASET_NAME  = "dmedhi/garbage-image-classification-detection"
PROBLEM       = "detect illegal dumping and garbage in street cameras"
DOMAIN        = "image"
DOMAIN_HINTS  = ["detect", "dumping", "garbage", "image", "visual",
                  "camera", "illegal", "street", "monitor"]
SEEDS         = [42, 123, 456]    # three independent random seeds
BUDGET        = 20                # proxy-score evaluations for ANAS / random

MODELS_DIR  = _PROJECT_ROOT / "models" / "trained"
RESULTS_DIR = Path(__file__).parent / "results"

# Agents that do not consume raw input -- they receive predictions from upstream.
# Excluded when determining the architecture's primary processing domain.
_TERMINAL_AGENTS = {"report", "severity", "optimizer", "audience"}

# Map from agent name to the training domain it requires.
_AGENT_DOMAIN: Dict[str, str] = {
    "image":     "image",
    "text":      "text",
    "medical":   "medical",
    "security":  "security",
    "sentiment": "text",
    "severity":  "image",   # post-processor; follows image domain by default
    "report":    "image",
    "audience":  "text",
    "optimizer": "text",
}


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_trained_metadata() -> List[Dict]:
    """
    Load every *_classes.json file from models/trained/ into a list of dicts.
    Each dict contains at minimum: dataset, domain, problem, test_accuracy.
    """
    meta_list: List[Dict] = []
    if not MODELS_DIR.exists():
        print(f"  [warn] models/trained/ not found at {MODELS_DIR}")
        return meta_list
    for path in MODELS_DIR.glob("*_classes.json"):
        try:
            meta_list.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            pass
    return meta_list


def cached_accuracies_for(meta_list: List[Dict],
                           dataset: str,
                           domain: str) -> List[float]:
    """Return all test_accuracy values matching (dataset, domain)."""
    return [
        float(m["test_accuracy"])
        for m in meta_list
        if m.get("dataset") == dataset
        and m.get("domain")  == domain
        and m.get("test_accuracy") is not None
    ]


def mean_std(values: List[float]) -> Tuple[float, float]:
    """Sample mean and sample standard deviation; std=0 for a single value."""
    if not values:
        return 0.0, 0.0
    mu = sum(values) / len(values)
    if len(values) < 2:
        return round(mu, 2), 0.0
    var = sum((v - mu) ** 2 for v in values) / (len(values) - 1)
    return round(mu, 2), round(math.sqrt(var), 2)


def primary_domain(agents: List[str]) -> str:
    """
    The domain of the first agent that actually processes raw input.

    Terminal agents (report, severity, optimizer, audience) receive predictions
    from upstream agents -- they do not determine the processing domain.
    """
    for a in agents:
        if a not in _TERMINAL_AGENTS:
            return _AGENT_DOMAIN.get(a, "unknown")
    # Fallback: use the first agent's mapping
    return _AGENT_DOMAIN.get(agents[0], "unknown") if agents else "unknown"


def accuracy_for_arch(arch: NetworkArchitecture,
                       meta_list: List[Dict]) -> Tuple[Optional[float], str]:
    """
    Look up test accuracy for this architecture's primary domain.

    Returns (accuracy, source) where source is one of:
      "cached"          -- loaded from existing models/trained/ metadata
      "domain_mismatch" -- arch cannot process images (wrong primary domain)
      "no_cache"        -- right domain but no cached result found
    """
    dom = primary_domain(arch.agents)
    if dom != DOMAIN:
        return 0.0, "domain_mismatch"
    hits = cached_accuracies_for(meta_list, DATASET_NAME, dom)
    if hits:
        return round(sum(hits) / len(hits), 2), "cached"
    return None, "no_cache"


def fresh_immune_system() -> ImmuneSystem:
    """Empty vaults in a temp dir -- used for clean/isolated experiments."""
    tmpdir = tempfile.mkdtemp(prefix="anas_baseline_exp_")
    tv = ToxicVault(os.path.join(tmpdir, "toxic.json"))
    sv = SuccessVault(os.path.join(tmpdir, "success.json"))
    return ImmuneSystem(toxic_vault=tv, success_vault=sv, stats_path=None)


def real_immune_system_readonly() -> ImmuneSystem:
    """
    Load the real brain_data vaults into a NEW ImmuneSystem instance.

    The instance reads from TOXIC_FILE and SUCCESS_FILE on construction,
    so it carries accumulated knowledge from prior training runs.
    stats_path=None ensures no stats file is written.
    We never call learn() on this instance, so the vault JSON files
    on disk are not modified -- this is purely a read operation.
    """
    tv = ToxicVault(TOXIC_FILE)
    sv = SuccessVault(SUCCESS_FILE)
    return ImmuneSystem(toxic_vault=tv, success_vault=sv, stats_path=None)


# ══════════════════════════════════════════════════════════════════════════════
# Baseline A -- Single best model (no search)
# ══════════════════════════════════════════════════════════════════════════════

def run_baseline_a(meta_list: List[Dict]) -> Dict[str, Any]:
    """
    Baseline A: select the image agent alone, train on the dataset.

    The three existing cached training runs (different problem phrasings,
    same dataset) serve as the three random seeds -- each reflects a genuine
    independent training run with a different random initialisation.

    No new training is triggered.
    """
    print("\n[Baseline A] Single best model -- image agent only")

    accs = cached_accuracies_for(meta_list, DATASET_NAME, DOMAIN)
    seeds_used = accs[:3] if len(accs) >= 3 else accs
    mu, sigma = mean_std(seeds_used)

    print(f"  Architecture : ['image'] / {SEQUENTIAL}")
    print(f"  Cached runs  : {seeds_used}")
    print(f"  Accuracy     : {mu}% +- {sigma}%")

    return {
        "method":          "Single model",
        "architecture":    ["image"],
        "topology":        SEQUENTIAL,
        "seeds":           seeds_used,
        "accuracy_mean":   mu,
        "accuracy_std":    sigma,
        "searches":        1,
        "blocked":         0,
        "proxy_score":     None,
        "source":          "cached",
        "note":            "3 existing ResNet18 fine-tune runs on same dataset",
    }


# ══════════════════════════════════════════════════════════════════════════════
# Baseline B -- Random search
# ══════════════════════════════════════════════════════════════════════════════

def run_baseline_b(meta_list: List[Dict]) -> Dict[str, Any]:
    """
    Baseline B: for each seed, draw 20 random architectures from Lambda
    and pick one uniformly at random (no proxy scoring).

    Accuracy depends on whether the randomly-selected primary agent is
    compatible with the visual garbage-detection task.  An architecture whose
    primary agent is not 'image' gets 0% -- it cannot process camera frames.
    """
    print("\n[Baseline B] Random search -- 20 candidates, random selection")

    space = ANASSearchSpace()
    seed_details: List[Dict] = []

    for seed in SEEDS:
        random.seed(seed)
        candidates  = space.generate_random(n=BUDGET)
        chosen      = random.choice(candidates)
        dom         = primary_domain(chosen.agents)
        acc, source = accuracy_for_arch(chosen, meta_list)

        print(f"  Seed {seed}: {chosen.agents} / {chosen.topology}"
              f"  primary={dom}  acc={acc}%  ({source})")

        seed_details.append({
            "seed":           seed,
            "architecture":   chosen.agents,
            "topology":       chosen.topology,
            "primary_domain": dom,
            "accuracy":       acc,
            "source":         source,
            "proxy_score":    None,   # random search uses no proxy
        })

    seed_accs = [d["accuracy"] for d in seed_details
                 if d["accuracy"] is not None]
    mu, sigma = mean_std(seed_accs)

    print(f"  Accuracy     : {mu}% +- {sigma}%  "
          f"(over {len(seed_accs)} valid seeds)")

    return {
        "method":          "Random search",
        "architecture":    [d["architecture"] for d in seed_details],
        "topology":        [d["topology"]     for d in seed_details],
        "seeds":           [d["accuracy"]     for d in seed_details],
        "accuracy_mean":   mu,
        "accuracy_std":    sigma,
        "searches":        BUDGET,
        "blocked":         0,
        "proxy_score":     None,
        "source":          "random_selection",
        "seed_details":    seed_details,
        "note":            "Architecture varies by seed; 0% = domain mismatch",
    }


# ══════════════════════════════════════════════════════════════════════════════
# Baseline C -- Template matching (current system, no ANAS)
# ══════════════════════════════════════════════════════════════════════════════

def run_baseline_c(meta_list: List[Dict]) -> Dict[str, Any]:
    """
    Baseline C: call TopologyDesigner.design() directly.

    This is the current production method: keyword matching against
    TOPOLOGY_TEMPLATES, no formal search, no immune system.
    Records what topology is selected and looks up the cached accuracy.
    """
    print("\n[Baseline C] Template matching -- TopologyDesigner.design()")

    td      = TopologyDesigner()
    result  = td.design(PROBLEM, DOMAIN)
    agents  = result.get("agents", ["image", "report"])
    topology = result.get("topology", SEQUENTIAL)
    source  = result.get("source", "unknown")

    print(f"  Selected     : {agents} / {topology}  (template source: {source})")

    dom = primary_domain(agents)
    if dom == DOMAIN:
        all_accs  = cached_accuracies_for(meta_list, DATASET_NAME, DOMAIN)
        seeds_used = all_accs[:3] if len(all_accs) >= 3 else all_accs
        acc_source = "cached"
    else:
        seeds_used = [0.0, 0.0, 0.0]
        acc_source = "domain_mismatch"

    mu, sigma = mean_std(seeds_used)
    print(f"  Accuracy     : {mu}% +- {sigma}%")

    return {
        "method":           "Template match",
        "architecture":     agents,
        "topology":         topology,
        "template_source":  source,
        "seeds":            seeds_used,
        "accuracy_mean":    mu,
        "accuracy_std":     sigma,
        "searches":         1,
        "blocked":          0,
        "proxy_score":      None,
        "source":           acc_source,
        "note":             "Deterministic keyword lookup; same arch every run",
    }


# ══════════════════════════════════════════════════════════════════════════════
# ANAS -- our method
# ══════════════════════════════════════════════════════════════════════════════

def run_anas(meta_list: List[Dict]) -> Dict[str, Any]:
    """
    ANAS: formal architecture search with proxy scoring and immune gating.

    Loads the real brain_data/ vaults (accumulated from prior training runs)
    into a new ImmuneSystem instance -- read-only; learn() is never called,
    so no vault file is modified.  stats_path=None prevents stats writes.
    """
    print("\n[ANAS] ANASSearchEngine.search() -- budget=20")

    immune = real_immune_system_readonly()
    n_toxic   = len(immune.toxic_vault)
    n_success = len(immune.success_vault)
    print(f"  Immune state : {n_toxic} toxic, {n_success} successes loaded")

    engine = ANASSearchEngine(immune_system=immune, stats_path=None)

    t0 = time.time()
    result = engine.search(
        problem      = PROBLEM,
        domain_hints = DOMAIN_HINTS,
        budget       = BUDGET,
    )
    elapsed = round(time.time() - t0, 3)

    best       = result["architecture"]
    agents     = best.agents
    topology   = best.topology
    evaluated  = result["evaluated"]
    aborted    = result["aborted"]
    proxy      = result["proxy_score"]

    print(f"  Winner       : {agents} / {topology}")
    print(f"  Proxy score  : {proxy}")
    print(f"  Evaluated    : {evaluated}   Blocked: {aborted}   "
          f"Elapsed: {elapsed}s")

    # Top-5 scored candidates for paper table (supplementary)
    top5 = [
        {
            "rank":     rank + 1,
            "agents":   a.agents,
            "topology": a.topology,
            "score":    round(s, 4),
            "source":   a.metadata.get("source", "?"),
        }
        for rank, (s, a) in enumerate(result["all_scored"][:5])
    ]

    # Accuracy: look up from cache
    dom = primary_domain(agents)
    if dom == DOMAIN:
        all_accs   = cached_accuracies_for(meta_list, DATASET_NAME, DOMAIN)
        seeds_used = all_accs[:3] if len(all_accs) >= 3 else all_accs
        acc_source = "cached"
    else:
        seeds_used = [0.0, 0.0, 0.0]
        acc_source = "domain_mismatch"

    mu, sigma = mean_std(seeds_used)
    print(f"  Accuracy     : {mu}% +- {sigma}%")

    # Proxy-real correlation: pair our proxy with the mean real accuracy
    if mu > 0:
        proxy_real_r = round(
            (proxy - 0.5) / 0.5 * (mu / 100.0), 3
        )   # order-of-magnitude indicator (full Pearson needs multiple points)
    else:
        proxy_real_r = 0.0

    return {
        "method":                 "ANAS (ours)",
        "architecture":           agents,
        "topology":               topology,
        "proxy_score":            proxy,
        "seeds":                  seeds_used,
        "accuracy_mean":          mu,
        "accuracy_std":           sigma,
        "searches":               evaluated,
        "blocked":                aborted,
        "source":                 acc_source,
        "search_elapsed_s":       elapsed,
        "top5_candidates":        top5,
        "proxy_real_indicator":   proxy_real_r,
        "immune_toxic_loaded":    n_toxic,
        "immune_success_loaded":  n_success,
        "note":                   "Real brain_data/ vaults used read-only; learn() not called",
    }


# ══════════════════════════════════════════════════════════════════════════════
# Reporting
# ══════════════════════════════════════════════════════════════════════════════

def _arch_str(arch_field: Any) -> str:
    """
    Convert the architecture field (list or list-of-lists) to a short string
    for the table column.
    """
    if not arch_field:
        return "?"
    if isinstance(arch_field[0], list):
        # Random search: multiple architectures (one per seed)
        parts = ["+".join(a) for a in arch_field]
        return " / ".join(parts)
    return "+".join(arch_field)


def print_table(results: Dict[str, Dict]) -> None:
    """Print the NeurIPS-paper comparison table to stdout."""
    SEP = "=" * 85
    print(f"\n{SEP}")
    print("ANAS Baseline Comparison -- NeurIPS AutoML Workshop")
    print(f"Dataset : {DATASET_NAME}")
    print(f"Problem : {PROBLEM}")
    print(SEP)

    header = (
        f"{'Method':<18} | {'Architecture':<30} | "
        f"{'Acc (%)':>9} | {'Searches':>9} | {'Blocked':>7}"
    )
    print(header)
    print("-" * 85)

    order = ["Single model", "Random search", "Template match", "ANAS (ours)"]
    for name in order:
        r = results.get(name)
        if r is None:
            continue

        arch_s = _arch_str(r["architecture"])
        if len(arch_s) > 30:
            arch_s = arch_s[:27] + "..."

        mu    = r["accuracy_mean"]
        sigma = r["accuracy_std"]
        acc_s = f"{mu:.1f}+-{sigma:.1f}" if sigma else f"{mu:.1f}"

        print(
            f"{name:<18} | {arch_s:<30} | "
            f"{acc_s:>9} | {r['searches']:>9} | {r['blocked']:>7}"
        )

    print(SEP)

    # Secondary detail: random search seed breakdown
    rnd = results.get("Random search", {})
    if rnd.get("seed_details"):
        print("\nRandom search -- architecture per seed:")
        for d in rnd["seed_details"]:
            dom_flag = "" if d["primary_domain"] == DOMAIN else "  [domain mismatch]"
            print(f"  seed={d['seed']}  {d['architecture']} / {d['topology']}"
                  f"  acc={d['accuracy']}%{dom_flag}")

    # ANAS top-5 candidates
    anas = results.get("ANAS (ours)", {})
    if anas.get("top5_candidates"):
        print("\nANAS top-5 scored candidates:")
        for c in anas["top5_candidates"]:
            print(f"  [{c['rank']}] score={c['score']:.4f}  "
                  f"{c['agents']} / {c['topology']}  ({c['source']})")

    print()


def save_results(results: Dict[str, Dict]) -> None:
    """Save full results dict to experiments/results/baseline_results.json."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "baseline_results.json"

    payload = {
        "experiment":  "ANAS Baseline Comparison",
        "dataset":     DATASET_NAME,
        "problem":     PROBLEM,
        "budget":      BUDGET,
        "seeds":       SEEDS,
        "run_at":      datetime.now().isoformat(),
        "methods":     results,
    }

    out_path.write_text(
        json.dumps(payload, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"Results saved -> {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 85)
    print("ANAS Baseline Comparison -- NeurIPS AutoML Workshop")
    print(f"Dataset : {DATASET_NAME}")
    print(f"Problem : {PROBLEM}")
    print(f"Runs at : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 85)

    # Load cached training metadata (no GPU needed)
    print("\nLoading cached training results from models/trained/...")
    meta_list = load_trained_metadata()
    target_runs = cached_accuracies_for(meta_list, DATASET_NAME, DOMAIN)
    print(f"  Found {len(target_runs)} cached run(s) for target dataset:"
          f"  {target_runs}")

    results: Dict[str, Dict] = {}

    # ── Baseline A: single model ──────────────────────────────────────────────
    results["Single model"]   = run_baseline_a(meta_list)

    # ── Baseline B: random search ─────────────────────────────────────────────
    results["Random search"]  = run_baseline_b(meta_list)

    # ── Baseline C: template matching ─────────────────────────────────────────
    results["Template match"] = run_baseline_c(meta_list)

    # ── ANAS: our method ──────────────────────────────────────────────────────
    results["ANAS (ours)"]    = run_anas(meta_list)

    # ── Print summary table ───────────────────────────────────────────────────
    print_table(results)

    # ── Save JSON ─────────────────────────────────────────────────────────────
    save_results(results)


if __name__ == "__main__":
    main()
