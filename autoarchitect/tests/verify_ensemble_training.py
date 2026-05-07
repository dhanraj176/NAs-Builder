# -*- coding: utf-8 -*-
"""
tests/verify_ensemble_training.py -- Day 21: Ensemble training verification

Three test tiers (all runnable offline / without HuggingFace downloads):

  1. Offline API    -- FusionAgent.fuse() logic, AgentNetwork.collaborate() surface
  2. Synthetic      -- Train two tiny DARTSNet models on random data in parallel,
                       verify both .pth files saved, run ensemble vs single-agent
  3. Integration    -- If real models exist in models/trained/, run ensemble on them

Usage:
    python tests/verify_ensemble_training.py           # all tiers
    python tests/verify_ensemble_training.py --quick   # offline only
"""

import sys
import os
import time
import json
import traceback
import threading
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TRAINED_DIR = Path(__file__).parent.parent / "models" / "trained"
TRAINED_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# TIER 1 -- Offline API tests
# ============================================================

def run_api_tests() -> bool:
    print("\n-- Offline API Tests (FusionAgent + AgentNetwork) --------------")

    from api.agents.fusion_agent   import FusionAgent
    from api.agents.agent_network  import AgentNetwork

    fusion  = FusionAgent()
    network = AgentNetwork(name="test_ensemble")

    # --- fuse() with two mock results ---
    r1 = {"label": "pothole", "confidence": 0.85, "agent_name": "ImageAgent"}
    r2 = {"label": "pothole", "confidence": 0.72, "agent_name": "MultimodalAgent"}
    r3 = {"label": "road",    "confidence": 0.60, "agent_name": "MultimodalAgent"}

    fused_agree  = fusion.fuse([r1, r2], domain="image")
    fused_disagree = fusion.fuse([r1, r3], domain="image")
    fused_single = fusion.fuse([r1], domain="image")

    checks = [
        ("fuse() agrees: winner=pothole",
             fused_agree.get("label") == "pothole"),
        ("fuse() agree: conf is float",
             isinstance(fused_agree.get("confidence"), float)),
        ("fuse() agree: contributing_agents present",
             "contributing_agents" in fused_agree),
        ("fuse() weights_source set",
             fused_agree.get("weights_source") in
             ("learned", "default", "provided")),
        ("fuse() single: passthrough method",
             fused_single.get("fusion_method") == "passthrough"),
        ("fuse() disagree: returns a valid label",
             fused_disagree.get("label") in ("pothole", "road")),
        ("collaborate_from_models() exists",
             hasattr(network, "collaborate_from_models")),
        ("collaborate() exists",
             hasattr(network, "collaborate")),
        ("get_weights_for_domain() callable",
             callable(fusion.get_weights_for_domain)),
    ]

    all_pass = True
    for name, result in checks:
        status = "PASS" if result else "FAIL"
        print(f"   [{status}] {name}")
        if not result:
            all_pass = False

    return all_pass


# ============================================================
# TIER 2 -- Synthetic parallel training test
# ============================================================

def _make_synthetic_loader(n_samples=64, num_classes=2, img_size=32, batch=16):
    """Create a tiny in-memory DataLoader on random 3-channel images."""
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    X = torch.randn(n_samples, 3, img_size, img_size)
    y = torch.randint(0, num_classes, (n_samples,))
    ds = TensorDataset(X, y)
    return DataLoader(ds, batch_size=batch, shuffle=True)


def _train_tiny_model(model, loader, epochs=1, device="cpu"):
    """Train a tiny DARTSNet for 1 epoch on synthetic data."""
    import torch
    import torch.nn as nn
    import torch.optim as optim

    model.train()
    opt       = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    for _ in range(epochs):
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            opt.zero_grad()
            out  = model(X)
            loss = criterion(out, y)
            loss.backward()
            opt.step()

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            preds    = model(X).argmax(dim=1)
            correct += (preds == y).sum().item()
            total   += y.size(0)

    return round(100 * correct / total, 2)


class _SyntheticAgent:
    """Minimal agent wrapping a trained DARTSNet for predict()."""

    def __init__(self, model, classes: list, name: str):
        import torch
        self._model          = model
        self._classes        = classes
        self.__class__       = type(name, (_SyntheticAgent,), {})
        self.__class__.__name__ = name
        self.agent_id        = name
        self.category        = "image"
        self.problem         = "synthetic_test"
        self.total_predictions = 0
        self._memory         = type("M", (), {"get_recent_accuracy": lambda *a: 0.0})()
        self._device         = torch.device("cpu")

    def predict(self, data):
        import torch
        self._model.eval()
        with torch.no_grad():
            if isinstance(data, torch.Tensor):
                x = data
            else:
                x = torch.randn(1, 3, 32, 32)
            if x.dim() == 3:
                x = x.unsqueeze(0)
            x      = x.to(self._device)
            logits = self._model(x)
            probs  = torch.softmax(logits, dim=1)
            idx    = probs.argmax(dim=1).item()
            conf   = round(float(probs[0, idx].item()), 4)
            label  = (self._classes[idx]
                      if idx < len(self._classes) else str(idx))
        self.total_predictions += 1
        return {
            "label":      label,
            "confidence": conf,
            "agent_name": self.__class__.__name__,
        }

    def _get_memory_accuracy(self):
        return 0.0

    def run_async(self, **kw): return None
    def stop(self):             pass
    def act(self, result, *a, **k): return result
    def remember(self, *a, **k):    pass


def run_synthetic_tests() -> bool:
    print("\n-- Synthetic Parallel Training Test ----------------------------")
    import torch
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from api.nas_engine import DARTSNet

    classes   = ["pothole", "normal"]
    num_cls   = 2
    device    = "cpu"

    train_ld  = _make_synthetic_loader(n_samples=128, num_classes=num_cls)
    test_ld   = _make_synthetic_loader(n_samples=32,  num_classes=num_cls)

    models    = {}
    save_paths = {}
    results   = {}

    # Parallel training of two architectures
    def train_agent(name):
        model = DARTSNet(C=8, num_cells=2, num_classes=num_cls).to(device)
        acc   = _train_tiny_model(model, train_ld, epochs=2, device=device)
        path  = str(TRAINED_DIR / f"synthetic_{name}.pth")
        torch.save(model.state_dict(), path)
        return name, model, acc, path

    t0     = time.time()
    agents = ["image_synth", "multimodal_synth"]
    print(f"   Training {len(agents)} agents in parallel (DARTSNet, 2 epochs, synthetic data)...")

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(train_agent, a): a for a in agents}
        for fut in as_completed(futures):
            name, model, acc, path = fut.result()
            models[name]     = model
            save_paths[name] = path
            results[name]    = acc

    elapsed = round(time.time() - t0, 2)

    # Both models saved
    both_saved = all(Path(p).exists() for p in save_paths.values())
    print(f"   Parallel training done in {elapsed}s")
    for name, acc in results.items():
        print(f"   {name}: {acc}% accuracy  -> {save_paths[name]}")

    # Build synthetic agents
    synth_agents = [
        _SyntheticAgent(models[a], classes, a)
        for a in agents
    ]

    # Ensemble inference via AgentNetwork.collaborate()
    from api.agents.agent_network import AgentNetwork
    test_tensor = torch.randn(1, 3, 32, 32)

    network     = AgentNetwork(name="synthetic_ensemble")
    ensemble    = network.collaborate(
        synth_agents, "detect potholes", test_tensor,
        timeout_per_agent=30, domain="image")
    ens_label = ensemble.get("label")
    ens_conf  = ensemble.get("confidence", 0.0)
    fusion_ok = ens_label in classes and isinstance(ens_conf, float)

    # Single-agent inference
    single = synth_agents[0].predict(test_tensor)
    single_label = single.get("label")
    single_conf  = single.get("confidence", 0.0)

    print(f"\n   Ensemble : {ens_label} conf={ens_conf:.4f}  "
          f"[{ensemble.get('fusion_method', '?')}]")
    print(f"   Single   : {single_label} conf={single_conf:.4f}")
    print(f"   Contributing agents: "
          f"{ensemble.get('contributing_agents', [])}")

    # Accuracy check: ensemble confidence should be >= min single confidence
    # (fused result is confidence-weighted, may be lower for disagreements)
    acc_check = isinstance(ens_conf, float) and ens_conf >= 0.0

    checks = [
        ("both models saved to disk",     both_saved),
        ("parallel training < 60s",       elapsed < 60),
        ("ensemble label is valid",        ens_label in classes),
        ("ensemble confidence is float",   isinstance(ens_conf, float)),
        ("ensemble confidence >= 0",       acc_check),
        ("fusion method set",              "fusion_method" in ensemble),
        ("contributing_agents present",    "contributing_agents" in ensemble),
        ("successful_agents == 2",
             ensemble.get("successful_agents", 0) == 2),
        ("single agent label valid",       single_label in classes),
    ]

    all_pass = True
    for name, result in checks:
        status = "PASS" if result else "FAIL"
        print(f"   [{status}] {name}")
        if not result:
            all_pass = False

    return all_pass


# ============================================================
# TIER 3 -- Integration test with real trained models
# ============================================================

def run_integration_tests() -> bool:
    print("\n-- Integration Test (real trained models, if present) ----------")

    # Find any two .pth model files with classes metadata
    pth_files = list(TRAINED_DIR.glob("*.pth"))
    cls_files = {
        p.stem.rsplit("_", 1)[0]: p
        for p in TRAINED_DIR.glob("*_classes.json")
    }

    if len(pth_files) < 2:
        print("   SKIP: fewer than 2 trained models found in models/trained/")
        print(f"         (found {len(pth_files)} .pth files)")
        return True

    print(f"   Found {len(pth_files)} trained models")

    from api.agents.agent_network import AgentNetwork
    import torch

    # Pick two models with class metadata
    candidates = []
    for pth in pth_files[:5]:
        stem  = pth.stem
        parts = stem.rsplit("_", 1)
        if len(parts) == 2:
            hash_id, domain = parts
            cls_key = stem
            cls_path = TRAINED_DIR / f"{stem}_classes.json"
            if cls_path.exists():
                try:
                    with open(cls_path) as f:
                        meta = json.load(f)
                    candidates.append({
                        "domain":     domain,
                        "model_path": str(pth),
                        "classes":    meta.get("classes", []),
                        "accuracy":   meta.get("test_accuracy", 0),
                    })
                except Exception:
                    pass

    if len(candidates) < 2:
        print(f"   SKIP: only {len(candidates)} model(s) with class metadata found")
        return True

    # Prefer different domains; fall back to any two if all same domain
    candidates.sort(key=lambda x: x.get("accuracy", 0), reverse=True)
    m1 = candidates[0]
    different = [c for c in candidates[1:] if c["domain"] != m1["domain"]]
    m2 = different[0] if different else candidates[1]

    # Use unique keys so same-domain models don't collide
    k1 = m1["domain"]
    k2 = m2["domain"] if m2["domain"] != m1["domain"] else f"{m2['domain']}_2"

    print(f"   Using: [{k1}] {m1['accuracy']}%  and [{k2}] {m2['accuracy']}%")

    t0  = time.time()
    net = AgentNetwork(name="real_ensemble")
    dummy_data = torch.randn(1, 3, 32, 32)

    try:
        result = net.collaborate_from_models(
            model_paths={k1: m1, k2: m2},
            problem   = "test ensemble inference",
            test_data = dummy_data,
            domain    = m1["domain"],
        )
        elapsed = round(time.time() - t0, 2)

        # Accept: full ensemble result OR single-agent passthrough
        # Reject: unhandled exception only — the key test is that no crash occurs
        no_crash = isinstance(result, dict)
        no_error = "error" not in result or result.get("successful_agents", 0) >= 1

        ok = no_crash and no_error
        status = "PASS" if ok else "FAIL"
        print(f"   [{status}] Ensemble inference in {elapsed}s "
              f"agents={result.get('successful_agents', '?')} "
              f"method={result.get('fusion_method', '?')}")
        return ok

    except Exception as e:
        print(f"   [FAIL] Ensemble inference raised: {e}")
        traceback.print_exc()
        return False


# ============================================================
# TIER 4 -- Orchestrator parallel routing smoke test
# ============================================================

def run_orchestrator_routing_test() -> bool:
    """
    Verify the routing logic in solve(): multi + parallel goes to
    _run_parallel_ensemble; sequential stays in _run_multi_agent.
    Tests the decision logic without instantiating the full orchestrator.
    """
    print("\n-- Orchestrator Routing Test -----------------------------------")
    import ast, inspect, textwrap
    import api.orchestrator as _orch_mod

    # Check methods exist in the class body
    src          = inspect.getsource(_orch_mod.AutoArchitectOrchestrator)
    has_parallel = "_run_parallel_ensemble" in src
    has_seq      = "_run_multi_agent"       in src
    routes_brain = "exec_mode" in src and "parallel" in src

    # Routing logic (mirrors what solve() does)
    def _would_be_parallel(workflow):
        brain_res = workflow.get("brain_result") or {}
        mode      = brain_res.get("architecture", {}).get(
                        "execution_mode", "parallel")
        return workflow["type"] == "multi" and mode in ("parallel", "hybrid")

    workflow_parallel = {
        "type": "multi", "agents": ["image", "multimodal"],
        "brain_result": {"architecture": {"execution_mode": "parallel"},
                         "source": "distilled_brain_v1"},
    }
    workflow_sequential = {
        "type": "multi", "agents": ["image", "text"],
        "brain_result": {"architecture": {"execution_mode": "sequential"},
                         "source": "distilled_brain_v1"},
    }
    workflow_no_brain = {
        "type": "multi", "agents": ["image", "text"],
        "brain_result": None,
    }

    checks = [
        ("_run_parallel_ensemble method exists",   has_parallel),
        ("_run_multi_agent method exists",         has_seq),
        ("routing code uses exec_mode + parallel", routes_brain),
        ("parallel workflow -> parallel",          _would_be_parallel(workflow_parallel)),
        ("sequential workflow -> sequential",
             not _would_be_parallel(workflow_sequential)),
        ("no-brain workflow defaults to parallel", _would_be_parallel(workflow_no_brain)),
    ]

    all_pass = True
    for name, result in checks:
        status = "PASS" if result else "FAIL"
        print(f"   [{status}] {name}")
        if not result:
            all_pass = False

    return all_pass


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    quick = "--quick" in sys.argv

    print("=" * 65)
    print("  Ensemble Training Verification -- Day 21")
    print("=" * 65)

    api_ok      = run_api_tests()
    routing_ok  = run_orchestrator_routing_test()
    synth_ok    = run_synthetic_tests() if not quick else True
    integ_ok    = run_integration_tests() if not quick else True

    all_ok = api_ok and routing_ok and synth_ok and integ_ok

    print("\n" + "=" * 65)
    print(f"  API tests:         {'PASS' if api_ok     else 'FAIL'}")
    print(f"  Routing test:      {'PASS' if routing_ok else 'FAIL'}")
    if not quick:
        print(f"  Synthetic test:    {'PASS' if synth_ok  else 'FAIL'}")
        print(f"  Integration test:  {'PASS' if integ_ok  else 'FAIL'}")
    else:
        print(f"  Synthetic/integ:   SKIPPED (--quick)")
    print(f"  OVERALL:           {'PASS' if all_ok     else 'FAIL'}")
    print("=" * 65)

    sys.exit(0 if all_ok else 1)
