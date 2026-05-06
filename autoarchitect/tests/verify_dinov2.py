# -*- coding: utf-8 -*-
"""
tests/verify_dinov2.py -- Day 15: DINOv2 vs ResNet18 comparison

Uses CIFAR-10 (binary subset: airplane vs automobile, 400 train / 100 test)
to benchmark DINOv2 frozen embeddings + linear head against ResNet18
transfer learning.

Tests:
  1.  DINOv2 backbone loads without error
  2.  Embedding extraction produces (N, 384) tensors
  3.  DINOv2 trains in < 60 seconds
  4.  DINOv2 accuracy > 70%
  5.  Model file size < 200 KB
  6.  predict_image() works end-to-end after DINOv2 training
  7.  predict_image() returns label in known classes
  8.  load_trained_model() reloads DINOv2 weights correctly
  9.  ResNet18 also trains successfully (baseline sanity check)
 10.  DINOv2 file is at least 10x smaller than ResNet18
 11.  All 12 system agents still pass
"""

import sys
import os
import time
import tempfile
import shutil
import subprocess

import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
from PIL import Image as PILImage

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from api.agents.image_agent import ImageAgent, DINOv2LinearClassifier
from api.transfer_trainer import train_transfer


# ── Dataset helpers ───────────────────────────────────────────────────────────

_TFM = T.Compose([
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Binary CIFAR-10 subset: class 0 (airplane) vs class 1 (automobile)
_CLASSES = ["airplane", "automobile"]

def _make_loaders(n_train=400, n_test=100):
    """Return (train_loader, test_loader, data_dict) from CIFAR-10 subset."""
    def _binary_indices(ds, max_per_class):
        idx = []
        counts = {0: 0, 1: 0}
        for i, (_, lbl) in enumerate(ds):
            if lbl in counts and counts[lbl] < max_per_class:
                idx.append(i)
                counts[lbl] += 1
            if sum(counts.values()) >= max_per_class * 2:
                break
        return idx

    ds_train = torchvision.datasets.CIFAR10(
        'datasets', train=True,  download=False, transform=_TFM)
    ds_test  = torchvision.datasets.CIFAR10(
        'datasets', train=False, download=False, transform=_TFM)

    tr_idx = _binary_indices(ds_train, n_train // 2)
    te_idx = _binary_indices(ds_test,  n_test  // 2)

    tr_loader = DataLoader(Subset(ds_train, tr_idx), batch_size=32, shuffle=True)
    te_loader = DataLoader(Subset(ds_test,  te_idx), batch_size=32, shuffle=False)

    data_dict = {
        "name":        "cifar10_binary",
        "train_size":  len(tr_idx),
        "test_size":   len(te_idx),
        "num_classes": 2,
        "classes":     _CLASSES,
        "train_loader": tr_loader,
        "test_loader":  te_loader,
    }
    return tr_loader, te_loader, data_dict


def _check(name, passed, detail=""):
    sym = "PASS" if passed else "FAIL"
    msg = f"  [{sym}] {name}"
    if detail:
        msg += f"  ({detail})"
    print(msg)
    return passed


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_dinov2(tr_loader, te_loader):
    print("\n-- DINOv2 training (CIFAR-10 binary, 400 train) --")
    results = {}
    agent = ImageAgent()

    # Test 1: backbone loads
    try:
        agent._load_dinov2()
        results["dinov2_backbone_loads"] = _check(
            "DINOv2 backbone loads", True, "facebook/dinov2-small")
    except Exception as e:
        results["dinov2_backbone_loads"] = _check(
            "DINOv2 backbone loads", False, str(e)[:80])
        print("  Cannot continue DINOv2 tests without backbone.")
        for k in ["embedding_shape_384", "trains_under_60s", "accuracy_above_70",
                  "model_size_under_200kb", "predict_returns_label",
                  "predict_label_in_classes", "reload_works"]:
            results[k] = _check(k, False, "backbone unavailable")
        return results, None, None

    # Test 2: embedding shape
    try:
        sample_batch = next(iter(tr_loader))[0][:4]
        emb = agent._extract_embeddings_from_tensors(sample_batch)
        ok  = emb.shape == (4, 384)
        results["embedding_shape_384"] = _check(
            "Embedding extraction returns (N, 384)", ok, str(emb.shape))
    except Exception as e:
        results["embedding_shape_384"] = _check(
            "Embedding extraction returns (N, 384)", False, str(e)[:60])

    # Tests 3-5: full training
    metrics = None
    try:
        metrics = agent.train_with_dinov2(
            tr_loader, te_loader, num_classes=2,
            hash_id="d15test", classes=_CLASSES)

        results["trains_under_60s"] = _check(
            "DINOv2 trains in < 60 seconds",
            metrics["training_time"] < 60,
            f"{metrics['training_time']}s")

        results["accuracy_above_70"] = _check(
            "DINOv2 accuracy > 70%",
            metrics["test_accuracy"] > 70,
            f"{metrics['test_accuracy']}%")

        results["model_size_under_200kb"] = _check(
            "Model file < 200 KB",
            metrics["model_size_kb"] < 200,
            f"{metrics['model_size_kb']:.0f} KB")
    except Exception as e:
        print(f"  [FAIL] DINOv2 training raised: {e}")
        for k in ["trains_under_60s", "accuracy_above_70", "model_size_under_200kb"]:
            results[k] = False

    # Tests 6-7: predict_image end-to-end
    tmp = tempfile.mkdtemp(prefix="d15_pred_")
    try:
        test_img_path = os.path.join(tmp, "test.png")
        PILImage.new("RGB", (64, 64), color=(120, 80, 40)).save(test_img_path)
        pred = agent.predict_image(test_img_path)

        results["predict_returns_label"] = _check(
            "predict_image() returns a label",
            pred.get("label") not in ("error", None),
            str(pred.get("label")))

        results["predict_label_in_classes"] = _check(
            "predict_image() label is in known classes",
            pred.get("label") in _CLASSES,
            f"label={pred.get('label')}  classes={_CLASSES}")
    except Exception as e:
        results["predict_returns_label"]    = _check("predict_image() returns a label",    False, str(e)[:60])
        results["predict_label_in_classes"] = _check("predict_image() label in classes",   False, str(e)[:60])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # Test 8: reload from disk
    try:
        from pathlib import Path
        weights = str(Path(__file__).parent.parent / "models" / "trained"
                      / "d15test_image_linear.pth")
        agent2 = ImageAgent()
        agent2.load_trained_model(weights, _CLASSES, 2)
        ok = agent2.model_arch == "dinov2_linear" and agent2._dinov2_linear is not None
        results["reload_works"] = _check(
            "load_trained_model() reloads DINOv2 correctly",
            ok, f"arch={agent2.model_arch}")
    except Exception as e:
        results["reload_works"] = _check(
            "load_trained_model() reloads DINOv2 correctly", False, str(e)[:60])

    return results, metrics, agent


def test_resnet18(tr_loader, te_loader, data_dict):
    print("\n-- ResNet18 baseline (same CIFAR-10 binary split) --")
    results = {}
    try:
        t0 = time.time()
        tr = train_transfer("binary image classification",
                            data_dict, epochs=3, device=None)
        elapsed = time.time() - t0

        ok = tr.get("test_accuracy", 0) > 50
        results["resnet18_trains_ok"] = _check(
            "ResNet18 trains successfully",
            ok, f"test={tr.get('test_accuracy')}%  time={elapsed:.1f}s")

        # Model size
        import pickle
        tmp = tempfile.mkdtemp(prefix="d15_rn_")
        try:
            pth = os.path.join(tmp, "resnet.pth")
            torch.save(tr["model"].state_dict(), pth)
            size_mb = os.path.getsize(pth) / (1024 * 1024)
            results["resnet18_trains_ok"] = _check(
                "ResNet18 trains successfully",
                ok, f"test={tr.get('test_accuracy')}%  size={size_mb:.1f}MB")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

        return results, tr, elapsed, size_mb
    except Exception as e:
        results["resnet18_trains_ok"] = _check(
            "ResNet18 trains successfully", False, str(e)[:80])
        return results, None, 0, 0


def test_size_ratio(dinov2_kb, resnet_mb):
    print("\n-- Model size comparison --")
    results = {}
    ratio = (resnet_mb * 1024) / dinov2_kb if dinov2_kb > 0 else 0
    ok    = ratio >= 10
    results["dinov2_10x_smaller"] = _check(
        "DINOv2 model at least 10x smaller than ResNet18",
        ok, f"DINOv2={dinov2_kb:.0f}KB  ResNet18={resnet_mb:.1f}MB  "
            f"ratio={ratio:.0f}x")
    return results


def test_system_agents():
    print("\n-- System agents still pass --")
    script = os.path.join(os.path.dirname(__file__), "verify_all_agents.py")
    try:
        r = subprocess.run(
            [sys.executable, script],
            capture_output=True, text=True, timeout=300)
        passed = r.returncode == 0
        if not passed:
            for line in r.stdout.splitlines():
                if "FAIL" in line or "OVERALL" in line:
                    print(f"    {line}")
        _check("All 12 system agents pass", passed, f"exit={r.returncode}")
        return {"system_agents_pass": passed}
    except Exception as e:
        _check("All 12 system agents pass", False, str(e))
        return {"system_agents_pass": False}


def print_comparison_table(dinov2_m, dinov2_t, dinov2_kb,
                            resnet_m, resnet_t, resnet_mb):
    d_acc  = dinov2_m.get("test_accuracy", 0) if dinov2_m else 0
    d_time = dinov2_m.get("training_time", 0) if dinov2_m else 0
    r_acc  = resnet_m.get("test_accuracy",  0) if resnet_m else 0
    r_size = resnet_mb
    d_size = f"{dinov2_kb:.0f} KB" if dinov2_kb else "--"
    r_size_str = f"{r_size:.1f} MB"
    speedup    = round(resnet_t / d_time, 1) if d_time > 0 else 0
    acc_delta  = round(d_acc - r_acc, 1)
    ratio      = round((r_size * 1024) / dinov2_kb, 0) if dinov2_kb else 0

    print()
    print("  +--------------------+----------+-----------+----------+")
    print("  |  Method            | Time     | Accuracy  | Size     |")
    print("  +--------------------+----------+-----------+----------+")
    print(f"  |  ResNet18          | {resnet_t:>6.1f}s  | {r_acc:>7.1f}%   | {r_size_str:<8} |")
    print(f"  |  DINOv2 + Linear   | {d_time:>6.1f}s  | {d_acc:>7.1f}%   | {d_size:<8} |")
    print(f"  |  Improvement       | {speedup:>5.1f}x   | {acc_delta:>+6.1f}%   | {ratio:.0f}x smaller |")
    print("  +--------------------+----------+-----------+----------+")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  DAY 15 - DINOv2 vs ResNet18 Comparison")
    print("=" * 60)

    print("\n  Building CIFAR-10 binary loaders (airplane vs automobile)...")
    tr_loader, te_loader, data_dict = _make_loaders(n_train=400, n_test=100)
    print(f"  Train: {data_dict['train_size']} samples  "
          f"Test: {data_dict['test_size']} samples")

    all_results = {}

    r1, dinov2_metrics, _ = test_dinov2(tr_loader, te_loader)
    all_results.update(r1)

    r2, resnet_tr, resnet_elapsed, resnet_mb = test_resnet18(
        tr_loader, te_loader, data_dict)
    all_results.update(r2)

    if dinov2_metrics and resnet_mb:
        r3 = test_size_ratio(dinov2_metrics["model_size_kb"], resnet_mb)
        all_results.update(r3)
        print_comparison_table(
            dinov2_metrics, None, dinov2_metrics["model_size_kb"],
            resnet_tr,      resnet_elapsed, resnet_mb)
    else:
        all_results["dinov2_10x_smaller"] = False

    r4 = test_system_agents()
    all_results.update(r4)

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  DAY 15 VERIFICATION SUMMARY - DINOv2")
    print("=" * 60)
    all_pass = True
    for name, passed in all_results.items():
        status = "PASS" if passed else "FAIL"
        print("  {:<45} {}".format(name, status))
        if not passed:
            all_pass = False
    print("=" * 60)
    print(f"  OVERALL: {'ALL PASS' if all_pass else 'FAILURES DETECTED'}")
    print("=" * 60)
    sys.exit(0 if all_pass else 1)
