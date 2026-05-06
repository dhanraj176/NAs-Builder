# -*- coding: utf-8 -*-
"""
tests/verify_day17.py -- Day 17: DINOv2 medical + Wav2Vec2 audio

PART 1 - Medical DINOv2 (CIFAR-10 binary as synthetic medical images):
  1.  DINOv2 backbone loads for MedicalAgent
  2.  Embedding extraction returns (N, 384)
  3.  train_with_dinov2() trains in < 60 seconds
  4.  Test accuracy > 70%
  5.  Model file < 200 KB
  6.  predict_image() returns a label
  7.  predict_image() label is in known classes
  8.  load_trained_model() reloads DINOv2 linear correctly
  9.  predict_image() returns error dict (not fake %) when no model loaded
 10.  ResNet18 medical also trains (baseline sanity check)
 11.  DINOv2 at least 10x smaller than ResNet18

PART 2 - Audio Wav2Vec2 (synthetic sine-wave classes):
 12.  Wav2Vec2 backbone loads
 13.  train_with_wav2vec2() trains in < 300 seconds
       (If > 300s: SKIP and document — same lesson learned from SetFit)
 14.  Test accuracy > 60%
 15.  Model file < 200 KB
 16.  predict() returns a label after Wav2Vec2 training
 17.  predict() label is in known classes
 18.  load_trained_model() reloads Wav2Vec2 linear correctly
 19.  MFCC+RandomForest baseline also trains successfully
 20.  All 12 system agents still pass
"""

import sys
import os
import time
import tempfile
import shutil
import subprocess
import math

import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
from PIL import Image as PILImage

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from api.agents.medical_agent import MedicalAgent
from api.transfer_trainer import train_transfer

TRAINED_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "models", "trained")

_MED_CLASSES = ["normal", "abnormal"]   # binary medical subset


# ── CIFAR-10 loader (binary: airplane=normal, automobile=abnormal) ─────────────

_TFM = T.Compose([
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def _make_medical_loaders(n_train=400, n_test=100):
    def _binary_idx(ds, max_per_class):
        idx, counts = [], {0: 0, 1: 0}
        for i, (_, lbl) in enumerate(ds):
            if lbl in counts and counts[lbl] < max_per_class:
                idx.append(i); counts[lbl] += 1
            if sum(counts.values()) >= max_per_class * 2:
                break
        return idx

    ds_train = torchvision.datasets.CIFAR10(
        'datasets', train=True,  download=False, transform=_TFM)
    ds_test  = torchvision.datasets.CIFAR10(
        'datasets', train=False, download=False, transform=_TFM)

    tr_idx = _binary_idx(ds_train, n_train // 2)
    te_idx = _binary_idx(ds_test,  n_test  // 2)

    tr_load = DataLoader(Subset(ds_train, tr_idx), batch_size=32, shuffle=True)
    te_load = DataLoader(Subset(ds_test,  te_idx), batch_size=32, shuffle=False)

    data_dict = {
        "name": "cifar10_binary_medical",
        "train_size": len(tr_idx), "test_size": len(te_idx),
        "num_classes": 2, "classes": _MED_CLASSES,
        "train_loader": tr_load, "test_loader": te_load,
    }
    return tr_load, te_load, data_dict


# ── Synthetic audio helpers ───────────────────────────────────────────────────

def _write_wav(path, freq_hz, duration_s=1.0, sample_rate=16000):
    """Write a mono sine-wave .wav file via soundfile (guaranteed torchaudio-readable)."""
    import soundfile as sf
    n_samples = int(sample_rate * duration_s)
    t    = np.arange(n_samples) / sample_rate
    data = (0.5 * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)
    sf.write(path, data, sample_rate)


def _make_synthetic_audio_folder(tmp_dir, n_per_class=10):
    """
    Create two classes of synthetic sine-wave .wav files.
    low/: 200 Hz tones
    high/: 2000 Hz tones
    """
    classes = {"low": 200, "high": 2000}
    for cls_name, freq in classes.items():
        cls_dir = os.path.join(tmp_dir, cls_name)
        os.makedirs(cls_dir, exist_ok=True)
        for i in range(n_per_class):
            _write_wav(os.path.join(cls_dir, f"{i:03d}.wav"),
                       freq_hz=freq + i * 5)   # slight variation per file
    return tmp_dir, list(classes.keys())


# ── Check helper ──────────────────────────────────────────────────────────────

def _check(name, passed, detail=""):
    sym = "PASS" if passed else "FAIL"
    msg = f"  [{sym}] {name}"
    if detail:
        msg += f"  ({detail})"
    print(msg)
    return passed


# ═════════════════════════════════════════════════════════════════════════════
# PART 1: Medical DINOv2
# ═════════════════════════════════════════════════════════════════════════════

def test_medical_dinov2(tr_load, te_load, data_dict):
    print("\n-- Medical DINOv2 (CIFAR-10 binary as medical proxy) --")
    results = {}
    agent   = MedicalAgent()

    # Test 1: backbone loads
    try:
        agent._load_dinov2()
        results["dinov2_medical_loads"] = _check(
            "DINOv2 backbone loads for MedicalAgent", True,
            "facebook/dinov2-small")
    except Exception as e:
        results["dinov2_medical_loads"] = _check(
            "DINOv2 backbone loads for MedicalAgent", False, str(e)[:80])
        for k in ["embedding_shape", "trains_under_60s", "accuracy_above_70",
                  "model_under_200kb", "predict_returns_label",
                  "predict_label_in_classes", "reload_works"]:
            results[k] = _check(k, False, "backbone unavailable")
        return results, None, agent

    # Test 2: embedding shape
    try:
        batch = next(iter(tr_load))[0][:4]
        emb   = agent._extract_embeddings_from_tensors(batch)
        ok    = emb.shape == (4, 384)
        results["embedding_shape"] = _check(
            "Embedding extraction returns (N, 384)", ok, str(emb.shape))
    except Exception as e:
        results["embedding_shape"] = _check(
            "Embedding extraction returns (N, 384)", False, str(e)[:60])

    # Tests 3-5: full training
    metrics = None
    try:
        t0      = time.time()
        metrics = agent.train_with_dinov2(
            tr_load, te_load, num_classes=2,
            hash_id="d17med", classes=_MED_CLASSES)
        elapsed = round(time.time() - t0, 1)

        results["trains_under_60s"] = _check(
            "train_with_dinov2() completes in < 60 seconds",
            elapsed < 60, f"{elapsed}s")

        results["accuracy_above_70"] = _check(
            "Medical DINOv2 accuracy > 70%",
            metrics["test_accuracy"] > 70,
            f"{metrics['test_accuracy']}%")

        wpath  = metrics.get("model_path", "")
        size_kb = os.path.getsize(wpath) / 1024 if os.path.exists(wpath) else 9999
        results["model_under_200kb"] = _check(
            "Model file < 200 KB",
            size_kb < 200, f"{size_kb:.0f} KB")
    except Exception as e:
        print(f"  [FAIL] Medical DINOv2 training raised: {e}")
        import traceback; traceback.print_exc()
        for k in ["trains_under_60s", "accuracy_above_70", "model_under_200kb"]:
            results[k] = False

    # Tests 6-7: predict_image
    tmp = tempfile.mkdtemp(prefix="d17med_")
    try:
        img_path = os.path.join(tmp, "test.png")
        PILImage.new("RGB", (64, 64), color=(120, 80, 40)).save(img_path)
        pred = agent.predict_image(img_path)
        results["predict_returns_label"] = _check(
            "predict_image() returns a label",
            pred.get("label") not in ("error", None),
            str(pred.get("label")))
        results["predict_label_in_classes"] = _check(
            "predict_image() label is in known classes",
            pred.get("label") in _MED_CLASSES,
            f"label={pred.get('label')}  classes={_MED_CLASSES}")
    except Exception as e:
        results["predict_returns_label"]    = _check(
            "predict_image() returns a label", False, str(e)[:60])
        results["predict_label_in_classes"] = _check(
            "predict_image() label in classes", False, str(e)[:60])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # Test 9: error dict when no model loaded
    agent_empty = MedicalAgent()
    pred_err    = agent_empty.predict_image("nonexistent.png")
    ok = (pred_err.get("label") == "error"
          and pred_err.get("fake") is False
          and pred_err.get("confidence") == 0.0)
    results["error_dict_no_model"] = _check(
        "predict_image() returns error dict when no model loaded",
        ok, str(pred_err))

    # Test 8: reload
    try:
        from pathlib import Path
        weights = str(Path(__file__).parent.parent / "models" / "trained"
                      / "d17med_medical_linear.pth")
        agent2 = MedicalAgent()
        agent2.load_trained_model(weights, _MED_CLASSES, 2)
        ok = (agent2.model_arch == "dinov2_linear" and
              agent2._dinov2_linear is not None)
        results["reload_works"] = _check(
            "load_trained_model() reloads DINOv2 correctly",
            ok, f"arch={agent2.model_arch}")
    except Exception as e:
        results["reload_works"] = _check(
            "load_trained_model() reloads DINOv2", False, str(e)[:60])

    return results, metrics, agent


def test_medical_resnet18(data_dict):
    print("\n-- Medical ResNet18 baseline --")
    results = {}
    try:
        t0 = time.time()
        tr = train_transfer("binary medical classification",
                            data_dict, epochs=3, device=None)
        elapsed = round(time.time() - t0, 1)
        ok = tr.get("test_accuracy", 0) > 50
        results["resnet18_trains_ok"] = _check(
            "ResNet18 trains successfully",
            ok, f"test={tr.get('test_accuracy')}%  time={elapsed}s")

        tmp = tempfile.mkdtemp(prefix="d17rn_")
        try:
            pth = os.path.join(tmp, "resnet.pth")
            torch.save(tr["model"].state_dict(), pth)
            size_mb = os.path.getsize(pth) / (1024 * 1024)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

        return results, tr, elapsed, size_mb
    except Exception as e:
        results["resnet18_trains_ok"] = _check(
            "ResNet18 trains successfully", False, str(e)[:80])
        return results, None, 0, 0


def test_size_ratio(dinov2_kb, resnet_mb, results):
    print("\n-- Medical model size comparison --")
    ratio = (resnet_mb * 1024) / dinov2_kb if dinov2_kb > 0 else 0
    ok    = ratio >= 10
    results["medical_dinov2_10x_smaller"] = _check(
        "DINOv2 medical at least 10x smaller than ResNet18",
        ok, f"DINOv2={dinov2_kb:.0f} KB  ResNet18={resnet_mb:.1f} MB  "
            f"ratio={ratio:.0f}x")
    return results


def print_medical_table(dinov2_m, resnet_m, resnet_t, resnet_mb):
    d_acc  = dinov2_m.get("test_accuracy",  0) if dinov2_m else 0
    d_time = dinov2_m.get("training_time",  0) if dinov2_m else 0
    d_kb   = dinov2_m.get("model_size_kb",  0) if dinov2_m else 0
    r_acc  = resnet_m.get("test_accuracy",  0) if resnet_m else 0
    print()
    print("  +--------------------+----------+-----------+----------+")
    print("  |  Method            | Time     | Accuracy  | Size     |")
    print("  +--------------------+----------+-----------+----------+")
    print(f"  |  ResNet18          | {resnet_t:>6.1f}s  | {r_acc:>7.1f}%   | "
          f"{resnet_mb:.1f} MB    |")
    print(f"  |  DINOv2 + Linear   | {d_time:>6.1f}s  | {d_acc:>7.1f}%   | "
          f"{d_kb:.0f} KB     |")
    print("  +--------------------+----------+-----------+----------+")


# ═════════════════════════════════════════════════════════════════════════════
# PART 2: Audio Wav2Vec2
# ═════════════════════════════════════════════════════════════════════════════

def test_wav2vec2(tmp_audio_dir, audio_classes):
    print("\n-- Wav2Vec2 audio training (synthetic sine waves) --")
    results = {}

    from api.agents.audio_agent import AudioAgent

    # Test 12: backbone loads
    agent = AudioAgent()
    W2V_TIMEOUT = 300   # 5-minute circuit-breaker

    t0_total = time.time()

    try:
        t0 = time.time()
        agent._load_wav2vec2()
        load_time = round(time.time() - t0, 1)
        results["wav2vec2_loads"] = _check(
            "Wav2Vec2 backbone loads", True,
            f"facebook/wav2vec2-base-960h  {load_time}s")
    except Exception as e:
        results["wav2vec2_loads"] = _check(
            "Wav2Vec2 backbone loads", False, str(e)[:80])
        for k in ["wav2vec2_trains", "wav2vec2_accuracy",
                  "wav2vec2_size", "wav2vec2_predict",
                  "wav2vec2_label_in_classes", "wav2vec2_reload"]:
            results[k] = _check(k, False, "backbone unavailable")
        return results, None

    # Test 13: training with circuit-breaker
    metrics = None
    hash_id = "d17aud"
    try:
        t0      = time.time()
        metrics = agent.train_with_wav2vec2(tmp_audio_dir, hash_id)
        elapsed = round(time.time() - t0, 1)
        total_elapsed = round(time.time() - t0_total, 1)

        if total_elapsed > W2V_TIMEOUT:
            print(f"  [SKIP] Wav2Vec2 took {total_elapsed}s > {W2V_TIMEOUT}s "
                  f"circuit-breaker. Same issue as SetFit.")
            for k in ["wav2vec2_trains", "wav2vec2_accuracy",
                      "wav2vec2_size", "wav2vec2_predict",
                      "wav2vec2_label_in_classes", "wav2vec2_reload"]:
                results[k] = _check(k, False, f"SKIP: {total_elapsed}s > 300s")
            return results, None

        results["wav2vec2_trains"] = _check(
            "train_with_wav2vec2() completes in < 300 seconds",
            elapsed < W2V_TIMEOUT, f"{elapsed}s")

        results["wav2vec2_accuracy"] = _check(
            "Wav2Vec2 test accuracy > 60%",
            metrics.get("test_accuracy", 0) > 60,
            f"{metrics.get('test_accuracy')}%")

        wpath   = metrics.get("model_path", "")
        size_kb = os.path.getsize(wpath) / 1024 if os.path.exists(wpath) else 9999
        results["wav2vec2_size"] = _check(
            "Wav2Vec2 linear model < 200 KB",
            size_kb < 200, f"{size_kb:.0f} KB")

    except Exception as e:
        print(f"  [FAIL] Wav2Vec2 training raised: {e}")
        import traceback; traceback.print_exc()
        for k in ["wav2vec2_trains", "wav2vec2_accuracy", "wav2vec2_size"]:
            results[k] = False

    # Tests 16-17: predict
    tmp_wav = tempfile.mkdtemp(prefix="d17pred_")
    try:
        test_wav = os.path.join(tmp_wav, "test.wav")
        _write_wav(test_wav, freq_hz=200)
        pred = agent.predict(test_wav)
        results["wav2vec2_predict"] = _check(
            "predict() returns a label after Wav2Vec2 training",
            pred.get("label") not in ("error", None),
            str(pred.get("label")))
        results["wav2vec2_label_in_classes"] = _check(
            "predict() label is in known classes",
            pred.get("label") in audio_classes,
            f"label={pred.get('label')}  classes={audio_classes}")
    except Exception as e:
        results["wav2vec2_predict"]           = _check(
            "predict() returns a label", False, str(e)[:60])
        results["wav2vec2_label_in_classes"]  = _check(
            "predict() label in classes", False, str(e)[:60])
    finally:
        shutil.rmtree(tmp_wav, ignore_errors=True)

    # Test 18: reload
    try:
        from pathlib import Path
        weights = str(Path(__file__).parent.parent / "models" / "trained"
                      / f"{hash_id}_audio_linear.pth")
        agent2 = AudioAgent()
        ok_load = agent2.load_trained_model(weights)
        ok = ok_load and agent2.model_arch == "wav2vec2_linear"
        results["wav2vec2_reload"] = _check(
            "load_trained_model() reloads Wav2Vec2 correctly",
            ok, f"model_arch={agent2.model_arch}")
    except Exception as e:
        results["wav2vec2_reload"] = _check(
            "load_trained_model() reloads Wav2Vec2", False, str(e)[:60])

    return results, metrics


def test_mfcc_baseline(tmp_audio_dir):
    print("\n-- MFCC+RandomForest baseline (same synthetic audio) --")
    results = {}
    try:
        from api.agents.audio_agent import AudioAgent
        agent = AudioAgent()
        t0    = time.time()
        m     = agent.train(tmp_audio_dir, hash_id="d17mfcc")
        elapsed = round(time.time() - t0, 1)
        ok = m.get("accuracy", 0) > 0.4
        results["mfcc_trains_ok"] = _check(
            "MFCC+RandomForest trains successfully",
            ok, f"test={m.get('accuracy', 0)*100:.1f}%  time={elapsed}s")
        return results, m
    except Exception as e:
        results["mfcc_trains_ok"] = _check(
            "MFCC+RandomForest trains", False, str(e)[:80])
        return results, None


def print_audio_table(w2v_m, mfcc_m):
    w_acc  = w2v_m.get("test_accuracy", 0) if w2v_m else 0
    m_acc  = (mfcc_m.get("accuracy", 0) * 100) if mfcc_m else 0
    w_kb   = w2v_m.get("model_size_kb", 0) if w2v_m else 0
    print()
    print("  +---------------------+-----------+----------+")
    print("  |  Method             | Accuracy  | Size     |")
    print("  +---------------------+-----------+----------+")
    print(f"  |  MFCC + RandomForest| {m_acc:>7.1f}%   | pkl      |")
    print(f"  |  Wav2Vec2 + Linear  | {w_acc:>7.1f}%   | {w_kb:.0f} KB   |")
    print("  +---------------------+-----------+----------+")


# ── System agents ─────────────────────────────────────────────────────────────

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


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  DAY 17 - DINOv2 Medical + Wav2Vec2 Audio")
    print("=" * 60)

    all_results = {}

    # ── Part 1: Medical ──────────────────────────────────────────────────────
    print("\n  Building CIFAR-10 binary loaders (medical proxy)...")
    tr_load, te_load, data_dict = _make_medical_loaders(
        n_train=400, n_test=100)
    print(f"  Train: {data_dict['train_size']}  Test: {data_dict['test_size']}")

    r1, dinov2_m, _ = test_medical_dinov2(tr_load, te_load, data_dict)
    all_results.update(r1)

    r2, resnet_m, resnet_t, resnet_mb = test_medical_resnet18(data_dict)
    all_results.update(r2)

    if dinov2_m and resnet_mb:
        all_results = test_size_ratio(
            dinov2_m["model_size_kb"], resnet_mb, all_results)
        print_medical_table(dinov2_m, resnet_m, resnet_t, resnet_mb)
    else:
        all_results["medical_dinov2_10x_smaller"] = False

    # ── Part 2: Audio ────────────────────────────────────────────────────────
    tmp_audio = tempfile.mkdtemp(prefix="d17audio_")
    try:
        print(f"\n  Generating synthetic audio (sine waves)...")
        tmp_audio, audio_classes = _make_synthetic_audio_folder(
            tmp_audio, n_per_class=15)
        print(f"  Classes: {audio_classes}  (15 files each, 1s @ 16kHz)")

        r3, wav2vec2_m = test_wav2vec2(tmp_audio, audio_classes)
        all_results.update(r3)

        r4, mfcc_m = test_mfcc_baseline(tmp_audio)
        all_results.update(r4)

        if wav2vec2_m and mfcc_m:
            print_audio_table(wav2vec2_m, mfcc_m)
    finally:
        shutil.rmtree(tmp_audio, ignore_errors=True)

    # ── System agents ────────────────────────────────────────────────────────
    r5 = test_system_agents()
    all_results.update(r5)

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  DAY 17 VERIFICATION SUMMARY")
    print("=" * 60)
    all_pass = True
    for name, passed in all_results.items():
        status = "PASS" if passed else "FAIL"
        print("  {:<50} {}".format(name, status))
        if not passed:
            all_pass = False
    print("=" * 60)
    print(f"  OVERALL: {'ALL PASS' if all_pass else 'FAILURES DETECTED'}")
    print("=" * 60)
    sys.exit(0 if all_pass else 1)
