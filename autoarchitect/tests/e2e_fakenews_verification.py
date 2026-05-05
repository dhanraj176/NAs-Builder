"""
tests/e2e_fakenews_verification.py — AutoArchitect Text Domain Verification

Fresh train → fresh test → real accuracy on text never seen during training.

Steps:
  1. Load GonzaloA/fake_news (capped at 5000 samples), enforce 70/15/15
     Save 15% holdout to tests/results/fakenews_holdout.json BEFORE training
  2. Build vocabulary from train texts only, train DARTSNet on 70% split
  3. Generate agent ZIP → tests/output/fakenews_agent.zip (vocab bundled)
  4. Extract ZIP, load model + vocab, run inference on 15% holdout
  5. Print per-sample results + final verification report
  6. Save tests/results/fakenews_verification.json
"""

import sys
import io

if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'buffer'):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import os
import json
import hashlib
import re
import random
import shutil
import time
import zipfile
import io as _io
from collections import Counter
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Paths & constants ──────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent.parent
TESTS_DIR   = Path(__file__).parent
RESULTS_DIR = TESTS_DIR / "results"
OUTPUT_DIR  = TESTS_DIR / "output"
TEMP_DIR    = TESTS_DIR / "temp" / "fakenews_test"

HOLDOUT_JSON = RESULTS_DIR / "fakenews_holdout.json"
REPORT_JSON  = RESULTS_DIR / "fakenews_verification.json"
ZIP_PATH     = OUTPUT_DIR  / "fakenews_agent.zip"

PROBLEM    = "classify fake news articles"
DOMAIN     = "text"
CLASSES    = ["real", "fake"]       # label 0=real, 1=fake
EPOCHS     = 5
SEED       = 42
CAP        = 5000                   # samples to use from 40K dataset
VOCAB_SIZE = 1000

# Model paths (same hash convention as self_trainer.py)
_cleaned    = re.sub(r'[^\w\s]', '', PROBLEM)
_normalized = ' '.join(_cleaned.lower().split())
PROB_HASH   = hashlib.md5(_normalized.encode()).hexdigest()[:10]
MODEL_PATH  = BASE_DIR / "models" / "trained" / f"{PROB_HASH}_text.pth"
CLS_PATH    = BASE_DIR / "models" / "trained" / f"{PROB_HASH}_text_classes.json"
VOCAB_PATH  = BASE_DIR / "models" / "trained" / f"{PROB_HASH}_text_vocab.json"

ARROW_DIR = BASE_DIR / "datasets" / "hf_cache" / "GonzaloA___fake_news"


# ── Bag-of-words dataset ───────────────────────────────────────────────────────

class TextBowDataset(torch.utils.data.Dataset):
    def __init__(self, texts: list, labels: list, w2i: dict):
        self.texts  = texts
        self.labels = labels
        self.w2i    = w2i

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        vec = torch.zeros(VOCAB_SIZE)
        for word in str(self.texts[idx]).lower().split():
            if word in self.w2i:
                vec[self.w2i[word]] += 1
        if vec.sum() > 0:
            vec = vec / vec.sum()
        padded = torch.zeros(3 * 32 * 32)
        padded[:VOCAB_SIZE] = vec[:3 * 32 * 32]
        return padded.reshape(3, 32, 32), int(self.labels[idx])


def text_to_tensor(text: str, w2i: dict) -> torch.Tensor:
    vec = torch.zeros(VOCAB_SIZE)
    for word in str(text).lower().split():
        if word in w2i:
            vec[w2i[word]] += 1
    if vec.sum() > 0:
        vec = vec / vec.sum()
    padded = torch.zeros(3 * 32 * 32)
    padded[:VOCAB_SIZE] = vec[:3 * 32 * 32]
    return padded.reshape(1, 3, 32, 32)


# ── Step 1 — Load & split ──────────────────────────────────────────────────────

def load_and_split():
    import pyarrow as pa

    print("\n" + "=" * 55)
    print("  STEP 1 — Load dataset + enforce 70/15/15 split")
    print("=" * 55)

    all_rows = []
    for arrow_path in sorted(ARROW_DIR.rglob("*.arrow")):
        r = pa.ipc.open_stream(str(arrow_path))
        for batch in r:
            texts  = batch.column("text").to_pylist()
            labels = batch.column("label").to_pylist()
            for text, lbl in zip(texts, labels):
                if text and str(text).strip():
                    all_rows.append({"text": str(text), "label": int(lbl)})

    print(f"  Total in dataset : {len(all_rows)}")

    # Cap and shuffle deterministically
    random.seed(SEED)
    random.shuffle(all_rows)
    all_rows = all_rows[:CAP]

    n_total = len(all_rows)
    n_train = int(0.70 * n_total)
    n_val   = int(0.15 * n_total)
    n_test  = n_total - n_train - n_val

    train_rows = all_rows[:n_train]
    val_rows   = all_rows[n_train : n_train + n_val]
    test_rows  = all_rows[n_train + n_val :]

    lbl_counts = Counter(r["label"] for r in all_rows)
    print(f"  Using (capped)   : {n_total}")
    print(f"  Label 0 (real)   : {lbl_counts[0]}")
    print(f"  Label 1 (fake)   : {lbl_counts[1]}")
    print(f"  Split            : {n_train} train / {n_val} val / {n_test} test  (70/15/15)")

    # ── Save holdout BEFORE training starts ───────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    holdout = {
        "problem":  PROBLEM,
        "domain":   DOMAIN,
        "classes":  CLASSES,
        "saved_at": datetime.now().isoformat(),
        "note":     "Saved BEFORE training. These samples are never seen during training.",
        "total":    n_test,
        "samples": [
            {
                "idx":        i + 1,
                "snippet":    r["text"][:120].replace("\n", " "),
                "true_label": r["label"],
                "true_class": CLASSES[r["label"]],
            }
            for i, r in enumerate(test_rows)
        ],
    }
    with open(HOLDOUT_JSON, "w", encoding="utf-8") as f:
        json.dump(holdout, f, indent=2, ensure_ascii=False)

    print(f"\n  ** Holdout saved to {HOLDOUT_JSON.name} **")
    print(f"     {n_test} samples locked away. Training will never touch them.")

    return train_rows, val_rows, test_rows, n_train, n_val, n_test


# ── Step 2 — Train ────────────────────────────────────────────────────────────

def train_model(train_rows: list, val_rows: list):
    print("\n" + "=" * 55)
    print("  STEP 2 — Train DARTSNet text classifier")
    print("=" * 55)

    from api.nas_engine import DARTSNet

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device     : {device}")
    print(f"  Train      : {len(train_rows)} samples")
    print(f"  Val        : {len(val_rows)} samples")
    print(f"  Vocab size : {VOCAB_SIZE}")
    print(f"  Epochs     : {EPOCHS}")

    # Build vocabulary from training texts ONLY
    train_texts = [r["text"] for r in train_rows]
    all_words   = " ".join(train_texts).lower().split()
    vocab_words = [w for w, _ in Counter(all_words).most_common(VOCAB_SIZE)]
    w2i         = {w: i for i, w in enumerate(vocab_words)}
    print(f"  Unique vocab words found: {len(w2i)}")

    train_labels = [r["label"] for r in train_rows]
    val_labels   = [r["label"] for r in val_rows]
    val_texts    = [r["text"]  for r in val_rows]

    train_ds = TextBowDataset(train_texts, train_labels, w2i)
    val_ds   = TextBowDataset(val_texts,   val_labels,   w2i)
    tr_ld    = torch.utils.data.DataLoader(
        train_ds, batch_size=64, shuffle=True,  num_workers=0)
    va_ld    = torch.utils.data.DataLoader(
        val_ds,   batch_size=64, shuffle=False, num_workers=0)

    num_classes = len(CLASSES)
    model       = DARTSNet(C=16, num_cells=3, num_classes=num_classes).to(device)
    params      = sum(p.numel() for p in model.parameters())
    print(f"  Parameters : {params:,}")

    net_opt  = optim.Adam(
        [p for n, p in model.named_parameters() if "arch_weights" not in n],
        lr=0.001)
    arch_opt = optim.Adam(
        [p for n, p in model.named_parameters() if "arch_weights" in n],
        lr=0.01)
    criterion    = nn.CrossEntropyLoss()
    t0           = time.time()
    epoch_history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        correct = total_n = 0
        for tensors, lbls in tr_ld:
            tensors, lbls = tensors.to(device), lbls.to(device)

            net_opt.zero_grad()
            out  = model(tensors)
            loss = criterion(out, lbls)
            loss.backward()
            net_opt.step()

            arch_opt.zero_grad()
            out  = model(tensors)
            loss = criterion(out, lbls)
            loss.backward()
            arch_opt.step()

            correct  += (out.argmax(1) == lbls).sum().item()
            total_n  += lbls.size(0)
        train_acc = round(100 * correct / total_n, 2)

        model.eval()
        vc = vt = 0
        with torch.no_grad():
            for tensors, lbls in va_ld:
                tensors, lbls = tensors.to(device), lbls.to(device)
                vc += (model(tensors).argmax(1) == lbls).sum().item()
                vt += lbls.size(0)
        val_acc = round(100 * vc / vt, 2)

        epoch_history.append({"epoch": epoch, "train_acc": train_acc, "val_acc": val_acc})
        print(f"  Epoch {epoch}/{EPOCHS} — train: {train_acc}%  val: {val_acc}%")

    duration = round(time.time() - t0, 1)

    # Save model + vocabulary + metadata
    (BASE_DIR / "models" / "trained").mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(MODEL_PATH))

    with open(VOCAB_PATH, "w") as f:
        json.dump(w2i, f)

    meta = {
        "problem":        PROBLEM,
        "domain":         DOMAIN,
        "classes":        CLASSES,
        "num_classes":    num_classes,
        "vocab_size":     len(w2i),
        "train_accuracy": epoch_history[-1]["train_acc"],
        "test_accuracy":  epoch_history[-1]["val_acc"],
        "method":         "darts_nas",
        "dataset":        "GonzaloA/fake_news",
        "trained_at":     datetime.now().isoformat(),
    }
    with open(CLS_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n  Model saved : {MODEL_PATH.name}")
    print(f"  Vocab saved : {VOCAB_PATH.name}")
    print(f"  Final val   : {epoch_history[-1]['val_acc']}%")
    print(f"  Time        : {duration}s")

    return {
        "model":          model,
        "device":         device,
        "w2i":            w2i,
        "train_accuracy": epoch_history[-1]["train_acc"],
        "val_accuracy":   epoch_history[-1]["val_acc"],
        "epoch_history":  epoch_history,
        "parameters":     params,
        "time":           duration,
    }


# ── Step 3 — Generate ZIP (with vocab bundled) ────────────────────────────────

def generate_zip():
    print("\n" + "=" * 55)
    print("  STEP 3 — Generate agent ZIP")
    print("=" * 55)

    from api.brain.network_zip_generator import NetworkZipGenerator
    from api.agents.agent_factory import get_factory

    topology = {
        "agents":      ["text"],
        "topology":    "sequential",
        "connections": [],
    }
    gen       = NetworkZipGenerator()
    zip_bytes = gen.generate(
        problem        = PROBLEM,
        topology       = topology,
        trained_models = {"text": str(MODEL_PATH)},
    )

    # Bundle vocabulary into the ZIP (needed for inference)
    factory    = get_factory()
    agent_mod  = factory.generate_name(PROBLEM)         # e.g. "classify_fake_agent"
    vocab_key  = f"models/{agent_mod[:-6]}_vocab.json"  # e.g. "models/classify_fake_vocab.json"

    buf = _io.BytesIO(zip_bytes)
    with zipfile.ZipFile(buf, "a", zipfile.ZIP_DEFLATED) as zf:
        zf.write(str(VOCAB_PATH), vocab_key)
    zip_bytes = buf.getvalue()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(ZIP_PATH, "wb") as f:
        f.write(zip_bytes)

    contents = zipfile.ZipFile(_io.BytesIO(zip_bytes)).namelist()
    print(f"\n  ZIP written : {ZIP_PATH.name} ({len(zip_bytes):,} bytes)")
    print(f"  Contents    : {contents}")
    return zip_bytes


# ── Step 4 — Verify on unseen holdout ────────────────────────────────────────

def verify_on_holdout(zip_bytes: bytes, test_rows: list):
    print("\n" + "=" * 55)
    print("  STEP 4 — Inference on unseen holdout samples")
    print("=" * 55)

    # Extract ZIP
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(_io.BytesIO(zip_bytes)) as zf:
        zf.extractall(TEMP_DIR)

    # Load vocabulary from ZIP
    vocab_files = list((TEMP_DIR / "models").glob("*_vocab.json"))
    if not vocab_files:
        raise FileNotFoundError("No _vocab.json in extracted ZIP models/")
    with open(vocab_files[0]) as f:
        w2i = json.load(f)

    # Load model from ZIP
    model_files = list((TEMP_DIR / "models").glob("*.pth"))
    if not model_files:
        raise FileNotFoundError("No .pth in extracted ZIP models/")

    from api.nas_engine import DARTSNet
    num_classes = len(CLASSES)
    model       = DARTSNet(C=16, num_cells=3, num_classes=num_classes)
    model.load_state_dict(
        torch.load(str(model_files[0]), map_location="cpu", weights_only=True))
    model.eval()

    print(f"  Model loaded from ZIP : {model_files[0].name}")
    print(f"  Vocab loaded from ZIP : {vocab_files[0].name} ({len(w2i)} words)")
    print(f"  Running inference on {len(test_rows)} unseen samples\n")

    results = []
    correct = 0

    for i, row in enumerate(test_rows, 1):
        text     = row["text"]
        true_lbl = row["label"]
        true_cls = CLASSES[true_lbl]
        snippet  = text[:55].replace("\n", " ")

        try:
            tensor = text_to_tensor(text, w2i)
            with torch.no_grad():
                out   = model(tensor)
                probs = torch.softmax(out, dim=1)
                conf  = float(probs.max())
                idx   = int(probs.argmax())
            pred_cls = CLASSES[idx] if idx < len(CLASSES) else str(idx)
            passed   = (idx == true_lbl)
        except Exception as e:
            pred_cls, conf, passed = "error", 0.0, False

        if passed:
            correct += 1

        status   = "PASS" if passed else "FAIL"
        conf_pct = f"{conf * 100:.1f}%"
        print(f'  [{i:3}] "{snippet:<55}" -> '
              f"Predicted: {pred_cls:<6} -> "
              f"Confidence: {conf_pct:<8} -> {status}")

        results.append({
            "idx":        i,
            "snippet":    snippet,
            "true_class": true_cls,
            "pred_class": pred_cls,
            "confidence": round(conf * 100, 1),
            "pass":       passed,
        })

    accuracy = round(100 * correct / len(test_rows), 1)
    print(f"\n  Correct : {correct} / {len(test_rows)}")
    print(f"  Accuracy: {accuracy}%")
    return results, accuracy


# ── Step 5 — Final report ─────────────────────────────────────────────────────

def print_and_save_report(train_result, inference_results, accuracy,
                           n_train, n_val, n_test):
    status = "VERIFIED" if accuracy >= 60.0 else "FAILED"

    line = "=" * 44
    print(f"\n{line}")
    print(f"  AUTOARCHITECT VERIFICATION REPORT")
    print(f"{line}")
    print(f"  Domain   : Text Classification")
    print(f"  Dataset  : GonzaloA/fake_news")
    print(f"  Train size: {n_train} (70%) | Val: {n_val} (15%) | Test: {n_test} (15%)")
    print(f"  Train acc (val set): {train_result['val_accuracy']}%")
    print(f"  Test accuracy on UNSEEN text: {accuracy}%")
    print(f"  Agent ZIP: {ZIP_PATH.name}")
    print(f"  Status   : {status}")
    print(f"{line}")

    report = {
        "generated_at": datetime.now().isoformat(),
        "problem":      PROBLEM,
        "domain":       DOMAIN,
        "dataset":      "GonzaloA/fake_news",
        "classes":      CLASSES,
        "split": {
            "train_n":   n_train,
            "val_n":     n_val,
            "test_n":    n_test,
            "train_pct": 70,
            "val_pct":   15,
            "test_pct":  15,
            "cap":       CAP,
            "seed":      SEED,
        },
        "training": {
            "method":         "darts_nas",
            "epochs":         EPOCHS,
            "vocab_size":     len(train_result["w2i"]),
            "parameters":     train_result["parameters"],
            "train_accuracy": train_result["train_accuracy"],
            "val_accuracy":   train_result["val_accuracy"],
            "epoch_history":  train_result["epoch_history"],
            "time_seconds":   train_result["time"],
        },
        "verification": {
            "test_accuracy": accuracy,
            "correct":       sum(1 for r in inference_results if r["pass"]),
            "total":         len(inference_results),
            "status":        status,
            "note":          "Evaluated on holdout set. No overlap with training data.",
        },
        "artifacts": {
            "model_path":    str(MODEL_PATH),
            "vocab_path":    str(VOCAB_PATH),
            "zip_path":      str(ZIP_PATH),
            "holdout_json":  str(HOLDOUT_JSON),
            "report_json":   str(REPORT_JSON),
        },
        "per_sample_results": inference_results,
    }

    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n  Report saved: {REPORT_JSON}")
    print(f"  Holdout    : {HOLDOUT_JSON}")
    return status


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 55)
    print("  AUTOARCHITECT END-TO-END VERIFICATION  (TEXT)")
    print(f"  Problem : {PROBLEM}")
    print(f"  Hash    : {PROB_HASH}")
    print(f"  Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 55)

    t_start = time.time()

    train_rows, val_rows, test_rows, n_train, n_val, n_test = load_and_split()
    train_result = train_model(train_rows, val_rows)
    zip_bytes    = generate_zip()
    inference_results, accuracy = verify_on_holdout(zip_bytes, test_rows)
    status = print_and_save_report(
        train_result, inference_results, accuracy, n_train, n_val, n_test)

    total_time = round(time.time() - t_start, 1)
    print(f"\n  Total time  : {total_time}s")
    print(f"  Final status: {status}\n")

    shutil.rmtree(TEMP_DIR, ignore_errors=True)
    return 0 if status == "VERIFIED" else 1


if __name__ == "__main__":
    sys.exit(main())
