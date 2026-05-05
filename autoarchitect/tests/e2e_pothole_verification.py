"""
tests/e2e_pothole_verification.py — AutoArchitect End-to-End Verification

Fresh train → fresh test → real accuracy on images never seen during training.

Steps:
  1. Load taroii/pothole-detection (437 images), enforce 70/15/15 split
     Save 15% holdout to tests/results/pothole_holdout.json BEFORE training
  2. Train ResNet18 transfer learning on 70% train split only
  3. Generate agent ZIP → tests/output/pothole_agent.zip
  4. Extract ZIP, load model from it, run inference on 15% holdout
  5. Print per-image results + final verification report
  6. Save tests/results/pothole_verification.json
"""

import sys
import io

# Force UTF-8 so emoji from library code don't crash cp1252 terminals
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
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Paths & constants ──────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent.parent
TESTS_DIR   = Path(__file__).parent
RESULTS_DIR = TESTS_DIR / "results"
OUTPUT_DIR  = TESTS_DIR / "output"
TEMP_DIR    = TESTS_DIR / "temp" / "pothole_test"

HOLDOUT_JSON = RESULTS_DIR / "pothole_holdout.json"
REPORT_JSON  = RESULTS_DIR / "pothole_verification.json"
ZIP_PATH     = OUTPUT_DIR  / "pothole_agent.zip"

PROBLEM = "detect potholes in road surface"
DOMAIN  = "image"
CLASSES = ["no_pothole", "pothole"]   # label 0 / label 1
EPOCHS  = 5
SEED    = 42

# Derive model path using same hash convention as self_trainer.py
_cleaned    = re.sub(r'[^\w\s]', '', PROBLEM)
_normalized = ' '.join(_cleaned.lower().split())
PROB_HASH   = hashlib.md5(_normalized.encode()).hexdigest()[:10]
MODEL_PATH  = BASE_DIR / "models" / "trained" / f"{PROB_HASH}_image.pth"
CLS_PATH    = BASE_DIR / "models" / "trained" / f"{PROB_HASH}_image_classes.json"

ARROW_DIR = BASE_DIR / "datasets" / "hf_cache" / "taroii___pothole-detection"

TRANSFORM_TRAIN = T.Compose([
    T.Resize((224, 224)),
    T.RandomHorizontalFlip(),
    T.RandomRotation(10),
    T.ColorJitter(brightness=0.2, contrast=0.2),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
TRANSFORM_EVAL = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ── Dataset ────────────────────────────────────────────────────────────────────

class PathDataset(Dataset):
    def __init__(self, rows: list, transform):
        self.rows      = rows
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        img = Image.open(row["path"]).convert("RGB")
        return self.transform(img), row["label"]


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
            imgs   = batch.column("image").to_pylist()
            labels = batch.column("label").to_pylist()
            for img, lbl in zip(imgs, labels):
                img_path = img.get("path", "") if isinstance(img, dict) else ""
                if img_path and Path(img_path).exists():
                    all_rows.append({"path": img_path, "label": int(lbl)})

    n_total   = len(all_rows)
    n_label0  = sum(1 for r in all_rows if r["label"] == 0)
    n_label1  = sum(1 for r in all_rows if r["label"] == 1)
    print(f"  Total images    : {n_total}")
    print(f"  Label 0 (no_pothole): {n_label0}")
    print(f"  Label 1 (pothole)   : {n_label1}")

    random.seed(SEED)
    random.shuffle(all_rows)

    n_train = int(0.70 * n_total)
    n_val   = int(0.15 * n_total)
    n_test  = n_total - n_train - n_val

    train_rows = all_rows[:n_train]
    val_rows   = all_rows[n_train : n_train + n_val]
    test_rows  = all_rows[n_train + n_val :]

    print(f"  Split: {n_train} train / {n_val} val / {n_test} test  (70/15/15)")

    # ── Save holdout BEFORE training starts ───────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    holdout = {
        "problem":    PROBLEM,
        "domain":     DOMAIN,
        "classes":    CLASSES,
        "saved_at":   datetime.now().isoformat(),
        "note":       "Saved BEFORE training. These images are never seen during training.",
        "total":      n_test,
        "images": [
            {
                "idx":        i + 1,
                "path":       r["path"],
                "filename":   Path(r["path"]).name,
                "true_label": r["label"],
                "true_class": CLASSES[r["label"]],
            }
            for i, r in enumerate(test_rows)
        ],
    }
    with open(HOLDOUT_JSON, "w") as f:
        json.dump(holdout, f, indent=2)

    print(f"\n  ** Holdout saved to {HOLDOUT_JSON.name} **")
    print(f"     {n_test} images locked away. Training will never touch them.")

    return train_rows, val_rows, test_rows, n_train, n_val, n_test


# ── Step 2 — Train ────────────────────────────────────────────────────────────

def train_model(train_rows: list, val_rows: list):
    print("\n" + "=" * 55)
    print("  STEP 2 — Train ResNet18 (transfer learning)")
    print("=" * 55)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device : {device}")
    print(f"  Train  : {len(train_rows)} images")
    print(f"  Val    : {len(val_rows)} images (internal eval only)")
    print(f"  Epochs : {EPOCHS}")

    train_ds = PathDataset(train_rows, TRANSFORM_TRAIN)
    val_ds   = PathDataset(val_rows,   TRANSFORM_EVAL)
    tr_ld    = DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=0)
    va_ld    = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=0)

    num_classes = len(CLASSES)
    model       = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    for p in model.parameters():
        p.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    for p in model.layer4.parameters():    # unfreeze last block
        p.requires_grad = True
    model = model.to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Params : {total:,} total / {trainable:,} trainable")

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    t0             = time.time()
    epoch_history  = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        correct = total_n = 0
        for imgs, lbls in tr_ld:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            out  = model(imgs)
            loss = criterion(out, lbls)
            loss.backward()
            optimizer.step()
            correct  += (out.argmax(1) == lbls).sum().item()
            total_n  += lbls.size(0)
        train_acc = round(100 * correct / total_n, 2)

        model.eval()
        vc = vt = 0
        with torch.no_grad():
            for imgs, lbls in va_ld:
                imgs, lbls = imgs.to(device), lbls.to(device)
                vc += (model(imgs).argmax(1) == lbls).sum().item()
                vt += lbls.size(0)
        val_acc = round(100 * vc / vt, 2)

        epoch_history.append({"epoch": epoch, "train_acc": train_acc, "val_acc": val_acc})
        print(f"  Epoch {epoch}/{EPOCHS} — train: {train_acc}%  val: {val_acc}%")
        scheduler.step()

    duration = round(time.time() - t0, 1)

    # Save model + metadata
    (BASE_DIR / "models" / "trained").mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(MODEL_PATH))
    meta = {
        "problem":        PROBLEM,
        "domain":         DOMAIN,
        "classes":        CLASSES,
        "num_classes":    num_classes,
        "train_accuracy": epoch_history[-1]["train_acc"],
        "test_accuracy":  epoch_history[-1]["val_acc"],
        "method":         "transfer_learning_resnet18",
        "dataset":        "taroii/pothole-detection",
        "trained_at":     datetime.now().isoformat(),
    }
    with open(CLS_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n  Model saved : {MODEL_PATH.name}")
    print(f"  Final val   : {epoch_history[-1]['val_acc']}%")
    print(f"  Time        : {duration}s")

    return {
        "model":          model,
        "device":         device,
        "train_accuracy": epoch_history[-1]["train_acc"],
        "val_accuracy":   epoch_history[-1]["val_acc"],
        "epoch_history":  epoch_history,
        "time":           duration,
    }


# ── Step 3 — Generate ZIP ─────────────────────────────────────────────────────

def generate_zip():
    print("\n" + "=" * 55)
    print("  STEP 3 — Generate agent ZIP")
    print("=" * 55)

    from api.brain.network_zip_generator import NetworkZipGenerator

    topology = {
        "agents":      ["image"],
        "topology":    "sequential",
        "connections": [],
    }
    gen       = NetworkZipGenerator()
    zip_bytes = gen.generate(
        problem        = PROBLEM,
        topology       = topology,
        trained_models = {"image": str(MODEL_PATH)},
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(ZIP_PATH, "wb") as f:
        f.write(zip_bytes)

    print(f"\n  ZIP written : {ZIP_PATH.name} ({len(zip_bytes):,} bytes)")
    contents = zipfile.ZipFile(_io.BytesIO(zip_bytes)).namelist()
    print(f"  Contents    : {contents}")
    return zip_bytes


# ── Step 4 — Verify on unseen images ─────────────────────────────────────────

def verify_on_holdout(zip_bytes: bytes, test_rows: list):
    print("\n" + "=" * 55)
    print("  STEP 4 — Inference on unseen holdout images")
    print("=" * 55)

    # Extract ZIP
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(_io.BytesIO(zip_bytes)) as zf:
        zf.extractall(TEMP_DIR)

    # Load model from ZIP
    model_files = list((TEMP_DIR / "models").glob("*.pth"))
    if not model_files:
        raise FileNotFoundError("No .pth in extracted ZIP models/")
    extracted_model = model_files[0]

    num_classes = len(CLASSES)
    model       = models.resnet18(weights=None)
    model.fc    = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(
        torch.load(str(extracted_model), map_location="cpu", weights_only=True))
    model.eval()

    print(f"  Model loaded from ZIP: {extracted_model.name}")
    print(f"  Running inference on {len(test_rows)} images never seen during training\n")

    results = []
    correct = 0

    for i, row in enumerate(test_rows, 1):
        img_path  = row["path"]
        true_lbl  = row["label"]
        true_cls  = CLASSES[true_lbl]
        filename  = Path(img_path).name

        try:
            img    = Image.open(img_path).convert("RGB")
            tensor = TRANSFORM_EVAL(img).unsqueeze(0)
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
        fn_trunc = filename[:38]
        print(f"  [{i:3}] {fn_trunc:<38} -> "
              f"Predicted: {pred_cls:<12} -> "
              f"Confidence: {conf_pct:<8} -> {status}")

        results.append({
            "idx":        i,
            "filename":   filename,
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
    badge  = "VERIFIED" if accuracy >= 60.0 else "FAILED"

    line = "=" * 44
    print(f"\n{line}")
    print(f"  AUTOARCHITECT VERIFICATION REPORT")
    print(f"{line}")
    print(f"  Domain   : Image Classification")
    print(f"  Dataset  : taroii/pothole-detection")
    print(f"  Train size: {n_train} (70%) | Val: {n_val} (15%) | Test: {n_test} (15%)")
    print(f"  Train acc (val set): {train_result['val_accuracy']}%")
    print(f"  Test accuracy on UNSEEN images: {accuracy}%")
    print(f"  Agent ZIP: {ZIP_PATH.name}")
    print(f"  Status   : {badge}")
    print(f"{line}")

    report = {
        "generated_at": datetime.now().isoformat(),
        "problem":      PROBLEM,
        "domain":       DOMAIN,
        "dataset":      "taroii/pothole-detection",
        "classes":      CLASSES,
        "split": {
            "train_n":  n_train,
            "val_n":    n_val,
            "test_n":   n_test,
            "train_pct": 70,
            "val_pct":   15,
            "test_pct":  15,
            "seed":      SEED,
        },
        "training": {
            "method":         "transfer_learning_resnet18",
            "epochs":         EPOCHS,
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
            "zip_path":      str(ZIP_PATH),
            "holdout_json":  str(HOLDOUT_JSON),
            "report_json":   str(REPORT_JSON),
        },
        "per_image_results": inference_results,
    }

    with open(REPORT_JSON, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n  Report saved: {REPORT_JSON}")
    print(f"  Holdout    : {HOLDOUT_JSON}")
    return status


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 55)
    print("  AUTOARCHITECT END-TO-END VERIFICATION")
    print(f"  Problem : {PROBLEM}")
    print(f"  Hash    : {PROB_HASH}")
    print(f"  Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 55)

    t_start = time.time()

    # Step 1 — load + split + save holdout
    train_rows, val_rows, test_rows, n_train, n_val, n_test = load_and_split()

    # Step 2 — train (holdout never touched)
    train_result = train_model(train_rows, val_rows)

    # Step 3 — generate ZIP
    zip_bytes = generate_zip()

    # Step 4 — verify on unseen holdout
    inference_results, accuracy = verify_on_holdout(zip_bytes, test_rows)

    # Step 5 — report
    status = print_and_save_report(
        train_result, inference_results, accuracy,
        n_train, n_val, n_test)

    total_time = round(time.time() - t_start, 1)
    print(f"\n  Total time: {total_time}s")
    print(f"  Final status: {status}\n")

    # Clean up temp extract dir
    shutil.rmtree(TEMP_DIR, ignore_errors=True)

    return 0 if status == "VERIFIED" else 1


if __name__ == "__main__":
    sys.exit(main())
