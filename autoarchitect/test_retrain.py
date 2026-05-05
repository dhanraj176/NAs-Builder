"""
test_retrain.py — Test NetworkZipGenerator.retrain()

Extracts 10 sample images from the garbage Arrow dataset,
organises them into class subfolders, then calls retrain()
on the existing 1309624efe_image.pth model.
"""

import sys
import io
import os
import shutil
from pathlib import Path

# Force UTF-8 so emoji in library code don't crash cp1252 terminals
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'buffer'):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).parent))

BASE_DIR   = Path(__file__).parent
ARROW_PATH = (BASE_DIR / "datasets" / "hf_cache"
              / "dmedhi___garbage-image-classification-detection"
              / "default" / "0.0.0"
              / "39569aebf4e9ae7b75b1832b8b0167607172f201"
              / "garbage-image-classification-detection-train.arrow")
OUT_DIR    = BASE_DIR / "test_retrain_data"
PROBLEM    = "detect illegal dumping in Oakland street cameras"
DOMAIN     = "image"
N_SAMPLES  = 10


def extract_sample_images():
    """Load 10 images from the Arrow stream dataset, save to class-labelled subfolders."""
    import pyarrow as pa
    import io as _io
    from PIL import Image as PILImage
    from collections import defaultdict

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)

    print(f"[Test] Loading Arrow stream: {ARROW_PATH.name}")
    reader = pa.ipc.open_stream(str(ARROW_PATH))
    print(f"[Test] Schema: {reader.schema.names}")

    # Read all batches into memory
    all_images      = []
    all_class_names = []
    for batch in reader:
        imgs   = batch.column("image").to_pylist()
        labels = batch.column("class_name").to_pylist()
        all_images.extend(imgs)
        all_class_names.extend(labels)

    print(f"[Test] Total rows: {len(all_images)}")

    # Sample N_SAMPLES spread evenly across classes
    by_class = defaultdict(list)
    for i, cls in enumerate(all_class_names):
        by_class[cls].append(i)

    selected    = []
    per_class   = max(1, N_SAMPLES // len(by_class))
    for cls, idxs in sorted(by_class.items()):
        selected.extend(idxs[:per_class])
        if len(selected) >= N_SAMPLES:
            break
    selected = selected[:N_SAMPLES]

    print(f"[Test] Saving {len(selected)} sample images -> {OUT_DIR}/")
    saved = 0
    for i in selected:
        cls     = all_class_names[i]
        raw     = all_images[i].get("bytes") if isinstance(all_images[i], dict) \
                  else all_images[i]
        if not raw:
            print(f"  [Test] Skip index {i} — empty bytes")
            continue
        cls_dir  = OUT_DIR / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        out_path = cls_dir / f"img_{i:05d}.jpg"
        try:
            img = PILImage.open(_io.BytesIO(raw)).convert("RGB")
            img.save(str(out_path))
            saved += 1
            print(f"  [Test] {cls}/{out_path.name}  ({img.size[0]}x{img.size[1]})")
        except Exception as e:
            print(f"  [Test] Failed index {i}: {e}")

    if saved == 0:
        raise RuntimeError("[Test] No images saved — check Arrow schema")

    print(f"\n[Test] Extracted {saved} images across "
          f"{len(list(OUT_DIR.iterdir()))} classes\n")
    return saved


def run():
    print("=" * 60)
    print("  AutoArchitect — retrain() test")
    print(f"  Problem : {PROBLEM}")
    print(f"  Domain  : {DOMAIN}")
    print(f"  Samples : {N_SAMPLES}")
    print("=" * 60)

    # Step 1: extract sample images
    try:
        extract_sample_images()
    except Exception as e:
        print(f"\n[Test] Image extraction failed: {e}")
        import traceback; traceback.print_exc()
        return

    # Step 2: call retrain()
    print("\n[Test] Calling NetworkZipGenerator.retrain() ...\n")
    try:
        from api.brain.network_zip_generator import NetworkZipGenerator
        gen    = NetworkZipGenerator()
        result = gen.retrain(
            problem       = PROBLEM,
            domain        = DOMAIN,
            new_data_path = str(OUT_DIR),
        )
    except Exception as e:
        print(f"\n[Test] retrain() raised: {e}")
        import traceback; traceback.print_exc()
        return

    # Step 3: print results
    print("\n" + "=" * 60)
    print("  RESULT")
    print("=" * 60)
    for k, v in result.items():
        print(f"  {k:<16}: {v}")
    print("=" * 60)

    # Verify expected keys
    assert result["status"]       == "retrained",   "status must be 'retrained'"
    assert result["epochs"]       == 5,              "epochs must be 5"
    assert isinstance(result["new_accuracy"], float), "new_accuracy must be float"
    print("\n[Test] ✅ All assertions passed")

    # Cleanup
    shutil.rmtree(OUT_DIR, ignore_errors=True)
    print("[Test] Cleaned up test data folder")


if __name__ == "__main__":
    run()
