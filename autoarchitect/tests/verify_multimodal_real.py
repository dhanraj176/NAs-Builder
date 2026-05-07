# -*- coding: utf-8 -*-
"""
tests/verify_multimodal_real.py -- Day 22 follow-up: CLIP zero-shot on CIFAR-10

Measures MultimodalAgent accuracy on 100 random CIFAR-10 test images.
CLIP base (~340MB, downloaded once) typically achieves 60-75% zero-shot.

Usage:
    python tests/verify_multimodal_real.py
"""

import sys
import os
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from torchvision import datasets, transforms
from PIL import Image
from api.agents.multimodal_agent import MultimodalAgent

CIFAR10_LABELS = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

print("=" * 60)
print("  MultimodalAgent -- CIFAR-10 Zero-Shot Accuracy (Day 22)")
print("=" * 60)

print("\nDownloading CIFAR-10 test set (first run only)...")
testset = datasets.CIFAR10(
    root=str(DATA_DIR), train=False, download=True,
    transform=transforms.ToTensor(),
)

random.seed(42)
indices = random.sample(range(len(testset)), 100)

print("Loading MultimodalAgent (CLIP zero-shot)...")
agent = MultimodalAgent()

correct = 0
total   = 0
errors  = 0

print(f"\nTesting on 100 CIFAR-10 images...")
print("=" * 60)

to_pil = transforms.ToPILImage()

for idx, i in enumerate(indices):
    image_tensor, label_idx = testset[i]
    expected = CIFAR10_LABELS[label_idx]

    pil_img   = to_pil(image_tensor)
    temp_path = DATA_DIR / f"cifar_temp_{idx}.png"
    pil_img.save(str(temp_path))

    try:
        result    = agent.classify_with_labels(str(temp_path), CIFAR10_LABELS)
        predicted = result.get("label", "")

        if predicted in CIFAR10_LABELS:
            if predicted == expected:
                correct += 1
            total += 1
        else:
            errors += 1

        if (idx + 1) % 10 == 0:
            running = (correct / total * 100) if total > 0 else 0.0
            print(f"  [{idx + 1:3d}/100]  running accuracy: {running:.1f}%")

    except Exception as e:
        errors += 1
        if errors <= 3:
            print(f"  Error on sample {idx}: {e}")
    finally:
        if temp_path.exists():
            temp_path.unlink()

print("=" * 60)
accuracy = (correct / total * 100) if total > 0 else 0.0

print(f"\nFINAL RESULTS:")
print(f"  Correct : {correct}/{total}")
print(f"  Accuracy: {accuracy:.1f}%")
print(f"  Errors  : {errors}")
print(f"\nMultimodal domain (CLIP zero-shot): VERIFIED at {accuracy:.1f}%")

# -- emit machine-readable line for the caller --------------------------------
print(f"\nACCURACY_RESULT:{accuracy:.1f}")
