"""
test_downloaded_agent.py — Standalone inference tester for downloaded agent zips.

Usage:
    python api/test_downloaded_agent.py                     # auto-discovers latest zip
    python api/test_downloaded_agent.py agent.zip           # explicit zip
    python api/test_downloaded_agent.py agent.zip image.jpg # explicit zip + image
"""

import sys
import os
import json
import zipfile
import tempfile
import glob
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image


TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def find_latest_zip() -> str:
    patterns = ["*.zip", "downloads/*.zip", "agents/*.zip", "../*.zip"]
    zips = []
    for pat in patterns:
        zips.extend(glob.glob(pat))
    if not zips:
        return None
    return max(zips, key=os.path.getmtime)


def build_resnet(num_classes: int, device: torch.device) -> nn.Module:
    from torchvision.models import resnet18
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device)


def load_agent(zip_path: str):
    """Unzip, load .pth model and metadata. Returns (model, classes, device)."""
    tmp = tempfile.mkdtemp(prefix="agent_test_")
    print(f"Extracting {zip_path} -> {tmp}")

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(tmp)

    pth_files = list(Path(tmp).rglob("*.pth"))
    if not pth_files:
        raise FileNotFoundError(f"No .pth file found inside {zip_path}")
    pth_path = pth_files[0]
    print(f"Model file  : {pth_path.name}")

    classes = None
    meta_files = list(Path(tmp).rglob("metadata.json"))
    if meta_files:
        with open(meta_files[0]) as f:
            meta = json.load(f)
        classes = meta.get("classes") or None
        if classes:
            print(f"Classes     : {classes}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(pth_path, map_location=device)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
        n_cls = checkpoint.get("num_classes",
                               len(classes) if classes else 2)
        model = build_resnet(n_cls, device)
        model.load_state_dict(state)
    elif isinstance(checkpoint, dict) and any(
            k.startswith("fc.") for k in checkpoint):
        fc_w = checkpoint.get("fc.weight")
        n_cls = fc_w.shape[0] if fc_w is not None else (
            len(classes) if classes else 2)
        model = build_resnet(n_cls, device)
        model.load_state_dict(checkpoint)
    else:
        model = checkpoint
        model = model.to(device)

    model.eval()
    print(f"Device      : {device}")
    return model, classes, device


def make_sample_image() -> Image.Image:
    """Solid grey 224x224 synthetic image used when no real image is provided."""
    return Image.new("RGB", (224, 224), color=(128, 128, 128))


def run_inference(model: nn.Module, image: Image.Image,
                  classes: list, device: torch.device):
    tensor = TRANSFORM(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]

    top_idx  = probs.argmax().item()
    top_prob = probs[top_idx].item()
    label    = (classes[top_idx]
                if classes and top_idx < len(classes)
                else str(top_idx))

    print("\n── Inference result ─────────────────────────")
    print(f"  Prediction : {label}")
    print(f"  Confidence : {top_prob:.1%}")
    print("  All scores :")
    for i, p in enumerate(probs.tolist()):
        cls = classes[i] if classes and i < len(classes) else str(i)
        bar = "█" * int(p * 30)
        print(f"    [{i}] {cls:<22} {p:.4f}  {bar}")
    print("─────────────────────────────────────────────")
    return label, top_prob


def main():
    args = sys.argv[1:]

    if args:
        zip_path = args[0]
    else:
        zip_path = find_latest_zip()
        if not zip_path:
            print("Usage: python api/test_downloaded_agent.py <agent.zip> [image.jpg]")
            print("No .zip file found in the current directory.")
            sys.exit(1)
        print(f"Auto-discovered: {zip_path}")

    if not os.path.exists(zip_path):
        print(f"Error: {zip_path} not found")
        sys.exit(1)

    if len(args) >= 2:
        img_path = args[1]
        if not os.path.exists(img_path):
            print(f"Error: image {img_path} not found")
            sys.exit(1)
        image = Image.open(img_path).convert("RGB")
        print(f"Image       : {img_path}")
    else:
        print("No image provided — using synthetic grey sample")
        image = make_sample_image()

    model, classes, device = load_agent(zip_path)
    run_inference(model, image, classes, device)


if __name__ == "__main__":
    main()
