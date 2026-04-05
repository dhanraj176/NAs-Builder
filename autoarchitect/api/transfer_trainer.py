# ============================================
# AutoArchitect — Transfer Learning Trainer
# Uses pretrained ResNet18 for image problems
# Replaces training from scratch → 20% accuracy
# With transfer learning → 70-90% accuracy
# ============================================

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
import time
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))


def build_transfer_model(num_classes: int, device):
    """
    ResNet18 pretrained on ImageNet.
    Replace final layer for our num_classes.
    Freeze all layers except final → fast fine-tuning.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # Replace final layer — only this trains
    in_features       = model.fc.in_features
    model.fc          = nn.Linear(in_features, num_classes)

    # Unfreeze last block too for better accuracy
    for param in model.layer4.parameters():
        param.requires_grad = True

    return model.to(device)


def get_transform():
    """ResNet18 expects 224x224 normalized images"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std= [0.229, 0.224, 0.225]
        )
    ])


def train_transfer(problem: str, data: dict,
                   epochs: int = 3,
                   progress_callback=None,
                   device=None) -> dict:
    """
    Full transfer learning pipeline.
    Accepts same data dict format as self_trainer.
    Returns same result format as self_trainer.

    Much faster than training from scratch because:
    - ResNet18 already knows image features
    - Only fine-tuning last layer + layer4
    - 3 epochs is enough for good accuracy
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    start      = time.time()
    num_classes = data['num_classes']

    print(f"   🔥 Transfer learning: ResNet18 → {num_classes} classes")
    print(f"   📦 Dataset: {data['name']} ({data['train_size']} samples)")

    # Build pretrained model
    model     = build_transfer_model(num_classes, device)
    params    = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   🧠 ResNet18: {params:,} total, {trainable:,} trainable")

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.001
    )
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)

    epoch_results = []

    for epoch in range(epochs):
        model.train()
        correct = total = 0

        for images, labels in data['train_loader']:
            # Resize to 224x224 for ResNet18 if needed
            if images.shape[-1] != 224:
                images = torch.nn.functional.interpolate(
                    images, size=(224, 224), mode='bilinear',
                    align_corners=False)

            # Handle grayscale → RGB
            if images.shape[1] == 1:
                images = images.repeat(1, 3, 1, 1)

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            out   = model(images)
            loss  = criterion(out, labels)
            loss.backward()
            optimizer.step()

            preds    = out.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

        acc = round(100 * correct / total, 2)
        epoch_results.append(acc)
        print(f"   Epoch {epoch+1}/{epochs} → Accuracy: {acc}%")
        scheduler.step()

        if progress_callback:
            progress_callback(4, 6,
                f"Transfer learning epoch {epoch+1}/{epochs} — {acc}%")

    # Evaluate
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in data['test_loader']:
            if images.shape[-1] != 224:
                images = torch.nn.functional.interpolate(
                    images, size=(224, 224), mode='bilinear',
                    align_corners=False)
            if images.shape[1] == 1:
                images = images.repeat(1, 3, 1, 1)
            images  = images.to(device)
            labels  = labels.to(device)
            preds   = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

    test_acc = round(100 * correct / total, 2)
    duration = round(time.time() - start, 1)

    print(f"   ✅ Transfer learning complete!")
    print(f"   Train accuracy: {epoch_results[-1]}%")
    print(f"   Test accuracy:  {test_acc}%  ← ResNet18 boost!")
    print(f"   Time: {duration}s")

    return {
        "train_accuracy":  epoch_results[-1],
        "test_accuracy":   test_acc,
        "epoch_history":   epoch_results,
        "parameters":      params,
        "model":           model,
        "time":            duration,
        "method":          "transfer_learning_resnet18",
    }
