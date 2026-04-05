# ============================================
# AutoArchitect — NAS Engine (with real model)
# ============================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os

# Available operations
OPS = {
    'skip':    lambda C: nn.Identity(),
    'conv3x3': lambda C: nn.Sequential(
                   nn.Conv2d(C, C, 3, padding=1, bias=False),
                   nn.BatchNorm2d(C), nn.ReLU()),
    'conv5x5': lambda C: nn.Sequential(
                   nn.Conv2d(C, C, 5, padding=2, bias=False),
                   nn.BatchNorm2d(C), nn.ReLU()),
    'maxpool': lambda C: nn.MaxPool2d(3, stride=1, padding=1),
    'avgpool': lambda C: nn.AvgPool2d(3, stride=1, padding=1),
}

class MixedOp(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.ops          = nn.ModuleList([OPS[n](C) for n in OPS])
        self.arch_weights = nn.Parameter(
            torch.ones(len(OPS)) / len(OPS))

    def forward(self, x):
        w = F.softmax(self.arch_weights, dim=0)
        return sum(wi * op(x) for wi, op in zip(w, self.ops))

class DARTSCell(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.ops = nn.ModuleList([MixedOp(C) for _ in range(4)])

    def forward(self, x):
        for op in self.ops:
            x = op(x)
        return x

class DARTSNet(nn.Module):
    def __init__(self, C=16, num_cells=3, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, C, 3, padding=1, bias=False),
            nn.BatchNorm2d(C), nn.ReLU()
        )
        self.cells      = nn.ModuleList(
            [DARTSCell(C) for _ in range(num_cells)])
        self.gap        = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C, num_classes)

    def forward(self, x):
        x = self.stem(x)
        for cell in self.cells:
            x = cell(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

def get_architecture(model):
    """Extract what operations AI chose"""
    arch     = []
    op_names = list(OPS.keys())
    for i, cell in enumerate(model.cells):
        cell_ops = []
        for op in cell.ops:
            w    = F.softmax(op.arch_weights, dim=0)
            best = op_names[w.argmax().item()]
            cell_ops.append({
                'operation':  best,
                'confidence': round(w.max().item() * 100, 1),
                'weights':    {n: round(wi.item(), 3)
                              for n, wi in zip(op_names, w)}
            })
        arch.append({'cell': i+1, 'operations': cell_ops})
    return arch

def load_trained_model():
    """Load the real trained model from Colab"""
    model_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'models', 'nas_model.pth'
    )
    model = DARTSNet(C=16, num_cells=3, num_classes=10)
    if os.path.exists(model_path):
        model.load_state_dict(
            torch.load(model_path,
                      map_location=torch.device('cpu'),
                      weights_only=True)
        )
        print(f"✅ Loaded real trained model!")
    else:
        print(f"⚠️ No trained model found, using random weights")
    return model

def run_quick_nas(num_classes=10, progress_callback=None):
    """Load real model + run quick search for demo"""
    device = torch.device('cpu')

    # Try loading real trained model first
    model = load_trained_model().to(device)

    # Run quick additional search
    net_opt  = torch.optim.Adam(
        [p for n, p in model.named_parameters()
         if 'arch_weights' not in n], lr=0.001)
    arch_opt = torch.optim.Adam(
        [p for n, p in model.named_parameters()
         if 'arch_weights' in n], lr=0.01)
    criterion = nn.CrossEntropyLoss()

    start  = time.time()
    epochs = 3

    for epoch in range(epochs):
        for _ in range(10):
            x = torch.randn(8, 3, 32, 32)
            y = torch.randint(0, num_classes, (8,))

            net_opt.zero_grad()
            out  = model(x)
            loss = criterion(out, y)
            loss.backward()
            net_opt.step()

            arch_opt.zero_grad()
            out  = model(x)
            loss = criterion(out, y)
            loss.backward()
            arch_opt.step()

        if progress_callback:
            progress_callback(epoch+1, epochs)

    duration = round(time.time() - start, 1)
    arch     = get_architecture(model)
    params   = sum(p.numel() for p in model.parameters())

    return {
        'architecture': arch,
        'parameters':   params,
        'search_time':  duration,
        'status':       'success',
        'model_source': 'trained' if os.path.exists(
            os.path.join(os.path.dirname(
                os.path.dirname(__file__)),
                'models', 'nas_model.pth')) else 'demo'
    }
