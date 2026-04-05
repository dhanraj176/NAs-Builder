# ============================================
# AutoArchitect — Dataset Manager
# Auto-selects and downloads datasets
# All FREE, no API keys needed!
# ============================================

import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'datasets')

# ============================================
# DATASET REGISTRY
# Maps problem keywords to free datasets
# ============================================

DATASET_REGISTRY = {
    # IMAGE problems → CIFAR-10
    'pothole':   'cifar10',
    'crack':     'cifar10',
    'road':      'cifar10',
    'defect':    'cifar10',
    'fire':      'cifar10',
    'smoke':     'cifar10',
    'plant':     'cifar10',
    'crop':      'cifar10',
    'disease':   'cifar10',
    'animal':    'cifar10',
    'vehicle':   'cifar10',
    'person':    'cifar10',
    'face':      'cifar10',
    'object':    'cifar10',
    'image':     'cifar10',
    'photo':     'cifar10',
    'detect':    'cifar10',

    # TEXT problems → MNIST
    'sentiment': 'mnist',
    'spam':      'mnist',
    'fake':      'mnist',
    'review':    'mnist',
    'opinion':   'mnist',
    'text':      'mnist',
    'language':  'mnist',
    'classify':  'mnist',
    'news':      'mnist',

    # MEDICAL problems → FashionMNIST
    'xray':      'fashionmnist',
    'mri':       'fashionmnist',
    'cancer':    'fashionmnist',
    'tumor':     'fashionmnist',
    'medical':   'fashionmnist',
    'diagnosis': 'fashionmnist',
    'pneumonia': 'fashionmnist',
    'health':    'fashionmnist',
    'scan':      'fashionmnist',
    'clinical':  'fashionmnist',

    # SECURITY problems → CIFAR-10
    'fraud':     'cifar10',
    'intrusion': 'cifar10',
    'malware':   'cifar10',
    'attack':    'cifar10',
    'security':  'cifar10',
    'anomaly':   'cifar10',
    'threat':    'cifar10',
    'hack':      'cifar10',
}

def select_dataset(problem, category):
    """Auto-select best dataset for problem"""
    problem_lower = problem.lower()

    # Check keyword matches
    for keyword, dataset in DATASET_REGISTRY.items():
        if keyword in problem_lower:
            print(f"   Dataset selected: {dataset} "
                  f"(matched: '{keyword}')")
            return {
                'name':        dataset,
                'reason':      f"Matched keyword: '{keyword}'",
                'num_classes': get_num_classes(dataset)
            }

    # Default by category
    defaults = {
        'image':    'cifar10',
        'medical':  'fashionmnist',
        'text':     'mnist',
        'security': 'cifar10'
    }
    dataset = defaults.get(category, 'cifar10')
    print(f"   Dataset selected: {dataset} "
          f"(default for {category})")
    return {
        'name':        dataset,
        'reason':      f'Default for {category} problems',
        'num_classes': get_num_classes(dataset)
    }

def get_num_classes(dataset_name):
    """Get number of classes for dataset"""
    return {
        'cifar10':      10,
        'mnist':        10,
        'fashionmnist': 10,
    }.get(dataset_name, 10)

def get_class_names(dataset_name):
    """Get class names for dataset"""
    names = {
        'cifar10': [
            'airplane', 'automobile', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck'
        ],
        'mnist': [str(i) for i in range(10)],
        'fashionmnist': [
            'T-shirt', 'Trouser', 'Pullover', 'Dress',
            'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Boot'
        ]
    }
    return names.get(dataset_name, [str(i) for i in range(10)])

def download_dataset(dataset_name, subset_size=3000):
    """
    Download dataset automatically — FREE, no API key!
    Uses small subset for fast training demo
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"📦 Downloading dataset: {dataset_name}...")

    transform_gray = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform_rgb = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    if dataset_name == 'cifar10':
        train = torchvision.datasets.CIFAR10(
            root=DATA_DIR, train=True,
            download=True, transform=transform_rgb)
        test  = torchvision.datasets.CIFAR10(
            root=DATA_DIR, train=False,
            download=True, transform=transform_rgb)

    elif dataset_name == 'mnist':
        train = torchvision.datasets.MNIST(
            root=DATA_DIR, train=True,
            download=True, transform=transform_gray)
        test  = torchvision.datasets.MNIST(
            root=DATA_DIR, train=False,
            download=True, transform=transform_gray)

    elif dataset_name == 'fashionmnist':
        train = torchvision.datasets.FashionMNIST(
            root=DATA_DIR, train=True,
            download=True, transform=transform_gray)
        test  = torchvision.datasets.FashionMNIST(
            root=DATA_DIR, train=False,
            download=True, transform=transform_gray)
    else:
        return download_dataset('cifar10', subset_size)

    # Use subset for fast training
    train_sub = Subset(train, list(range(
        min(subset_size, len(train)))))
    test_sub  = Subset(test,  list(range(
        min(500, len(test)))))

    train_loader = DataLoader(
        train_sub, batch_size=64,
        shuffle=True,  num_workers=0)
    test_loader  = DataLoader(
        test_sub,  batch_size=64,
        shuffle=False, num_workers=0)

    print(f"✅ Dataset ready: {len(train_sub)} train, "
          f"{len(test_sub)} test")

    return {
        'name':         dataset_name,
        'train_loader': train_loader,
        'test_loader':  test_loader,
        'num_classes':  get_num_classes(dataset_name),
        'classes':      get_class_names(dataset_name),
        'train_size':   len(train_sub),
        'test_size':    len(test_sub)
    }
