# ============================================
# AutoArchitect — Auto Training Pipeline
# Wired to real self_trainer.py
# No fake sleep loops, no random accuracy
# ============================================

import os
import time
from datetime import datetime

BASE_DIR  = os.path.dirname(os.path.dirname(__file__))
CACHE_DIR = os.path.join(BASE_DIR, 'cache')

# ============================================
# MODEL REGISTRY
# ============================================

MODEL_REGISTRY = {
    'image': {
        'keywords': [
            'pothole', 'crack', 'road', 'defect',
            'detect', 'identify', 'classify', 'spot',
            'crop', 'plant', 'disease', 'fire', 'smoke',
            'face', 'person', 'people', 'count', 'track',
            'animal', 'object', 'vehicle', 'car', 'fruit'
        ],
        'base_model':  'ResNet18',
        'model_type':  'resnet18',
        'description': 'ResNet18 transfer learning — best for visual detection'
    },
    'medical': {
        'keywords': [
            'xray', 'mri', 'scan', 'diagnosis', 'cancer',
            'tumor', 'disease', 'medical', 'health', 'ct',
            'pneumonia', 'fracture', 'retina', 'skin'
        ],
        'base_model':  'ResNet18',
        'model_type':  'resnet18',
        'description': 'ResNet18 transfer learning — best for medical imaging'
    },
    'text': {
        'keywords': [
            'sentiment', 'review', 'spam', 'fake', 'news',
            'classify', 'text', 'language', 'opinion',
            'feedback', 'comment', 'tweet', 'email'
        ],
        'base_model':  'DARTS NAS',
        'model_type':  'darts',
        'description': 'DARTS NAS — best for text classification'
    },
    'security': {
        'keywords': [
            'fraud', 'intrusion', 'malware', 'attack',
            'suspicious', 'anomaly', 'threat', 'hack',
            'phishing', 'ddos', 'unauthorized', 'breach'
        ],
        'base_model':  'DARTS NAS',
        'model_type':  'darts',
        'description': 'DARTS NAS — best for anomaly detection'
    }
}


def select_base_model(problem, category):
    problem_lower = problem.lower()
    registry      = MODEL_REGISTRY.get(category, MODEL_REGISTRY['image'])
    matches = sum(1 for kw in registry['keywords'] if kw in problem_lower)
    return {
        'base_model':  registry['base_model'],
        'model_type':  registry['model_type'],
        'description': registry['description'],
        'matches':     matches,
        'category':    category
    }


def run_yolo_detection(image_path, model_name='yolov8n.pt'):
    """Run YOLOv8 detection on an image."""
    try:
        from ultralytics import YOLO
        model   = YOLO(model_name)
        results = model(image_path, verbose=False)
        boxes   = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf  = float(box.conf[0])
                cls   = int(box.cls[0])
                label = model.names[cls]
                boxes.append({
                    'x':          int(x1),
                    'y':          int(y1),
                    'w':          int(x2 - x1),
                    'h':          int(y2 - y1),
                    'confidence': round(conf * 100, 1),
                    'label':      label
                })
        return {'status': 'success', 'boxes': boxes, 'count': len(boxes), 'model_used': model_name}
    except Exception as e:
        return {'status': 'error', 'error': str(e), 'boxes': []}


def train_new_model(problem, category, progress_callback=None, epochs=3):
    """
    Train a real model using self_trainer.py pipeline.
    No fake sleep loops. No random accuracy.
    Returns real training results.
    """
    start    = time.time()
    selected = select_base_model(problem, category)

    print(f"Auto-training for: {problem[:40]}")
    print(f"   Base model: {selected['base_model']}")
    print(f"   Type:       {selected['model_type']}")

    # Progress callbacks for UI
    if progress_callback:
        progress_callback(1, 6, "Analyzing problem requirements...")
    if progress_callback:
        progress_callback(2, 6, "Selecting optimal base model...")

    try:
        # ── Call REAL self_trainer ────────────────────────────────────
        from api.self_trainer import self_train

        if progress_callback:
            progress_callback(3, 6, "Fetching real dataset...")

        trained = self_train(
            problem  = problem,
            category = category,
            epochs   = epochs
        )

        if progress_callback:
            progress_callback(4, 6, "Neural Architecture Search...")
        if progress_callback:
            progress_callback(5, 6, "Evaluating performance...")
        if progress_callback:
            progress_callback(6, 6, "Saving to knowledge base...")

        duration = round(time.time() - start, 1)

        print(f"Real training complete!")
        print(f"   Train: {trained.get('train_accuracy', 0)}%")
        print(f"   Test:  {trained.get('test_accuracy', 0)}%")
        print(f"   Time:  {duration}s")

        return {
            'status':         'success',
            'base_model':     selected['base_model'],
            'model_type':     selected['model_type'],
            'description':    selected['description'],
            'accuracy':       trained.get('test_accuracy', 0),
            'train_accuracy': trained.get('train_accuracy', 0),
            'test_accuracy':  trained.get('test_accuracy', 0),
            'train_size':     trained.get('train_size', 0),
            'dataset':        trained.get('dataset', 'unknown'),
            'method':         trained.get('method', 'resnet18'),
            'model_path':     trained.get('model_path', ''),
            'classes':        trained.get('classes', []),
            'real_training':  True,
            'train_time':     duration,
            'problem':        problem,
            'category':       category,
            'trained_at':     datetime.now().isoformat()
        }

    except Exception as e:
        print(f"Real training failed: {e} — returning fallback")
        duration = round(time.time() - start, 1)

        # Honest fallback — don't fake accuracy
        return {
            'status':        'partial',
            'base_model':    selected['base_model'],
            'model_type':    selected['model_type'],
            'description':   selected['description'],
            'accuracy':      0,
            'train_accuracy': 0,
            'test_accuracy':  0,
            'real_training':  False,
            'error':          str(e),
            'train_time':     duration,
            'problem':        problem,
            'category':       category,
            'trained_at':     datetime.now().isoformat(),
            'message':        'Training failed — upload your own data for real accuracy'
        }
