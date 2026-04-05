"""
AutoArchitect — Self Training Agent
Trains real models and saves them to a known location
so network_zip_generator can package them correctly.

Model save path: models/trained/{problem_hash}_{domain}.pth
Classes saved:   models/trained/{problem_hash}_{domain}_classes.json
"""

import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
import hashlib
import json
import re
from datetime import datetime

from api.nas_engine import DARTSNet

BASE_DIR      = os.path.dirname(os.path.dirname(__file__))
CACHE_DIR     = os.path.join(BASE_DIR, 'cache')
TRAINED_DIR   = os.path.join(BASE_DIR, 'models', 'trained')
os.makedirs(TRAINED_DIR, exist_ok=True)

# ── Domain correction map ─────────────────────────────────────────────────
DOMAIN_OVERRIDES = {
    # Infrastructure / pipes → image
    "pipe":            "image",
    "pipeline":        "image",
    "sewage":          "image",
    "sewer":           "image",
    "blockage":        "image",
    "leak":            "image",
    "corrosion":       "image",
    "crack":           "image",
    "pothole":         "image",
    "road":            "image",
    "pavement":        "image",
    "bridge":          "image",
    "building":        "image",
    "construction":    "image",
    "drone":           "image",
    "satellite":       "image",
    "camera":          "image",
    # Water / environment → image
    "water quality":   "image",
    "water pollution": "image",
    "oil spill":       "image",
    "algae bloom":     "image",
    "toxic algae":     "image",
    "bay water":       "image",
    "ocean":           "image",
    "river":           "image",
    "lake":            "image",
    "contamination":   "image",
    "discharge":       "image",
    # Medical → medical
    "xray":            "medical",
    "x-ray":           "medical",
    "mri":             "medical",
    "tumor":           "medical",
    "cancer":          "medical",
    "pneumonia":       "medical",
    "diagnosis":       "medical",
    "pathology":       "medical",
    "radiology":       "medical",
    # Text
    "sentiment":       "text",
    "spam":            "text",
    "fake news":       "text",
    "toxic comment":   "text",
    "toxic review":    "text",
    "hate speech":     "text",
    "nlp":             "text",
    # Security
    "intrusion":       "security",
    "malware":         "security",
    "fraud":           "security",
    "phishing":        "security",
    "vulnerability":   "security",
    "cyber":           "security",
}

# Track datasets used in current run to avoid collisions
_used_datasets_this_run = set()


def _correct_domain(problem: str, bert_domain: str) -> str:
    p = problem.lower()
    for keyword, correct_domain in DOMAIN_OVERRIDES.items():
        if keyword in p:
            if correct_domain != bert_domain:
                print(f"   🔧 Domain corrected: {bert_domain} → "
                      f"{correct_domain} (keyword: '{keyword}')")
            return correct_domain
    return bert_domain


def _problem_hash(problem: str) -> str:
    cleaned = re.sub(r'[^\w\s]', '', problem)
    normalized = ' '.join(cleaned.lower().split())
    return hashlib.md5(normalized.encode()).hexdigest()[:10]


def _fetch_dataset_smart(problem: str, category: str,
                          subset_size: int = 2000) -> dict:
    global _used_datasets_this_run
    groq_key = os.getenv("GROQ_API_KEY", "")

    # 1. Discovery engine
    try:
        from api.brain.data_discovery_engine import DataDiscoveryEngine
        engine = DataDiscoveryEngine(groq_api_key=groq_key)
        result = engine.find(problem, category, subset_size)
        if result and result.get("name") != "clip_zero_shot":
            if result["name"] not in _used_datasets_this_run:
                _used_datasets_this_run.add(result["name"])
                print(f"   ✅ Discovery engine found: {result['name']}")
                return result
            else:
                print(f"   ⚠️  Dataset {result['name']} already used — searching alternative")
    except Exception as e:
        print(f"   ⚠️  Discovery engine error: {e}")

    # 2. Registry fallback
    try:
        from api.dataset_fetcher import fetch_dataset
        result = fetch_dataset(problem, category, subset_size)
        if result and result.get("real_dataset"):
            if result["name"] not in _used_datasets_this_run:
                _used_datasets_this_run.add(result["name"])
                print(f"   ✅ Registry found: {result['name']}")
                return result
    except Exception as e:
        print(f"   ⚠️  Registry error: {e}")

    # 3. Honest last resort
    print(f"   ⚠️  No real dataset found")
    print(f"   💡 Upload your own data for 85%+ accuracy")
    return None


class SelfTrainingAgent:

    def __init__(self):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🤖 SelfTrainingAgent ready on {self.device}")

    def train(self, problem, category,
              epochs=3, progress_callback=None):

        start   = time.time()
        results = {
            'problem':  problem,
            'category': category,
            'status':   'training',
            'steps':    []
        }

        # Step 1
        self._update(results, progress_callback, 1, 6,
            "🔍 Analyzing problem requirements...")
        print(f"   Problem:  {problem[:40]}")
        print(f"   Category: {category}")

        # Fix domain
        category = _correct_domain(problem, category)
        results['category'] = category
        print(f"   Domain:   {category}")

        # Step 2 — fetch dataset
        self._update(results, progress_callback, 2, 6,
            f"📦 Fetching best dataset for: {problem[:30]}...")

        data = _fetch_dataset_smart(problem, category, subset_size=2000)

        if data is None or data.get("train_loader") is None:
            print(f"   ⚠️  No training data available")
            results.update({
                'dataset':        'none',
                'train_size':     0,
                'test_size':      0,
                'classes':        [],
                'real_dataset':   False,
                'train_accuracy': 0,
                'test_accuracy':  0,
                'parameters':     0,
                'method':         'no_data',
                'time':           round(time.time() - start, 1),
                'status':         'no_data',
                'trained_at':     datetime.now().isoformat(),
                'model_path':     None,
                'classes_path':   None,
            })
            return results

        results['dataset']      = data['name']
        results['train_size']   = data['train_size']
        results['test_size']    = data['test_size']
        results['classes']      = data['classes']
        results['real_dataset'] = data.get('real_dataset', False)

        # Step 3 — architecture
        self._update(results, progress_callback, 3, 6,
            "🧠 Running Neural Architecture Search...")

        num_classes  = data['num_classes']
        use_transfer = category in ('image', 'medical')

        if use_transfer:
            print(f"   🔥 Image/Medical → ResNet18 transfer learning")
            params = 11_177_538
        else:
            model  = DARTSNet(C=16, num_cells=3,
                              num_classes=num_classes).to(self.device)
            params = sum(p.numel() for p in model.parameters())

        results['parameters'] = params
        print(f"   Architecture: {params:,} parameters")

        # Step 4 — train
        self._update(results, progress_callback, 4, 6,
            f"⚡ Training on {data['train_size']} samples...")

        if use_transfer:
            from api.transfer_trainer import train_transfer
            tr = train_transfer(
                problem, data,
                epochs=epochs,
                progress_callback=progress_callback,
                device=self.device
            )
            results['train_accuracy']    = tr['train_accuracy']
            results['test_accuracy']     = tr['test_accuracy']
            results['epoch_history']     = tr['epoch_history']
            results['method']            = tr['method']
            results['expected_accuracy'] = data.get('expected_accuracy', 75)

            # Save model to known location
            self._update(results, progress_callback, 6, 6,
                "💾 Saving to knowledge base...")
            model_path, classes_path = self._save_trained_model(
                tr['model'], problem, category, data['classes'], results)
            results['model_path']   = model_path
            results['classes_path'] = classes_path
            self._save_cache(tr['model'], problem, results, data)

            duration              = round(time.time() - start, 1)
            results['time']       = duration
            results['status']     = 'complete'
            results['trained_at'] = datetime.now().isoformat()

            print(f"\n✅ Self-training complete!")
            print(f"   Dataset: {data['name']} "
                  f"({'REAL' if data.get('real_dataset') else 'synthetic'})")
            print(f"   Method:  ResNet18 Transfer Learning")
            print(f"   Train:   {results['train_accuracy']}%")
            print(f"   Test:    {results['test_accuracy']}%")
            print(f"   Model:   {model_path}")
            return results

        # NAS training (text/security)
        net_opt  = optim.Adam(
            [p for n, p in model.named_parameters()
             if 'arch_weights' not in n], lr=0.001)
        arch_opt = optim.Adam(
            [p for n, p in model.named_parameters()
             if 'arch_weights' in n], lr=0.01)
        criterion = nn.CrossEntropyLoss()

        epoch_results = []
        for epoch in range(epochs):
            model.train()
            correct = total = 0
            for images, labels in data['train_loader']:
                if images.shape[1] == 1:
                    images = images.repeat(1, 3, 1, 1)
                images = images.to(self.device)
                labels = labels.to(self.device)

                net_opt.zero_grad()
                out  = model(images)
                loss = criterion(out, labels)
                loss.backward()
                net_opt.step()

                arch_opt.zero_grad()
                out  = model(images)
                loss = criterion(out, labels)
                loss.backward()
                arch_opt.step()

                preds     = out.argmax(dim=1)
                correct  += (preds == labels).sum().item()
                total    += labels.size(0)

            acc = round(100 * correct / total, 2)
            epoch_results.append(acc)
            print(f"   Epoch {epoch+1}/{epochs} → Accuracy: {acc}%")
            if progress_callback:
                progress_callback(4, 6,
                    f"Training epoch {epoch+1}/{epochs} — {acc}%")

        results['train_accuracy'] = epoch_results[-1]
        results['epoch_history']  = epoch_results

        # Step 5 — evaluate
        self._update(results, progress_callback, 5, 6,
            "📊 Evaluating on test data...")
        test_acc = self._evaluate(model, data['test_loader'])
        results['test_accuracy']     = test_acc
        results['expected_accuracy'] = data.get('expected_accuracy', 50)
        print(f"   Test Accuracy: {test_acc}%")

        # Step 6 — save
        self._update(results, progress_callback, 6, 6,
            "💾 Saving to knowledge base...")
        model_path, classes_path = self._save_trained_model(
            model, problem, category, data['classes'], results)
        results['model_path']   = model_path
        results['classes_path'] = classes_path
        self._save_cache(model, problem, results, data)

        duration              = round(time.time() - start, 1)
        results['time']       = duration
        results['status']     = 'complete'
        results['trained_at'] = datetime.now().isoformat()

        print(f"\n✅ Self-training complete!")
        print(f"   Dataset: {data['name']} "
              f"({'REAL' if data.get('real_dataset') else 'synthetic'})")
        print(f"   Train:   {results['train_accuracy']}%")
        print(f"   Test:    {test_acc}%")
        print(f"   Model:   {model_path}")
        return results

    def _save_trained_model(self, model, problem: str,
                             domain: str, classes: list,
                             results: dict):
        """
        Save trained model to known location.
        network_zip_generator uses this exact path.
        """
        h          = _problem_hash(problem)
        model_path = os.path.join(TRAINED_DIR, f"{h}_{domain}.pth")
        cls_path   = os.path.join(TRAINED_DIR, f"{h}_{domain}_classes.json")

        # Save model weights
        torch.save(model.state_dict(), model_path)

        # Save classes + metadata
        meta = {
            "problem":        problem,
            "domain":         domain,
            "classes":        classes,
            "num_classes":    len(classes),
            "train_accuracy": results.get('train_accuracy', 0),
            "test_accuracy":  results.get('test_accuracy', 0),
            "method":         results.get('method', 'darts_nas'),
            "dataset":        results.get('dataset', 'unknown'),
            "trained_at":     datetime.now().isoformat(),
        }
        with open(cls_path, 'w') as f:
            json.dump(meta, f, indent=2)

        print(f"   💾 Model saved: {model_path}")
        return model_path, cls_path

    def _save_cache(self, model, problem, results, data):
        """Also save to old cache location for backwards compatibility."""
        import re
        cleaned    = re.sub(r'[^\w\s]', '', problem)
        normalized = ' '.join(cleaned.lower().split())
        h          = hashlib.md5(normalized.encode()).hexdigest()[:10]
        cache_path = os.path.join(CACHE_DIR, h)
        os.makedirs(cache_path, exist_ok=True)

        torch.save(model.state_dict(),
                   os.path.join(cache_path, 'model.pth'))

        metadata = {
            'problem':           problem,
            'category':          results.get('category'),
            'dataset':           results.get('dataset'),
            'real_dataset':      results.get('real_dataset', False),
            'train_accuracy':    results.get('train_accuracy'),
            'test_accuracy':     results.get('test_accuracy'),
            'expected_accuracy': results.get('expected_accuracy', 75),
            'parameters':        results.get('parameters'),
            'train_size':        results.get('train_size'),
            'classes':           data.get('classes'),
            'num_classes':       data.get('num_classes'),
            'method':            results.get('method', 'darts_nas'),
            'time':              results.get('time'),
            'trained_at':        results.get('trained_at'),
            'model_path':        results.get('model_path'),
            'classes_path':      results.get('classes_path'),
            'use_count':         1,
            'self_trained':      True,
        }
        with open(os.path.join(cache_path, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"   Saved to cache: {h}")
        results['cache_model_path'] = os.path.join(cache_path, 'model.pth')

    def _evaluate(self, model, test_loader):
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                if images.shape[1] == 1:
                    images = images.repeat(1, 3, 1, 1)
                images  = images.to(self.device)
                labels  = labels.to(self.device)
                preds   = model(images).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total   += labels.size(0)
        return round(100 * correct / total, 2)

    def _update(self, results, callback, step, total, message):
        results['steps'].append(message)
        print(f"\n[{step}/{total}] {message}")
        if callback:
            callback(step, total, message)


# Global instance
agent = SelfTrainingAgent()


def self_train(problem, category,
               epochs=3, progress_callback=None):
    global _used_datasets_this_run
    _used_datasets_this_run = set()  # reset for each new problem
    return agent.train(problem, category, epochs, progress_callback)
