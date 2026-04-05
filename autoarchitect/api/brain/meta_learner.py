# ============================================
# AutoArchitect — Meta-Learning Brain
# An AI that learns how to build AI
#
# Research Contribution:
# "We treat pipeline selection as a supervised
#  learning problem where each solved problem
#  becomes a training example for predicting
#  optimal agent configurations on future problems"
#
# How it works:
# 1. Every solved problem → extract BERT embedding
# 2. Store (embedding, outcome) as training example
# 3. Train small neural net on all past examples
# 4. Next problem → predict best pipeline BEFORE running
# 5. Gets smarter with every single problem solved
# ============================================

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datetime import datetime

BRAIN_DIR    = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    'brain_data'
)
META_FILE    = os.path.join(BRAIN_DIR, 'meta_examples.json')
MODEL_FILE   = os.path.join(BRAIN_DIR, 'meta_model.pth')
INSIGHT_FILE = os.path.join(BRAIN_DIR, 'meta_insights.json')

# All possible agent combinations
AGENT_COMBOS = [
    ["image"],
    ["text"],
    ["medical"],
    ["security"],
    ["image", "text"],
    ["image", "medical"],
    ["text", "security"],
    ["medical", "text"],
    ["image", "security"],
    ["image", "text", "security"],
]

# All datasets the system can select
DATASETS = [
    "sms_spam",
    "imdb",
    "GonzaloA/fake_news",
    "synthetic_fraud",
    "synthetic_intrusion",
    "cifar10",
    "fashionmnist",
    "user_data",
]

# Training methods
METHODS = [
    "transfer_learning_resnet18",
    "darts_nas",
]


# ============================================
# META-LEARNER NEURAL NETWORK
# 768 → 256 → 128 → 64 → outputs
# ============================================

class MetaNet(nn.Module):
    def __init__(self,
                 input_dim:  int = 768,
                 hidden_dim: int = 256,
                 n_combos:   int = len(AGENT_COMBOS),
                 n_datasets: int = len(DATASETS),
                 n_methods:  int = len(METHODS)):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        self.head_agents   = nn.Linear(64, n_combos)
        self.head_dataset  = nn.Linear(64, n_datasets)
        self.head_method   = nn.Linear(64, n_methods)
        self.head_accuracy = nn.Linear(64, 1)

    def forward(self, x):
        features = self.encoder(x)
        return {
            "agents":   self.head_agents(features),
            "dataset":  self.head_dataset(features),
            "method":   self.head_method(features),
            "accuracy": self.head_accuracy(features).squeeze(-1),
        }


# ============================================
# META-LEARNER — main class
# ============================================

class MetaLearner:
    MIN_EXAMPLES_TO_TRAIN = 5
    RETRAIN_EVERY         = 3

    def __init__(self):
        os.makedirs(BRAIN_DIR, exist_ok=True)
        self.examples = self._load_examples()
        self.model    = MetaNet()
        self.trained  = False
        self.device   = torch.device('cpu')

        # Try loading existing trained model
        if os.path.exists(MODEL_FILE) and \
           len(self.examples) >= self.MIN_EXAMPLES_TO_TRAIN:
            try:
                self.model.load_state_dict(
                    torch.load(MODEL_FILE,
                               map_location='cpu',
                               weights_only=True))
                self.trained = True
                print(f"🧠 Meta-learner loaded! "
                      f"Trained on {len(self.examples)} examples")
            except Exception:
                print(f"🧠 Meta-learner: retraining needed")

        # Force train if we have enough examples but no saved model
        if len(self.examples) >= self.MIN_EXAMPLES_TO_TRAIN \
                and not self.trained:
            print(f"  🔄 Force training on "
                  f"{len(self.examples)} existing examples...")
            self._train()

        print(f"🧠 Meta-learner ready — "
              f"{len(self.examples)} training examples")

    # ── PREDICT ──────────────────────────────────────────────
    def predict(self, problem: str,
                bert_embedding: list = None) -> dict:
        if not self.trained or \
           len(self.examples) < self.MIN_EXAMPLES_TO_TRAIN:
            return {
                "predicted":  False,
                "reason":     f"Need {self.MIN_EXAMPLES_TO_TRAIN} "
                              f"examples (have {len(self.examples)})",
                "confidence": 0.0,
            }

        embedding = bert_embedding or self._get_embedding(problem)
        if not embedding:
            return {"predicted": False,
                    "reason": "embedding failed",
                    "confidence": 0.0}

        self.model.eval()
        with torch.no_grad():
            x      = torch.tensor(embedding,
                                  dtype=torch.float32).unsqueeze(0)
            output = self.model(x)

            combo_idx   = output["agents"].argmax(dim=1).item()
            dataset_idx = output["dataset"].argmax(dim=1).item()
            method_idx  = output["method"].argmax(dim=1).item()
            acc_pred    = round(output["accuracy"].item(), 1)

            combo_conf = torch.softmax(
                output["agents"],  dim=1).max().item()
            ds_conf    = torch.softmax(
                output["dataset"], dim=1).max().item()
            m_conf     = torch.softmax(
                output["method"],  dim=1).max().item()
            confidence = round(
                (combo_conf + ds_conf + m_conf) / 3, 3)

        predicted_agents   = AGENT_COMBOS[combo_idx]
        predicted_dataset  = DATASETS[dataset_idx]
        predicted_method   = METHODS[method_idx]
        predicted_accuracy = max(0, min(100, acc_pred))

        print(f"  🔮 Meta-learner prediction:")
        print(f"     Agents:     {predicted_agents}")
        print(f"     Dataset:    {predicted_dataset}")
        print(f"     Method:     {predicted_method}")
        print(f"     Expected:   ~{predicted_accuracy}%")
        print(f"     Confidence: {confidence:.1%}")

        return {
            "predicted":  True,
            "agents":     predicted_agents,
            "dataset":    predicted_dataset,
            "method":     predicted_method,
            "accuracy":   predicted_accuracy,
            "confidence": confidence,
            "trained_on": len(self.examples),
        }

    # ── LEARN ────────────────────────────────────────────────
    def learn(self, problem: str,
              agents_used:     list,
              dataset_used:    str,
              method_used:     str,
              actual_accuracy: float,
              bert_embedding:  list = None):

        embedding = bert_embedding or self._get_embedding(problem)
        if not embedding:
            print("  ⚠️ Meta-learner: no embedding, skipping")
            return
        
        if actual_accuracy == 0.0:
            print(f"  ⚠️  Skipping meta-learner — no dataset found, not storing 0% result")
            return

        example = {
            "problem":       problem,
            "embedding":     embedding,
            "agents":        agents_used,
            "dataset":       dataset_used,
            "method":        method_used,
            "accuracy":      actual_accuracy,
            "combo_label":   self._encode_combo(agents_used),
            "dataset_label": self._encode_dataset(dataset_used),
            "method_label":  self._encode_method(method_used),
            "learned_at":    datetime.now().isoformat(),
        }

        self.examples.append(example)
        self._save_examples()

        print(f"  🧠 Meta-learner: stored example #{len(self.examples)}")
        print(f"     Problem: {problem[:40]}")
        print(f"     Result:  {agents_used} → {actual_accuracy}%")

        # Retrain every N examples once we have enough
        n = len(self.examples)
        if n >= self.MIN_EXAMPLES_TO_TRAIN and \
                n % self.RETRAIN_EVERY == 0:
            print(f"  🔄 Retraining meta-model on {n} examples...")
            self._train()
        elif n >= self.MIN_EXAMPLES_TO_TRAIN and not self.trained:
            print(f"  🔄 First training triggered on {n} examples...")
            self._train()

        self._save_insights()

    # ── TRAIN ────────────────────────────────────────────────
    def _train(self, epochs: int = 50):
        if len(self.examples) < self.MIN_EXAMPLES_TO_TRAIN:
            return

        self.model.train()
        optimizer = optim.Adam(self.model.parameters(),
                               lr=0.001, weight_decay=1e-4)
        ce_loss  = nn.CrossEntropyLoss()
        mse_loss = nn.MSELoss()

        X       = torch.tensor(
            [e["embedding"] for e in self.examples],
            dtype=torch.float32)
        y_combo = torch.tensor(
            [e["combo_label"] for e in self.examples],
            dtype=torch.long)
        y_ds    = torch.tensor(
            [e["dataset_label"] for e in self.examples],
            dtype=torch.long)
        y_meth  = torch.tensor(
            [e["method_label"] for e in self.examples],
            dtype=torch.long)
        y_acc   = torch.tensor(
            [e["accuracy"] / 100.0 for e in self.examples],
            dtype=torch.float32)

        best_loss = float('inf')
        for epoch in range(epochs):
            optimizer.zero_grad()
            out = self.model(X)
            loss = (
                ce_loss(out["agents"],   y_combo) * 1.0 +
                ce_loss(out["dataset"],  y_ds)    * 0.8 +
                ce_loss(out["method"],   y_meth)  * 0.6 +
                mse_loss(out["accuracy"], y_acc)  * 0.5
            )
            loss.backward()
            optimizer.step()
            if loss.item() < best_loss:
                best_loss = loss.item()

        torch.save(self.model.state_dict(), MODEL_FILE)
        self.trained = True

        self.model.eval()
        with torch.no_grad():
            out   = self.model(X)
            preds = out["agents"].argmax(dim=1)
            acc   = (preds == y_combo).float().mean().item()

        print(f"  ✅ Meta-model trained!")
        print(f"     Examples: {len(self.examples)}")
        print(f"     Agent prediction accuracy: {acc:.1%}")
        print(f"     Loss: {best_loss:.4f}")

    # ── INSIGHTS ─────────────────────────────────────────────
    def get_insights(self) -> dict:
        if not self.examples:
            return {
                "status":   "learning",
                "examples": 0,
                "message":  "Collecting training examples..."
            }

        combo_accs = {}
        combo_cnts = {}
        for e in self.examples:
            key = str(e["agents"])
            combo_accs[key] = combo_accs.get(key, 0) + e["accuracy"]
            combo_cnts[key] = combo_cnts.get(key, 0) + 1

        best_combo = max(
            combo_accs.items(),
            key=lambda x: x[1] / combo_cnts[x[0]]
        )[0] if combo_accs else "unknown"

        ds_accs = {}
        ds_cnts = {}
        for e in self.examples:
            ds = e["dataset"]
            ds_accs[ds] = ds_accs.get(ds, 0) + e["accuracy"]
            ds_cnts[ds] = ds_cnts.get(ds, 0) + 1

        best_dataset = max(
            ds_accs.items(),
            key=lambda x: x[1] / ds_cnts[x[0]]
        )[0] if ds_accs else "unknown"

        avg_acc = round(
            sum(e["accuracy"] for e in self.examples) /
            len(self.examples), 1)

        accuracy_trend = []
        if len(self.examples) >= 3:
            window = max(3, len(self.examples) // 3)
            for i in range(0, len(self.examples), window):
                chunk = self.examples[i:i+window]
                accuracy_trend.append(round(
                    sum(e["accuracy"] for e in chunk) /
                    len(chunk), 1))

        return {
            "status":          "trained" if self.trained else "collecting",
            "examples":        len(self.examples),
            "trained":         self.trained,
            "avg_accuracy":    avg_acc,
            "best_combo":      best_combo,
            "best_dataset":    best_dataset,
            "accuracy_trend":  accuracy_trend,
            "until_retrain":   self.RETRAIN_EVERY - (
                len(self.examples) % self.RETRAIN_EVERY),
            "until_first_train": max(0,
                self.MIN_EXAMPLES_TO_TRAIN - len(self.examples)),
            "combo_performance": {
                k: round(combo_accs[k] / combo_cnts[k], 1)
                for k in combo_accs
            },
            "dataset_performance": {
                k: round(ds_accs[k] / ds_cnts[k], 1)
                for k in ds_accs
            },
        }

    # ── HELPERS ──────────────────────────────────────────────
    def _get_embedding(self, text: str) -> list:
        try:
            from api.cache_manager import get_embedding
            return get_embedding(text)
        except Exception as e:
            print(f"  ⚠️ Embedding error: {e}")
            return []

    def _encode_combo(self, agents: list) -> int:
        agents_sorted = sorted(agents)
        for i, combo in enumerate(AGENT_COMBOS):
            if sorted(combo) == agents_sorted:
                return i
        for i, combo in enumerate(AGENT_COMBOS):
            if any(a in combo for a in agents):
                return i
        return 0

    def _encode_dataset(self, dataset: str) -> int:
        for i, ds in enumerate(DATASETS):
            if ds in dataset or dataset in ds:
                return i
        return len(DATASETS) - 1

    def _encode_method(self, method: str) -> int:
        for i, m in enumerate(METHODS):
            if m in method or method in m:
                return i
        return 1

    def _load_examples(self) -> list:
        if os.path.exists(META_FILE):
            try:
                with open(META_FILE, 'r') as f:
                    return json.load(f)
            except Exception:
                return []
        return []

    def _save_examples(self):
        with open(META_FILE, 'w') as f:
            json.dump([{
                "problem":       e["problem"],
                "embedding":     e["embedding"],
                "agents":        e["agents"],
                "dataset":       e["dataset"],
                "method":        e["method"],
                "accuracy":      e["accuracy"],
                "combo_label":   e["combo_label"],
                "dataset_label": e["dataset_label"],
                "method_label":  e["method_label"],
                "learned_at":    e["learned_at"],
            } for e in self.examples], f, indent=2)

    def _save_insights(self):
        with open(INSIGHT_FILE, 'w') as f:
            json.dump(self.get_insights(), f, indent=2)


# ── Global singleton ─────────────────────────────────────────
_meta_learner = None

def get_meta_learner() -> MetaLearner:
    global _meta_learner
    if _meta_learner is None:
        _meta_learner = MetaLearner()
    return _meta_learner
