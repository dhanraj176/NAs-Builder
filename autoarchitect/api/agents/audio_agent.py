"""
audio_agent.py -- AutoArchitect AudioAgent
MFCC + RandomForest (primary) with Wav2Vec2 foundation model upgrade.
Whisper for transcription.
"""

import os
import json
import pickle
import hashlib
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn

BASE_DIR    = Path(__file__).parent.parent.parent
TRAINED_DIR = BASE_DIR / "models" / "trained"
TRAINED_DIR.mkdir(parents=True, exist_ok=True)

_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


# ── Wav2Vec2 linear head ──────────────────────────────────────────────────────

class Wav2Vec2LinearClassifier(nn.Module):
    """Tiny linear head on top of Wav2Vec2 frozen embeddings (768-dim)."""
    def __init__(self, num_classes, embedding_dim=768):
        super().__init__()
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, embeddings):
        return self.classifier(embeddings)


# ── Agent ─────────────────────────────────────────────────────────────────────

class AudioAgent:
    NAME = "Audio Agent"

    def __init__(self, name="AudioAgent"):
        self.name            = name
        self.model           = None
        self.classifier_type = None
        self.whisper_model   = None
        self.classes         = None
        self.feature_dim     = 40
        self._hash_id        = None
        self._label_encoder  = None
        # Wav2Vec2 state
        self.model_arch      = None   # None | "wav2vec2_linear"
        self._w2v_processor  = None
        self._w2v_model      = None
        self._w2v_linear     = None
        print(f"  [AudioAgent] {name} loaded")

    # ── Wav2Vec2 backbone (lazy) ──────────────────────────────────────────────

    def _load_wav2vec2(self):
        """Load Wav2Vec2-base once; subsequent calls are no-ops."""
        if self._w2v_model is not None:
            return
        print("[Wav2Vec2] Loading foundation model (one-time)...")
        from transformers import Wav2Vec2Processor, Wav2Vec2Model
        self._w2v_processor = Wav2Vec2Processor.from_pretrained(
            "facebook/wav2vec2-base-960h")
        self._w2v_model = Wav2Vec2Model.from_pretrained(
            "facebook/wav2vec2-base-960h")
        self._w2v_model.eval()
        for p in self._w2v_model.parameters():
            p.requires_grad = False
        print("[Wav2Vec2] Ready.")

    def _extract_audio_embeddings(self, audio_array, sample_rate=16000):
        """
        Extract mean-pooled Wav2Vec2 embeddings from a 1-D numpy array.
        Returns (1, 768) tensor.
        """
        self._load_wav2vec2()
        inputs = self._w2v_processor(
            audio_array,
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True,
        )
        with torch.no_grad():
            outputs = self._w2v_model(**inputs)
        # Mean-pool over time dimension -> (1, 768)
        return outputs.last_hidden_state.mean(dim=1)

    # ── Wav2Vec2 training ─────────────────────────────────────────────────────

    def train_with_wav2vec2(self, audio_folder, hash_id=None):
        """
        Foundation model training for audio:
          1. Walk audio_folder (subfolder = class label)
          2. Resample to 16 kHz, extract Wav2Vec2 embeddings (frozen)
          3. Train tiny linear classifier on embeddings (20 epochs)
          4. Save linear weights only (~5 KB)

        Falls back to MFCC+RandomForest on any failure.
        Returns metrics dict.
        """
        try:
            import torchaudio
        except ImportError:
            raise RuntimeError(
                "torchaudio not installed. Run: pip install torchaudio")

        from sklearn.preprocessing import LabelEncoder
        from sklearn.model_selection import train_test_split

        audio_folder = Path(audio_folder)
        if not audio_folder.is_dir():
            raise ValueError(f"audio_folder not found: {audio_folder}")

        self._load_wav2vec2()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        X_emb, y_labels = [], []
        TARGET_SR = 16000

        import soundfile as _sf

        for class_dir in sorted(audio_folder.iterdir()):
            if not class_dir.is_dir():
                continue
            class_name = class_dir.name
            for audio_file in sorted(class_dir.iterdir()):
                if audio_file.suffix.lower() not in _AUDIO_EXTS:
                    continue
                try:
                    data, sr = _sf.read(str(audio_file), dtype='float32',
                                        always_2d=False)
                    if data.ndim > 1:
                        data = data.mean(axis=1)
                    if sr != TARGET_SR:
                        wf = torch.tensor(data[None, :])
                        wf = torchaudio.functional.resample(wf, sr, TARGET_SR)
                        data = wf.squeeze().numpy()
                    emb = self._extract_audio_embeddings(data, TARGET_SR)
                    X_emb.append(emb.cpu())
                    y_labels.append(class_name)
                except Exception as e:
                    print(f"  [Wav2Vec2] skipping {audio_file.name}: {e}")

        if len(X_emb) < 4:
            raise ValueError(
                f"Need at least 4 audio samples, got {len(X_emb)}")
        n_classes = len(set(y_labels))
        if n_classes < 2:
            raise ValueError(
                f"Need at least 2 classes, got {n_classes}")

        print(f"  [Wav2Vec2] Loaded {len(X_emb)} samples, "
              f"{n_classes} classes: {sorted(set(y_labels))}")

        le     = LabelEncoder()
        y_enc  = le.fit_transform(y_labels)
        classes = list(le.classes_)

        X_arr = torch.cat(X_emb)  # (N, 768)
        y_t   = torch.tensor(y_enc, dtype=torch.long)

        try:
            from sklearn.model_selection import train_test_split as _tts
            tr_idx, te_idx = _tts(
                range(len(X_arr)), test_size=0.2,
                random_state=42, stratify=y_enc)
        except ValueError:
            n = len(X_arr)
            tr_idx = list(range(int(0.8 * n)))
            te_idx = list(range(int(0.8 * n), n))

        train_emb = X_arr[tr_idx]
        train_lbl = y_t[tr_idx]
        test_emb  = X_arr[te_idx]
        test_lbl  = y_t[te_idx]

        print(f"  [Wav2Vec2] Training linear head ({n_classes} classes)...")
        linear    = Wav2Vec2LinearClassifier(n_classes, 768).to(device)
        optimizer = torch.optim.Adam(linear.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        BATCH = 32
        N     = len(train_emb)
        last_train_acc = 0.0
        for epoch in range(20):
            perm    = torch.randperm(N)
            correct = total = 0
            for i in range(0, N, BATCH):
                idx   = perm[i:i+BATCH]
                emb_b = train_emb[idx].to(device)
                lbl_b = train_lbl[idx].to(device)
                optimizer.zero_grad()
                out   = linear(emb_b)
                loss  = criterion(out, lbl_b)
                loss.backward()
                optimizer.step()
                preds    = out.argmax(1)
                correct += (preds == lbl_b).sum().item()
                total   += len(lbl_b)
            last_train_acc = round(100 * correct / total, 2)
            if (epoch + 1) % 5 == 0:
                print(f"  [Wav2Vec2] Epoch {epoch+1}/20 -> {last_train_acc}%")

        linear.eval()
        with torch.no_grad():
            out      = linear(test_emb.to(device))
            preds    = out.argmax(1)
            test_acc = round(
                100 * (preds == test_lbl.to(device)).sum().item()
                / len(test_lbl), 2)

        if hash_id is None:
            hash_id = hashlib.md5(
                str(audio_folder.resolve()).encode()).hexdigest()[:10]
        self._hash_id = hash_id

        weights_path = str(TRAINED_DIR / f"{hash_id}_audio_linear.pth")
        meta_path    = str(TRAINED_DIR / f"{hash_id}_audio_linear_meta.json")

        torch.save(linear.state_dict(), weights_path)
        model_size_kb = os.path.getsize(weights_path) / 1024

        with open(meta_path, 'w') as f:
            json.dump({
                "model_type":       "wav2vec2_linear",
                "method":           "wav2vec2_linear",
                "num_classes":      n_classes,
                "embedding_dim":    768,
                "foundation_model": "facebook/wav2vec2-base-960h",
                "classes":          classes,
                "train_accuracy":   last_train_acc,
                "test_accuracy":    test_acc,
                "hash_id":          hash_id,
                "sample_rate":      TARGET_SR,
            }, f, indent=2)

        self._w2v_linear     = linear.cpu()
        self.model_arch      = "wav2vec2_linear"
        self.classes         = classes
        self._label_encoder  = le

        print(f"  [Wav2Vec2] Done! Test: {test_acc}%  "
              f"Size: {model_size_kb:.0f} KB")

        return {
            "accuracy":        test_acc / 100,
            "test_accuracy":   test_acc,
            "train_accuracy":  last_train_acc,
            "classes":         classes,
            "classifier_type": "wav2vec2_linear",
            "model_path":      weights_path,
            "model_size_kb":   round(model_size_kb, 1),
        }

    # ── MFCC training (primary, unchanged) ───────────────────────────────────

    def train(self, audio_folder, hash_id=None):
        """
        Walk audio_folder (subfolder = class label), extract 40-band MFCC,
        train RandomForest; fallback to GradientBoosting if val_acc < 0.60.
        """
        import librosa
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.preprocessing import LabelEncoder
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score

        audio_folder = Path(audio_folder)
        if not audio_folder.is_dir():
            raise ValueError(f"audio_folder not found: {audio_folder}")

        X, y = [], []
        for class_dir in sorted(audio_folder.iterdir()):
            if not class_dir.is_dir():
                continue
            class_name = class_dir.name
            for audio_file in sorted(class_dir.iterdir()):
                if audio_file.suffix.lower() not in _AUDIO_EXTS:
                    continue
                try:
                    signal, sr = librosa.load(str(audio_file), sr=22050, mono=True)
                    mfcc       = librosa.feature.mfcc(y=signal, sr=sr,
                                                      n_mfcc=self.feature_dim)
                    X.append(mfcc.mean(axis=1))
                    y.append(class_name)
                except Exception as e:
                    print(f"  [AudioAgent] skipping {audio_file.name}: {e}")

        if len(X) < 4:
            raise ValueError(f"Need at least 4 audio samples, got {len(X)}")
        n_classes = len(set(y))
        if n_classes < 2:
            raise ValueError(f"Need at least 2 classes, got {n_classes}")

        print(f"  [AudioAgent] Loaded {len(X)} samples, {n_classes} classes: "
              f"{sorted(set(y))}")

        X_arr = np.array(X)
        le    = LabelEncoder()
        y_enc = le.fit_transform(y)
        self.classes        = list(le.classes_)
        self._label_encoder = le

        try:
            X_tr, X_tmp, y_tr, y_tmp = train_test_split(
                X_arr, y_enc, test_size=0.30, random_state=42, stratify=y_enc)
        except ValueError:
            X_tr, X_tmp, y_tr, y_tmp = train_test_split(
                X_arr, y_enc, test_size=0.30, random_state=42)
        try:
            X_val, X_te, y_val, y_te = train_test_split(
                X_tmp, y_tmp, test_size=0.50, random_state=42, stratify=y_tmp)
        except ValueError:
            X_val, X_te, y_val, y_te = train_test_split(
                X_tmp, y_tmp, test_size=0.50, random_state=42)

        print(f"  [AudioAgent] Split: train={len(X_tr)}, val={len(X_val)}, "
              f"test={len(X_te)}")

        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_tr, y_tr)
        rf_val_acc = accuracy_score(y_val, rf.predict(X_val))
        print(f"  [AudioAgent] RandomForest val accuracy: {rf_val_acc:.3f}")

        best_model   = rf
        best_val_acc = rf_val_acc
        self.classifier_type = "random_forest"

        if rf_val_acc < 0.60:
            print(f"  [AudioAgent] RF < 0.60, trying GradientBoosting...")
            gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
            gb.fit(X_tr, y_tr)
            gb_val_acc = accuracy_score(y_val, gb.predict(X_val))
            print(f"  [AudioAgent] GradientBoosting val accuracy: {gb_val_acc:.3f}")
            if gb_val_acc > best_val_acc:
                best_model   = gb
                best_val_acc = gb_val_acc
                self.classifier_type = "gradient_boosting"

        self.model = best_model
        test_acc   = accuracy_score(y_te, best_model.predict(X_te))
        print(f"  [AudioAgent] Test accuracy: {test_acc:.3f} ({self.classifier_type})")

        if hash_id is None:
            hash_id = hashlib.md5(str(audio_folder.resolve()).encode()).hexdigest()[:10]
        self._hash_id = hash_id

        model_path = TRAINED_DIR / f"{hash_id}_audio.pkl"
        meta_path  = TRAINED_DIR / f"{hash_id}_audio_meta.json"

        with open(model_path, "wb") as f:
            pickle.dump({
                "model":         best_model,
                "classes":       self.classes,
                "label_encoder": le,
                "feature_dim":   self.feature_dim,
            }, f)

        meta = {
            "hash_id":         hash_id,
            "classifier_type": self.classifier_type,
            "classes":         self.classes,
            "feature_dim":     self.feature_dim,
            "num_samples":     len(X),
            "train_size":      len(X_tr),
            "test_size":       len(X_te),
            "val_accuracy":    round(float(best_val_acc), 4),
            "test_accuracy":   round(float(test_acc), 4),
            "sample_rate":     22050,
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        print(f"  [AudioAgent] Saved: {model_path}")

        return {
            "accuracy":        round(float(test_acc), 4),
            "val_accuracy":    round(float(best_val_acc), 4),
            "num_samples":     len(X),
            "classes":         self.classes,
            "classifier_type": self.classifier_type,
            "sample_rate":     22050,
            "model_path":      str(model_path),
            "train_size":      len(X_tr),
            "test_size":       len(X_te),
        }

    # ── LOAD ──────────────────────────────────────────────────────────────────

    def load_trained_model(self, model_path):
        """Load from disk. Detects Wav2Vec2 linear or MFCC pkl."""
        try:
            model_path = Path(model_path)
            if not model_path.exists():
                print(f"  [AudioAgent] Model not found: {model_path}")
                return False

            # Wav2Vec2 linear .pth
            if model_path.suffix == '.pth':
                meta_p = str(model_path).replace(".pth", "_meta.json")
                saved  = {}
                if os.path.exists(meta_p):
                    with open(meta_p) as f:
                        saved = json.load(f)
                if saved.get("method") == "wav2vec2_linear":
                    self._load_wav2vec2()
                    state  = torch.load(str(model_path), map_location="cpu",
                                        weights_only=True)
                    nc     = state["classifier.weight"].shape[0]
                    linear = Wav2Vec2LinearClassifier(nc, 768)
                    linear.load_state_dict(state)
                    linear.eval()
                    self._w2v_linear = linear
                    self.model_arch  = "wav2vec2_linear"
                    self.classes     = saved.get("classes", [str(i) for i in range(nc)])
                    print(f"  [AudioAgent] Loaded Wav2Vec2 linear — "
                          f"{nc} classes: {self.classes}")
                    return True

            # MFCC pkl (legacy)
            with open(model_path, "rb") as f:
                data = pickle.load(f)

            self.model           = data["model"]
            self.classes         = data["classes"]
            self.feature_dim     = data.get("feature_dim", 40)
            self._label_encoder  = data.get("label_encoder")
            self.classifier_type = type(self.model).__name__

            stem      = model_path.stem
            meta_path = model_path.parent / f"{stem}_meta.json"
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                self.classifier_type = meta.get("classifier_type",
                                                 self.classifier_type)
                self._hash_id = meta.get("hash_id")

            print(f"  [AudioAgent] Loaded {self.classifier_type} model -- "
                  f"{len(self.classes)} classes: {self.classes}")
            return True
        except Exception as e:
            print(f"  [AudioAgent] Load failed: {e}")
            return False

    # ── PREDICT ───────────────────────────────────────────────────────────────

    def predict(self, audio_path):
        """
        Classify audio file.
        Routes to Wav2Vec2 linear if available, else MFCC+sklearn.
        """
        # Wav2Vec2 path
        if self.model_arch == "wav2vec2_linear" and self._w2v_linear is not None:
            try:
                import soundfile as _sf
                import torchaudio
                data, sr = _sf.read(str(audio_path), dtype='float32',
                                    always_2d=False)
                if data.ndim > 1:
                    data = data.mean(axis=1)
                if sr != 16000:
                    wf = torch.tensor(data[None, :])
                    wf = torchaudio.functional.resample(wf, sr, 16000)
                    data = wf.squeeze().numpy()
                audio_np = data
                emb = self._extract_audio_embeddings(audio_np, 16000)
                with torch.no_grad():
                    out   = self._w2v_linear(emb)
                    probs = torch.softmax(out, dim=1).squeeze()
                    idx   = int(probs.argmax())
                    conf  = float(probs[idx])
                label    = (self.classes[idx] if self.classes and
                            idx < len(self.classes) else str(idx))
                top3_idx = probs.argsort(descending=True)[:min(3, len(probs))]
                top3 = [
                    {
                        "label":      (self.classes[int(i)] if self.classes and
                                       int(i) < len(self.classes) else str(int(i))),
                        "confidence": round(float(probs[int(i)]), 3),
                    }
                    for i in top3_idx
                ]
                return {
                    "label":      label,
                    "confidence": round(conf, 3),
                    "top3":       top3,
                    "agent_used": "AudioAgent/Wav2Vec2",
                }
            except Exception as e:
                return {"label": "error", "confidence": 0.0,
                        "error": str(e), "fake": False}

        # MFCC path (legacy)
        if self.model is None:
            return {"label": "error", "confidence": 0.0,
                    "error": "No trained model. Run training first.",
                    "fake": False}
        try:
            import librosa
            signal   = librosa.load(str(audio_path), sr=22050, mono=True)[0]
            mfcc     = librosa.feature.mfcc(y=signal, sr=22050,
                                             n_mfcc=self.feature_dim)
            features = mfcc.mean(axis=1).reshape(1, -1)

            probs = self.model.predict_proba(features)[0]
            idx   = int(np.argmax(probs))
            conf  = float(probs[idx])
            label = (self.classes[idx] if self.classes and
                     idx < len(self.classes) else str(idx))

            top3_idx = np.argsort(probs)[::-1][:min(3, len(probs))]
            top3 = [
                {
                    "label":      (self.classes[int(i)] if self.classes and
                                   int(i) < len(self.classes) else str(int(i))),
                    "confidence": round(float(probs[int(i)]), 3),
                }
                for i in top3_idx
            ]
            return {
                "label":      label,
                "confidence": round(conf, 3),
                "top3":       top3,
                "agent_used": "AudioAgent",
            }
        except Exception as e:
            return {"label": "error", "confidence": 0.0,
                    "error": str(e), "fake": False}

    # ── TRANSCRIBE ────────────────────────────────────────────────────────────

    def transcribe(self, audio_path):
        """Transcribe audio with Whisper base model."""
        try:
            if self.whisper_model is None:
                import whisper
                print("  [AudioAgent] Loading Whisper base model...")
                self.whisper_model = whisper.load_model("base")
            result = self.whisper_model.transcribe(str(audio_path))
            return {
                "transcript": result.get("text", "").strip(),
                "language":   result.get("language", "unknown"),
                "segments":   result.get("segments", []),
            }
        except Exception as e:
            return {
                "transcript": "",
                "language":   "unknown",
                "segments":   [],
                "error":      str(e),
            }

    def transcribe_and_classify(self, audio_path, text_classifier=None):
        """Transcribe then optionally classify the transcript."""
        transcription = self.transcribe(audio_path)
        result = {
            "transcript": transcription.get("transcript", ""),
            "language":   transcription.get("language", "unknown"),
        }
        if "error" in transcription:
            result["error"] = transcription["error"]
            return result
        if text_classifier is not None:
            try:
                pred              = text_classifier.predict(result["transcript"])
                result["label"]   = pred.get("label", "unknown")
                result["confidence"] = pred.get("confidence", 0.0)
            except Exception as e:
                result["label"]      = "error"
                result["confidence"] = 0.0
                result["error"]      = str(e)
        return result

    # ── ORCHESTRATOR STUB ─────────────────────────────────────────────────────

    def run(self, problem, image_data=""):
        return {
            "status":       "success",
            "agent":        self.NAME,
            "type":         "audio_classification",
            "model_loaded": self.model is not None or self._w2v_linear is not None,
            "classes":      self.classes or [],
            "message": (
                f"AudioAgent ready. Call train(audio_folder) to train. "
                f"Problem: {problem[:60]}"
            ),
        }
