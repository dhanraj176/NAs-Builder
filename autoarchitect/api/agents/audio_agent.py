"""
audio_agent.py -- AutoArchitect AudioAgent
MFCC feature extraction + sklearn classifier for audio classification.
Whisper for transcription.
"""

import os
import json
import pickle
import hashlib
import numpy as np
from pathlib import Path

BASE_DIR    = Path(__file__).parent.parent.parent
TRAINED_DIR = BASE_DIR / "models" / "trained"
TRAINED_DIR.mkdir(parents=True, exist_ok=True)

_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


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
        print(f"  [AudioAgent] {name} loaded")

    # -- TRAINING --------------------------------------------------------------

    def train(self, audio_folder, hash_id=None):
        """
        Walk audio_folder (subfolder = class label), extract 40-band MFCC features,
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

        print(f"  [AudioAgent] Loaded {len(X)} samples, {n_classes} classes: {sorted(set(y))}")

        X_arr = np.array(X)
        le    = LabelEncoder()
        y_enc = le.fit_transform(y)
        self.classes        = list(le.classes_)
        self._label_encoder = le

        # 70/15/15 stratified split with non-stratified fallback
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

        print(f"  [AudioAgent] Split: train={len(X_tr)}, val={len(X_val)}, test={len(X_te)}")

        # RandomForest primary
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

        # Save model + meta
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

    # -- LOAD ------------------------------------------------------------------

    def load_trained_model(self, model_path):
        """Load pkl + meta JSON from disk. Returns False (not raises) on failure."""
        try:
            model_path = Path(model_path)
            if not model_path.exists():
                print(f"  [AudioAgent] Model not found: {model_path}")
                return False

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
                self.classifier_type = meta.get("classifier_type", self.classifier_type)
                self._hash_id        = meta.get("hash_id")

            print(f"  [AudioAgent] Loaded {self.classifier_type} model -- "
                  f"{len(self.classes)} classes: {self.classes}")
            return True
        except Exception as e:
            print(f"  [AudioAgent] Load failed: {e}")
            return False

    # -- PREDICT ---------------------------------------------------------------

    def predict(self, audio_path):
        """
        Extract MFCC from audio_path and classify.
        Returns {label, confidence, top3, agent_used} or error dict.
        """
        if self.model is None:
            return {"label": "error", "confidence": 0.0,
                    "error": "No trained model. Run training first.", "fake": False}
        try:
            import librosa
            signal   = librosa.load(str(audio_path), sr=22050, mono=True)[0]
            mfcc     = librosa.feature.mfcc(y=signal, sr=22050, n_mfcc=self.feature_dim)
            features = mfcc.mean(axis=1).reshape(1, -1)

            probs = self.model.predict_proba(features)[0]
            idx   = int(np.argmax(probs))
            conf  = float(probs[idx])
            label = (self.classes[idx]
                     if self.classes and idx < len(self.classes) else str(idx))

            top3_idx = np.argsort(probs)[::-1][:min(3, len(probs))]
            top3 = [
                {
                    "label":      (self.classes[int(i)]
                                   if self.classes and int(i) < len(self.classes)
                                   else str(int(i))),
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

    # -- TRANSCRIBE ------------------------------------------------------------

    def transcribe(self, audio_path):
        """Transcribe audio with Whisper base model. Returns error dict on failure."""
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

    # -- ORCHESTRATOR STUB -----------------------------------------------------

    def run(self, problem, image_data=""):
        return {
            "status":       "success",
            "agent":        self.NAME,
            "type":         "audio_classification",
            "model_loaded": self.model is not None,
            "classes":      self.classes or [],
            "message": (
                f"AudioAgent ready. Call train(audio_folder) to train. "
                f"Problem: {problem[:60]}"
            ),
        }
