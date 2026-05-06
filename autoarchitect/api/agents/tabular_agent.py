"""
tabular_agent.py — AutoArchitect TabularAgent
TabPFN (primary, zero-shot) / XGBoost / LightGBM fallback on CSV data.
Saves model + encoders + meta to models/trained/.
"""

import os
import json
import pickle
import hashlib
import time
import numpy as np
import pandas as pd
import psutil

from pathlib import Path

try:
    from tabpfn import TabPFNClassifier
    _TABPFN_AVAILABLE = True
except ImportError:
    _TABPFN_AVAILABLE = False

BASE_DIR    = Path(__file__).parent.parent.parent
TRAINED_DIR = BASE_DIR / "models" / "trained"
TRAINED_DIR.mkdir(parents=True, exist_ok=True)

_TARGET_CANDIDATES = ["label", "target", "class", "y", "output", "result"]


class TabularAgent:
    NAME = "Tabular Agent"

    def __init__(self, name: str = "TabularAgent"):
        self.name            = name
        self.model           = None
        self.model_type      = None          # "tabpfn", "xgboost", or "lightgbm"
        self.feature_columns = None
        self.target_column   = None
        self.label_encoders  = {}
        self.classes         = None
        self._hash_id        = None
        print(f"  [TabularAgent] {name} loaded")

    # ── GUARDRAIL ─────────────────────────────────────────────────────────────

    def _should_use_tabpfn(self, X_train):
        """Determine if TabPFN is appropriate for this dataset."""
        if not _TABPFN_AVAILABLE:
            return False, "tabpfn not installed"
        if X_train.shape[0] > 10000:
            return False, f"too many rows ({X_train.shape[0]})"
        if X_train.shape[1] > 100:
            return False, f"too many features ({X_train.shape[1]})"
        available_ram_gb = psutil.virtual_memory().available / 1e9
        if available_ram_gb < 8:
            return False, f"insufficient RAM ({available_ram_gb:.1f}GB)"
        return True, "ok"

    # ── SAVE HELPER ───────────────────────────────────────────────────────────

    def _save_model(self, hash_id, meta_extras=None):
        """Pickle current model + encoders and write meta JSON."""
        model_path = TRAINED_DIR / f"{hash_id}_tabular.pkl"
        enc_path   = TRAINED_DIR / f"{hash_id}_tabular_encoders.pkl"
        meta_path  = TRAINED_DIR / f"{hash_id}_tabular_meta.json"

        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)
        with open(enc_path, "wb") as f:
            pickle.dump(self.label_encoders, f)

        meta = {
            "hash_id":         hash_id,
            "model_type":      self.model_type,
            "feature_columns": self.feature_columns,
            "target_column":   self.target_column,
            "classes":         self.classes,
            "num_classes":     len(self.classes),
            "num_features":    len(self.feature_columns),
            "tabpfn_used":     self.model_type == "tabpfn",
            "label_encoder_classes": {
                col: list(le.classes_)
                for col, le in self.label_encoders.items()
            },
        }
        if meta_extras:
            meta.update(meta_extras)

        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        print(f"  [TabularAgent] Saved: {model_path}")
        return model_path

    # ── TRAINING ──────────────────────────────────────────────────────────────

    def train(self, csv_path: str, target_column: str = None,
              hash_id: str = None) -> dict:
        """
        TabPFN primary (zero-shot) with XGBoost / LightGBM fallback.
        Returns metrics dict.
        """
        import xgboost as xgb
        import lightgbm as lgb
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing   import LabelEncoder
        from sklearn.metrics         import accuracy_score

        # 1. Load ──────────────────────────────────────────────────────────────
        df = pd.read_csv(csv_path)
        print(f"  [TabularAgent] Loaded {len(df)} rows x {len(df.columns)} cols")

        # 2. Auto-detect target ────────────────────────────────────────────────
        if target_column is None:
            for cand in _TARGET_CANDIDATES:
                if cand in df.columns:
                    target_column = cand
                    break
            else:
                target_column = df.columns[-1]
        self.target_column = target_column
        print(f"  [TabularAgent] Target: '{target_column}'")

        X = df.drop(columns=[target_column]).copy()
        y = df[target_column].copy()

        # 3. Missing values ────────────────────────────────────────────────────
        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                X[col] = X[col].fillna(X[col].median())
            else:
                mode_val = X[col].mode()
                X[col]   = X[col].fillna(mode_val.iloc[0] if len(mode_val) > 0
                                          else "unknown")

        # 4. Label encode categoricals ─────────────────────────────────────────
        self.label_encoders  = {}
        self.feature_columns = []
        for col in X.columns:
            if not pd.api.types.is_numeric_dtype(X[col]):
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self.label_encoders[col] = le
            self.feature_columns.append(col)

        # Encode target
        target_le = LabelEncoder()
        y_enc     = target_le.fit_transform(y.astype(str))
        self.label_encoders["__target__"] = target_le
        self.classes = list(target_le.classes_)
        num_classes  = len(self.classes)

        X_arr = X[self.feature_columns].values.astype(float)

        # 5. 70 / 15 / 15 stratified split ────────────────────────────────────
        X_tr, X_tmp, y_tr, y_tmp = train_test_split(
            X_arr, y_enc, test_size=0.30, random_state=42, stratify=y_enc)
        X_val, X_te, y_val, y_te = train_test_split(
            X_tmp, y_tmp, test_size=0.50, random_state=42, stratify=y_tmp)
        print(f"  [TabularAgent] Split: train={len(X_tr)}, "
              f"val={len(X_val)}, test={len(X_te)}")

        if hash_id is None:
            hash_id = hashlib.md5(
                os.path.abspath(csv_path).encode()).hexdigest()[:10]
        self._hash_id = hash_id

        # 6. TabPFN primary attempt ────────────────────────────────────────────
        use_tabpfn, reason = self._should_use_tabpfn(X_tr)

        if use_tabpfn:
            try:
                start = time.time()
                clf = TabPFNClassifier(device='cpu')
                clf.fit(X_tr, y_tr)
                training_time = time.time() - start

                val_acc  = clf.score(X_val, y_val)
                test_acc = clf.score(X_te, y_te)
                print(f"  [TabPFN] val={val_acc:.3f}  test={test_acc:.3f}  "
                      f"time={training_time:.2f}s")

                if val_acc >= 0.65:
                    self.model      = clf
                    self.model_type = "tabpfn"
                    model_path = self._save_model(hash_id, {
                        "val_accuracy":          round(float(val_acc), 4),
                        "test_accuracy":         round(float(test_acc), 4),
                        "training_time_seconds": round(training_time, 2),
                        "num_samples":           X_tr.shape[0],
                    })
                    return {
                        "accuracy":      round(float(test_acc), 4),
                        "val_accuracy":  round(float(val_acc), 4),
                        "model_type":    "tabpfn",
                        "training_time": round(training_time, 2),
                        "num_samples":   X_tr.shape[0],
                        "num_features":  X_tr.shape[1],
                        "num_classes":   num_classes,
                        "train_size":    len(X_tr),
                        "test_size":     len(X_te),
                        "tabpfn_used":   True,
                        "model_path":    str(model_path),
                        "classes":       self.classes,
                    }
                else:
                    print(f"  [TabPFN] Low accuracy {val_acc:.2%}, fallback")
            except Exception as e:
                print(f"  [TabPFN] Failed: {e}, fallback to XGBoost")
        else:
            print(f"  [TabPFN] Skipped: {reason}")

        # 7. Train XGBoost (fallback) ──────────────────────────────────────────
        xgb_start = time.time()
        xgb_model = xgb.XGBClassifier(
            n_estimators  = 200,
            max_depth     = 5,
            learning_rate = 0.1,
            eval_metric   = "logloss",
            random_state  = 42,
            n_jobs        = -1,
        )
        xgb_model.fit(X_tr, y_tr, verbose=False)
        xgb_val_acc = accuracy_score(y_val, xgb_model.predict(X_val))
        xgb_time    = time.time() - xgb_start
        print(f"  [TabularAgent] XGBoost val accuracy: {xgb_val_acc:.3f}  "
              f"time={xgb_time:.2f}s")

        # 8. Fallback to LightGBM if accuracy < 0.65 ──────────────────────────
        best_model   = xgb_model
        best_val_acc = xgb_val_acc
        best_time    = xgb_time
        self.model_type = "xgboost"

        if xgb_val_acc < 0.65:
            print(f"  [TabularAgent] XGBoost < 0.65, trying LightGBM...")
            lgb_start = time.time()
            lgb_model = lgb.LGBMClassifier(
                n_estimators  = 200,
                max_depth     = 5,
                learning_rate = 0.1,
                random_state  = 42,
                n_jobs        = -1,
                verbose       = -1,
            )
            lgb_model.fit(X_tr, y_tr)
            lgb_val_acc = accuracy_score(y_val, lgb_model.predict(X_val))
            lgb_time    = time.time() - lgb_start
            print(f"  [TabularAgent] LightGBM val accuracy: {lgb_val_acc:.3f}")
            if lgb_val_acc > best_val_acc:
                best_model      = lgb_model
                best_val_acc    = lgb_val_acc
                best_time       = lgb_time
                self.model_type = "lightgbm"

        self.model = best_model

        test_acc = accuracy_score(y_te, best_model.predict(X_te))
        print(f"  [TabularAgent] Test accuracy: {test_acc:.3f} ({self.model_type})")

        imps     = best_model.feature_importances_
        feat_imp = sorted(zip(self.feature_columns, imps.tolist()),
                          key=lambda x: x[1], reverse=True)
        top5 = feat_imp[:5]
        print(f"  [TabularAgent] Top features: {[f[0] for f in top5]}")

        model_path = self._save_model(hash_id, {
            "val_accuracy":          round(float(best_val_acc), 4),
            "test_accuracy":         round(float(test_acc), 4),
            "training_time_seconds": round(best_time, 2),
            "feature_importances":   {col: round(float(imp), 6)
                                      for col, imp in feat_imp},
        })

        return {
            "accuracy":            round(float(test_acc), 4),
            "val_accuracy":        round(float(best_val_acc), 4),
            "feature_importances": {col: round(float(imp), 6)
                                    for col, imp in feat_imp},
            "top_5_features":      [
                {"feature": col, "importance": round(float(imp), 6)}
                for col, imp in top5
            ],
            "model_type":   self.model_type,
            "num_features": len(self.feature_columns),
            "num_classes":  num_classes,
            "train_size":   len(X_tr),
            "test_size":    len(X_te),
            "tabpfn_used":  False,
            "model_path":   str(model_path),
            "classes":      self.classes,
        }

    # ── LOAD ──────────────────────────────────────────────────────────────────

    def load_trained_model(self, model_path: str) -> bool:
        """Load .pkl model + _meta.json + _encoders.pkl from disk."""
        try:
            model_path = Path(model_path)
            if not model_path.exists():
                print(f"  [TabularAgent] Model not found: {model_path}")
                return False

            stem      = model_path.stem
            meta_path = model_path.parent / f"{stem}_meta.json"
            enc_path  = model_path.parent / f"{stem}_encoders.pkl"

            if not meta_path.exists():
                print(f"  [TabularAgent] Meta not found: {meta_path}")
                return False

            with open(model_path, "rb") as f:
                self.model = pickle.load(f)

            with open(meta_path) as f:
                meta = json.load(f)

            self.feature_columns = meta["feature_columns"]
            self.target_column   = meta["target_column"]
            self.classes         = meta["classes"]
            self.model_type      = meta["model_type"]  # "tabpfn", "xgboost", or "lightgbm"
            self._hash_id        = meta.get("hash_id")

            if enc_path.exists():
                with open(enc_path, "rb") as f:
                    self.label_encoders = pickle.load(f)
            else:
                from sklearn.preprocessing import LabelEncoder
                self.label_encoders = {}
                for col, classes in meta.get("label_encoder_classes", {}).items():
                    le          = LabelEncoder()
                    le.classes_ = np.array(classes)
                    self.label_encoders[col] = le

            print(f"  [TabularAgent] Loaded {self.model_type} model -- "
                  f"{len(self.feature_columns)} features, "
                  f"{len(self.classes)} classes")
            return True

        except Exception as e:
            print(f"  [TabularAgent] Load failed: {e}")
            return False

    # ── PREDICT (single) ──────────────────────────────────────────────────────

    def predict(self, data) -> dict:
        """
        data: dict of feature values, pandas Series/DataFrame row, or list.
        Returns {label, confidence, top3, model_type, agent_used}.
        """
        if self.model is None:
            return {
                "label":      "error",
                "confidence": 0.0,
                "error":      "No trained model. Run training first.",
                "fake":       False,
            }
        try:
            X = self._build_feature_array(data)

            if self.model_type == "tabpfn":
                proba     = self.model.predict_proba(X)[0]
                label_idx = int(proba.argmax())
                label     = (self.classes[label_idx]
                             if self.classes and label_idx < len(self.classes)
                             else str(label_idx))
                confidence = float(proba[label_idx])
                top3_idx  = proba.argsort()[-3:][::-1]
                top3 = [
                    {
                        "label": (self.classes[int(i)]
                                  if self.classes and int(i) < len(self.classes)
                                  else str(int(i))),
                        "score": float(proba[int(i)]),
                    }
                    for i in top3_idx
                ]
            else:
                probs     = self.model.predict_proba(X)[0]
                idx       = int(np.argmax(probs))
                confidence = float(probs[idx])
                label     = (self.classes[idx]
                             if self.classes and idx < len(self.classes)
                             else str(idx))
                top3_idx  = np.argsort(probs)[::-1][:min(3, len(probs))]
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
                "confidence": round(confidence, 3),
                "top3":       top3,
                "model_type": self.model_type,
                "agent_used": "TabularAgent",
            }
        except Exception as e:
            return {"label": "error", "confidence": 0.0,
                    "error": str(e), "fake": False}

    # ── PREDICT BATCH ─────────────────────────────────────────────────────────

    def predict_batch(self, csv_path: str) -> list:
        """Run inference over every row in csv_path. Returns list of dicts."""
        if self.model is None:
            return [{"error": "No trained model. Run training first.",
                     "fake": False}]
        try:
            df = pd.read_csv(csv_path)
            if self.target_column and self.target_column in df.columns:
                df = df.drop(columns=[self.target_column])

            results = []
            for idx, row in df.iterrows():
                pred = self.predict(row.to_dict())
                pred["row"] = int(idx)
                results.append(pred)
            return results
        except Exception as e:
            return [{"error": str(e), "fake": False}]

    # ── RUN (orchestrator compat) ─────────────────────────────────────────────

    def run(self, problem: str, image_data: str = "") -> dict:
        """Orchestrator-compatible stub — real work is in train() + predict()."""
        return {
            "status":      "success",
            "agent":       self.NAME,
            "type":        "tabular_classification",
            "model_type":  self.model_type or "not trained",
            "model_loaded": self.model is not None,
            "classes":     self.classes or [],
            "message": (
                f"TabularAgent ready. Call train(csv_path) to train. "
                f"Problem: {problem[:60]}"
            ),
        }

    # ── HELPERS ───────────────────────────────────────────────────────────────

    def _build_feature_array(self, data) -> np.ndarray:
        """
        Convert dict / Series / DataFrame / list → (1, n_features) float array
        using the same encoding applied at training time.
        """
        if isinstance(data, pd.Series):
            row = data.to_dict()
        elif isinstance(data, pd.DataFrame):
            row = data.iloc[0].to_dict()
        elif isinstance(data, list):
            row = dict(zip(self.feature_columns, data))
        elif isinstance(data, dict):
            row = data
        else:
            row = dict(data)

        features = []
        for col in self.feature_columns:
            val = row.get(col, None)

            if col in self.label_encoders:
                le      = self.label_encoders[col]
                val_str = str(val) if val is not None else "unknown"
                if val_str in le.classes_:
                    val = int(le.transform([val_str])[0])
                else:
                    val = 0
            else:
                if val is None:
                    val = 0.0
                try:
                    val = float(val)
                    if np.isnan(val):
                        val = 0.0
                except (ValueError, TypeError):
                    val = 0.0

            features.append(val)

        return np.array([features], dtype=float)
