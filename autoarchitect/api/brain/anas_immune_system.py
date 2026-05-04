"""
anas_immune_system.py — Structural Immune System for ANAS
Agent Network Architecture Search — NeurIPS AutoML Workshop

Novel contribution: a persistent, cosine-similarity-based memory that
prevents re-evaluating architectures structurally similar to known failures,
and steers search toward regions near past successes.

Two vaults
----------
ToxicVault   — quarantines failure patterns; blocks similar candidates before
               any GPU time is spent on them.
SuccessVault — archives high-performing architectures; surfaces them as
               warm-start candidates for future problems.

ImmuneSystem — composes both vaults into a single interface; tracks compute
               saved for paper reporting.

Similarity metric: cosine similarity over the 26-dim structural_vector()
defined in anas_search_space.py.  Cosine is preferred over Jaccard because
it respects agent execution-order (encoded in the position sub-vector) and
topology type (the one-hot sub-vector), not just set membership.

Persistence
-----------
brain_data/toxic_vault.json   — failure archive
brain_data/success_vault.json — success archive
brain_data/immune_stats.json  — cumulative check/abort counters
"""

from __future__ import annotations

import sys as _sys
import pathlib as _pathlib

# Allow `python api/brain/anas_immune_system.py` (script mode) to resolve
# api.* imports by inserting the project root before any package imports.
if __name__ == "__main__":
    _sys.path.insert(
        0, str(_pathlib.Path(__file__).resolve().parent.parent.parent)
    )

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from api.brain.anas_search_space import NetworkArchitecture, STRUCTURAL_DIM


# ── File paths (absolute, mirrors meta_learner.py convention) ─────────────────

BRAIN_DIR    = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "brain_data",
)
TOXIC_FILE   = os.path.join(BRAIN_DIR, "toxic_vault.json")
SUCCESS_FILE = os.path.join(BRAIN_DIR, "success_vault.json")
STATS_FILE   = os.path.join(BRAIN_DIR, "immune_stats.json")


# ── Domain constants ──────────────────────────────────────────────────────────

# Architecture accuracy below this is treated as a failure and quarantined.
FAILURE_ACCURACY_THRESHOLD: float = 30.0

# Compute cost model (T4 GPU, spot pricing — used for paper reporting).
GPU_MINUTES_PER_RUN:   float = 15.0   # average ResNet18 fine-tune on ~10k samples
COST_PER_GPU_HOUR_USD: float = 0.50   # cloud T4 spot rate (USD)


# ── Module-level similarity function ──────────────────────────────────────────

def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Cosine similarity ∈ [-1, 1] between two structural vectors.

    Returns 0.0 when either vector is all-zeros (degenerate case for empty
    architectures — avoids a divide-by-zero NaN propagating into vault logic).
    """
    norm_a = float(np.linalg.norm(vec_a))
    norm_b = float(np.linalg.norm(vec_b))
    if norm_a < 1e-9 or norm_b < 1e-9:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


# ═════════════════════════════════════════════════════════════════════════════
# ToxicVault — quarantine for failed architecture patterns
# ═════════════════════════════════════════════════════════════════════════════

class ToxicVault:
    """
    Persistent store of failed architectures.

    Every entry records the structural_vector so that similarity checks are
    O(|vault|) vector operations rather than requiring architecture
    re-instantiation.  At typical vault sizes (< 10k entries) this is
    fast enough to run synchronously before each candidate evaluation.

    Persistence: brain_data/toxic_vault.json
    """

    def __init__(self, vault_path: str = TOXIC_FILE):
        self._path    = vault_path
        os.makedirs(os.path.dirname(vault_path), exist_ok=True)
        self._entries: List[Dict[str, Any]] = self._load()
        # Pre-build numpy matrix for fast batch similarity — rebuilt on write.
        self._vec_matrix: Optional[np.ndarray] = self._build_matrix()

    # ── Public API ────────────────────────────────────────────────────────────

    def store_failure(
        self,
        arch:           NetworkArchitecture,
        failure_reason: str,
        accuracy:       float,
    ) -> None:
        """
        Quarantine arch as a known failure.

        Deduplicates by architecture_id — storing the same structural failure
        twice would double-count its influence in similarity checks.
        """
        aid = arch.architecture_id()
        if any(e["arch_id"] == aid for e in self._entries):
            return  # already recorded

        entry: Dict[str, Any] = {
            "arch_id":          aid,
            "agents":           arch.agents,
            "topology":         arch.topology,
            "structural_vector": arch.structural_vector().tolist(),
            "failure_reason":   failure_reason,
            "accuracy":         round(float(accuracy), 3),
            "stored_at":        datetime.now().isoformat(),
        }
        self._entries.append(entry)
        self._vec_matrix = self._build_matrix()
        self._save()
        print(f"  [ToxicVault] quarantined {aid[:8]} "
              f"(acc={accuracy:.1f}%, reason='{failure_reason}')")

    def is_toxic(
        self,
        arch:      NetworkArchitecture,
        threshold: float = 0.90,
    ) -> bool:
        """
        Return True if arch is >= threshold cosine-similar to any known failure.

        The default threshold of 0.90 was chosen to catch structural near-
        duplicates (same agents, minor reordering, or topology swap) while
        avoiding false positives for genuinely different architectures.
        See ablation study Table 3 in the paper for threshold sensitivity.
        """
        sim, _ = self._find_best_match(arch.structural_vector(), threshold)
        return sim >= threshold

    def similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Cosine similarity between two structural vectors (delegates to module fn)."""
        return cosine_similarity(vec_a, vec_b)

    def find_match(
        self,
        arch:      NetworkArchitecture,
        threshold: float = 0.90,
    ) -> Optional[Dict[str, Any]]:
        """
        Return the most similar toxic entry if similarity >= threshold, else None.

        Returns a copy of the vault entry augmented with ``"similarity"`` key.
        """
        sim, entry = self._find_best_match(arch.structural_vector(), threshold)
        if sim >= threshold and entry is not None:
            return {**entry, "similarity": round(sim, 4)}
        return None

    def __len__(self) -> int:
        return len(self._entries)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _find_best_match(
        self,
        query_vec: np.ndarray,
        threshold: float,
    ) -> Tuple[float, Optional[Dict[str, Any]]]:
        """Return (best_similarity, best_entry) — O(|vault|) scan."""
        if self._vec_matrix is None or len(self._entries) == 0:
            return 0.0, None

        q_norm = float(np.linalg.norm(query_vec))
        if q_norm < 1e-9:
            return 0.0, None

        # Matrix-vector cosine: (M @ q) / (row_norms * q_norm)
        dots       = self._vec_matrix @ query_vec               # shape (n,)
        row_norms  = np.linalg.norm(self._vec_matrix, axis=1)   # shape (n,)
        denom      = row_norms * q_norm
        safe_denom = np.where(denom < 1e-9, 1e-9, denom)
        sims       = dots / safe_denom                           # shape (n,)

        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])
        return best_sim, self._entries[best_idx]

    def _build_matrix(self) -> Optional[np.ndarray]:
        if not self._entries:
            return None
        return np.array(
            [e["structural_vector"] for e in self._entries],
            dtype=np.float32,
        )

    def _load(self) -> List[Dict[str, Any]]:
        if os.path.exists(self._path):
            try:
                with open(self._path, "r") as f:
                    data = json.load(f)
                    return data.get("entries", [])
            except Exception:
                pass
        return []

    def _save(self) -> None:
        with open(self._path, "w") as f:
            json.dump({"version": "1.0", "entries": self._entries}, f, indent=2)


# ═════════════════════════════════════════════════════════════════════════════
# SuccessVault — archive of high-performing architectures
# ═════════════════════════════════════════════════════════════════════════════

class SuccessVault:
    """
    Persistent store of successful architectures.

    Used to warm-start search in new domains: given a query architecture,
    get_similar_successes() returns past winners that are structurally similar,
    ranked by a combined score (cosine_similarity × normalised_accuracy).
    This biases future search toward regions of the space that have already
    produced good results.

    Persistence: brain_data/success_vault.json
    """

    def __init__(self, vault_path: str = SUCCESS_FILE):
        self._path    = vault_path
        os.makedirs(os.path.dirname(vault_path), exist_ok=True)
        self._entries: List[Dict[str, Any]] = self._load()
        self._vec_matrix: Optional[np.ndarray] = self._build_matrix()

    # ── Public API ────────────────────────────────────────────────────────────

    def store_success(
        self,
        arch:             NetworkArchitecture,
        accuracy:         float,
        dataset:          str,
        task_description: str,
    ) -> None:
        """
        Archive a successful architecture.

        If the same arch_id already exists, update it only if this run
        achieved higher accuracy — keeping the best known result per topology.
        """
        aid = arch.architecture_id()
        for entry in self._entries:
            if entry["arch_id"] == aid:
                if accuracy > entry["accuracy"]:
                    entry["accuracy"]         = round(float(accuracy), 3)
                    entry["dataset"]          = dataset
                    entry["task_description"] = task_description
                    entry["updated_at"]       = datetime.now().isoformat()
                    self._vec_matrix = self._build_matrix()
                    self._save()
                return

        entry: Dict[str, Any] = {
            "arch_id":          aid,
            "agents":           arch.agents,
            "topology":         arch.topology,
            "structural_vector": arch.structural_vector().tolist(),
            "accuracy":         round(float(accuracy), 3),
            "dataset":          dataset,
            "task_description": task_description,
            "stored_at":        datetime.now().isoformat(),
        }
        self._entries.append(entry)
        self._vec_matrix = self._build_matrix()
        self._save()
        print(f"  [SuccessVault] archived {aid[:8]} "
              f"(acc={accuracy:.1f}%, dataset='{dataset}')")

    def get_similar_successes(
        self,
        arch:  NetworkArchitecture,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Return top_k past successes most similar to arch.

        Ranking key: similarity × (accuracy / 100), so a candidate that is
        90% similar with 95% accuracy outranks one that is 95% similar with
        60% accuracy.  This prevents the vault from locking search into
        high-similarity but mediocre regions.

        Each returned dict extends the vault entry with:
          "similarity"     — cosine similarity to query
          "combined_score" — similarity × (accuracy / 100)
        """
        if self._vec_matrix is None or not self._entries:
            return []

        query_vec = arch.structural_vector()
        q_norm    = float(np.linalg.norm(query_vec))
        if q_norm < 1e-9:
            return []

        dots      = self._vec_matrix @ query_vec
        row_norms = np.linalg.norm(self._vec_matrix, axis=1)
        denom     = np.where(row_norms * q_norm < 1e-9, 1e-9, row_norms * q_norm)
        sims      = dots / denom

        scored = []
        for i, (sim, entry) in enumerate(zip(sims, self._entries)):
            combined = float(sim) * (entry["accuracy"] / 100.0)
            scored.append((combined, float(sim), entry))

        scored.sort(key=lambda x: x[0], reverse=True)

        return [
            {
                **entry,
                "similarity":     round(sim, 4),
                "combined_score": round(combined, 4),
            }
            for combined, sim, entry in scored[:top_k]
        ]

    def __len__(self) -> int:
        return len(self._entries)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_matrix(self) -> Optional[np.ndarray]:
        if not self._entries:
            return None
        return np.array(
            [e["structural_vector"] for e in self._entries],
            dtype=np.float32,
        )

    def _load(self) -> List[Dict[str, Any]]:
        if os.path.exists(self._path):
            try:
                with open(self._path, "r") as f:
                    data = json.load(f)
                    return data.get("entries", [])
            except Exception:
                pass
        return []

    def _save(self) -> None:
        with open(self._path, "w") as f:
            json.dump({"version": "1.0", "entries": self._entries}, f, indent=2)


# ═════════════════════════════════════════════════════════════════════════════
# ImmuneSystem — orchestrates ToxicVault + SuccessVault
# ═════════════════════════════════════════════════════════════════════════════

class ImmuneSystem:
    """
    The ANAS structural immune system.

    Workflow per candidate architecture lambda:
      1. immune.check(lambda)        — abort early if structurally toxic
      2. [evaluate lambda on GPU]
      3. immune.learn(lambda, acc)   — route to toxic or success vault

    The check() step costs ~1ms (matrix multiply over vault) vs ~15 GPU-min
    per evaluation, so the amortised cost of maintaining the vault is
    negligible even at vault sizes of 10k entries.
    """

    # Threshold below which an architecture is considered a failure.
    FAILURE_THRESHOLD: float = FAILURE_ACCURACY_THRESHOLD

    def __init__(
        self,
        toxic_vault:   Optional[ToxicVault]   = None,
        success_vault: Optional[SuccessVault]  = None,
        stats_path:    Optional[str]           = STATS_FILE,
        toxic_threshold: float                 = 0.90,
    ):
        os.makedirs(BRAIN_DIR, exist_ok=True)
        self.toxic_vault     = toxic_vault   if toxic_vault   is not None else ToxicVault()
        self.success_vault   = success_vault if success_vault is not None else SuccessVault()
        self._stats_path     = stats_path
        self._toxic_threshold = toxic_threshold
        self._total_checks, self._total_aborts = self._load_stats()
        print(
            f"[ImmuneSystem] ready — "
            f"{len(self.toxic_vault)} toxic, "
            f"{len(self.success_vault)} successes, "
            f"{self._total_aborts} lifetime aborts"
        )

    # ── Core interface ────────────────────────────────────────────────────────

    def check(
        self,
        arch: NetworkArchitecture,
    ) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """
        Gate a candidate architecture before evaluation.

        Returns
        -------
        is_safe        : False means skip evaluation — arch is too similar to
                         a known failure.
        reason         : human-readable explanation (useful for paper logging).
        similar_failure: the matching vault entry if is_safe=False, else None.

        This is the primary compute-saving mechanism.  A False return avoids
        a full GPU training run at the cost of one matrix multiply.
        """
        self._total_checks += 1
        match = self.toxic_vault.find_match(arch, threshold=self._toxic_threshold)

        if match is not None:
            self._total_aborts += 1
            self._save_stats()
            reason = (
                f"Cosine similarity {match['similarity']:.3f} >= "
                f"{self._toxic_threshold:.2f} with known failure "
                f"'{match['arch_id'][:8]}' "
                f"(acc={match['accuracy']:.1f}%, "
                f"reason='{match['failure_reason']}')"
            )
            return False, reason, match

        self._save_stats()
        return True, "clear", None

    def learn(
        self,
        arch:             NetworkArchitecture,
        accuracy:         float,
        dataset:          str  = "unknown",
        task:             str  = "",
    ) -> None:
        """
        Update immune memory from a completed evaluation.

        Routes arch to the toxic vault if accuracy < FAILURE_THRESHOLD,
        otherwise to the success vault.  Call this after every real training
        run — not after cache hits (consistent with meta_learner.learn).
        """
        if accuracy < self.FAILURE_THRESHOLD:
            reason = (
                f"accuracy={accuracy:.1f}% below "
                f"threshold={self.FAILURE_THRESHOLD:.1f}%"
            )
            self.toxic_vault.store_failure(arch, reason, accuracy)
        else:
            self.success_vault.store_success(arch, accuracy, dataset, task)

    # ── Paper reporting ───────────────────────────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        """
        Summary dict for paper Table: Immune System Performance.

        Keys match the column headers in the NeurIPS submission.
        """
        n_toxic   = len(self.toxic_vault)
        n_success = len(self.success_vault)
        abort_rate = round(
            self._total_aborts / max(self._total_checks, 1), 4
        )

        # Failure reason breakdown
        reason_counts: Dict[str, int] = {}
        for e in self.toxic_vault._entries:
            key = e.get("failure_reason", "unknown")[:60]
            reason_counts[key] = reason_counts.get(key, 0) + 1

        return {
            "toxic_architectures":    n_toxic,
            "successful_architectures": n_success,
            "total_checks":           self._total_checks,
            "total_aborts":           self._total_aborts,
            "abort_rate":             abort_rate,
            "toxic_threshold":        self._toxic_threshold,
            "failure_threshold_pct":  self.FAILURE_THRESHOLD,
            "compute_saved":          self.compute_saved_estimate(),
            "failure_reasons":        reason_counts,
        }

    def compute_saved_estimate(self) -> Dict[str, Any]:
        """
        Estimate GPU compute avoided by immune system aborts.

        Assumptions (stated explicitly in paper Section 4.3):
          - GPU_MINUTES_PER_RUN = 15 min  (ResNet18 fine-tune, T4 GPU, ~10k samples)
          - COST_PER_GPU_HOUR_USD = $0.50 (cloud T4 spot price)

        These are conservative lower bounds — larger datasets and deeper
        backbones would increase savings proportionally.
        """
        n           = self._total_aborts
        mins_saved  = n * GPU_MINUTES_PER_RUN
        hours_saved = mins_saved / 60.0
        cost_saved  = hours_saved * COST_PER_GPU_HOUR_USD

        return {
            "aborts":                        n,
            "gpu_minutes_saved":             round(mins_saved,  1),
            "gpu_hours_saved":               round(hours_saved, 3),
            "cost_saved_usd":                round(cost_saved,  4),
            "assumption_gpu_min_per_run":    GPU_MINUTES_PER_RUN,
            "assumption_cost_per_gpu_hour":  COST_PER_GPU_HOUR_USD,
        }

    # ── Stats persistence ─────────────────────────────────────────────────────

    def _load_stats(self) -> Tuple[int, int]:
        if self._stats_path and os.path.exists(self._stats_path):
            try:
                with open(self._stats_path, "r") as f:
                    d = json.load(f)
                    return d.get("total_checks", 0), d.get("total_aborts", 0)
            except Exception:
                pass
        return 0, 0

    def _save_stats(self) -> None:
        if self._stats_path is None:
            return
        try:
            with open(self._stats_path, "w") as f:
                json.dump(
                    {
                        "total_checks": self._total_checks,
                        "total_aborts": self._total_aborts,
                        "saved_at":     datetime.now().isoformat(),
                    },
                    f, indent=2,
                )
        except Exception:
            pass


# ── Global singleton (mirrors meta_learner pattern) ───────────────────────────

_immune_system: Optional[ImmuneSystem] = None


def get_immune_system() -> ImmuneSystem:
    global _immune_system
    if _immune_system is None:
        _immune_system = ImmuneSystem()
    return _immune_system


# ═════════════════════════════════════════════════════════════════════════════
# Verification — run as script
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import tempfile

    SEP = "=" * 64

    print(SEP)
    print("ANAS Immune System  --  NeurIPS AutoML Workshop")
    print("Structural similarity-based failure quarantine + success archive")
    print(SEP)

    # Use a temporary directory so verification never pollutes real brain_data.
    tmp = tempfile.mkdtemp(prefix="anas_immune_verify_")
    tv  = ToxicVault(os.path.join(tmp, "toxic_vault.json"))
    sv  = SuccessVault(os.path.join(tmp, "success_vault.json"))
    immune = ImmuneSystem(
        toxic_vault   = tv,
        success_vault = sv,
        stats_path    = None,   # no persistence for test run
    )

    # ── Build test architectures ──────────────────────────────────────────────
    print("\n[0] Test architectures")

    # Known bad: text->report via PARALLEL — topology mismatch for this combo
    toxic_a = NetworkArchitecture(
        agents=["text", "image", "report"], topology="parallel"
    )
    # Known bad: security alone with no output — under-specified
    toxic_b = NetworkArchitecture(
        agents=["security", "report"], topology="conditional"
    )
    # Known bad: optimizer as first agent — semantically invalid ordering
    toxic_c = NetworkArchitecture(
        agents=["optimizer", "text", "report"], topology="sequential"
    )
    # High performer to store in success vault
    success_a = NetworkArchitecture(
        agents=["image", "severity", "report"], topology="sequential"
    )
    success_b = NetworkArchitecture(
        agents=["medical", "severity", "report"], topology="sequential"
    )
    success_c = NetworkArchitecture(
        agents=["text", "sentiment", "audience", "report"], topology="hierarchical"
    )

    for arch in [toxic_a, toxic_b, toxic_c]:
        print(f"  TOXIC   : {arch}")
    for arch in [success_a, success_b, success_c]:
        print(f"  SUCCESS : {arch}")

    # ── Teach the immune system about failures ────────────────────────────────
    print(f"\n[1] Teaching failures (accuracy < {FAILURE_ACCURACY_THRESHOLD}%)")
    immune.learn(toxic_a, accuracy=18.2, dataset="cifar10",
                 task="image+text multimodal parallel — collapsed")
    immune.learn(toxic_b, accuracy=22.5, dataset="synthetic_intrusion",
                 task="security conditional — under-powered")
    immune.learn(toxic_c, accuracy=11.0, dataset="imdb",
                 task="optimizer-first pipeline — inverted data flow")

    # ── Teach successes ───────────────────────────────────────────────────────
    print(f"\n[2] Teaching successes (accuracy >= {FAILURE_ACCURACY_THRESHOLD}%)")
    immune.learn(success_a, accuracy=88.4, dataset="cifar10",
                 task="pothole visual detection with severity triage")
    immune.learn(success_b, accuracy=91.2, dataset="user_data",
                 task="chest X-ray pneumonia screening pipeline")
    immune.learn(success_c, accuracy=84.7, dataset="imdb",
                 task="full sentiment marketing automation network")

    assert len(tv) == 3,  f"Expected 3 toxic entries, got {len(tv)}"
    assert len(sv) == 3,  f"Expected 3 success entries, got {len(sv)}"
    print(f"  Vault sizes confirmed: {len(tv)} toxic, {len(sv)} successes")

    # ── Test 1: Exact match must be flagged ───────────────────────────────────
    print(f"\n[3] Test 1 — exact match is flagged (similarity = 1.000)")
    is_safe, reason, match = immune.check(toxic_a)
    sim = match["similarity"] if match else 0.0
    print(f"  Query   : {toxic_a}")
    print(f"  is_safe : {is_safe}  (expected False)")
    print(f"  sim     : {sim:.4f}  (expected 1.0000)")
    print(f"  reason  : {reason}")
    assert not is_safe,     "FAIL: exact toxic match should be unsafe"
    assert abs(sim - 1.0) < 1e-4, f"FAIL: sim should be 1.0, got {sim}"
    print("  PASS")

    # ── Test 2: Near-identical (agent reorder) is flagged ────────────────────
    print(f"\n[4] Test 2 -- near-identical (adjacent reorder) is flagged (sim >= 0.90)")
    # toxic_a = ["text", "image", "report"] parallel
    # near = ["image", "text", "report"] parallel  (swap first two)
    near_toxic = NetworkArchitecture(
        agents=["image", "text", "report"], topology="parallel"
    )
    is_safe2, reason2, match2 = immune.check(near_toxic)
    sim2 = match2["similarity"] if match2 else 0.0
    sv2  = near_toxic.structural_vector()
    sv_a = toxic_a.structural_vector()
    raw_sim = cosine_similarity(sv2, sv_a)
    print(f"  Query   : {near_toxic}")
    print(f"  Closest : {toxic_a}")
    print(f"  raw cos : {raw_sim:.4f}")
    print(f"  is_safe : {is_safe2}  (expected False, threshold=0.90)")
    print(f"  sim     : {sim2:.4f}")
    print(f"  reason  : {reason2}")
    assert not is_safe2, "FAIL: near-identical arch should be flagged"
    print("  PASS")

    # ── Test 3: Dissimilar architecture passes freely ─────────────────────────
    print(f"\n[5] Test 3 -- dissimilar architecture passes (sim < 0.90)")
    safe_arch = NetworkArchitecture(
        agents=["medical", "severity", "report"], topology="sequential"
    )
    is_safe3, reason3, match3 = immune.check(safe_arch)
    # Compute similarity to every toxic entry for display
    sims_to_toxic = [
        cosine_similarity(safe_arch.structural_vector(),
                          np.array(e["structural_vector"], dtype=np.float32))
        for e in tv._entries
    ]
    print(f"  Query          : {safe_arch}")
    print(f"  Sims to toxic  : {[round(s, 4) for s in sims_to_toxic]}")
    print(f"  Max sim        : {max(sims_to_toxic):.4f}  (expected < 0.90)")
    print(f"  is_safe        : {is_safe3}   (expected True)")
    print(f"  reason         : {reason3}")
    assert is_safe3, f"FAIL: dissimilar arch flagged with sims {sims_to_toxic}"
    print("  PASS")

    # ── Test 4: Completely different domain passes ────────────────────────────
    print(f"\n[6] Test 4 -- completely different domain passes")
    diff_arch = NetworkArchitecture(
        agents=["audience", "optimizer", "report"], topology="hierarchical"
    )
    is_safe4, reason4, _ = immune.check(diff_arch)
    sims_to_toxic4 = [
        cosine_similarity(diff_arch.structural_vector(),
                          np.array(e["structural_vector"], dtype=np.float32))
        for e in tv._entries
    ]
    print(f"  Query          : {diff_arch}")
    print(f"  Sims to toxic  : {[round(s, 4) for s in sims_to_toxic4]}")
    print(f"  Max sim        : {max(sims_to_toxic4):.4f}  (expected < 0.90)")
    print(f"  is_safe        : {is_safe4}   (expected True)")
    assert is_safe4, "FAIL: completely different arch should pass"
    print("  PASS")

    # ── Test 5: get_similar_successes for warm-starting ───────────────────────
    print(f"\n[7] get_similar_successes (warm-start candidates)")
    query = NetworkArchitecture(
        agents=["image", "severity", "report"], topology="pipeline"
    )
    similar = immune.success_vault.get_similar_successes(query, top_k=3)
    print(f"  Query   : {query}")
    print(f"  Top-3 similar successes:")
    for s in similar:
        print(f"    id={s['arch_id'][:8]}  agents={s['agents']}  "
              f"acc={s['accuracy']:.1f}%  sim={s['similarity']:.4f}  "
              f"combined={s['combined_score']:.4f}")
    assert len(similar) > 0, "FAIL: should return at least one similar success"
    assert similar[0]["similarity"] > 0.5, "FAIL: top hit should be reasonably similar"
    print("  PASS")

    # ── Compute saved estimate ────────────────────────────────────────────────
    print(f"\n[8] Compute saved estimate")
    compute = immune.compute_saved_estimate()
    print(f"  Total checks         : {immune._total_checks}")
    print(f"  Total aborts         : {compute['aborts']}")
    print(f"  GPU minutes saved    : {compute['gpu_minutes_saved']:.1f} min")
    print(f"  GPU hours saved      : {compute['gpu_hours_saved']:.3f} h")
    print(f"  Estimated cost saved : ${compute['cost_saved_usd']:.4f} USD")
    print(f"  (@ {compute['assumption_gpu_min_per_run']} min/run, "
          f"${compute['assumption_cost_per_gpu_hour']}/GPU-hr)")

    # ── Full stats dict ───────────────────────────────────────────────────────
    print(f"\n[9] Full stats() for paper")
    s = immune.stats()
    for k, v in s.items():
        if k not in ("compute_saved", "failure_reasons"):
            print(f"  {k:<32}: {v}")
    print(f"  compute_saved:")
    for k, v in s["compute_saved"].items():
        print(f"    {k:<32}: {v}")

    # ── Structural vector similarity matrix ───────────────────────────────────
    print(f"\n[10] Pairwise similarity matrix (3 toxic architectures)")
    toxics = [toxic_a, toxic_b, toxic_c]
    labels = ["text+image+report/par", "security+rep/cond", "opt+text+rep/seq"]
    vecs   = np.stack([a.structural_vector() for a in toxics])
    norms  = np.linalg.norm(vecs, axis=1, keepdims=True)
    sim_mat = (vecs @ vecs.T) / (norms @ norms.T + 1e-9)
    header  = "                      " + "  ".join(f"{l[:10]:>10}" for l in labels)
    print(f"  {header}")
    for i, row_label in enumerate(labels):
        row = "  ".join(f"{sim_mat[i, j]:10.4f}" for j in range(len(labels)))
        print(f"  {row_label[:20]:<22}{row}")

    # Cleanup
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n[OK] Immune system verification complete.")
    print(f"  All 4 safety checks passed.")
    print(f"  {compute['aborts']} aborts -> {compute['gpu_minutes_saved']} GPU-min "
          f"-> ${compute['cost_saved_usd']} saved.")
    print(SEP)
