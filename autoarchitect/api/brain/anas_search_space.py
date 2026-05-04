"""
anas_search_space.py — Formal Search Space for ANAS
Agent Network Architecture Search — NeurIPS AutoML Workshop

Defines the search space:

    Λ = { λ = (A, T) | A ⊆ AGENTS, T ∈ TOPOLOGIES, valid(λ, constraints) }

where A is an ordered sequence of agents (execution order matters) and T is a
topology type that determines how data flows between agents.

Key properties
--------------
* NetworkArchitecture — a single point λ ∈ Λ with fingerprinting, structural
  encoding for immune-system similarity, compatibility scoring, and hard
  constraint checking.
* SearchConstraints   — hardware-aware feasibility region; shrinks Λ to what
  the target machine can actually run.
* ANASSearchSpace     — operations over Λ: random sampling, local neighborhood
  mutations, warm-start seeding, and exact cardinality analysis for paper
  reporting.
"""

from __future__ import annotations

import sys as _sys
import pathlib as _pathlib

# Allow `python api/brain/anas_search_space.py` (script mode) to resolve
# api.* imports by inserting the project root before any package imports.
if __name__ == "__main__":
    _sys.path.insert(
        0, str(_pathlib.Path(__file__).resolve().parent.parent.parent)
    )

import hashlib
import itertools
import json
import math
import os
import platform
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from api.brain.topology_designer import (
    AGENT_CATALOG,
    TOPOLOGY_TEMPLATES,
    SEQUENTIAL,
    PARALLEL,
    CONDITIONAL,
    PIPELINE,
    HIERARCHICAL,
)


# ── Canonical index mappings (never reorder — breaks structural vectors) ───────

AGENT_INDEX: Dict[str, int] = {
    name: idx for idx, name in enumerate(sorted(AGENT_CATALOG.keys()))
}
# sorted: audience=0, image=1, medical=2, optimizer=3,
#         report=4,   security=5, sentiment=6, severity=7, text=8

TOPOLOGY_INDEX: Dict[str, int] = {
    SEQUENTIAL:   0,
    PARALLEL:     1,
    CONDITIONAL:  2,
    PIPELINE:     3,
    HIERARCHICAL: 4,
}

ALL_TOPOLOGIES = list(TOPOLOGY_INDEX.keys())
N_AGENTS       = len(AGENT_CATALOG)      # 9
N_TOPOLOGIES   = len(TOPOLOGY_INDEX)     # 5

# structural_vector layout:
#   [0        .. N_AGENTS-1]               agent multi-hot          (9 dims)
#   [N_AGENTS .. N_AGENTS+N_TOP-1]         topology one-hot         (5 dims)
#   [N_AGENTS+N_TOP .. 2*N_AGENTS+N_TOP-1] agent position normalised (9 dims)
#   [-3]                                   connection density        (1 dim)
#   [-2]                                   agent count normalised    (1 dim)
#   [-1]                                   has_report flag           (1 dim)
STRUCTURAL_DIM = N_AGENTS + N_TOPOLOGIES + N_AGENTS + 3   # 26


# ── Data-flow compatibility graph ─────────────────────────────────────────────
# Valid downstream agents for each agent, derived from input/output contracts
# in AGENT_CATALOG.  Used by compatibility_score() and neighborhood pruning.

_COMPAT: Dict[str, List[str]] = {
    "image":     ["severity", "report", "text", "medical"],
    "text":      ["sentiment", "report", "security", "severity"],
    "sentiment": ["audience", "report", "optimizer"],
    "severity":  ["report"],
    "medical":   ["severity", "report"],
    "security":  ["severity", "report"],
    "audience":  ["optimizer", "report"],
    "optimizer": ["report"],
    "report":    [],   # terminal — no valid downstream
}

# Agents that should appear at the end of a pipeline
_TERMINAL_AGENTS = {"report", "optimizer"}


# ── Per-agent hyperparameter bounds ───────────────────────────────────────────

_HP_BOUNDS: Dict[str, Dict[str, Any]] = {
    "image":     {"backbone":              ["resnet18", "resnet50", "mobilenet_v2"],
                  "confidence_threshold":  (0.30, 0.90)},
    "text":      {"max_length":            [64, 128, 256],
                  "num_labels":            [2, 3, 5]},
    "sentiment": {"granularity":           ["binary", "ternary", "5-class"]},
    "severity":  {"levels":               [3, 5]},
    "medical":   {"backbone":              ["densenet121", "resnet18"],
                  "confidence_threshold":  (0.50, 0.95)},
    "security":  {"anomaly_threshold":     (0.10, 0.50),
                  "mode":                  ["detection", "classification"]},
    "audience":  {"score_bins":           [5, 10, 20]},
    "optimizer": {"strategy":             ["greedy", "beam_search", "random"]},
    "report":    {"format":               ["json", "markdown", "html"],
                  "alert":                [True, False]},
}


# ═════════════════════════════════════════════════════════════════════════════
# NetworkArchitecture — a single point λ ∈ Λ
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class NetworkArchitecture:
    """
    One candidate agent network λ = (agents, topology, hyperparams).

    ``agents`` is an *ordered* list — the execution order matters for
    SEQUENTIAL and PIPELINE topologies and encodes data-flow direction.
    """

    agents:      List[str]
    topology:    str
    hyperparams: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    metadata:    Dict[str, Any]            = field(default_factory=dict)

    # ── Identity ──────────────────────────────────────────────────────────────

    def architecture_id(self) -> str:
        """
        16-char SHA-256 fingerprint of (agents, topology).

        Hyperparams are excluded so that the same topology with different
        hyperparameter settings maps to the same structural ID.  This is
        consistent with the NAS convention of separating architecture search
        from hyperparameter optimisation.
        """
        canonical = json.dumps(
            {"agents": self.agents, "topology": self.topology},
            sort_keys=True,
        )
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    # ── Structural representation ─────────────────────────────────────────────

    def structural_vector(self) -> np.ndarray:
        """
        Fixed-length float32 vector ∈ ℝ^{STRUCTURAL_DIM} (26 dims).

        Suitable for cosine / Euclidean similarity in:
          - Clonal selection (immune system) diversity pressure
          - Kernel density estimation over the visited archive
          - t-SNE / UMAP visualisation of the search trajectory

        Layout
        ------
        [0..8]   agent multi-hot        — which agents are present
        [9..13]  topology one-hot       — what connection pattern
        [14..22] agent position vector  — normalised execution rank per agent
        [23]     connection density     — valid edges / max possible edges
        [24]     agent count normalised — |A| / N_AGENTS
        [25]     has_report flag        — 1.0 if terminal output agent present
        """
        vec = np.zeros(STRUCTURAL_DIM, dtype=np.float32)
        n   = len(self.agents)

        # Agent multi-hot [0..N_AGENTS-1]
        for a in self.agents:
            if a in AGENT_INDEX:
                vec[AGENT_INDEX[a]] = 1.0

        # Topology one-hot [N_AGENTS..N_AGENTS+N_TOP-1]
        t_offset = N_AGENTS
        vec[t_offset + TOPOLOGY_INDEX.get(self.topology, 0)] = 1.0

        # Agent position — normalised rank in execution order [N_AGENTS+N_TOP..]
        p_offset = N_AGENTS + N_TOPOLOGIES
        for rank, a in enumerate(self.agents):
            if a in AGENT_INDEX:
                vec[p_offset + AGENT_INDEX[a]] = (rank + 1) / max(n, 1)

        # Scalar features
        valid_edges  = sum(
            1 for i, a in enumerate(self.agents)
            for b in self.agents[i + 1:]
            if b in _COMPAT.get(a, [])
        )
        max_edges    = max(n * (n - 1), 1)
        vec[-3] = valid_edges / max_edges              # density ∈ [0, 1]
        vec[-2] = n / N_AGENTS                         # count normalised
        vec[-1] = float("report" in self.agents)       # terminal flag

        return vec

    # ── Compatibility ─────────────────────────────────────────────────────────

    def compatibility_score(self) -> float:
        """
        Score ∈ [0, 1] measuring agent inter-operability.

        Combines two signals:
          1. Data-flow validity  — fraction of edges that respect _COMPAT types
          2. Ordering bonus/penalty — terminal agents (report/optimizer) should
             appear last; penalise -0.1 per misplaced terminal agent.

        A score of 1.0 means every data-flow edge is type-valid and all
        terminal agents are correctly positioned.
        """
        n = len(self.agents)
        if n == 0:
            return 0.0
        if n == 1:
            return 1.0

        flow_score = 0.0
        edge_count = 0

        if self.topology in (SEQUENTIAL, PIPELINE):
            for i in range(n - 1):
                a, b = self.agents[i], self.agents[i + 1]
                edge_count += 1
                if b in _COMPAT.get(a, []):
                    flow_score += 1.0

        elif self.topology == PARALLEL:
            merger = self.agents[-1]
            for a in self.agents[:-1]:
                edge_count += 1
                if merger in _COMPAT.get(a, []):
                    flow_score += 1.0

        elif self.topology == HIERARCHICAL:
            if n >= 3:
                source = self.agents[0]
                sink   = self.agents[-1]
                middle = self.agents[1:-1]
                for m in middle:
                    edge_count += 2
                    flow_score += float(m    in _COMPAT.get(source, []))
                    flow_score += float(sink in _COMPAT.get(m, []))
            else:
                a, b = self.agents[0], self.agents[1]
                edge_count = 1
                flow_score = float(b in _COMPAT.get(a, []))

        elif self.topology == CONDITIONAL:
            a, b = self.agents[0], self.agents[1]
            edge_count = 1
            flow_score = float(b in _COMPAT.get(a, []))

        base = flow_score / edge_count if edge_count else 0.5

        # Ordering bonus: +0.1 for each terminal agent correctly at the end,
        # -0.1 for each terminal agent appearing before the last position.
        order_delta = 0.0
        for i, a in enumerate(self.agents):
            if a in _TERMINAL_AGENTS:
                order_delta += 0.1 if i == n - 1 else -0.1

        return float(np.clip(base + order_delta, 0.0, 1.0))

    # ── Validity ──────────────────────────────────────────────────────────────

    def is_valid(self, constraints: Optional["SearchConstraints"] = None) -> bool:
        """
        Hard constraint checker.

        Rules enforced regardless of constraints
        ----------------------------------------
        * agents list must be non-empty
        * no duplicate agents
        * all agents in AGENT_CATALOG
        * topology in TOPOLOGY_INDEX
        * PARALLEL / CONDITIONAL / HIERARCHICAL require ≥ 2 agents
        * HIERARCHICAL requires ≥ 3 agents

        Rules enforced when constraints are supplied
        --------------------------------------------
        * min_agents ≤ |agents| ≤ max_agents
        * all agents in allowed_agents
        * topology in allowed_topologies
        * "report" present if require_output_agent is True
        """
        if not self.agents:
            return False
        if len(self.agents) != len(set(self.agents)):
            return False
        if not all(a in AGENT_CATALOG for a in self.agents):
            return False
        if self.topology not in TOPOLOGY_INDEX:
            return False
        if self.topology in (PARALLEL, CONDITIONAL) and len(self.agents) < 2:
            return False
        if self.topology == HIERARCHICAL and len(self.agents) < 3:
            return False

        if constraints is not None:
            c = constraints
            if len(self.agents) < c.min_agents:
                return False
            if len(self.agents) > c.max_agents:
                return False
            if not set(self.agents).issubset(set(c.allowed_agents)):
                return False
            if self.topology not in c.allowed_topologies:
                return False
            if c.require_output_agent and "report" not in self.agents:
                return False

        return True

    # ── Dunder helpers ────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"NetworkArchitecture("
            f"id={self.architecture_id()[:8]}, "
            f"agents={self.agents}, "
            f"topology={self.topology}, "
            f"compat={self.compatibility_score():.3f})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, NetworkArchitecture):
            return NotImplemented
        return self.architecture_id() == other.architecture_id()

    def __hash__(self) -> int:
        return hash(self.architecture_id())


# ═════════════════════════════════════════════════════════════════════════════
# SearchConstraints — hardware-aware feasibility region
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class SearchConstraints:
    """
    Defines which points in Λ are feasible on the target hardware.

    Hardware tiers
    --------------
    minimal  — ≤ 4 CPU cores **or** ≤ 4 GB RAM
               max 3 agents, no parallel/hierarchical topologies
    standard — ≤ 8 cores **or** ≤ 16 GB RAM
               max 4 agents, all topologies
    high     — > 8 cores **and** > 16 GB RAM
               max 5 agents, all topologies
    """

    min_agents:           int       = 1
    max_agents:           int       = 5
    allowed_agents:       List[str] = field(
        default_factory=lambda: list(AGENT_CATALOG.keys())
    )
    allowed_topologies:   List[str] = field(
        default_factory=lambda: ALL_TOPOLOGIES[:]
    )
    require_output_agent: bool      = True    # "report" must be present
    max_connections:      int       = 20
    hardware_tier:        str       = "standard"
    hardware_info:        Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_hardware_detection(cls) -> "SearchConstraints":
        """
        Inspect the host machine and return appropriately-sized constraints.

        Falls back gracefully if psutil is unavailable.
        """
        cpu_count = os.cpu_count() or 4
        ram_gb    = _detect_ram_gb()

        if cpu_count <= 4 or ram_gb <= 4.0:
            tier       = "minimal"
            max_agents = 3
            topologies = [SEQUENTIAL, PIPELINE, CONDITIONAL]
        elif cpu_count <= 8 or ram_gb <= 16.0:
            tier       = "standard"
            max_agents = 4
            topologies = ALL_TOPOLOGIES[:]
        else:
            tier       = "high"
            max_agents = 5
            topologies = ALL_TOPOLOGIES[:]

        return cls(
            min_agents           = 1,
            max_agents           = max_agents,
            allowed_agents       = list(AGENT_CATALOG.keys()),
            allowed_topologies   = topologies,
            require_output_agent = True,
            max_connections      = max_agents * (max_agents - 1),
            hardware_tier        = tier,
            hardware_info        = {
                "cpu_cores": cpu_count,
                "ram_gb":    round(ram_gb, 1),
                "platform":  platform.platform(),
            },
        )


def _detect_ram_gb() -> float:
    """Best-effort total RAM detection; returns 8.0 on failure."""
    try:
        import psutil
        return psutil.virtual_memory().total / (1024 ** 3)
    except ImportError:
        pass
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    return int(line.split()[1]) / (1024 ** 2)
    except Exception:
        pass
    return 8.0


# ═════════════════════════════════════════════════════════════════════════════
# ANASSearchSpace — operations over Λ
# ═════════════════════════════════════════════════════════════════════════════

class ANASSearchSpace:
    """
    The formal ANAS search space with sampling, mutation, and analysis methods.

    All methods return only valid architectures (validated against
    self.constraints) so callers never need to filter.
    """

    def __init__(self, constraints: Optional[SearchConstraints] = None):
        self.constraints = constraints or SearchConstraints.from_hardware_detection()
        self._rng        = random.Random()

    # ── Random baseline ───────────────────────────────────────────────────────

    def generate_random(
        self,
        n: int,
        seed: Optional[int] = None,
    ) -> List[NetworkArchitecture]:
        """
        Sample n structurally distinct valid architectures uniformly at random.

        Used as an unbiased baseline for NAS benchmarking — a random search
        that matches ANAS in wall-clock budget should be beatable by any
        reasonable search strategy.

        Parameters
        ----------
        n    : number of distinct architectures to return
        seed : optional RNG seed for reproducibility
        """
        rng        = random.Random(seed)
        pool:  List[NetworkArchitecture] = []
        seen  = set()
        c     = self.constraints
        pool_agents = [a for a in c.allowed_agents if a in AGENT_CATALOG]
        max_attempts = n * 200

        for _ in range(max_attempts):
            if len(pool) >= n:
                break

            k      = rng.randint(c.min_agents, min(c.max_agents, len(pool_agents)))
            agents = rng.sample(pool_agents, k)

            # Enforce output agent at tail position
            if c.require_output_agent and "report" in pool_agents:
                if "report" not in agents:
                    agents[-1] = "report"
                elif agents[-1] != "report":
                    agents.remove("report")
                    agents.append("report")

            topology = rng.choice(c.allowed_topologies)
            hp       = _sample_hyperparams(agents, rng)

            arch = NetworkArchitecture(
                agents=agents, topology=topology, hyperparams=hp,
                metadata={"source": "random", "seed": seed},
            )
            aid = arch.architecture_id()
            if arch.is_valid(c) and aid not in seen:
                seen.add(aid)
                pool.append(arch)

        return pool

    # ── Neighborhood (local search) ───────────────────────────────────────────

    def neighborhood(
        self,
        arch: NetworkArchitecture,
        radius: int = 1,
    ) -> List[NetworkArchitecture]:
        """
        Enumerate all valid 1-edit neighbours of arch.

        Mutation operators (5 types)
        ----------------------------
        add_agent       — insert one agent not currently in the network
        remove_agent    — drop one agent (if |A| > min_agents)
        swap_agent      — replace one agent with a different one
        change_topology — switch to a different topology type
        reorder         — swap two adjacent agents in the execution order

        These five operators form a connected neighbourhood graph over Λ,
        guaranteeing that any two valid architectures are reachable from each
        other via a sequence of single mutations (necessary for completeness
        proofs in the paper).
        """
        c         = self.constraints
        seen      = {arch.architecture_id()}
        neighbours: List[NetworkArchitecture] = []
        available = [a for a in c.allowed_agents
                     if a in AGENT_CATALOG and a not in arch.agents]

        def _try(agents: List[str], topology: str, op: str) -> None:
            hp   = _sample_hyperparams(agents, self._rng)
            cand = NetworkArchitecture(
                agents=list(agents), topology=topology, hyperparams=hp,
                metadata={"source": op, "parent": arch.architecture_id()[:8]},
            )
            aid = cand.architecture_id()
            if cand.is_valid(c) and aid not in seen:
                seen.add(aid)
                neighbours.append(cand)

        n = len(arch.agents)

        # 1. add_agent
        if n < c.max_agents:
            for a in available:
                _try(arch.agents + [a], arch.topology, "add_agent")

        # 2. remove_agent
        if n > c.min_agents:
            for i in range(n):
                _try(arch.agents[:i] + arch.agents[i + 1:],
                     arch.topology, "remove_agent")

        # 3. swap_agent
        for i in range(n):
            for new_a in available:
                mutated      = list(arch.agents)
                mutated[i]   = new_a
                _try(mutated, arch.topology, "swap_agent")

        # 4. change_topology
        for t in c.allowed_topologies:
            if t != arch.topology:
                _try(arch.agents, t, "change_topology")

        # 5. reorder (adjacent transposition — covers all permutations via
        #    composition, which is sufficient for local search convergence)
        for i in range(n - 1):
            mutated          = list(arch.agents)
            mutated[i], mutated[i + 1] = mutated[i + 1], mutated[i]
            _try(mutated, arch.topology, "reorder")

        return neighbours

    # ── Warm-start ────────────────────────────────────────────────────────────

    def warm_start_candidates(
        self,
        domain_hints: Optional[List[str]] = None,
    ) -> List[NetworkArchitecture]:
        """
        Return high-quality starting architectures to seed the search.

        Sources (merged in priority order, deduplicated by architecture_id)
        -------------------------------------------------------------------
        1. TOPOLOGY_TEMPLATES matching domain_hints keywords
        2. All TOPOLOGY_TEMPLATES (domain-agnostic seed)
        3. AGENT_COMBOS from meta_learner (empirically validated combos)

        Using warm starts instead of random initialisation typically reduces
        the number of evaluations needed to reach a fixed quality threshold,
        which is reported as ``warm_start_speedup`` in the ablation study.
        """
        from api.brain.meta_learner import AGENT_COMBOS

        candidates: List[NetworkArchitecture] = []
        seen = set()
        c    = self.constraints

        def _add(agents: List[str], topology: str, source: str) -> None:
            arch = NetworkArchitecture(
                agents=list(agents), topology=topology,
                hyperparams={a: {} for a in agents},
                metadata={"source": source},
            )
            aid = arch.architecture_id()
            if arch.is_valid(c) and aid not in seen:
                seen.add(aid)
                candidates.append(arch)

        # 1. Domain-filtered templates
        if domain_hints:
            hints_lower = [h.lower() for h in domain_hints]
            for name, tmpl in TOPOLOGY_TEMPLATES.items():
                kws = [kw.lower() for kw in tmpl["keywords"]]
                if any(h in kw or kw in h
                       for h in hints_lower for kw in kws):
                    _add(tmpl["agents"], tmpl["topology"],
                         f"template:{name}:domain_match")

        # 2. All templates
        for name, tmpl in TOPOLOGY_TEMPLATES.items():
            _add(tmpl["agents"], tmpl["topology"], f"template:{name}")

        # 3. Meta-learner combos — pair with heuristic topology
        for combo in AGENT_COMBOS:
            filtered = [a for a in combo if a in c.allowed_agents
                        and a in AGENT_CATALOG]
            if not filtered:
                continue
            if c.require_output_agent and "report" in c.allowed_agents:
                if "report" not in filtered:
                    filtered = filtered + ["report"]
                elif filtered[-1] != "report":
                    filtered.remove("report")
                    filtered.append("report")
            topology = _infer_topology(filtered)
            _add(filtered, topology, "meta_combo")

        return candidates

    # ── Cardinality analysis ──────────────────────────────────────────────────

    def size(self, with_hyperparams: bool = False) -> Dict[str, Any]:
        """
        Exact cardinality of Λ under current constraints.

        Enumerates all (ordered_subset, topology) pairs and validates each one,
        so the count reflects the true feasible set — not an approximation.

        Returns a dict ready for a LaTeX table in the paper.

        Parameters
        ----------
        with_hyperparams : if True, also compute |Λ| × |H| where H is the
                           joint hyperparam space (discrete choices only).
        """
        c           = self.constraints
        agents_pool = [a for a in c.allowed_agents if a in AGENT_CATALOG]
        n_pool      = len(agents_pool)
        n_top       = len(c.allowed_topologies)

        topology_counts: Dict[str, int] = {t: 0 for t in c.allowed_topologies}
        compat_sum    = 0.0
        total_valid   = 0
        total_checked = 0

        for k in range(c.min_agents, min(c.max_agents, n_pool) + 1):
            for subset in itertools.combinations(agents_pool, k):
                # fast-path: skip subsets that cannot satisfy require_output_agent
                if c.require_output_agent and "report" not in subset:
                    continue
                for perm in itertools.permutations(subset):
                    perm_list = list(perm)
                    for t in c.allowed_topologies:
                        total_checked += 1
                        arch = NetworkArchitecture(
                            agents=perm_list, topology=t, hyperparams={}
                        )
                        if arch.is_valid(c):
                            total_valid += 1
                            topology_counts[t] += 1
                            compat_sum         += arch.compatibility_score()

        # Theoretical upper bound (no validity filter)
        theoretical = sum(
            math.perm(n_pool, k) * n_top
            for k in range(c.min_agents, min(c.max_agents, n_pool) + 1)
        )

        # Discrete hyperparam multiplier (over allowed agents)
        hp_multiplier = 1
        if with_hyperparams:
            for a in agents_pool:
                for v in _HP_BOUNDS.get(a, {}).values():
                    if isinstance(v, list):
                        hp_multiplier *= len(v)

        avg_compat = round(compat_sum / max(total_valid, 1), 4)

        return {
            "valid_architectures":     total_valid,
            "valid_with_hyperparams":  total_valid * hp_multiplier,
            "hp_multiplier":           hp_multiplier,
            "theoretical_upper_bound": theoretical,
            "pruning_ratio":           round(1.0 - total_valid / max(theoretical, 1), 4),
            "avg_compatibility":       avg_compat,
            "by_topology":             topology_counts,
            "constraints": {
                "min_agents":    c.min_agents,
                "max_agents":    c.max_agents,
                "n_agent_pool":  n_pool,
                "n_topologies":  n_top,
                "hardware_tier": c.hardware_tier,
            },
        }


# ── Internal helpers ──────────────────────────────────────────────────────────

def _sample_hyperparams(agents: List[str], rng: random.Random) -> Dict[str, Dict[str, Any]]:
    hp: Dict[str, Dict[str, Any]] = {}
    for a in agents:
        hp[a] = {}
        for param, choices in _HP_BOUNDS.get(a, {}).items():
            if isinstance(choices, list):
                hp[a][param] = rng.choice(choices)
            elif isinstance(choices, tuple) and len(choices) == 2:
                hp[a][param] = round(rng.uniform(choices[0], choices[1]), 3)
    return hp


def _infer_topology(agents: List[str]) -> str:
    """Mirror TopologyDesigner._infer_topology_type for consistency."""
    n = len(agents)
    if n == 1:
        return SEQUENTIAL
    if ("optimizer" in agents or "report" in agents) and n >= 4:
        return HIERARCHICAL
    if "image" in agents and "text" in agents and n == 3:
        return PARALLEL
    return SEQUENTIAL


# ═════════════════════════════════════════════════════════════════════════════
# Verification — run as script to confirm search space properties
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    SEP = "=" * 64

    print(SEP)
    print("ANAS Search Space  —  NeurIPS AutoML Workshop")
    print("Formal verification of Lambda = {(A, T) | valid(lambda, constraints)}")
    print(SEP)

    # ── Hardware-aware constraints ────────────────────────────────────────────
    constraints = SearchConstraints.from_hardware_detection()
    print(f"\n[0] Hardware detection")
    print(f"  Tier        : {constraints.hardware_tier}")
    for k, v in constraints.hardware_info.items():
        print(f"  {k:<11}: {v}")
    print(f"  max_agents  : {constraints.max_agents}")
    print(f"  topologies  : {constraints.allowed_topologies}")

    space = ANASSearchSpace(constraints)

    # ── Search space cardinality ──────────────────────────────────────────────
    print(f"\n[1] Search space cardinality (exact enumeration)")
    stats = space.size(with_hyperparams=True)
    print(f"  Valid architectures      : {stats['valid_architectures']:>10,}")
    print(f"  Hyperparam multiplier    : {stats['hp_multiplier']:>10,}")
    print(f"  Valid x hyperparams      : {stats['valid_with_hyperparams']:>10,}")
    print(f"  Theoretical upper bound  : {stats['theoretical_upper_bound']:>10,}")
    print(f"  Pruning ratio            : {stats['pruning_ratio']:>10.1%}")
    print(f"  Avg compatibility score  : {stats['avg_compatibility']:>10.4f}")
    print(f"  By topology:")
    for topo, count in stats["by_topology"].items():
        bar = "#" * (count // max(max(stats["by_topology"].values()) // 20, 1))
        print(f"    {topo:<14}: {count:>6,}  {bar}")

    # ── Random sample ─────────────────────────────────────────────────────────
    print(f"\n[2] Random sample (n=5, seed=42)")
    randoms = space.generate_random(5, seed=42)
    for arch in randoms:
        sv   = arch.structural_vector()
        print(f"  {arch}")
        print(f"    valid={arch.is_valid(constraints)}  "
              f"norm={np.linalg.norm(sv):.3f}  "
              f"v[:5]={np.round(sv[:5], 2).tolist()}")

    # ── Warm-start candidates ─────────────────────────────────────────────────
    print(f"\n[3] Warm-start candidates  (domain_hints=['security', 'threat'])")
    warm = space.warm_start_candidates(domain_hints=["security", "threat"])
    print(f"  Total candidates: {len(warm)}")
    for arch in warm[:6]:
        print(f"  {arch}  <- {arch.metadata.get('source')}")

    # ── Neighborhood exploration ──────────────────────────────────────────────
    print(f"\n[4] Neighborhood of first random arch")
    if randoms:
        parent = randoms[0]
        nbrs   = space.neighborhood(parent)
        print(f"  Parent     : {parent}")
        print(f"  Neighbours : {len(nbrs)} valid 1-edit mutations")
        ops: Dict[str, int] = {}
        for nbr in nbrs:
            op = nbr.metadata.get("source", "?")
            ops[op] = ops.get(op, 0) + 1
        for op, cnt in sorted(ops.items()):
            print(f"    {op:<20}: {cnt}")

    # ── Structural vector properties ──────────────────────────────────────────
    print(f"\n[5] Structural vector properties")
    if randoms:
        vecs = np.stack([a.structural_vector() for a in randoms])
        print(f"  Shape          : {vecs.shape}  ({STRUCTURAL_DIM} dims per arch)")
        print(f"  Dim layout     : "
              f"agent_hot[0:{N_AGENTS}]  "
              f"topo_hot[{N_AGENTS}:{N_AGENTS+N_TOPOLOGIES}]  "
              f"position[{N_AGENTS+N_TOPOLOGIES}:{2*N_AGENTS+N_TOPOLOGIES}]  "
              f"scalars[-3:]")
        # pairwise cosine similarity
        norms  = np.linalg.norm(vecs, axis=1, keepdims=True)
        cosine = (vecs @ vecs.T) / (norms @ norms.T + 1e-9)
        off_diag = cosine[np.triu_indices(len(randoms), k=1)]
        print(f"  Pairwise cosine similarity (n={len(randoms)} sample):")
        print(f"    mean={off_diag.mean():.3f}  "
              f"min={off_diag.min():.3f}  "
              f"max={off_diag.max():.3f}")

    # ── Architecture identity ─────────────────────────────────────────────────
    print(f"\n[6] Architecture identity checks")
    if len(randoms) >= 2:
        a1 = randoms[0]
        a2 = NetworkArchitecture(
            agents=a1.agents[:], topology=a1.topology,
            hyperparams={"image": {"backbone": "resnet50"}},  # different hp
            metadata={"source": "clone"},
        )
        print(f"  Same topology, different hyperparams -> same id? "
              f"{a1.architecture_id() == a2.architecture_id()}")
        print(f"  a1 == a2 : {a1 == a2}")
        print(f"  hash(a1) == hash(a2) : {hash(a1) == hash(a2)}")

    print(f"\n[OK] ANAS search space verification complete.")
    print(f"    Lambda contains {stats['valid_architectures']:,} valid architectures "
          f"({stats['valid_architectures'] * stats['hp_multiplier']:,} "
          f"with discrete hyperparams) on a {constraints.hardware_tier} machine.")
    print(SEP)
