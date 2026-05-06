# AutoArchitect Week 2 Backlog
Generated: 2026-05-06 | Day 7 stress test

---

## Summary of Issues Found

| # | Severity | Area | Issue |
|---|----------|------|-------|
| 1 | HIGH     | Routing | Cache poisoning: spam routes to image agents |
| 2 | HIGH     | Architecture | AudioAgent/MultimodalAgent/TabularAgent bypassed by orchestrator |
| 3 | MEDIUM   | Routing | "identify plant species" missing pipeline template |
| 4 | MEDIUM   | Encoding | 146 emoji print statements crash on direct import (Windows) |
| 5 | MEDIUM   | Architecture | 5 AGENT_CATALOG entries have no dedicated agent file |
| 6 | MEDIUM   | Agent | tabular_agent.run() is an orchestrator stub, not real inference |
| 7 | LOW      | Data  | topology_history.json has 2 confirmed wrong-routing entries |
| 8 | LOW      | Data  | brain_data growth — no cap/pruning logic |
| 9 | LOW      | Git   | brain_cache/ not in .gitignore (chroma.sqlite3, 184KB) |
| 10| LOW      | Docs  | verify_zip.py moved to tests/ but may be referenced at root |

---

## Issue Details

---

### [1] HIGH — Cache Poisoning: Wrong Routing for Spam Problems

**Symptom:**
```
"classify spam text messages"  ->  ['image', 'severity', 'report']   WRONG
                               should be  ['text', 'report']
```

**Root cause:**
`topology_history.json` contains a stale entry from 2026-04-07:
```json
{ "problem": "Classify spam text messages", "topology": {"agents": ["image","severity","report"]} }
```
`TopologyDesigner._check_cache()` uses word-Jaccard similarity. Comparing
`"classify spam text messages"` (lowercase) to `"Classify spam text messages"` (lowercase)
yields similarity = **1.000** (identical word sets), so the wrong cached entry wins immediately.

Also affected: `"detect fake news articles"` cached with `['image', 'report']` (2026-04-04).

**Fix (Week 2):**
- Delete the 2 wrong entries from `brain_data/topology_history.json`
- In `_store()`: validate that agents are domain-consistent before saving
  (text problems must not store image agents, etc.)
- Consider inserting new entries at position 0 and searching reversed, so most
  recent results win over old wrong entries

---

### [2] HIGH — AudioAgent/MultimodalAgent/TabularAgent Bypassed in Orchestrator

**Symptom:**
When topology routes to `audio`, `multimodal`, or `tabular` domain, the specialized
agents' real methods are never called.

**Root cause:**
`orchestrator._run_multi_agent()` does:
```python
agent  = self._wake_agent(domain)   # calls factory.create(problem, domain)
result = agent.run(problem, image_data)  # calls DynamicAgent.run() stub
# then immediately overrides with:
trained = self_train(problem, category=domain, epochs=3)  # DARTSNet NAS
```

`agent_factory._model_type_for_domain()`:
```python
def _model_type_for_domain(self, domain):
    if domain in ("image", "medical"):
        return "resnet18"
    return "darts"   # audio/multimodal/tabular all get DARTSNet!
```

So `factory.create("classify calls", "audio")` creates a `DynamicAgent` with a DARTS model.
`AudioAgent`, `MultimodalAgent`, `TabularAgent` are **never instantiated** via the orchestrator.

**Fix (Week 2):**
- Add cases in `_model_type_for_domain()`:
  `"audio" -> "audio_mfcc"`, `"multimodal" -> "clip"`, `"tabular" -> "xgboost"`
- In `factory.create()`: when domain is audio/multimodal/tabular, instantiate the
  real specialized agent, not DynamicAgent
- In `orchestrator._run_single_agent()`: detect specialized domains and call the
  correct train/predict flow instead of self_train()

**Files:** `api/agents/agent_factory.py:152`, `api/orchestrator.py:580-632`

---

### [3] MEDIUM — "Identify Plant Species" Missing Pipeline Template

**Symptom:**
```
"identify plant species from photos"  ->  ['image']   source: rule_based
```
Returns only the image agent with no severity or report stage.

**Root cause:**
No TOPOLOGY_TEMPLATE covers "species", "plant", "flower", or "crop" keywords
with a full pipeline. The rule-based fallback only assigns one agent.

**Fix (Week 2):**
Add keywords `["species", "plant", "flower", "crop", "leaf", "botany", "agriculture"]`
to the `detect_classify_report` template in `topology_designer.py`.

---

### [4] MEDIUM — 146 Emoji Print Statements Break Direct Imports on Windows

**Symptom:**
```python
from api.analyzer import ProblemAnalyzer
# UnicodeEncodeError: 'charmap' codec can't encode '✅' (position 0)
```

`app.py` is safe because it wraps stdout with UTF-8 before importing anything.
But any script or test that imports these modules directly fails on Windows CP1252.

**Affected files (sample):**
```
api/analyzer.py:61          print("checkmark BERT analyzer ready!")
api/cache_manager.py:28     print("brain Loading BERT...")
api/dataset_fetcher.py:391  print(f"target Registry match: ...")
api/dataset_manager.py:135  print(f"package Downloading dataset: ...")
... 146 total across 15+ files
```

**Fix (Week 2, option A — minimal):**
Add UTF-8 stdout wrapper to every script/test entry point:
```python
import sys, io
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
```
**Fix (Week 2, option B — clean sweep):**
Replace all emoji in print() with ASCII tags `[OK]`, `[INFO]`, `[WARN]`, `[+]`.
Consistent with the style already used in agents added Days 1-6.

---

### [5] MEDIUM — 5 AGENT_CATALOG Entries Have No Dedicated Agent File

**AGENT_CATALOG entries without a `*_agent.py` file:**
```
sentiment  — no sentiment_agent.py
severity   — no severity_agent.py
report     — no report_agent.py
audience   — no audience_agent.py
optimizer  — no optimizer_agent.py
```

These 5 are routed to DynamicAgent at runtime, which uses DARTSNet and returns
generic predictions. They are referenced in 12 of 14 topology templates.

**Agent files with no AGENT_CATALOG entry (support agents, expected):**
```
base_agent.py       -- abstract base class
dynamic_agent.py    -- runtime agent, instantiated by factory
evaluator_agent.py  -- evaluation pipeline
fusion_agent.py     -- multi-agent fusion
```

**Fix (Week 2 priority order):**
1. `report_agent.py` — most-used in templates (appears in 11/14); needs real alerting
2. `sentiment_agent.py` — clear BERT/VADER implementation
3. `severity_agent.py` — rule-based severity classifier
4. `audience_agent.py` / `optimizer_agent.py` — lower priority

---

### [6] MEDIUM — tabular_agent.run() Is an Orchestrator Stub

`api/agents/tabular_agent.py:332`:
```python
def run(self, problem, image_data=""):
    """Orchestrator-compatible stub -- real work is in train() + predict()."""
    return {"status": "success", "agent": self.NAME, "type": "tabular_classification", ...}
```

When the orchestrator calls `agent.run()` for a tabular domain, it gets a stub dict
with no actual training or prediction — then immediately tries DARTSNet self-training
which is the wrong model for tabular data.

Same architectural gap exists in `audio_agent.py` and `multimodal_agent.py`.

**Fix:** See Issue [2] — wire specialized agents into the factory + orchestrator.

---

### [7] LOW — 2 Confirmed Wrong-Routing Entries in topology_history.json

From inspection of `brain_data/topology_history.json`:

| Problem | Stored agents | Expected agents | Date |
|---------|--------------|----------------|------|
| "Classify spam text messages" | `['image','severity','report']` | `['text','report']` | 2026-04-07 |
| "detect fake news articles" | `['image','report']` | `['text','report']` | 2026-04-04 |

These were stored from early ANAS runs before the templates were tuned.
Both cause future routing errors via cache hits.

**Immediate fix (safe to do in Week 2):**
Edit `brain_data/topology_history.json` and correct both entries.

---

### [8] LOW — brain_data Growth: No Cap or Pruning Logic

```
brain_data/topology_history.json  : 60 entries, 80KB  (20 added in May alone)
brain_data/meta_examples.json     : 22 entries, 488KB  (ML embeddings are large)
brain_data/history.json           : 1414 lines, 40KB
brain_data/strategies.json        : 1077 lines, 28KB
Total brain_data:                              ~1.7MB
```

At the current rate (5-10 topology entries per run), topology_history will exceed
500 entries (~650KB) within weeks. meta_examples could grow to 10MB+ if more
meta-training is done.

**Fix (Week 2):**
- Add `MAX_HISTORY = 200` cap in `TopologyDesigner._store()`: prune entries where
  `accuracy is None` or accuracy < 0.50 first, then oldest entries
- Add `MAX_META_EXAMPLES = 100` cap in MetaLearner
- Add `brain_data/*.json` to `.gitignore` (these are runtime-generated, not source)

---

### [9] LOW — brain_cache/ Not in .gitignore

`brain_cache/chroma.sqlite3` (184KB) is a runtime ChromaDB file.
It is untracked by git but not listed in `.gitignore`.

Current `.gitignore` covers `cache/` (old cache) and `brain_data/research_cache/`
but not `brain_cache/`.

**Fix:**
```
# add to .gitignore:
brain_cache/
```

---

### [10] LOW — verify_zip.py Path Reference Mismatch

`verify_zip.py` was moved to `tests/verify_zip.py` but any docs, scripts,
or CI configs that reference `python verify_zip.py` (root) will break.

**Fix:**
Search for `verify_zip` references in docs/scripts and update to `tests/verify_zip.py`.

---

## Stress Test Results

### Topology Routing (5 problems)

| Problem | Agents | Source | Correct? |
|---------|--------|--------|----------|
| detect potholes in road images | `['image','severity','report']` | cache | YES |
| classify spam text messages | `['image','severity','report']` | cache | **NO** (see Issue 1) |
| predict customer churn from transaction data | `['tabular','report']` | template | YES |
| identify plant species from photos | `['image']` | rule_based | PARTIAL (missing report) |
| transcribe and classify customer service calls | `['audio','report']` | template | YES |

3/5 fully correct, 1/5 wrong domain, 1/5 incomplete pipeline.

### verify_zip.py (end-to-end ZIP)

All 5 checks PASS:
- image predict: PASS
- text on image agent: PASS
- readme: PASS
- requirements: PASS
- model weights: PASS

ZIP deployment pipeline is unaffected by Days 1-6 changes.

### brain_data Assessment

- Total size: 1.7MB (acceptable for now)
- Growth rate: ~20 topology entries/month at current usage
- No pruning logic exists — needs a cap before production

---

## Week 2 Priority Order

1. **Fix cache poisoning** (edit 2 wrong JSON entries) — 15 min
2. **Wire AudioAgent/MultimodalAgent/TabularAgent into orchestrator** — 1 day
3. **Build report_agent.py and sentiment_agent.py** — 1 day
4. **ASCII-ify print statements** (or add UTF-8 wrapper to all entry points) — 2 hrs
5. **Add plant/species keywords to templates** — 10 min
6. **Add brain_cache/ to .gitignore and brain_data pruning cap** — 30 min
