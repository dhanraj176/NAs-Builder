# Brain Core 1 — Task Understander: Training Results

## Overview

| Field | Value |
|-------|-------|
| Core | Core 1: Task Understander |
| Base model | Qwen/Qwen2.5-1.5B-Instruct |
| Fine-tuning method | QLoRA (4-bit, Unsloth) |
| Teacher model | DeepSeek V3 (671B parameters) |
| Training examples | 672 (distilled from teacher) |
| Train / eval split | 90% / 10% (≈ 605 train, 67 eval) |

## Training Metrics (Colab, Day 18)

| Metric | Value |
|--------|-------|
| Final train loss | **0.345** |
| Schema pass rate | **10/10 (100%)** |
| Domain accuracy | **9/10 (90%)** |
| Training time | **6.2 minutes** |
| Hardware | Tesla T4 GPU (Colab free tier) |
| Adapter size (zip) | 81.4 MB |
| Cost | $0 |

## LoRA Configuration

| Setting | Value |
|---------|-------|
| LoRA rank (r) | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Epochs | 3 |
| Effective batch size | 16 (4 × 4 grad accum) |
| Learning rate | 2e-4 with cosine decay |
| Optimizer | adamw_8bit |
| Precision | bf16 |

## Output Schema

The model outputs structured JSON with these fields:

```json
{
  "primary_intent": "string",
  "domain": "image | text | tabular | audio | multimodal | medical | security",
  "complexity": "low | medium | high",
  "real_time_required": true | false,
  "multi_modal": true | false,
  "key_entities": ["list", "of", "strings"]
}
```

## Adapter Location

```
models/brain_cores/core1_task_understander_adapter/
```

Files committed to git (large files excluded via `.gitignore`):
- `adapter_config.json` (LoRA config)
- `special_tokens_map.json`
- `tokenizer_config.json`
- `vocab.json`, `merges.txt` (tokenizer vocabulary)

Files excluded by `.gitignore`:
- `adapter_model.safetensors` (~70.5 MB) — excluded via `*.safetensors`
- `tokenizer.json` (~10.9 MB) — excluded via `models/brain_cores/*/tokenizer.json`

## Integration

```python
from api.brain.cores.task_understander import get_task_understander

core1 = get_task_understander()
result = core1.understand("Detect potholes in dashcam footage in real time")
# Returns: {"primary_intent": "...", "domain": "image", ...}
```

First call triggers one-time model load (~30s, downloads Qwen2.5-1.5B base if not cached).

## Status

| Step | Status |
|------|--------|
| Dataset generation (672 examples) | DONE (Day 13) |
| Colab fine-tuning notebook | DONE (Day 18) |
| Adapter training on T4 GPU | DONE (Day 18, 6.2 min) |
| Integration code (`task_understander.py`) | DONE (Day 18) |
| Offline schema tests | DONE (Day 18) |
| Full inference test (`--infer`) | Pending (requires Qwen download) |
