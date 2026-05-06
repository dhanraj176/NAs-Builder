# Brain Core 3 — Architecture Advisor: Training Results

## Overview

| Field | Value |
|-------|-------|
| Core | Core 3: Architecture Advisor |
| Base model | Qwen/Qwen2.5-1.5B-Instruct |
| Fine-tuning method | QLoRA (4-bit, Unsloth) |
| Teacher model | DeepSeek V3 (671B parameters) |
| Training examples | 672 (distilled from teacher) |
| Train / eval split | 90% / 10% (≈ 605 train, 67 eval) |

## Training Metrics (Colab, Day 19)

| Metric | Value |
|--------|-------|
| Final train loss | _(fill in after Colab run)_ |
| Schema pass rate | _(fill in)_ |
| Execution mode accuracy | _(fill in)_ |
| Training time | _(fill in)_ minutes |
| Hardware | Tesla T4 GPU (Colab free tier) |
| Adapter size (zip) | _(fill in)_ MB |
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
  "execution_mode": "sequential | parallel | hybrid",
  "agent_topology": ["AgentA", "AgentB"],
  "expected_accuracy": 0.0,
  "rationale": "string explaining the topology decision"
}
```

## Adapter Location

```
models/brain_cores/core3_architecture_advisor_adapter/
```

Files excluded by `.gitignore`:
- `adapter_model.safetensors` — excluded via `*.safetensors`
- `tokenizer.json` — excluded via `models/brain_cores/*/tokenizer.json`

## Integration

```python
from api.brain.cores.architecture_advisor import get_architecture_advisor

core3 = get_architecture_advisor()
result = core3.advise("Run image and text analysis simultaneously")
# Returns: {"execution_mode": "parallel", "agent_topology": [...], ...}
```

## Status

| Step | Status |
|------|--------|
| Dataset generation (672 examples) | DONE (Day 13) |
| Colab fine-tuning notebook | DONE (Day 19) |
| Adapter training on T4 GPU | Pending Colab run |
| Integration code (`architecture_advisor.py`) | DONE (Day 19) |
| Offline schema tests | DONE (Day 19) |
| Full inference test (`--infer`) | Pending (requires adapter + Qwen download) |
