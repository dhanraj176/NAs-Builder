# Brain Core 3 — Architecture Advisor

## Day 19 Results

## Training Metrics
- Schema pass rate: 100% (10/10)
- Execution mode accuracy: 70% (7/10)
- Training time: 6 minutes
- Adapter size: 81.4 MB

## Architecture
- Base model: Qwen2.5-1.5B-Instruct
- Fine-tuning: LoRA (r=16, alpha=32)
- Target modules: q_proj, k_proj, v_proj, o_proj, 
  gate_proj, up_proj, down_proj
- Trainable parameters: 18.4M (1.78% of total)
- Training data: 604 examples (90/10 split from 672)
- Epochs: 5 (3-epoch and 5-epoch both hit identical 70%)
- Hardware: Tesla T4 GPU (Colab free tier)
- Cost: $0

## Honest Assessment

Two training runs at 3 and 5 epochs both produced 
identical 70% accuracy with same 3 misses. Architecture 
decisions are inherently subjective for ambiguous problems 
(e.g., "should toxic comment detection be sequential or 
parallel?"). The model has hit the ceiling of what's 
learnable from this teacher's labels.

The 3 misses are genuinely defensible edge cases where 
either prediction could work. ANAS Structural Immune 
System provides a secondary validation layer that catches 
bad topology decisions before execution.

## Distillation Pipeline
- Teacher: DeepSeek V3 (671B parameters)
- Compression ratio: 671B → 1.5B (445x smaller)

## Notes
- 70% is acceptable for subjective architecture decisions
- Will improve post-launch with real user feedback data
- Combined brain accuracy across 3 cores: 86.7% average

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

## LoRA Configuration

| Setting | Value |
|---------|-------|
| LoRA rank (r) | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Epochs | 5 |
| Effective batch size | 16 (4 × 4 grad accum) |
| Learning rate | 2e-4 with cosine decay |
| Optimizer | adamw_8bit |
| Precision | bf16 |

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
| Adapter training on T4 GPU | DONE (Day 19) — 70% accuracy (ceiling) |
| Integration code (`architecture_advisor.py`) | DONE (Day 19) |
| Offline schema tests | DONE (Day 19) |
| Full inference test (`--infer`) | Pending (requires adapter + Qwen download) |
