# Brain Core 2 — Domain Classifier

## Day 19 Results

## Training Metrics (Colab fine-tuning)
- Schema pass rate: 100% (10/10)
- Primary agent accuracy: 100% (10/10) — PERFECT
- Training time: 6 minutes
- Adapter size: 81.4 MB

## Architecture
- Base model: Qwen2.5-1.5B-Instruct
- Fine-tuning: LoRA (r=16, alpha=32)
- Target modules: q_proj, k_proj, v_proj, o_proj, 
  gate_proj, up_proj, down_proj
- Trainable parameters: 18.4M (1.78% of total)
- Training data: 604 examples (90/10 split from 672)
- Validation data: 68 examples
- Epochs: 5
- Hardware: Tesla T4 GPU (Colab free tier)
- Cost: $0

## Distillation Pipeline
- Teacher: DeepSeek V3 (671B parameters)
- Compression ratio: 671B → 1.5B (445x smaller)

## Notes
- Perfect 10/10 routing accuracy on unseen problems
- All 7 agent types correctly classified:
  ImageAgent, TextAgent, TabularAgent, AudioAgent,
  MultimodalAgent, MedicalAgent, SecurityAgent
- Core 2 is the most critical brain core — handles 
  agent routing decisions for the entire system

## Output Schema

The model outputs structured JSON with these fields:

```json
{
  "primary_agent": "ImageAgent | TextAgent | TabularAgent | AudioAgent | MultimodalAgent | MedicalAgent | SecurityAgent",
  "secondary_agents": ["list", "of", "agent", "names"],
  "confidence": 0.0,
  "reasoning": "string explaining the routing decision"
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
models/brain_cores/core2_domain_classifier_adapter/
```

Files excluded by `.gitignore`:
- `adapter_model.safetensors` — excluded via `*.safetensors`
- `tokenizer.json` — excluded via `models/brain_cores/*/tokenizer.json`

## Integration

```python
from api.brain.cores.domain_classifier import get_domain_classifier

core2 = get_domain_classifier()
result = core2.classify("Detect defects in PCB board images")
# Returns: {"primary_agent": "ImageAgent", "secondary_agents": [...], ...}
```

## Status

| Step | Status |
|------|--------|
| Dataset generation (672 examples) | DONE (Day 13) |
| Colab fine-tuning notebook | DONE (Day 19) |
| Adapter training on T4 GPU | DONE (Day 19) — 100% accuracy |
| Integration code (`domain_classifier.py`) | DONE (Day 19) |
| Offline schema tests | DONE (Day 19) |
| Full inference test (`--infer`) | Pending (requires adapter + Qwen download) |
