# AutoArchitect Brain — Core 1 Fine-tuning Setup Guide

## What this does

Fine-tunes **Qwen2.5-1.5B-Instruct** on 672 DeepSeek V3-distilled examples to create
**Core 1: Task Understander** — the brain module that parses ML problem descriptions
into structured JSON (domain, complexity, intent, entities).

Uses QLoRA (4-bit) so the full fine-tune runs on a free T4 GPU in ~30-45 minutes.
Produces a ~25 MB LoRA adapter, not the full 1.5B model.

---

## Prerequisites

- Google account (for Colab)
- The file `dataset_task_understander.jsonl` from `experiments/distillation/`

---

## Step-by-Step Instructions

### 1. Open Colab

Go to: **https://colab.research.google.com**

### 2. Upload the Notebook

- Click **File → Upload notebook**
- Select `experiments/distillation/finetune_core1.ipynb`

### 3. Set GPU Runtime

- Click **Runtime → Change runtime type**
- Set **Hardware accelerator** to **T4 GPU**
- Click **Save**

### 4. Run the Notebook

- Click **Runtime → Run all**  
  OR press `Ctrl+F9`

### 5. Upload the Dataset (Step 2 cell)

When the **Step 2** cell runs, a file picker appears.  
Select `experiments/distillation/dataset_task_understander.jsonl` from your computer.

### 6. Wait for Training

Training takes approximately **30-45 minutes** on T4 GPU.  
You will see loss values logged every 10 steps.

Expected final training loss: **< 0.5**

### 7. Review Test Results (Step 9)

After training, 10 unseen problems are evaluated.  
Record the schema validation pass rate and domain accuracy.

**Good results:** Schema pass rate >= 80%, Domain accuracy >= 70%

### 8. Download the Adapter

The **Step 10** cell automatically downloads `core1_adapter.zip`.  
Extract it to get the folder `core1_task_understander_adapter/`.

### 9. Place the Adapter

Move the extracted folder to:
```
models/brain_cores/core1_task_understander_adapter/
```

Create the directory if it doesn't exist:
```
mkdir -p models/brain_cores/
```

### 10. Notify Claude Code

Tell Claude Code: **"Core 1 is ready"**

Include these metrics from Step 9:
- Final train loss
- Schema pass rate (out of 10)
- Domain accuracy (out of 10)
- Training time (minutes)
- Adapter size (MB)

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| `No GPU found` | Runtime → Change runtime type → T4 GPU |
| `CUDA out of memory` | Reduce `per_device_train_batch_size` from 4 to 2 in Step 6 |
| Unsloth install fails | Run the install cell again; Colab sometimes needs a retry |
| Dataset not found | Make sure you uploaded `dataset_task_understander.jsonl` in Step 2 |
| JSON parse errors in Step 9 | Normal for first run; schema pass rate >= 70% is acceptable |

---

## What Comes Next

| Core | Dataset | Status |
|------|---------|--------|
| Core 1: Task Understander | `dataset_task_understander.jsonl` | **This notebook** |
| Core 2: Domain Classifier | `dataset_domain_classifier.jsonl` | Day 19 |
| Core 3: Architecture Advisor | `dataset_architecture_advisor.jsonl` | Day 20 |

Each subsequent core uses the same notebook structure — only the dataset and
system prompt change.

---

## Technical Details

| Setting | Value |
|---------|-------|
| Base model | Qwen/Qwen2.5-1.5B-Instruct |
| Quantization | 4-bit (QLoRA) |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate/up/down_proj |
| Epochs | 3 |
| Batch size | 4 × 4 gradient accumulation = 16 effective |
| Learning rate | 2e-4 with cosine decay |
| Optimizer | adamw_8bit |
| Training examples | ~605 (90% of 672) |
| Eval examples | ~67 (10% of 672) |
