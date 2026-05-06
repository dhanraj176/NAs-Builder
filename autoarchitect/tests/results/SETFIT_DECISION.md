# Day 16 — SetFit Evaluation Decision

## Decision: SKIP SetFit — BoW + DARTSNet remains primary text trainer

## What was tested

SetFit (`sentence-transformers/all-MiniLM-L6-v2`) was evaluated as a foundation
model upgrade for the TextAgent, replacing the BoW + DARTSNet pipeline.

## Findings

**SetFit is too slow on real production data sizes.**

- With 400 training examples and `num_iterations=20`, SetFit generates 16,000
  contrastive pairs. On CPU this takes ~90 minutes per training run.
- The "26x speedup" claim for SetFit only holds in FEW-SHOT scenarios
  (8–16 examples per class). AutoArchitect users routinely supply 100–1000
  examples per class.
- BoW + DARTSNet trains in ~20 seconds on the same 400-example dataset.

## Compatibility issues found (patched but not merged)

- setfit 1.1.3 incompatible with transformers 5.3.0: `default_logdir` import
  removed. Patched locally in `setfit/training_args.py`.
- setfit 1.1.3 incompatible with newer `huggingface_hub`: `EntryNotFoundError`
  not caught where `requests.exceptions.RequestException` was expected.
  Patched locally in `setfit/modeling.py`.

These are upstream bugs, not fixable in this codebase.

## What was kept

The `train_with_setfit()` and updated `predict()` / `load_trained_model()`
methods added to `text_agent.py` were **reverted** to the original BoW +
DARTSNet implementation.

The `self_trainer.py` SetFit integration block was also removed.

## Path forward

Text foundation model upgrade will be revisited in a future day with a model
that is fast on CPU at production scale (e.g., TF-IDF + LightGBM, or a
distilled BERT with batch inference acceleration).
