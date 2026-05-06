# Distillation Micro Batch Report

```
=======================================================
DISTILLATION MICRO BATCH REPORT
=======================================================
Teacher model: deepseek-chat (V3)
Total API calls: 150
Successful: 150 / 150
Failed: 0 / 150
Schema pass rate: 100.0%

Per-core breakdown:
  Task Understander: 50 calls, 50 passed, 0 failed
  Domain Classifier: 50 calls, 50 passed, 0 failed
  Architecture Advisor: 50 calls, 50 passed, 0 failed

Token usage:
  Input tokens:  19,172
  Output tokens: 9,500
  Cost (50 probs): $0.0156

Performance:
  Average response time: 1.5s
  Total elapsed time: 3.84 minutes

Projected cost for 1000 prompts (3 cores each):
  Estimated: $0.31

ACCEPTANCE GATE:
  Pass rate > 95%:           YES  (100.0%)
  Cost < $1 for 1000 prompts:YES  ($0.31)
  Avg response < 10s:        YES  (1.5s)

OVERALL: PROCEED TO FULL DATASET
=======================================================
```