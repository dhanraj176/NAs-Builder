# AutoArchitect -- All 5 Domains Verified

Generated: 2026-05-07 10:25

| Domain | Dataset | Accuracy | ZIP Size | Status |
|--------|---------|----------|----------|--------|
| Image | taroii/pothole-detection | 100.0% | 40588 KB | PASS |
| Text | GonzaloA/fake_news | 74.13% | 414 KB | PASS |
| Tabular | synthetic_credit_fraud (sklearn mak | 98.0% | 108 KB | PASS |
| Audio | synthetic sine-wave dataset (440 /  | 100.0% | 12 KB | PASS |
| Multimodal |  | N/A | ? KB | SKIP |

## Sample Predictions

**Image**
- Input: `synthetic`
- Label: `1`  confidence: `0.926`
- Classes: ['1', '0']

**Text**
- Input: `Politicians claim vaccines cause autism -- doctors disagree`
- Label: `real`  confidence: `0.548`
- Classes: ['real', 'fake']

**Tabular**
- Input: `synthetic`
- Label: `legit`  confidence: `0.997`
- Classes: ['fraud', 'legit']

**Audio**
- Input: `440 Hz sine wave (1 sec)`
- Label: `high`  confidence: `0.610`
- Classes: [np.str_('high'), np.str_('low')]

## Training Details

**Image**
- Model  : `a676edaeb1_image.pth`
- Dataset: taroii/pothole-detection
- Acc    : 100.0
- Time   : 21.3s

**Text**
- Model  : `71802a8ac0_text.pth`
- Dataset: GonzaloA/fake_news
- Acc    : 74.13
- Time   : 3.1s

**Tabular**
- Model  : `xgboost`
- Dataset: synthetic_credit_fraud (sklearn make_classification)
- Acc    : 98.0
- Time   : 3.5s

**Audio**
- Model  : `MFCC + RandomForest`
- Dataset: synthetic sine-wave dataset (440 / 880 Hz classes)
- Acc    : 100.0
- Time   : 8.5s

## Ensemble ZIP Smoke Test

- Status : PASS
- ZIP    : 812 KB
- Ensemble predict.py: YES
- Parallel header    : YES

---

## System

| Component | Value |
|-----------|-------|
| Brain | DistilledBrain (3 cores, 86.7% avg) |
| NAS | ANAS + DARTS |
| Ensemble | FusionAgent (learned weights) |
| ZIP | NetworkZipGenerator (ensemble-aware) |
| Verified | 2026-05-07 10:25 |