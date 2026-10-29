# TinyDerm Lab - Deterministic Improvement Analysis

**Research Status:** Automated Evidence-Based Diagnostic Report  
**Headline Macro F1:** 0.6930  
**Balanced Accuracy:** 0.7165  
**Ordinary Accuracy:** 0.8131  
**Expected Calibration Error (ECE):** 0.0824  

---

## 1. Class Imbalance & Representation Warnings

### ⚠️ Rare Class Warnings
- **AKIEC** (Support: 32): Sample count in the holdout is very small; statistical estimates carry high variance.
- **DF** (Support: 12): Sample count in the holdout is very small; statistical estimates carry high variance.
- **VASC** (Support: 14): Sample count in the holdout is very small; statistical estimates carry high variance.

### ⚠️ Low Recall Alerts
- **DF** (Recall: 41.67%, Support: 12): Class suffers from disproportionately low detection rate.

## 2. Learning Dynamics (Overfitting / Underfitting)

✓ Training convergence remained stable without severe divergence.

## 3. Major Confusion Pairs

Off-diagonal confusion matrix pairs ranked by error magnitude:

| True Class | Predicted Class | Count | Fraction of True Class |
| :--- | :--- | :--- | :--- |
| **NV** | **MEL** | 49 | 7.35% |
| **MEL** | **NV** | 24 | 20.51% |
| **MEL** | **BKL** | 16 | 13.68% |
| **NV** | **BKL** | 14 | 2.10% |
| **BKL** | **MEL** | 14 | 12.73% |

## 4. Confidence & Calibration Analysis

- **High-Confidence Errors (Confidence ≥ 80%):** 16 samples (1.59% of holdout).
- **Low-Confidence Correct (Confidence < 50%):** 67 samples.
- **Expected Calibration Error (ECE):** 0.0824

> [!NOTE] In clinical applications, high-confidence misclassifications represent primary safety risks. Temperature scaling or uncertainty thresholding is recommended.

## 5. Deployment Footprint & Latency Summary

| Target | Latency p50 | Latency p95 | Throughput | Model Size |
| :--- | :--- | :--- | :--- | :--- |
| fp32_multithread | 0.84 ms | 1.30 ms | 1105.7 img/s | 5.83 MB |
| fp32_singlethread | 1.69 ms | 2.42 ms | 553.4 img/s | 5.83 MB |
| int8_multithread | 22.15 ms | 23.55 ms | 45.6 img/s | 1.62 MB |
| int8_singlethread | 29.73 ms | 32.25 ms | 33.4 img/s | 1.62 MB |

## 6. Actionable Research Next Steps

1. **Distillation from Specialist Teacher (Phase 2):**
   Transfer softened logits from Derm Foundation or an ensemble teacher to boost minority class sensitivity (especially DF, VASC, AKIEC).
2. **Confidence-Gated Escalation:**
   Deploy an automated referral mechanism where cases with confidence < 0.70 or entropy > 1.0 are routed to human dermatologists or cloud foundation models.
3. **Focal Loss / Balanced Softmax Exploration:**
   Evaluate whether Focal Loss or Class-Balanced Cross-Entropy yields a superior macro-F1 trade-off over linear inverse weighting.