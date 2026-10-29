# TinyDerm Lab - Deterministic Improvement Analysis

**Research Status:** Automated Evidence-Based Diagnostic Report  
**Headline Macro F1:** 0.6503  
**Balanced Accuracy:** 0.6835  
**Ordinary Accuracy:** 0.8121  
**Expected Calibration Error (ECE):** 0.0844  

---

## 1. Class Imbalance & Representation Warnings

### ⚠️ Rare Class Warnings
- **AKIEC** (Support: 32): Sample count in the holdout is very small; statistical estimates carry high variance.
- **DF** (Support: 12): Sample count in the holdout is very small; statistical estimates carry high variance.
- **VASC** (Support: 14): Sample count in the holdout is very small; statistical estimates carry high variance.

### ⚠️ Low Recall Alerts
- **DF** (Recall: 25.00%, Support: 12): Class suffers from disproportionately low detection rate.

## 2. Learning Dynamics (Overfitting / Underfitting)

✓ Training convergence remained stable without severe divergence.

## 3. Major Confusion Pairs

Off-diagonal confusion matrix pairs ranked by error magnitude:

| True Class | Predicted Class | Count | Fraction of True Class |
| :--- | :--- | :--- | :--- |
| **NV** | **MEL** | 44 | 6.60% |
| **MEL** | **NV** | 23 | 19.66% |
| **NV** | **BKL** | 22 | 3.30% |
| **MEL** | **BKL** | 12 | 10.26% |
| **BKL** | **NV** | 11 | 10.00% |

## 4. Confidence & Calibration Analysis

- **High-Confidence Errors (Confidence ≥ 80%):** 78 samples (7.75% of holdout).
- **Low-Confidence Correct (Confidence < 50%):** 11 samples.
- **Expected Calibration Error (ECE):** 0.0844

> [!NOTE] In clinical applications, high-confidence misclassifications represent primary safety risks. Temperature scaling or uncertainty thresholding is recommended.

## 5. Deployment Footprint & Latency Summary

| Target | Latency p50 | Latency p95 | Throughput | Model Size |
| :--- | :--- | :--- | :--- | :--- |
| fp32_multithread | 0.79 ms | 0.84 ms | 1255.0 img/s | 5.83 MB |
| fp32_singlethread | 1.51 ms | 1.59 ms | 655.7 img/s | 5.83 MB |
| int8_multithread | 21.58 ms | 22.44 ms | 46.7 img/s | 1.62 MB |
| int8_singlethread | 26.21 ms | 30.28 ms | 37.3 img/s | 1.62 MB |

## 6. Actionable Research Next Steps

1. **Distillation from Specialist Teacher (Phase 2):**
   Transfer softened logits from Derm Foundation or an ensemble teacher to boost minority class sensitivity (especially DF, VASC, AKIEC).
2. **Confidence-Gated Escalation:**
   Deploy an automated referral mechanism where cases with confidence < 0.70 or entropy > 1.0 are routed to human dermatologists or cloud foundation models.
3. **Focal Loss / Balanced Softmax Exploration:**
   Evaluate whether Focal Loss or Class-Balanced Cross-Entropy yields a superior macro-F1 trade-off over linear inverse weighting.