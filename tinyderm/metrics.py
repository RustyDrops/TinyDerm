"""Evaluation metrics calculation for TinyDerm Lab."""

from typing import Any, Dict, List, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

OFFICIAL_CLASSES = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]


def compute_expected_calibration_error(
    confidences: np.ndarray,
    correctness: np.ndarray,
    n_bins: int = 10,
) -> Tuple[float, List[Dict[str, float]]]:
    """Computes Expected Calibration Error (ECE) and bin details."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_data = []

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(correctness[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin

            bin_data.append({
                "bin_center": float(round((bin_lower + bin_upper) / 2, 3)),
                "accuracy": float(round(accuracy_in_bin, 4)),
                "confidence": float(round(avg_confidence_in_bin, 4)),
                "count": int(np.sum(in_bin)),
            })
        else:
            bin_data.append({
                "bin_center": float(round((bin_lower + bin_upper) / 2, 3)),
                "accuracy": 0.0,
                "confidence": float(round((bin_lower + bin_upper) / 2, 3)),
                "count": 0,
            })

    return float(round(ece, 4)), bin_data


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    classes: List[str] = OFFICIAL_CLASSES,
) -> Dict[str, Any]:
    """Computes comprehensive classification metrics."""
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    balanced_acc = float(balanced_accuracy_score(y_true, y_pred))
    acc = float(accuracy_score(y_true, y_pred))
    macro_prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))

    # Per-class sensitivity/recall and specificity
    per_class_metrics = {}
    for i, c in enumerate(classes):
        tp = int(cm[i, i])
        fn = int(np.sum(cm[i, :]) - tp)
        fp = int(np.sum(cm[:, i]) - tp)
        tn = int(np.sum(cm) - tp - fn - fp)

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1_c = float(f1_score(y_true == i, y_pred == i, zero_division=0))

        per_class_metrics[c] = {
            "sensitivity": float(round(sensitivity, 4)),
            "specificity": float(round(specificity, 4)),
            "f1": float(round(f1_c, 4)),
            "support": int(tp + fn),
        }

    # Calibration
    confidences = np.max(y_prob, axis=1)
    correctness = (y_true == y_pred).astype(float)
    ece, calibration_bins = compute_expected_calibration_error(confidences, correctness)

    # ROC-AUC where possible
    roc_auc = None
    try:
        # Check if all classes present in y_true
        if len(np.unique(y_true)) == len(classes):
            roc_auc = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
    except Exception:
        pass

    return {
        "macro_f1": round(macro_f1, 4),
        "balanced_accuracy": round(balanced_acc, 4),
        "accuracy": round(acc, 4),
        "macro_precision": round(macro_prec, 4),
        "macro_recall": round(macro_rec, 4),
        "roc_auc_ovr": round(roc_auc, 4) if roc_auc is not None else None,
        "expected_calibration_error": ece,
        "calibration_bins": calibration_bins,
        "per_class": per_class_metrics,
        "confusion_matrix": cm.tolist(),
        "classes": classes,
    }
