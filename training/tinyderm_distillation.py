"""Confidence-weighted distillation helpers for TinyDerm."""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F


def teacher_confidence(probabilities: torch.Tensor) -> torch.Tensor:
    """Return normalized-entropy confidence in [0, 1] for class probabilities."""
    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise ValueError("Teacher probabilities must have shape [batch, classes] with at least two classes")
    if not torch.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError("Teacher probabilities must be finite and nonnegative")
    row_sums = probabilities.sum(dim=1)
    if (row_sums <= 0).any():
        raise ValueError("Each teacher probability row must have positive mass")
    probs = probabilities / row_sums.unsqueeze(1)
    entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=1)
    return (1.0 - entropy / math.log(probabilities.shape[1])).clamp(0.0, 1.0)


def confidence_weighted_kl(student_logits: torch.Tensor, teacher_probabilities: torch.Tensor) -> torch.Tensor:
    """Average per-example KL weighted by teacher certainty, with safe zero-confidence behavior."""
    if student_logits.shape != teacher_probabilities.shape:
        raise ValueError("Student logits and teacher probabilities must have identical shapes")
    weights = teacher_confidence(teacher_probabilities)
    per_example = F.kl_div(
        F.log_softmax(student_logits.float(), dim=1),
        (teacher_probabilities / teacher_probabilities.sum(dim=1, keepdim=True)).float(),
        reduction="none",
    ).sum(dim=1)
    return (per_example * weights).sum() / weights.sum().clamp_min(1e-8)


def calibration_metrics(labels, probabilities, classes, melanoma_label="MEL", bins=10):
    """Return multiclass Brier/NLL and MEL Brier/ECE without fitting on evaluation labels."""
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape != (len(labels), len(classes)):
        raise ValueError("Probability shape must match labels and class order")
    if not np.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError("Probabilities must be finite and nonnegative")
    row_sums = probabilities.sum(axis=1)
    if (row_sums <= 0).any():
        raise ValueError("Each probability row must have positive mass")
    probabilities = probabilities / row_sums[:, None]
    class_to_index = {label: i for i, label in enumerate(classes)}
    if melanoma_label not in class_to_index or any(label not in class_to_index for label in labels):
        raise ValueError("Labels must be present in the class order")
    truth = np.zeros_like(probabilities)
    truth[np.arange(len(labels)), [class_to_index[label] for label in labels]] = 1.0
    eps = 1e-12
    nll = -np.log(np.clip(probabilities[np.arange(len(labels)), truth.argmax(axis=1)], eps, 1.0)).mean()
    mel_idx = class_to_index[melanoma_label]
    mel_truth = truth[:, mel_idx]
    mel_prob = probabilities[:, mel_idx]
    ece = 0.0
    for lo in np.linspace(0.0, 1.0, bins + 1)[:-1]:
        hi = lo + 1.0 / bins
        in_bin = (mel_prob >= lo) & ((mel_prob < hi) if hi < 1.0 else (mel_prob <= hi))
        if in_bin.any():
            ece += in_bin.mean() * abs(float(mel_prob[in_bin].mean() - mel_truth[in_bin].mean()))
    return {"multiclass_brier": float(np.mean(np.sum((probabilities - truth) ** 2, axis=1))),
            "multiclass_nll": float(nll), "melanoma_brier": float(np.mean((mel_prob - mel_truth) ** 2)),
            "melanoma_ece_10bin": float(ece)}

