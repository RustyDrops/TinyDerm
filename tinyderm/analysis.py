"""Deterministic Improvement Analysis module for TinyDerm Lab.

Generates improvement_report.md based on empirical evidence without calling an LLM.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

OFFICIAL_CLASSES = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]


def generate_improvement_report(
    metrics: Dict[str, Any],
    predictions: List[Dict[str, Any]],
    epochs_history: List[Dict[str, Any]],
    benchmark_results: Dict[str, Any],
    output_dir: str | Path,
) -> str:
    """Generates deterministic improvement report analyzing error modes and bottlenecks."""
    output_dir = Path(output_dir)
    report_path = output_dir / "improvement_report.md"

    pred_df = pd.DataFrame(predictions)
    per_class = metrics.get("per_class", {})
    cm = np.array(metrics.get("confusion_matrix", []))

    lines = [
        "# TinyDerm Lab - Deterministic Improvement Analysis",
        "",
        "**Research Status:** Automated Evidence-Based Diagnostic Report  ",
        f"**Headline Macro F1:** {metrics.get('macro_f1', 0.0):.4f}  ",
        f"**Balanced Accuracy:** {metrics.get('balanced_accuracy', 0.0):.4f}  ",
        f"**Ordinary Accuracy:** {metrics.get('accuracy', 0.0):.4f}  ",
        f"**Expected Calibration Error (ECE):** {metrics.get('expected_calibration_error', 0.0):.4f}  ",
        "",
        "---",
        "",
        "## 1. Class Imbalance & Representation Warnings",
        "",
    ]

    # Imbalance and rare class checks
    rare_warnings = []
    low_recall_warnings = []
    for c in OFFICIAL_CLASSES:
        c_stats = per_class.get(c, {})
        support = c_stats.get("support", 0)
        sensitivity = c_stats.get("sensitivity", 0.0)

        if support < 50:
            rare_warnings.append(f"- **{c}** (Support: {support}): Sample count in the holdout is very small; statistical estimates carry high variance.")
        if sensitivity < 0.50:
            low_recall_warnings.append(f"- **{c}** (Recall: {sensitivity:.2%}, Support: {support}): Class suffers from disproportionately low detection rate.")

    if rare_warnings:
        lines.append("### ⚠️ Rare Class Warnings")
        lines.extend(rare_warnings)
        lines.append("")
    if low_recall_warnings:
        lines.append("### ⚠️ Low Recall Alerts")
        lines.extend(low_recall_warnings)
        lines.append("")

    # Overfitting / Underfitting analysis
    lines.extend([
        "## 2. Learning Dynamics (Overfitting / Underfitting)",
        "",
    ])
    if epochs_history:
        train_losses = [e["train_loss"] for e in epochs_history if e.get("train_loss") is not None]
        val_losses = [e["val_loss"] for e in epochs_history if e.get("val_loss") is not None]
        val_f1s = [e["val_f1"] for e in epochs_history if e.get("val_f1") is not None]

        if len(val_losses) >= 3:
            recent_val_loss = val_losses[-3:]
            recent_train_loss = train_losses[-3:]
            if recent_val_loss[2] > recent_val_loss[0] and recent_train_loss[2] < recent_train_loss[0]:
                lines.append("> [!WARNING] **Overfitting Detected**: Training loss continued decreasing while validation loss increased over recent epochs.")
            elif val_f1s[-1] < 0.40 and train_losses[-1] > 1.2:
                lines.append("> [!WARNING] **Underfitting Warning**: Model capacity or optimization appears constrained; both training and validation metrics remain low.")
            else:
                lines.append("✓ Training convergence remained stable without severe divergence.")
    else:
        lines.append("No multi-epoch history available for convergence check.")
    lines.append("")

    # Top Confusion Pairs
    lines.extend([
        "## 3. Major Confusion Pairs",
        "",
        "Off-diagonal confusion matrix pairs ranked by error magnitude:",
        "",
        "| True Class | Predicted Class | Count | Fraction of True Class |",
        "| :--- | :--- | :--- | :--- |",
    ])

    confusion_pairs = []
    if len(cm) == len(OFFICIAL_CLASSES):
        for i, true_c in enumerate(OFFICIAL_CLASSES):
            row_total = np.sum(cm[i, :])
            for j, pred_c in enumerate(OFFICIAL_CLASSES):
                if i != j and cm[i, j] > 0:
                    fraction = cm[i, j] / max(row_total, 1)
                    confusion_pairs.append((true_c, pred_c, int(cm[i, j]), fraction))

        confusion_pairs.sort(key=lambda x: x[2], reverse=True)
        for true_c, pred_c, count, fraction in confusion_pairs[:5]:
            lines.append(f"| **{true_c}** | **{pred_c}** | {count} | {fraction:.2%} |")
    lines.append("")

    # Confidence Warnings
    lines.extend([
        "## 4. Confidence & Calibration Analysis",
        "",
    ])
    if not pred_df.empty:
        high_conf_errors = pred_df[(~pred_df["correct"]) & (pred_df["confidence"] >= 0.80)]
        low_conf_correct = pred_df[(pred_df["correct"]) & (pred_df["confidence"] < 0.50)]

        lines.append(f"- **High-Confidence Errors (Confidence ≥ 80%):** {len(high_conf_errors)} samples ({len(high_conf_errors)/len(pred_df):.2%} of holdout).")
        lines.append(f"- **Low-Confidence Correct (Confidence < 50%):** {len(low_conf_correct)} samples.")
        lines.append(f"- **Expected Calibration Error (ECE):** {metrics.get('expected_calibration_error', 0.0):.4f}")
        if len(high_conf_errors) > 0:
            lines.append("")
            lines.append("> [!NOTE] In clinical applications, high-confidence misclassifications represent primary safety risks. Temperature scaling or uncertainty thresholding is recommended.")
    lines.append("")

    # Runtime and CPU Benchmarking
    lines.extend([
        "## 5. Deployment Footprint & Latency Summary",
        "",
        "| Target | Latency p50 | Latency p95 | Throughput | Model Size |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ])
    for key, data in benchmark_results.items():
        if isinstance(data, dict):
            lines.append(
                f"| {key} | {data.get('p50_latency_ms', 0):.2f} ms | {data.get('p95_latency_ms', 0):.2f} ms | "
                f"{data.get('throughput_images_per_sec', 0):.1f} img/s | {data.get('model_size_mb', 0):.2f} MB |"
            )
    lines.append("")

    # Actionable Recommendations
    lines.extend([
        "## 6. Actionable Research Next Steps",
        "",
        "1. **Distillation from Specialist Teacher (Phase 2):**",
        "   Transfer softened logits from Derm Foundation or an ensemble teacher to boost minority class sensitivity (especially DF, VASC, AKIEC).",
        "2. **Confidence-Gated Escalation:**",
        "   Deploy an automated referral mechanism where cases with confidence < 0.70 or entropy > 1.0 are routed to human dermatologists or cloud foundation models.",
        "3. **Focal Loss / Balanced Softmax Exploration:**",
        "   Evaluate whether Focal Loss or Class-Balanced Cross-Entropy yields a superior macro-F1 trade-off over linear inverse weighting.",
    ])

    report_content = "\n".join(lines)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    return report_content
