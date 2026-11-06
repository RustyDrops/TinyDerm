"""Cross-fit class-bias calibration on saved lesion-grouped OOF probabilities."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from hunch_experiment import development_rows
from tinyderm.dataset import OFFICIAL_CLASSES


def metric(y, probability):
    pred = probability.argmax(axis=1)
    matrix = np.bincount(y * len(OFFICIAL_CLASSES) + pred, minlength=len(OFFICIAL_CLASSES) ** 2).reshape(len(OFFICIAL_CLASSES), len(OFFICIAL_CLASSES))
    true_positive = np.diag(matrix).astype(np.float64)
    precision = true_positive / np.maximum(matrix.sum(axis=0), 1)
    recall = true_positive / np.maximum(matrix.sum(axis=1), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return {
        "macro_f1": float(f1.mean()),
        "balanced_accuracy": float(recall.mean()),
        "per_class": {name: {"precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i]), "support": int(matrix[i].sum())} for i, name in enumerate(OFFICIAL_CLASSES)},
    }


def fit_bias(log_prob, y, grid, penalty, rounds):
    bias = np.zeros(len(OFFICIAL_CLASSES), dtype=np.float64)
    best_value = metric(y, log_prob + bias).get("macro_f1")
    for _ in range(rounds):
        changed = False
        for cls in range(len(OFFICIAL_CLASSES)):
            current = bias[cls]
            best_class_value = best_value - penalty * float(np.dot(bias, bias))
            for candidate in grid:
                bias[cls] = candidate
                score = metric(y, log_prob + bias)["macro_f1"] - penalty * float(np.dot(bias, bias))
                if score > best_class_value:
                    best_class_value = score
                    best_value = score + penalty * float(np.dot(bias, bias))
                    current = candidate
            bias[cls] = current
            changed = changed or current != 0.0
    return bias


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=31415)
    parser.add_argument("--epoch", type=int, default=None, help="one-based epoch to select from fine-tune OOF arrays; default uses the final epoch")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--penalty", type=float, default=0.02)
    args = parser.parse_args()
    frame = development_rows(args.manifest)
    y = np.array([OFFICIAL_CLASSES.index(c) for c in frame.class_label])
    groups = frame.lesion_id.to_numpy()
    aligned_splits = []
    arrays = {name: np.empty((len(frame), len(OFFICIAL_CLASSES)), dtype=np.float64) for name in ("baseline", "hunch_assisted", "ensemble")}
    fold_index = np.full(len(frame), -1, dtype=np.int32)
    covered = np.zeros(len(frame), dtype=np.int8)
    all_indices = np.arange(len(frame))
    for fold in range(1, args.folds + 1):
        prediction_path = args.predictions / f"fold{fold}_oof_predictions.npz"
        if not prediction_path.exists():
            prediction_path = args.predictions / f"fold{fold}_tta_predictions.npz"
        saved = np.load(prediction_path)
        saved_idx = saved["held_indices"]
        if np.any(saved_idx < 0) or np.any(saved_idx >= len(frame)):
            raise ValueError(f"Fold {fold} contains invalid held-out indices")
        baseline = saved["baseline"]
        hunch_assisted = saved["hunch_assisted"]
        if baseline.ndim == 3:
            selected_epoch = args.epoch - 1 if args.epoch is not None else baseline.shape[0] - 1
            if selected_epoch < 0 or selected_epoch >= baseline.shape[0]:
                raise ValueError(f"Requested epoch {args.epoch} is not present in fold {fold}")
            baseline = baseline[selected_epoch]
            hunch_assisted = hunch_assisted[selected_epoch]
        elif args.epoch is not None:
            raise ValueError("--epoch can only be used with per-epoch fine-tuning predictions")
        arrays["baseline"][saved_idx] = baseline
        arrays["hunch_assisted"][saved_idx] = hunch_assisted
        arrays["ensemble"][saved_idx] = (baseline + hunch_assisted) / 2
        fold_index[saved_idx] = fold
        covered[saved_idx] += 1
        aligned_splits.append((np.setdiff1d(all_indices, saved_idx), saved_idx))
    if not np.all(covered == 1):
        raise ValueError("Saved folds must cover every development image exactly once")

    grid = np.linspace(-1.5, 1.5, 21)
    output = {"scope": "second-level cross-fitted calibration over lesion-grouped OOF development predictions", "calibration": {"method": "class-specific additive log-probability bias, coordinate search", "grid": grid.tolist(), "rounds": args.rounds, "l2_penalty": args.penalty}, "methods": {}}
    for name, probability in arrays.items():
        raw_probability = np.log(np.maximum(probability, 1e-8))
        adjusted = np.empty_like(probability)
        fold_details = []
        biases = []
        for fold, (train_idx, held_idx) in enumerate(aligned_splits, start=1):
            bias = fit_bias(raw_probability[train_idx], y[train_idx], grid, args.penalty, args.rounds)
            adjusted[held_idx] = np.exp(raw_probability[held_idx] + bias - (raw_probability[held_idx] + bias).max(axis=1, keepdims=True))
            adjusted[held_idx] /= adjusted[held_idx].sum(axis=1, keepdims=True)
            biases.append(bias.tolist())
            fold_details.append({"fold": fold, "heldout_images": len(held_idx), "uncalibrated": metric(y[held_idx], probability[held_idx]), "calibrated": metric(y[held_idx], adjusted[held_idx])})
        output["methods"][name] = {"uncalibrated": metric(y, probability), "cross_fitted_calibrated": metric(y, adjusted), "fold_biases": biases, "folds": fold_details}

    adaptive = np.empty_like(arrays["baseline"])
    adaptive_raw = np.empty_like(arrays["baseline"])
    blend_details = []
    blend_grid = np.linspace(0.0, 1.0, 11)
    for fold, (train_idx, held_idx) in enumerate(aligned_splits, start=1):
        best_alpha, best_macro = 0.0, -1.0
        for alpha in blend_grid:
            candidate = (1 - alpha) * arrays["baseline"][train_idx] + alpha * arrays["hunch_assisted"][train_idx]
            score = metric(y[train_idx], candidate)["macro_f1"]
            if score > best_macro or (score == best_macro and abs(alpha - 0.5) < abs(best_alpha - 0.5)):
                best_alpha, best_macro = float(alpha), score
        train_blend = (1 - best_alpha) * arrays["baseline"][train_idx] + best_alpha * arrays["hunch_assisted"][train_idx]
        bias = fit_bias(np.log(np.maximum(train_blend, 1e-8)), y[train_idx], grid, args.penalty, args.rounds)
        held_blend = (1 - best_alpha) * arrays["baseline"][held_idx] + best_alpha * arrays["hunch_assisted"][held_idx]
        adaptive_raw[held_idx] = held_blend
        log_adjusted = np.log(np.maximum(held_blend, 1e-8)) + bias
        adjusted_prob = np.exp(log_adjusted - log_adjusted.max(axis=1, keepdims=True))
        adjusted_prob /= adjusted_prob.sum(axis=1, keepdims=True)
        adaptive[held_idx] = adjusted_prob
        blend_details.append({"fold": fold, "hunch_weight": best_alpha, "train_macro_f1_at_selected_weight": best_macro, "heldout_images": len(held_idx), "bias": bias.tolist(), "raw": metric(y[held_idx], held_blend), "calibrated": metric(y[held_idx], adjusted_prob)})
    output["methods"]["cross_fitted_adaptive_blend"] = {"raw": metric(y, adaptive_raw), "calibrated": metric(y, adaptive), "folds": blend_details}
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    for name, result in output["methods"].items():
        raw = result.get("uncalibrated", result.get("raw"))
        calibrated = result.get("cross_fitted_calibrated", result.get("calibrated"))
        print(f"{name}: raw {raw['macro_f1']:.4f}/{raw['balanced_accuracy']:.4f}; calibrated {calibrated['macro_f1']:.4f}/{calibrated['balanced_accuracy']:.4f}", flush=True)


if __name__ == "__main__":
    main()

