"""Evaluation and Holdout Governance module for TinyDerm Lab."""

import datetime
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import scipy.stats
import torch
from torch.utils.data import DataLoader

from tinyderm.augment import get_eval_transforms
from tinyderm.dataset import CLASS_TO_IDX, DermDataset, IDX_TO_CLASS, OFFICIAL_CLASSES
from tinyderm.metrics import compute_classification_metrics
from tinyderm.provenance import compute_file_hash
from tinyderm.split import mark_holdout_consumed

logger = logging.getLogger(__name__)


def evaluate_model_on_dataset(
    model: torch.nn.Module,
    dataset: DermDataset,
    device: torch.device,
    batch_size: int = 64,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Evaluates model, returning full classification metrics and row-level prediction details."""
    model = model.to(device)
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    all_targets = []
    all_preds = []
    all_probs = []
    all_image_ids = []
    all_lesion_ids = []

    with torch.no_grad():
        for images, targets, img_ids, lesion_ids in loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_targets.append(targets.numpy())
            all_preds.append(preds)
            all_probs.append(probs)
            all_image_ids.extend(img_ids)
            all_lesion_ids.extend(lesion_ids)

    y_true = np.concatenate(all_targets, axis=0)
    y_pred = np.concatenate(all_preds, axis=0)
    y_prob = np.concatenate(all_probs, axis=0)

    metrics = compute_classification_metrics(y_true, y_pred, y_prob)

    predictions = []
    for i in range(len(y_true)):
        t_cls = IDX_TO_CLASS[int(y_true[i])]
        p_cls = IDX_TO_CLASS[int(y_pred[i])]
        p_vec = {OFFICIAL_CLASSES[j]: float(round(y_prob[i, j], 5)) for j in range(len(OFFICIAL_CLASSES))}
        prob_true = float(round(y_prob[i, int(y_true[i])], 5))
        prob_pred = float(round(y_prob[i, int(y_pred[i])], 5))
        conf = float(round(np.max(y_prob[i]), 5))
        entropy = float(round(scipy.stats.entropy(y_prob[i] + 1e-12), 4))
        is_corr = bool(y_true[i] == y_pred[i])

        predictions.append({
            "image_id": all_image_ids[i],
            "lesion_id": all_lesion_ids[i],
            "true_class": t_cls,
            "predicted_class": p_cls,
            "probability_true": prob_true,
            "probability_predicted": prob_pred,
            "confidence": conf,
            "entropy": entropy,
            "correct": is_corr,
            "probabilities": p_vec,
        })

    return metrics, predictions


def run_sealed_holdout_evaluation(
    model: torch.nn.Module,
    manifest_df: pd.DataFrame,
    splits_dir: str | Path,
    run_id: str,
    output_dir: str | Path,
    db,
    device: torch.device,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Executes the single, sealed 10% holdout benchmark with strict verification."""
    splits_dir = Path(splits_dir)
    output_dir = Path(output_dir) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_csv = splits_dir / "split_manifest.csv"
    manifest_sha = splits_dir / "split_manifest.sha256"
    lock_file = splits_dir / "holdout_lock.json"

    # Step 1: Verify split hash
    if not manifest_csv.exists() or not lock_file.exists():
        raise FileNotFoundError("Split manifest or holdout lock file missing!")

    current_hash = compute_file_hash(manifest_csv)
    with open(lock_file, "r", encoding="utf-8") as f:
        lock_data = json.load(f)

    if lock_data.get("split_hash") != current_hash:
        raise ValueError(
            f"Split integrity violation! Current hash ({current_hash}) does not match locked hash ({lock_data.get('split_hash')})"
        )

    # Step 2: Log unlock event
    db.log_event(
        run_id=run_id,
        event_type="HOLDOUT_UNLOCKED",
        stage="FINAL_HOLDOUT",
        message="Sealed 10% Holdout benchmark unlocked for final model evaluation.",
        data={"split_hash": current_hash, "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()},
    )

    # Step 3: Load holdout dataset with explicit permission
    eval_transform = get_eval_transforms()
    holdout_ds = DermDataset(manifest_df, split="holdout", transform=eval_transform, allow_holdout=True)

    # Step 4: Run evaluation
    metrics, predictions = evaluate_model_on_dataset(model, holdout_ds, device)

    # Step 5: Mark lock consumed
    updated_lock = mark_holdout_consumed(splits_dir)
    db.log_event(
        run_id=run_id,
        event_type="HOLDOUT_CONSUMED",
        stage="FINAL_HOLDOUT",
        message="Sealed 10% Holdout evaluation complete. Status updated to CONSUMED.",
        data=updated_lock,
    )

    # Step 6: Save predictions and metrics artifacts
    pred_df = pd.DataFrame(predictions)
    pred_csv_path = output_dir / "predictions_holdout.csv"
    pred_df.to_csv(pred_csv_path, index=False)

    metrics_json_path = output_dir / "final_metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    cm_df = pd.DataFrame(metrics["confusion_matrix"], index=OFFICIAL_CLASSES, columns=OFFICIAL_CLASSES)
    cm_csv_path = output_dir / "confusion_matrix.csv"
    cm_df.to_csv(cm_csv_path)

    # Persist in SQLite
    db.log_predictions(run_id, split="holdout", predictions_list=predictions)
    db.log_artifact(
        run_id=run_id,
        artifact_type="predictions_holdout",
        file_path=str(pred_csv_path),
        file_size_bytes=pred_csv_path.stat().st_size,
        sha256=compute_file_hash(pred_csv_path),
    )
    db.log_artifact(
        run_id=run_id,
        artifact_type="final_metrics",
        file_path=str(metrics_json_path),
        file_size_bytes=metrics_json_path.stat().st_size,
        sha256=compute_file_hash(metrics_json_path),
    )

    return metrics, predictions
