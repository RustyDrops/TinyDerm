"""Two-stage training loop for TinyDerm Lab."""

import copy
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from tinyderm.augment import get_eval_transforms, get_train_transforms
from tinyderm.dataset import DermDataset
from tinyderm.metrics import compute_classification_metrics
from tinyderm.model import TinyDermModel

logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: torch.device = torch.device("cpu"),
    is_training: bool = True,
    throughput_callback: Optional[Callable[[float], None]] = None,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, float]:
    """Execute a single train or evaluation epoch."""
    if is_training:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    all_preds = []
    all_targets = []
    all_probs = []

    total_images = 0
    t0 = time.time()

    for images, targets, _, _ in dataloader:
        images = images.to(device)
        targets = targets.to(device)
        batch_size = images.size(0)
        total_images += batch_size

        if is_training and optimizer is not None:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                outputs = model(images)
                loss = criterion(outputs, targets)

        running_loss += loss.item() * batch_size
        probs = torch.softmax(outputs, dim=1).detach().cpu().numpy()
        preds = np.argmax(probs, axis=1)

        all_preds.append(preds)
        all_targets.append(targets.cpu().numpy())
        all_probs.append(probs)

        elapsed = max(time.time() - t0, 0.001)
        if throughput_callback:
            throughput_callback(total_images / elapsed)

    epoch_loss = running_loss / max(total_images, 1)
    y_pred = np.concatenate(all_preds, axis=0) if all_preds else np.array([])
    y_true = np.concatenate(all_targets, axis=0) if all_targets else np.array([])
    y_prob = np.concatenate(all_probs, axis=0) if all_probs else np.array([])
    duration = max(time.time() - t0, 0.001)

    return epoch_loss, y_true, y_pred, y_prob, duration


def train_student_model(
    manifest_df: pd.DataFrame,
    config: Dict[str, Any],
    db,
    run_id: str,
    output_dir: str | Path = "runs",
    retrain_on_dev: bool = False,
    progress_callback: Optional[Callable[[str, int, int, Dict[str, float]], None]] = None,
    throughput_callback: Optional[Callable[[float], None]] = None,
) -> Tuple[TinyDermModel, Dict[str, Any]]:
    """Two-stage training (head freezing + full fine-tuning) with early stopping."""
    output_dir = Path(output_dir) / run_id
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    train_cfg = config.get("training", {})
    batch_size = train_cfg.get("batch_size", 64)
    num_workers = train_cfg.get("num_workers", 2)
    image_size = config.get("model", {}).get("image_size", 224)

    # Prepare datasets
    train_transform = get_train_transforms(image_size)
    eval_transform = get_eval_transforms(image_size)

    if retrain_on_dev:
        # Full development pool (train + val, excluding sealed holdout)
        train_ds = DermDataset(manifest_df, split="dev", transform=train_transform)
        val_ds = DermDataset(manifest_df, split="val", transform=eval_transform)
    else:
        train_ds = DermDataset(manifest_df, split="train", transform=train_transform)
        val_ds = DermDataset(manifest_df, split="val", transform=eval_transform)

    # Class-weighted loss derived strictly from training data
    class_weights = train_ds.get_class_weights().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Batch loader with OOM protection loop
    def make_loaders(bs: int):
        t_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=num_workers, pin_memory=(device.type == "cuda"))
        v_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=num_workers, pin_memory=(device.type == "cuda"))
        return t_loader, v_loader

    train_loader, val_loader = make_loaders(batch_size)

    # Initialize model
    model = TinyDermModel(
        model_name=config.get("model", {}).get("name", "mobilenetv3_small_100"),
        pretrained=config.get("model", {}).get("pretrained", True),
        num_classes=7,
    ).to(device)

    best_val_f1 = -1.0
    best_model_weights = None
    best_metrics = {}
    patience = train_cfg.get("early_stopping_patience", 5)
    patience_counter = 0

    # -------------------------------------------------------------
    # STAGE 1: Train classification head only
    # -------------------------------------------------------------
    head_epochs = train_cfg.get("head_epochs", 3)
    if head_epochs > 0:
        model.freeze_backbone()
        optimizer_head = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=train_cfg.get("head_lr", 1e-3),
            weight_decay=train_cfg.get("weight_decay", 0.01),
        )

        for ep in range(1, head_epochs + 1):
            try:
                t_loss, t_true, t_pred, t_prob, t_dur = run_epoch(
                    model, train_loader, criterion, optimizer_head, device, True, throughput_callback
                )
            except torch.cuda.OutOfMemoryError:
                logger.warning(f"CUDA OOM in Stage 1! Halving batch size from {batch_size} to {batch_size // 2}")
                torch.cuda.empty_cache()
                batch_size = max(8, batch_size // 2)
                train_loader, val_loader = make_loaders(batch_size)
                continue

            v_loss, v_true, v_pred, v_prob, v_dur = run_epoch(
                model, val_loader, criterion, None, device, False, throughput_callback
            )

            t_metrics = compute_classification_metrics(t_true, t_pred, t_prob)
            v_metrics = compute_classification_metrics(v_true, v_pred, v_prob)

            db.log_epoch(
                run_id=run_id,
                phase="RETRAIN_HEAD" if retrain_on_dev else "TRAIN_HEAD",
                epoch=ep,
                train_loss=t_loss,
                val_loss=v_loss,
                train_f1=t_metrics["macro_f1"],
                val_f1=v_metrics["macro_f1"],
                train_balanced_acc=t_metrics["balanced_accuracy"],
                val_balanced_acc=v_metrics["balanced_accuracy"],
                lr=train_cfg.get("head_lr", 1e-3),
                images_per_sec=len(train_ds) / max(t_dur, 0.001),
                duration_sec=t_dur + v_dur,
                metrics_dict={"train": t_metrics, "val": v_metrics},
            )

            if progress_callback:
                progress_callback(
                    "TRAINING_HEAD" if not retrain_on_dev else "RETRAINING_HEAD",
                    ep,
                    head_epochs,
                    {"val_f1": v_metrics["macro_f1"], "train_loss": t_loss, "val_loss": v_loss},
                )

            if v_metrics["macro_f1"] > best_val_f1:
                best_val_f1 = v_metrics["macro_f1"]
                best_model_weights = copy.deepcopy(model.state_dict())
                best_metrics = v_metrics

    # -------------------------------------------------------------
    # STAGE 2: Fine-tune complete network
    # -------------------------------------------------------------
    model.unfreeze_backbone()
    finetune_epochs = train_cfg.get("finetune_max_epochs", 20)
    optimizer_full = AdamW(
        [
            {"params": [p for n, p in model.model.named_parameters() if "classifier" not in n], "lr": train_cfg.get("backbone_lr", 1e-4)},
            {"params": [p for n, p in model.model.named_parameters() if "classifier" in n], "lr": train_cfg.get("head_lr", 1e-3)},
        ],
        weight_decay=train_cfg.get("weight_decay", 0.01),
    )
    scheduler = CosineAnnealingLR(optimizer_full, T_max=finetune_epochs, eta_min=1e-6)

    for ep in range(1, finetune_epochs + 1):
        global_ep = head_epochs + ep
        try:
            t_loss, t_true, t_pred, t_prob, t_dur = run_epoch(
                model, train_loader, criterion, optimizer_full, device, True, throughput_callback
            )
        except torch.cuda.OutOfMemoryError:
            logger.warning(f"CUDA OOM in Stage 2! Halving batch size from {batch_size} to {batch_size // 2}")
            torch.cuda.empty_cache()
            batch_size = max(8, batch_size // 2)
            train_loader, val_loader = make_loaders(batch_size)
            continue

        scheduler.step()

        v_loss, v_true, v_pred, v_prob, v_dur = run_epoch(
            model, val_loader, criterion, None, device, False, throughput_callback
        )

        t_metrics = compute_classification_metrics(t_true, t_pred, t_prob)
        v_metrics = compute_classification_metrics(v_true, v_pred, v_prob)

        db.log_epoch(
            run_id=run_id,
            phase="RETRAIN_FULL" if retrain_on_dev else "FINETUNE",
            epoch=global_ep,
            train_loss=t_loss,
            val_loss=v_loss,
            train_f1=t_metrics["macro_f1"],
            val_f1=v_metrics["macro_f1"],
            train_balanced_acc=t_metrics["balanced_accuracy"],
            val_balanced_acc=v_metrics["balanced_accuracy"],
            lr=float(scheduler.get_last_lr()[0]),
            images_per_sec=len(train_ds) / max(t_dur, 0.001),
            duration_sec=t_dur + v_dur,
            metrics_dict={"train": t_metrics, "val": v_metrics},
        )

        if progress_callback:
            progress_callback(
                "FINETUNING" if not retrain_on_dev else "RETRAINING_90_PERCENT",
                ep,
                finetune_epochs,
                {"val_f1": v_metrics["macro_f1"], "train_loss": t_loss, "val_loss": v_loss},
            )

        # Early stopping check on validation macro_f1
        if v_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = v_metrics["macro_f1"]
            best_model_weights = copy.deepcopy(model.state_dict())
            best_metrics = v_metrics
            patience_counter = 0
            # Save best checkpoint
            ckpt_name = "best_retrained_model.pt" if retrain_on_dev else "best_dev_model.pt"
            torch.save(best_model_weights, models_dir / ckpt_name)
        else:
            patience_counter += 1
            if patience_counter >= patience and not retrain_on_dev:
                logger.info(f"Early stopping triggered after {ep} finetuning epochs.")
                break

    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)

    return model, best_metrics
