"""Lesion-grouped CV with partial DINOv2 fine-tuning and Hunch distillation."""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'training-review'))
import numpy as np
import pandas as pd
os.environ.setdefault("HF_HOME", str(Path(__file__).parent / "training-review" / "model-cache"))
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from tinyderm_safe_data import development_identity_fingerprint, development_rows, load_hunch
from tinyderm_distillation import calibration_metrics, confidence_weighted_kl
from tinyderm.augment import get_eval_transforms, get_train_transforms
from tinyderm.dataset import OFFICIAL_CLASSES
from crossvalidate_hunch_dinov2 import Head, ordered_probs


class LabeledImages(Dataset):
    def __init__(self, frame, transform, soft_targets=None):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform
        self.soft = soft_targets

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, i):
        row = self.frame.iloc[i]
        with Image.open(row.source_path) as im:
            image = im.convert("RGB")
        image = self.transform(image)
        label = OFFICIAL_CLASSES.index(row.class_label)
        if self.soft is None:
            return image, label
        return image, label, self.soft[i]


class DinoStudent(nn.Module):
    def __init__(self, unfreeze_blocks=4):
        super().__init__()
        self.encoder = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True, num_classes=0, dynamic_img_size=True)
        for p in self.encoder.parameters():
            p.requires_grad = False
        for block in self.encoder.blocks[-unfreeze_blocks:]:
            for p in block.parameters():
                p.requires_grad = True
        for p in self.encoder.norm.parameters():
            p.requires_grad = True
        self.head = Head(self.encoder.num_features, len(OFFICIAL_CLASSES))

    def forward(self, x):
        return self.head(self.encoder(x))


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def metric(y, probability):
    prediction = np.asarray(OFFICIAL_CLASSES)[probability.argmax(axis=1)]
    return {
        "macro_f1": float(f1_score(y, prediction, labels=OFFICIAL_CLASSES, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "per_class": classification_report(y, prediction, labels=OFFICIAL_CLASSES, output_dict=True, zero_division=0),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--hunch-cache", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--unfreeze-blocks", type=int, default=4)
    parser.add_argument("--seed", type=int, default=52711)
    parser.add_argument("--teacher-c", type=float, default=0.001)
    parser.add_argument("--distill-weight", type=float, default=0.10)
    parser.add_argument("--class-weight-strength", type=float, default=0.5)
    parser.add_argument("--backbone-lr", type=float, default=0.00002)
    parser.add_argument("--head-lr", type=float, default=0.001)
    parser.add_argument("--melanoma-weight", type=float, default=0.5)
    parser.add_argument("--melanoma-positive-weight", type=float, default=2.0)
    parser.add_argument("--confidence-weighted-distillation", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)
    frame = development_rows(args.manifest)
    labels = frame.class_label.to_numpy()
    groups = frame.lesion_id.to_numpy()
    hunch = load_hunch(frame, args.hunch_cache)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits = list(StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed).split(hunch, labels, groups))
    records = {arm: [] for arm in ("baseline", "hunch_assisted")}
    teacher_probabilities = np.zeros((len(frame), len(OFFICIAL_CLASSES)), dtype=np.float32)
    reports = []
    started = time.monotonic()

    for fold, (train_idx, held_idx) in enumerate(splits, start=1):
        seed = args.seed + fold
        train_y = labels[train_idx]
        inner = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=seed)
        soft = np.zeros((len(train_idx), len(OFFICIAL_CLASSES)), dtype=np.float32)
        for fit_idx, soft_idx in inner.split(hunch[train_idx], train_y, groups[train_idx]):
            teacher = make_pipeline(StandardScaler(), LogisticRegression(C=args.teacher_c, class_weight="balanced", max_iter=2000, random_state=seed))
            teacher.fit(hunch[train_idx][fit_idx], train_y[fit_idx])
            soft[soft_idx] = ordered_probs(teacher, hunch[train_idx][soft_idx])
        outer_teacher = make_pipeline(StandardScaler(), LogisticRegression(C=args.teacher_c, class_weight="balanced", max_iter=2000, random_state=seed))
        outer_teacher.fit(hunch[train_idx], train_y)
        teacher_probabilities[held_idx] = ordered_probs(outer_teacher, hunch[held_idx])
        train_frame = frame.iloc[train_idx].reset_index(drop=True)
        held_frame = frame.iloc[held_idx].reset_index(drop=True)
        train_ds = LabeledImages(train_frame, get_train_transforms(args.image_size), soft)
        held_ds = LabeledImages(held_frame, get_eval_transforms(args.image_size))
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=device.type == "cuda")
        held_loader = DataLoader(held_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda")
        held_y = labels[held_idx]
        counts = np.array([(train_y == c).sum() for c in OFFICIAL_CLASSES], dtype=np.float64)
        class_weights = (counts.sum() / (len(OFFICIAL_CLASSES) * np.maximum(counts, 1))) ** args.class_weight_strength
        class_weights = torch.tensor(class_weights / class_weights.mean(), dtype=torch.float32, device=device)
        fold_report = {"fold": fold, "train_images": len(train_idx), "heldout_images": len(held_idx), "arms": {}}
        print(f"outer fold {fold}/{args.folds}: {len(train_idx)} train, {len(held_idx)} held-out", flush=True)

        for arm in records:
            seed_all(seed)
            model = DinoStudent(args.unfreeze_blocks).to(device)
            backbone_params = [p for p in model.encoder.parameters() if p.requires_grad]
            head_params = list(model.head.parameters())
            optimizer = torch.optim.AdamW([
                {"params": backbone_params, "lr": args.backbone_lr},
                {"params": head_params, "lr": args.head_lr},
            ], weight_decay=0.01)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
            scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
            mel_index = OFFICIAL_CLASSES.index('MEL')
            other_indices = [i for i in range(len(OFFICIAL_CLASSES)) if i != mel_index]
            mel_pos_weight = torch.tensor(args.melanoma_positive_weight, device=device)
            hist, epoch_probs = [], []
            for epoch in range(1, args.epochs + 1):
                model.train()
                loss_sum = 0.0
                for images, target, soft_target in train_loader:
                    images, target, soft_target = images.to(device, non_blocking=True), target.to(device), soft_target.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                        logits = model(images)
                        hard = F.cross_entropy(logits, target, weight=class_weights)
                        if arm == "hunch_assisted":
                            if args.confidence_weighted_distillation:
                                kd = confidence_weighted_kl(logits, soft_target)
                            else:
                                kd = F.kl_div(F.log_softmax(logits, dim=1), soft_target, reduction="batchmean")
                            loss = (1 - args.distill_weight) * hard + args.distill_weight * kd
                        else:
                            loss = hard
                        # The odds of MEL versus all other classes use the same seven-class logits.
                        mel_logit = logits.float()[:, mel_index] - torch.logsumexp(logits.float()[:, other_indices], dim=1)
                        mel_target = (target == mel_index).float()
                        loss = loss.float() + args.melanoma_weight * F.binary_cross_entropy_with_logits(mel_logit, mel_target, pos_weight=mel_pos_weight)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    loss_sum += float(loss.item()) * len(target)
                scheduler.step()
                model.eval()
                chunks = []
                with torch.inference_mode():
                    for images, _ in held_loader:
                        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                            chunks.append(torch.softmax(model(images.to(device, non_blocking=True)), dim=1).float().cpu().numpy())
                probability = np.concatenate(chunks)
                epoch_probs.append(probability)
                score = metric(held_y, probability)
                score.update(epoch=epoch, training_loss=loss_sum / len(train_ds))
                hist.append(score)
                print(f"fold {fold} {arm} epoch {epoch}: macro F1 {score['macro_f1']:.4f}; balanced accuracy {score['balanced_accuracy']:.4f}", flush=True)
            records[arm].append(np.stack(epoch_probs))
            fold_report["arms"][arm] = hist
            torch.save(model.state_dict(), args.output / f"fold{fold}_{arm}_epoch{args.epochs}.pt")
            del model, optimizer, scheduler
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        reports.append(fold_report)
        np.savez_compressed(args.output / f"fold{fold}_oof_predictions.npz", held_indices=held_idx, baseline=records["baseline"][-1], hunch_assisted=records["hunch_assisted"][-1])
        (args.output / "fold_reports.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")

    aggregate = {arm: [] for arm in records}
    for arm, folds in records.items():
        for epoch in range(args.epochs):
            oof = np.empty((len(frame), len(OFFICIAL_CLASSES)), dtype=np.float32)
            for f_idx, (_, held_idx) in enumerate(splits):
                oof[held_idx] = folds[f_idx][epoch]
            aggregate[arm].append(metric(labels, oof))
    summary = {
        "scope": f"{args.folds}-fold lesion-grouped OOF on development rows only; final holdout excluded",
        "image_encoder": "ImageNet-pretrained DINOv2-S/14; final transformer blocks fine-tuned",
        "device": str(device),
        "images": len(frame),
        "lesion_groups": int(frame.lesion_id.nunique()),
        "settings": vars(args) | {"manifest": str(args.manifest), "hunch_cache": str(args.hunch_cache), "output": str(args.output), "teacher_targets": "nested lesion-grouped Hunch/EmbeddingGemma logistic OOF probabilities"},
        "aggregate_oof": aggregate,
        "teacher_oof": metric(labels, teacher_probabilities),
        "teacher_oof_calibration": calibration_metrics(labels, teacher_probabilities, OFFICIAL_CLASSES),
        "teacher_oof_description": "outer-fold class-weighted logistic model fit on outer-train Hunch embeddings and scored on held-out development rows",
        "folds_detail": reports,
        "elapsed_seconds": time.monotonic() - started,
        "development_identity_sha256": development_identity_fingerprint(frame),
    }
    np.savez_compressed(args.output / "teacher_oof_predictions.npz", probabilities=teacher_probabilities)
    (args.output / "crossvalidation.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    for arm in records:
        score = aggregate[arm][-1]
        print(f"aggregate {arm} epoch {args.epochs}: macro F1 {score['macro_f1']:.4f}; balanced accuracy {score['balanced_accuracy']:.4f}", flush=True)


if __name__ == "__main__":
    main()

