"""Lesion-grouped CV of a staged MobileNet with and without Hunch distillation."""
import argparse
import hashlib
import io
import json
import random
import sqlite3
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from hunch_experiment import development_rows
from tinyderm.augment import get_eval_transforms, get_train_transforms
from tinyderm.dataset import DermDataset, OFFICIAL_CLASSES
from tinyderm.model import TinyDermModel


class WithTeacher(Dataset):
    def __init__(self, base, targets):
        self.base, self.targets = base, targets

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        image, label, image_id, _ = self.base[index]
        return image, label, torch.tensor(self.targets[image_id], dtype=torch.float32)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ordered_probabilities(model, x):
    p = model.predict_proba(x)
    return p[:, [list(model.classes_).index(c) for c in OFFICIAL_CLASSES]]


def metric(y, probabilities):
    predicted = np.asarray(OFFICIAL_CLASSES)[probabilities.argmax(axis=1)]
    return {
        "macro_f1": float(f1_score(y, predicted, labels=OFFICIAL_CLASSES, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "per_class": classification_report(y, predicted, labels=OFFICIAL_CLASSES, output_dict=True, zero_division=0),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--head-epochs", type=int, default=4)
    parser.add_argument("--fine-tune-epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=288)
    parser.add_argument("--seed", type=int, default=27182)
    parser.add_argument("--teacher-c", type=float, default=0.001)
    parser.add_argument("--distill-weight", type=float, default=0.35)
    parser.add_argument("--class-weight-strength", type=float, default=0.5)
    parser.add_argument("--head-lr", type=float, default=0.001)
    parser.add_argument("--backbone-lr", type=float, default=0.00015)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite output directory: {args.output}")
    args.output.mkdir(parents=True)
    frame = development_rows(args.manifest)
    labels = frame.class_label.to_numpy()
    groups = frame.lesion_id.to_numpy()
    connection = sqlite3.connect(args.cache)
    cache = {key: blob for key, blob in connection.execute("SELECT image_id, vector FROM vectors")}
    connection.close()
    if set(frame.image_id) - set(cache):
        raise ValueError("Hunch cache does not cover all development images")
    x = np.stack([np.load(io.BytesIO(cache[key]), allow_pickle=False) for key in frame.image_id])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outer = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    splits = list(outer.split(x, labels, groups))
    epochs = args.head_epochs + args.fine_tune_epochs
    arms = ("baseline", "hunch_assisted")
    predictions = {arm: [None] * args.folds for arm in arms}
    fold_reports = []
    start = time.monotonic()

    for fold, (train_idx, held_idx) in enumerate(splits, start=1):
        fold_seed = args.seed + fold
        print(f"outer fold {fold}/{args.folds}: {len(train_idx)} train, {len(held_idx)} held-out", flush=True)
        outer_x, outer_y = x[train_idx], labels[train_idx]
        inner = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=fold_seed)
        soft_train = np.zeros((len(train_idx), len(OFFICIAL_CLASSES)), dtype=np.float32)
        for inner_fold, (fit_idx, soft_idx) in enumerate(inner.split(outer_x, outer_y, groups[train_idx]), start=1):
            teacher = make_pipeline(StandardScaler(), LogisticRegression(C=args.teacher_c, class_weight="balanced", max_iter=2000, random_state=fold_seed))
            teacher.fit(outer_x[fit_idx], outer_y[fit_idx])
            soft_train[soft_idx] = ordered_probabilities(teacher, outer_x[soft_idx])
        teacher = make_pipeline(StandardScaler(), LogisticRegression(C=args.teacher_c, class_weight="balanced", max_iter=2000, random_state=fold_seed))
        teacher.fit(outer_x, outer_y)
        soft_held = ordered_probabilities(teacher, x[held_idx])

        train_frame = frame.iloc[train_idx].copy().reset_index(drop=True)
        held_frame = frame.iloc[held_idx].copy().reset_index(drop=True)
        train_frame["split"], held_frame["split"] = "train", "val"
        teacher_map = {str(key): value for key, value in zip(train_frame.image_id, soft_train)}
        train_dataset = WithTeacher(DermDataset(train_frame, "train", get_train_transforms(args.image_size)), teacher_map)
        held_dataset = DermDataset(held_frame, "val", get_eval_transforms(args.image_size))
        held_y = labels[held_idx]
        counts = np.array([(outer_y == c).sum() for c in OFFICIAL_CLASSES], dtype=np.float64)
        weights = (counts.sum() / (len(OFFICIAL_CLASSES) * np.maximum(counts, 1))) ** args.class_weight_strength
        weights = torch.tensor(weights / weights.mean(), dtype=torch.float32, device=device)
        history_fold = {"fold": fold, "train_images": len(train_idx), "heldout_images": len(held_idx), "hunch_teacher": metric(held_y, soft_held), "arms": {}}

        for arm in arms:
            seed_all(fold_seed)
            model = TinyDermModel(pretrained=True).to(device)
            base_parameters = [p for name, p in model.model.named_parameters() if "classifier" not in name]
            head_parameters = [p for name, p in model.model.named_parameters() if "classifier" in name]
            loaders = {
                "train": DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0),
                "held": DataLoader(held_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0),
            }
            history, epoch_prob = [], []
            phase_specs = [("head", args.head_epochs, args.head_lr, None), ("fine_tune", args.fine_tune_epochs, args.backbone_lr, args.head_lr)]
            current_epoch = 0
            for phase, phase_epochs, backbone_lr, phase_head_lr in phase_specs:
                if phase_epochs <= 0:
                    continue
                if phase == "head":
                    for parameter in base_parameters:
                        parameter.requires_grad = False
                    for parameter in head_parameters:
                        parameter.requires_grad = True
                    optimizer_parameters = [{"params": head_parameters, "lr": backbone_lr}]
                else:
                    for parameter in model.parameters():
                        parameter.requires_grad = True
                    optimizer_parameters = [
                        {"params": base_parameters, "lr": backbone_lr},
                        {"params": head_parameters, "lr": phase_head_lr},
                    ]
                optimizer = torch.optim.AdamW(optimizer_parameters, weight_decay=args.weight_decay)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=phase_epochs, eta_min=1e-6)
                for _ in range(phase_epochs):
                    current_epoch += 1
                    model.train()
                    if phase == "head":
                        for module in model.modules():
                            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                                module.eval()
                    loss_sum = 0.0
                    for images, target, soft in loaders["train"]:
                        images, target, soft = images.to(device), target.to(device), soft.to(device)
                        optimizer.zero_grad(set_to_none=True)
                        logits = model(images)
                        hard = F.cross_entropy(logits, target, weight=weights)
                        if arm == "hunch_assisted":
                            kd = F.kl_div(F.log_softmax(logits, dim=1), soft, reduction="batchmean")
                            loss = (1 - args.distill_weight) * hard + args.distill_weight * kd
                        else:
                            loss = hard
                        loss.backward()
                        optimizer.step()
                        loss_sum += float(loss.item()) * len(target)
                    scheduler.step()
                    model.eval()
                    chunks = []
                    with torch.inference_mode():
                        for images, _, _, _ in loaders["held"]:
                            chunks.append(torch.softmax(model(images.to(device)), dim=1).cpu().numpy())
                    probability = np.concatenate(chunks)
                    epoch_prob.append(probability)
                    score = metric(held_y, probability)
                    score.update(epoch=current_epoch, phase=phase, training_loss=loss_sum / len(train_dataset))
                    history.append(score)
                    print(f"fold {fold} {arm} epoch {current_epoch}: macro F1 {score['macro_f1']:.4f}; balanced accuracy {score['balanced_accuracy']:.4f}", flush=True)
                del optimizer, scheduler
            predictions[arm][fold - 1] = epoch_prob
            history_fold["arms"][arm] = history
            torch.save(model.state_dict(), args.output / f"fold{fold}_{arm}_epoch{epochs}.pt")
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        fold_reports.append(history_fold)
        np.savez_compressed(args.output / f"fold{fold}_oof_predictions.npz", held_indices=held_idx, baseline=np.stack(predictions["baseline"][fold - 1]), hunch_assisted=np.stack(predictions["hunch_assisted"][fold - 1]))
        (args.output / "fold_reports.json").write_text(json.dumps(fold_reports, indent=2), encoding="utf-8")

    aggregate = {arm: [] for arm in arms}
    for arm, folds in predictions.items():
        for epoch_index in range(epochs):
            ordered = np.empty((len(frame), len(OFFICIAL_CLASSES)), dtype=np.float32)
            for fold_index, (_, held_idx) in enumerate(splits):
                ordered[held_idx] = folds[fold_index][epoch_index]
            aggregate[arm].append(metric(labels, ordered))
    report = {
        "scope": "5-fold lesion-grouped out-of-fold development estimate; original holdout excluded",
        "initialization": "ImageNet-pretrained timm MobileNetV3-Small from local cache",
        "device": str(device),
        "images": len(frame),
        "lesion_groups": int(frame.lesion_id.nunique()),
        "settings": {"folds": args.folds, "head_epochs": args.head_epochs, "fine_tune_epochs": args.fine_tune_epochs, "batch_size": args.batch_size, "image_size": args.image_size, "seed": args.seed, "teacher_c": args.teacher_c, "distill_weight": args.distill_weight, "class_weight_strength": args.class_weight_strength, "head_lr": args.head_lr, "backbone_lr": args.backbone_lr, "weight_decay": args.weight_decay},
        "aggregate_oof": aggregate,
        "folds_detail": fold_reports,
        "elapsed_seconds": time.monotonic() - start,
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
    }
    (args.output / "crossvalidation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for epoch in (args.head_epochs, epochs):
        for arm in arms:
            score = aggregate[arm][epoch - 1]
            print(f"aggregate {arm} epoch {epoch}: macro F1 {score['macro_f1']:.4f}; balanced accuracy {score['balanced_accuracy']:.4f}", flush=True)


if __name__ == "__main__":
    main()

