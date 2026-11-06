"""Lesion-grouped out-of-fold comparison from cached Hunch embeddings.

Every image is evaluated by a model trained on other lesion groups. Both the
Hunch teacher and student are fitted inside each outer training fold. The
consumed holdout is excluded by development_rows().
"""
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
    def __init__(self, base, teacher):
        self.base = base
        self.teacher = teacher

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        image, label, image_id, _ = self.base[index]
        return image, label, torch.tensor(self.teacher[image_id], dtype=torch.float32)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def probs_in_class_order(estimator, matrix):
    prob = estimator.predict_proba(matrix)
    return prob[:, [list(estimator.classes_).index(c) for c in OFFICIAL_CLASSES]]


def metrics(y_true, probabilities):
    predicted = np.asarray(OFFICIAL_CLASSES)[probabilities.argmax(axis=1)]
    return {
        "macro_f1": float(f1_score(y_true, predicted, labels=OFFICIAL_CLASSES, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "per_class": classification_report(y_true, predicted, labels=OFFICIAL_CLASSES, output_dict=True, zero_division=0),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=288)
    parser.add_argument("--seed", type=int, default=31415)
    parser.add_argument("--teacher-c", type=float, default=0.001)
    parser.add_argument("--distill-weight", type=float, default=0.35)
    parser.add_argument("--class-weight-strength", type=float, default=0.5)
    parser.add_argument("--backbone-lr", type=float, default=1.5e-4)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    frame = development_rows(args.manifest)
    labels = frame.class_label.to_numpy()
    groups = frame.lesion_id.to_numpy()
    ids = frame.image_id.astype(str).to_numpy()
    database = sqlite3.connect(args.cache)
    cache = {key: blob for key, blob in database.execute("SELECT image_id, vector FROM vectors")}
    database.close()
    missing = set(ids) - set(cache)
    if missing:
        raise ValueError(f"Hunch cache is missing {len(missing)} development images")
    x = np.stack([np.load(io.BytesIO(cache[key]), allow_pickle=False) for key in ids])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fold_splitter = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    predictions = {arm: [None] * args.folds for arm in ("baseline", "hunch_assisted")}
    fold_reports = []
    start = time.monotonic()

    for fold, (train_idx, held_idx) in enumerate(fold_splitter.split(x, labels, groups), start=1):
        fold_seed = args.seed + fold
        print(f"outer fold {fold}/{args.folds}: {len(train_idx)} train, {len(held_idx)} held-out images", flush=True)
        outer_train, outer_held = x[train_idx], x[held_idx]
        train_labels, held_labels = labels[train_idx], labels[held_idx]

        inner_splitter = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=fold_seed)
        train_targets = np.zeros((len(train_idx), len(OFFICIAL_CLASSES)), dtype=np.float32)
        for inner, (fit_idx, target_idx) in enumerate(inner_splitter.split(outer_train, train_labels, groups[train_idx]), start=1):
            teacher = make_pipeline(StandardScaler(), LogisticRegression(C=args.teacher_c, class_weight="balanced", max_iter=2000, random_state=fold_seed))
            teacher.fit(outer_train[fit_idx], train_labels[fit_idx])
            train_targets[target_idx] = probs_in_class_order(teacher, outer_train[target_idx])
        validation_teacher = make_pipeline(StandardScaler(), LogisticRegression(C=args.teacher_c, class_weight="balanced", max_iter=2000, random_state=fold_seed))
        validation_teacher.fit(outer_train, train_labels)
        held_targets = probs_in_class_order(validation_teacher, outer_held)

        train_frame = frame.iloc[train_idx].copy().reset_index(drop=True)
        held_frame = frame.iloc[held_idx].copy().reset_index(drop=True)
        train_frame["split"] = "train"
        held_frame["split"] = "val"
        teacher_map = {image_id: probability for image_id, probability in zip(train_frame.image_id.astype(str), train_targets)}
        train_set = WithTeacher(DermDataset(train_frame, "train", get_train_transforms(args.image_size)), teacher_map)
        val_set = DermDataset(held_frame, "val", get_eval_transforms(args.image_size))
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
        counts = np.array([(train_labels == c).sum() for c in OFFICIAL_CLASSES], dtype=np.float64)
        weights = (counts.sum() / (len(OFFICIAL_CLASSES) * np.maximum(counts, 1))) ** args.class_weight_strength
        weights = torch.tensor(weights / weights.mean(), dtype=torch.float32, device=device)

        fold_history = {"fold": fold, "train_images": len(train_idx), "heldout_images": len(held_idx), "teacher": metrics(held_labels, held_targets), "arms": {}}
        for arm in ("baseline", "hunch_assisted"):
            seed_all(fold_seed)
            model = TinyDermModel(pretrained=True).to(device)
            groups_opt = [
                {"params": [p for n, p in model.model.named_parameters() if "classifier" not in n], "lr": args.backbone_lr},
                {"params": [p for n, p in model.model.named_parameters() if "classifier" in n], "lr": args.head_lr},
            ]
            optimizer = torch.optim.AdamW(groups_opt, weight_decay=0.01)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1), eta_min=1e-6)
            history = []
            held_predictions = []
            for epoch in range(1, args.epochs + 1):
                model.train()
                loss_total = 0.0
                for images, target, soft in train_loader:
                    images, target, soft = images.to(device), target.to(device), soft.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(images)
                    hard = F.cross_entropy(logits, target, weight=weights)
                    if arm == "hunch_assisted":
                        distill = F.kl_div(F.log_softmax(logits, dim=1), soft, reduction="batchmean")
                        loss = (1 - args.distill_weight) * hard + args.distill_weight * distill
                    else:
                        loss = hard
                    loss.backward()
                    optimizer.step()
                    loss_total += float(loss.item()) * len(target)
                scheduler.step()
                model.eval()
                epoch_probabilities = []
                with torch.inference_mode():
                    for images, _, _, _ in val_loader:
                        epoch_probabilities.append(torch.softmax(model(images.to(device)), dim=1).cpu().numpy())
                epoch_probabilities = np.concatenate(epoch_probabilities)
                held_predictions.append(epoch_probabilities)
                epoch_metrics = metrics(held_labels, epoch_probabilities)
                epoch_metrics["epoch"] = epoch
                epoch_metrics["training_loss"] = loss_total / len(train_set)
                history.append(epoch_metrics)
                print(f"fold {fold} {arm} epoch {epoch}: macro F1 {epoch_metrics['macro_f1']:.4f}; balanced accuracy {epoch_metrics['balanced_accuracy']:.4f}", flush=True)
            predictions[arm][fold - 1] = held_predictions
            fold_history["arms"][arm] = history
            torch.save(model.state_dict(), args.output / f"fold{fold}_{arm}_epoch{args.epochs}.pt")
        fold_reports.append(fold_history)
        np.savez_compressed(
            args.output / f"fold{fold}_oof_predictions.npz",
            held_indices=held_idx,
            baseline=np.stack(predictions["baseline"][fold - 1]),
            hunch_assisted=np.stack(predictions["hunch_assisted"][fold - 1]),
        )
        (args.output / "fold_reports.json").write_text(json.dumps(fold_reports, indent=2), encoding="utf-8")
        del model, optimizer, scheduler
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    aggregate = {arm: [] for arm in predictions}
    for arm, folds in predictions.items():
        for epoch_index in range(args.epochs):
            ordered = np.empty((len(frame), len(OFFICIAL_CLASSES)), dtype=np.float32)
            for fold_index, (_, held_idx) in enumerate(fold_splitter.split(x, labels, groups)):
                ordered[held_idx] = folds[fold_index][epoch_index]
            aggregate[arm].append(metrics(labels, ordered))
    report = {
        "scope": "5-fold lesion-grouped out-of-fold estimate over development data; consumed holdout excluded",
        "model_initialization": "ImageNet-pretrained timm MobileNetV3-Small from local Hugging Face cache",
        "device": str(device),
        "train_and_validation_images": len(frame),
        "lesion_groups": int(frame.lesion_id.nunique()),
        "settings": {"folds": args.folds, "epochs": args.epochs, "batch_size": args.batch_size, "image_size": args.image_size, "seed": args.seed, "teacher_c": args.teacher_c, "distill_weight": args.distill_weight, "class_weight_strength": args.class_weight_strength, "backbone_lr": args.backbone_lr, "head_lr": args.head_lr, "encoder_revision": "914f7f89142e33e77833254d9c9b90c3cef7303b", "hunch_revision": "6541bd157e11a5e7d7434360e99ea81f40a8c181"},
        "aggregate_oof": aggregate,
        "folds_detail": fold_reports,
        "elapsed_seconds": time.monotonic() - start,
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
    }
    (args.output / "crossvalidation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

