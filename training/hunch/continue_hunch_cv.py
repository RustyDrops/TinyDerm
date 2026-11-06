"""Continue paired grouped-CV fold checkpoints with a lower fine-tuning rate."""
import argparse
import json
import sqlite3
import io
import random
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


def predict(teacher, matrix):
    prob = teacher.predict_proba(matrix)
    return prob[:, [list(teacher.classes_).index(c) for c in OFFICIAL_CLASSES]]


def evaluate(y, prob):
    pred = np.asarray(OFFICIAL_CLASSES)[prob.argmax(axis=1)]
    return {
        "macro_f1": float(f1_score(y, pred, labels=OFFICIAL_CLASSES, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "per_class": classification_report(y, pred, labels=OFFICIAL_CLASSES, output_dict=True, zero_division=0),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--initial-checkpoints", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--initial-epoch", type=int, default=8)
    parser.add_argument("--extra-epochs", type=int, default=4)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=31415)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=288)
    parser.add_argument("--teacher-c", type=float, default=0.001)
    parser.add_argument("--distill-weight", type=float, default=0.35)
    parser.add_argument("--class-weight-strength", type=float, default=0.5)
    parser.add_argument("--backbone-lr", type=float, default=5e-5)
    parser.add_argument("--head-lr", type=float, default=3e-4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    frame = development_rows(args.manifest)
    labels, groups = frame.class_label.to_numpy(), frame.lesion_id.to_numpy()
    database = sqlite3.connect(args.cache)
    cache = {key: blob for key, blob in database.execute("SELECT image_id, vector FROM vectors")}
    database.close()
    x = np.stack([np.load(io.BytesIO(cache[str(key)]), allow_pickle=False) for key in frame.image_id])
    splitter = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    splits = list(splitter.split(x, labels, groups))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    oof = {arm: np.empty((len(frame), len(OFFICIAL_CLASSES)), dtype=np.float32) for arm in ("baseline", "hunch_assisted")}
    fold_reports = []

    for fold, (train_idx, held_idx) in enumerate(splits, start=1):
        seed = args.seed + fold
        outer_train, held_labels = x[train_idx], labels[held_idx]
        inner_splitter = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=seed)
        train_targets = np.zeros((len(train_idx), len(OFFICIAL_CLASSES)), dtype=np.float32)
        for fit_idx, held_teacher_idx in inner_splitter.split(outer_train, labels[train_idx], groups[train_idx]):
            teacher = make_pipeline(StandardScaler(), LogisticRegression(C=args.teacher_c, class_weight="balanced", max_iter=2000, random_state=seed))
            teacher.fit(outer_train[fit_idx], labels[train_idx][fit_idx])
            train_targets[held_teacher_idx] = predict(teacher, outer_train[held_teacher_idx])
        teacher = make_pipeline(StandardScaler(), LogisticRegression(C=args.teacher_c, class_weight="balanced", max_iter=2000, random_state=seed))
        teacher.fit(outer_train, labels[train_idx])
        held_targets = predict(teacher, x[held_idx])

        train_frame = frame.iloc[train_idx].copy().reset_index(drop=True)
        held_frame = frame.iloc[held_idx].copy().reset_index(drop=True)
        train_frame["split"], held_frame["split"] = "train", "val"
        target_map = {str(key): value for key, value in zip(train_frame.image_id, train_targets)}
        train_set = WithTeacher(DermDataset(train_frame, "train", get_train_transforms(args.image_size)), target_map)
        held_set = DermDataset(held_frame, "val", get_eval_transforms(args.image_size))
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
        held_loader = DataLoader(held_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
        counts = np.array([(labels[train_idx] == c).sum() for c in OFFICIAL_CLASSES], dtype=np.float64)
        weights = (counts.sum() / (len(OFFICIAL_CLASSES) * np.maximum(counts, 1))) ** args.class_weight_strength
        weights = torch.tensor(weights / weights.mean(), dtype=torch.float32, device=device)
        fold_report = {"fold": fold, "arms": {}}

        for arm in ("baseline", "hunch_assisted"):
            seed_all(seed)
            model = TinyDermModel(pretrained=False).to(device)
            source = args.initial_checkpoints / f"fold{fold}_{arm}_epoch{args.initial_epoch}.pt"
            model.load_state_dict(torch.load(source, map_location=device, weights_only=True))
            groups_opt = [
                {"params": [p for n, p in model.model.named_parameters() if "classifier" not in n], "lr": args.backbone_lr},
                {"params": [p for n, p in model.model.named_parameters() if "classifier" in n], "lr": args.head_lr},
            ]
            optimizer = torch.optim.AdamW(groups_opt, weight_decay=0.01)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.extra_epochs, 1), eta_min=1e-6)
            held_probs = []
            history = []
            for offset in range(1, args.extra_epochs + 1):
                model.train()
                total_loss = 0.0
                for images, target, soft in train_loader:
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
                    total_loss += float(loss.item()) * len(target)
                scheduler.step()
                model.eval()
                chunk = []
                with torch.inference_mode():
                    for images, _, _, _ in held_loader:
                        chunk.append(torch.softmax(model(images.to(device)), dim=1).cpu().numpy())
                probability = np.concatenate(chunk)
                held_probs.append(probability)
                epoch = args.initial_epoch + offset
                score = evaluate(held_labels, probability)
                score.update(epoch=epoch, training_loss=total_loss / len(train_set))
                history.append(score)
                print(f"fold {fold} {arm} epoch {epoch}: macro F1 {score['macro_f1']:.4f}; balanced accuracy {score['balanced_accuracy']:.4f}", flush=True)
            oof[arm][held_idx] = held_probs[-1]
            fold_report["arms"][arm] = history
            torch.save(model.state_dict(), args.output / f"fold{fold}_{arm}_epoch{args.initial_epoch + args.extra_epochs}.pt")
            del model, optimizer, scheduler
        fold_reports.append(fold_report)
        np.savez_compressed(args.output / f"fold{fold}_oof_predictions.npz", held_indices=held_idx, baseline=oof["baseline"][held_idx], hunch_assisted=oof["hunch_assisted"][held_idx])
        (args.output / "fold_reports.json").write_text(json.dumps(fold_reports, indent=2), encoding="utf-8")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    report = {
        "scope": "5-fold lesion-grouped out-of-fold estimate; continue from grouped CV checkpoints",
        "device": str(device),
        "initial_epoch": args.initial_epoch,
        "final_epoch": args.initial_epoch + args.extra_epochs,
        "settings": {"folds": args.folds, "extra_epochs": args.extra_epochs, "batch_size": args.batch_size, "image_size": args.image_size, "seed": args.seed, "teacher_c": args.teacher_c, "distill_weight": args.distill_weight, "class_weight_strength": args.class_weight_strength, "backbone_lr": args.backbone_lr, "head_lr": args.head_lr},
        "aggregate_oof": {arm: evaluate(labels, prob) for arm, prob in oof.items()},
        "folds_detail": fold_reports,
    }
    (args.output / "crossvalidation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for arm, result in report["aggregate_oof"].items():
        print(f"{arm}: macro F1 {result['macro_f1']:.4f}; balanced accuracy {result['balanced_accuracy']:.4f}", flush=True)


if __name__ == "__main__":
    main()

