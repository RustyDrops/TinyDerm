"""Fit a fixed-epoch Hunch-assisted student on train plus validation only."""
import argparse
import hashlib
import io
import json
import random
import sqlite3
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from hunch_experiment import development_rows
from tinyderm.augment import get_train_transforms
from tinyderm.dataset import DermDataset, OFFICIAL_CLASSES
from tinyderm.model import TinyDermModel


class WithTeacher(Dataset):
    def __init__(self, base, probabilities):
        self.base = base
        self.probabilities = probabilities

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        image, label, image_id, _ = self.base[index]
        return image, label, torch.tensor(self.probabilities[image_id], dtype=torch.float32)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--tokens", type=int, default=70)
    parser.add_argument("--teacher-c", type=float, default=0.001)
    parser.add_argument("--distill-weight", type=float, default=0.35)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--class-weight-strength", type=float, default=0.5)
    args = parser.parse_args()
    if args.output.exists():
        existing = {path.name for path in args.output.iterdir()}
        if existing != {"embeddings.sqlite"}:
            raise FileExistsError(f"Output directory must be empty except for embeddings.sqlite: {args.output}")
    else:
        args.output.mkdir(parents=True)
    frame = development_rows(args.manifest)
    train = frame.split.eq("train").to_numpy()
    val = frame.split.eq("val").to_numpy()
    labels = frame.class_label.to_numpy()
    if set(labels) != set(OFFICIAL_CLASSES):
        raise ValueError("All seven classes must be present across train plus validation")
    db = sqlite3.connect(args.cache)
    vectors = {key: blob for key, blob in db.execute("SELECT image_id, vector FROM vectors")}
    db.close()
    if set(frame.image_id) - set(vectors):
        raise ValueError("Hunch embedding cache does not cover development rows")
    x = np.stack([np.load(io.BytesIO(vectors[key]), allow_pickle=False) for key in frame.image_id])

    teacher = make_pipeline(StandardScaler(), LogisticRegression(C=args.teacher_c, class_weight="balanced", max_iter=2000, random_state=args.seed))
    train_rows = np.flatnonzero(train)
    targets = np.zeros((len(frame), len(OFFICIAL_CLASSES)), dtype=np.float32)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed)
    for fold, (fit_idx, held_idx) in enumerate(splitter.split(x[train], labels[train], frame.loc[train, "lesion_id"]), start=1):
        fold_model = make_pipeline(StandardScaler(), LogisticRegression(C=args.teacher_c, class_weight="balanced", max_iter=2000, random_state=args.seed))
        fold_model.fit(x[train][fit_idx], labels[train][fit_idx])
        prob = fold_model.predict_proba(x[train][held_idx])
        targets[train_rows[held_idx]] = prob[:, [list(fold_model.classes_).index(c) for c in OFFICIAL_CLASSES]]
        print(f"Hunch grouped teacher fold {fold}/5 complete", flush=True)
    teacher.fit(x[train], labels[train])
    val_prob = teacher.predict_proba(x[val])
    targets[val] = val_prob[:, [list(teacher.classes_).index(c) for c in OFFICIAL_CLASSES]]
    joblib.dump(teacher, args.output / "hunch_teacher.joblib")
    mapping = dict(zip(frame.image_id, targets))

    train_counts = np.array([(frame.loc[train, "class_label"] == c).sum() for c in OFFICIAL_CLASSES], dtype=np.float64)
    class_weights = (train_counts.sum() / (len(OFFICIAL_CLASSES) * np.maximum(train_counts, 1))) ** args.class_weight_strength
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_weights = torch.tensor(class_weights / class_weights.mean(), dtype=torch.float32, device=device)
    seed_all(args.seed)
    model = TinyDermModel(pretrained=False).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    dataset = WithTeacher(DermDataset(frame, "dev", get_train_transforms(288)), mapping)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for images, target, soft in loader:
            images, target, soft = images.to(device), target.to(device), soft.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            hard = F.cross_entropy(logits, target, weight=class_weights)
            distill = F.kl_div(F.log_softmax(logits, dim=1), soft, reduction="batchmean")
            loss = (1 - args.distill_weight) * hard + args.distill_weight * distill
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(target)
        checkpoint = args.output / f"hunch_assisted_full_dev_epoch{epoch}.pt"
        torch.save(model.state_dict(), checkpoint)
        history.append({"epoch": epoch, "training_loss": total_loss / len(dataset), "checkpoint": checkpoint.name})
        print(f"full-development assisted retrain epoch {epoch} training loss: {history[-1]['training_loss']:.4f}", flush=True)

    settings = {
        "scope": "train plus validation; no performance metrics computed",
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "tokens": args.tokens,
        "teacher_head": "standardized logistic regression with class balancing",
        "teacher_c": args.teacher_c,
        "teacher_train_targets": "5-fold lesion-grouped out-of-fold probabilities",
        "distill_weight": args.distill_weight,
        "learning_rate": args.learning_rate,
        "class_weight_strength": args.class_weight_strength,
        "train_images": int(train.sum()),
        "validation_images_in_training_pool": int(val.sum()),
        "embedding_dimensions": int(x.shape[1]),
        "device": str(device),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "embedding_revision": "914f7f89142e33e77833254d9c9b90c3cef7303b",
        "hunch_revision": "6541bd157e11a5e7d7434360e99ea81f40a8c181",
        "epochs_history": history,
    }
    (args.output / "settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

