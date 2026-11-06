"""Grouped CV of DINOv2 image features with a Hunch-guided classification head."""
import argparse
import hashlib
import io
import json
import os
import random
import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd
os.environ.setdefault("HF_HOME", str(Path(__file__).parent / "model-cache"))
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
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

from hunch_experiment import development_rows
from tinyderm.dataset import OFFICIAL_CLASSES


class ImageRows(Dataset):
    def __init__(self, frame, transform):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, i):
        row = self.frame.iloc[i]
        with Image.open(row.source_path) as im:
            image = im.convert("RGB")
        return self.transform(image), str(row.image_id)


class Head(nn.Module):
    def __init__(self, in_features=384, classes=7):
        super().__init__()
        self.layers = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, 256),
            nn.GELU(),
            nn.Dropout(0.30),
            nn.Linear(256, classes),
        )

    def forward(self, x):
        return self.layers(x)


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


def ordered_probs(model, x):
    probability = model.predict_proba(x)
    return probability[:, [list(model.classes_).index(c) for c in OFFICIAL_CLASSES]]


def get_hunch_features(frame, cache_path):
    with sqlite3.connect(cache_path) as db:
        cache = {key: blob for key, blob in db.execute("SELECT image_id, vector FROM vectors")}
    missing = set(frame.image_id) - set(cache)
    if missing:
        raise ValueError(f"Hunch embedding cache misses {len(missing)} development rows")
    return np.stack([np.load(io.BytesIO(cache[key]), allow_pickle=False) for key in frame.image_id])


def extract_dinov2(frame, cache_path, device, batch_size, multiview=False):
    manifest_hash = hashlib.sha256(pd.util.hash_pandas_object(frame[["image_id", "source_path"]], index=False).values.tobytes()).hexdigest()
    if cache_path.exists():
        saved = np.load(cache_path, allow_pickle=False)
        expected_ndim = 3 if multiview else 2
        if str(saved["manifest_hash"].item()) == manifest_hash and saved["image_ids"].tolist() == frame.image_id.tolist() and saved["features"].ndim == expected_ndim:
            print(f"loaded cached DINOv2 features from {cache_path}", flush=True)
            return saved["features"]
    print("loading cached ImageNet DINOv2-S/14 weights", flush=True)
    encoder = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True, num_classes=0, dynamic_img_size=True).to(device).eval()
    transform = T.Compose([
        T.Resize((518, 518), interpolation=InterpolationMode.BICUBIC, antialias=True),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    loader = DataLoader(ImageRows(frame, transform), batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    chunks = []
    with torch.inference_mode():
        for images, _ in loader:
            images = images.to(device, non_blocking=True)
            view_features = []
            views = [images, torch.flip(images, [3]), torch.flip(images, [2]), torch.flip(images, [2, 3])] if multiview else [images]
            for view in views:
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                    features = encoder(view)
                view_features.append(features.float())
            features = torch.stack(view_features, dim=1) if multiview else view_features[0]
            chunks.append(features.cpu().numpy())
            if len(chunks) % 40 == 0:
                print(f"extracted DINOv2 features for {min(len(chunks)*batch_size,len(frame))}/{len(frame)} images", flush=True)
    features = np.concatenate(chunks).astype(np.float32)
    np.savez_compressed(cache_path, features=features, image_ids=frame.image_id.to_numpy(dtype=str), manifest_hash=np.array(manifest_hash))
    return features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--hunch-cache", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--feature-cache", type=Path, default=None)
    parser.add_argument("--multiview", action="store_true", help="train with random flips and average four flip views at scoring")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--extract-batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42017)
    parser.add_argument("--teacher-c", type=float, default=0.001)
    parser.add_argument("--distill-weight", type=float, default=0.35)
    parser.add_argument("--class-weight-strength", type=float, default=0.5)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--baseline-only", action="store_true", help="run the control while a Hunch embedding cache is still being prepared")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)
    frame = development_rows(args.manifest)
    labels = frame.class_label.to_numpy()
    groups = frame.lesion_id.to_numpy()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_cache = args.feature_cache or (args.output / "dinov2_features.npz")
    features = extract_dinov2(frame, feature_cache, device, args.extract_batch_size, args.multiview)
    hunch = None if args.baseline_only else get_hunch_features(frame, args.hunch_cache)
    splits = list(StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed).split(features, labels, groups))
    arms = ("baseline",) if args.baseline_only else ("baseline", "hunch_assisted")
    records = {arm: [] for arm in arms}
    fold_reports = []
    started = time.monotonic()

    for fold, (train_idx, held_idx) in enumerate(splits, start=1):
        seed = args.seed + fold
        print(f"outer fold {fold}/{args.folds}: {len(train_idx)} train, {len(held_idx)} held-out", flush=True)
        train_y, held_y = labels[train_idx], labels[held_idx]
        # Hunch target probabilities are generated without training on the row's lesion.
        soft_train = None
        if not args.baseline_only:
            inner = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=seed)
            soft_train = np.zeros((len(train_idx), len(OFFICIAL_CLASSES)), dtype=np.float32)
            for fit_local, soft_local in inner.split(hunch[train_idx], train_y, groups[train_idx]):
                teacher = make_pipeline(StandardScaler(), LogisticRegression(C=args.teacher_c, class_weight="balanced", max_iter=2000, random_state=seed))
                teacher.fit(hunch[train_idx][fit_local], train_y[fit_local])
                soft_train[soft_local] = ordered_probs(teacher, hunch[train_idx][soft_local])

        train_features = features[train_idx]
        held_features = features[held_idx]
        if train_features.ndim == 3:
            scaler = StandardScaler().fit(train_features.reshape(-1, train_features.shape[-1]))
            x_train = scaler.transform(train_features.reshape(-1, train_features.shape[-1])).reshape(train_features.shape).astype(np.float32)
            x_held = scaler.transform(held_features.reshape(-1, held_features.shape[-1])).reshape(held_features.shape).astype(np.float32)
        else:
            scaler = StandardScaler().fit(train_features)
            x_train = scaler.transform(train_features).astype(np.float32)
            x_held = scaler.transform(held_features).astype(np.float32)
        counts = np.array([(train_y == c).sum() for c in OFFICIAL_CLASSES], dtype=np.float64)
        class_weights = (counts.sum() / (len(OFFICIAL_CLASSES) * np.maximum(counts, 1))) ** args.class_weight_strength
        class_weights = torch.tensor(class_weights / class_weights.mean(), dtype=torch.float32, device=device)
        fold_record = {"fold": fold, "train_images": len(train_idx), "heldout_images": len(held_idx), "arms": {}}

        for arm in arms:
            seed_all(seed)
            model = Head(features.shape[-1], len(OFFICIAL_CLASSES)).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
            tx = torch.from_numpy(x_train)
            ty = torch.tensor([OFFICIAL_CLASSES.index(c) for c in train_y], dtype=torch.long)
            ts = torch.from_numpy(soft_train) if soft_train is not None else None
            vx = torch.from_numpy(x_held).to(device)
            hist, epoch_probs = [], []
            generator = torch.Generator().manual_seed(seed)
            for epoch in range(1, args.epochs + 1):
                model.train()
                permutation = torch.randperm(len(tx), generator=generator)
                random_views = torch.randint(tx.shape[1], (len(tx),), generator=generator) if tx.ndim == 3 else None
                loss_total = 0.0
                for start in range(0, len(tx), args.batch_size):
                    idx = permutation[start:start + args.batch_size]
                    xb = tx[idx, random_views[idx]].to(device) if random_views is not None else tx[idx].to(device)
                    yb = ty[idx].to(device)
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(xb)
                    hard = F.cross_entropy(logits, yb, weight=class_weights)
                    if arm == "hunch_assisted":
                        sb = ts[idx].to(device)
                        kd = F.kl_div(F.log_softmax(logits, dim=1), sb, reduction="batchmean")
                        loss = (1 - args.distill_weight) * hard + args.distill_weight * kd
                    else:
                        loss = hard
                    loss.backward()
                    optimizer.step()
                    loss_total += float(loss.item()) * len(idx)
                scheduler.step()
                model.eval()
                with torch.inference_mode():
                    if vx.ndim == 3:
                        m, v, d = vx.shape
                        p = torch.softmax(model(vx.reshape(m * v, d)), dim=1).reshape(m, v, -1).mean(dim=1).cpu().numpy()
                    else:
                        p = torch.softmax(model(vx), dim=1).cpu().numpy()
                epoch_probs.append(p)
                score = metric(held_y, p)
                score.update(epoch=epoch, training_loss=loss_total / len(tx))
                hist.append(score)
                if epoch in (10, 20, 30, 40, args.epochs):
                    print(f"fold {fold} {arm} epoch {epoch}: macro F1 {score['macro_f1']:.4f}; balanced accuracy {score['balanced_accuracy']:.4f}", flush=True)
            records[arm].append(np.stack(epoch_probs))
            fold_record["arms"][arm] = hist
            torch.save(model.state_dict(), args.output / f"fold{fold}_{arm}_epoch{args.epochs}.pt")
            del model, optimizer, scheduler
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        fold_reports.append(fold_record)
        arrays = {"held_indices": held_idx}
        arrays.update({arm: records[arm][-1] for arm in arms})
        np.savez_compressed(args.output / f"fold{fold}_oof_predictions.npz", **arrays)
        (args.output / "fold_reports.json").write_text(json.dumps(fold_reports, indent=2), encoding="utf-8")

    aggregate = {arm: [] for arm in records}
    for arm, folds in records.items():
        for epoch in range(args.epochs):
            oof = np.empty((len(frame), len(OFFICIAL_CLASSES)), dtype=np.float32)
            for fold_idx, (_, held_idx) in enumerate(splits):
                oof[held_idx] = folds[fold_idx][epoch]
            aggregate[arm].append(metric(labels, oof))
    summary = {
        "scope": "5-fold lesion-grouped out-of-fold development estimate; original holdout excluded",
        "image_encoder": "timm vit_small_patch14_dinov2.lvd142m; frozen ImageNet-22k DINOv2-S/14 features",
        "image_encoder_license": "Apache-2.0",
        "device": str(device),
        "images": len(frame),
        "lesion_groups": int(frame.lesion_id.nunique()),
        "settings": {"folds": args.folds, "epochs": args.epochs, "batch_size": args.batch_size, "image_size": 518, "multiview_random_flip_training": args.multiview, "scoring_views": 4 if args.multiview else 1, "seed": args.seed, "teacher_c": args.teacher_c, "distill_weight": args.distill_weight, "class_weight_strength": args.class_weight_strength, "learning_rate": args.learning_rate, "teacher_targets": "not used in control-only run" if args.baseline_only else "fold-internal StratifiedGroupKFold logistic model on Hunch EmbeddingGemma image embeddings"},
        "aggregate_oof": aggregate,
        "folds_detail": fold_reports,
        "elapsed_seconds": time.monotonic() - started,
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
    }
    (args.output / "crossvalidation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for arm in records:
        for epoch in (9, args.epochs):
            score = aggregate[arm][epoch - 1]
            print(f"aggregate {arm} epoch {epoch}: macro F1 {score['macro_f1']:.4f}; balanced accuracy {score['balanced_accuracy']:.4f}", flush=True)


if __name__ == "__main__":
    main()

