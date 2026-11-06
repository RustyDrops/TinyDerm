"""Fit full-development DINOv2 classifier heads with cross-fitted Hunch targets."""
import argparse
import io
import json
import random
import sqlite3
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from crossvalidate_hunch_dinov2 import Head
from hunch_experiment import development_rows
from tinyderm.dataset import OFFICIAL_CLASSES


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ordered_probs(model, x):
    p = model.predict_proba(x)
    return p[:, [list(model.classes_).index(c) for c in OFFICIAL_CLASSES]]


def make_hunch_targets(frame, hunch, folds, seed, teacher_c):
    labels = frame.class_label.to_numpy()
    groups = frame.lesion_id.to_numpy()
    result = np.zeros((len(frame), len(OFFICIAL_CLASSES)), dtype=np.float32)
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    for fold, (fit_idx, target_idx) in enumerate(splitter.split(hunch, labels, groups), start=1):
        teacher = make_pipeline(StandardScaler(), LogisticRegression(C=teacher_c, class_weight="balanced", max_iter=2000, random_state=seed + fold))
        teacher.fit(hunch[fit_idx], labels[fit_idx])
        result[target_idx] = ordered_probs(teacher, hunch[target_idx])
        print(f"generated Hunch teacher targets for fold {fold}/{folds}", flush=True)
    return result


def train_head(features, labels, soft_targets, epochs, class_weights, device, seed, lr, alpha):
    seed_all(seed)
    model = Head(features.shape[-1], len(OFFICIAL_CLASSES)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    tx = torch.from_numpy(features)
    ty = torch.tensor([OFFICIAL_CLASSES.index(c) for c in labels], dtype=torch.long)
    ts = torch.from_numpy(soft_targets)
    generator = torch.Generator().manual_seed(seed)
    batch_size = 128
    for epoch in range(1, epochs + 1):
        model.train()
        permutation = torch.randperm(len(tx), generator=generator)
        random_views = torch.randint(tx.shape[1], (len(tx),), generator=generator) if tx.ndim == 3 else None
        loss_total = 0.0
        for start in range(0, len(tx), batch_size):
            idx = permutation[start:start + batch_size]
            xb = tx[idx, random_views[idx]].to(device) if random_views is not None else tx[idx].to(device)
            yb = ty[idx].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            hard = F.cross_entropy(logits, yb, weight=class_weights)
            if alpha:
                kd = F.kl_div(F.log_softmax(logits, dim=1), ts[idx].to(device), reduction="batchmean")
                loss = (1 - alpha) * hard + alpha * kd
            else:
                loss = hard
            loss.backward()
            optimizer.step()
            loss_total += float(loss.item()) * len(idx)
        scheduler.step()
        if epoch in (1, epochs):
            print(f"{('baseline' if not alpha else 'hunch_assisted')} epoch {epoch}/{epochs}; loss {loss_total/len(tx):.4f}", flush=True)
    return model.cpu()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--hunch-cache", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--teacher-folds", type=int, default=5)
    parser.add_argument("--teacher-c", type=float, default=0.001)
    parser.add_argument("--distill-weight", type=float, default=0.10)
    parser.add_argument("--class-weight-strength", type=float, default=0.5)
    parser.add_argument("--baseline-epochs", type=int, default=15)
    parser.add_argument("--hunch-epochs", type=int, default=29)
    parser.add_argument("--seed", type=int, default=42017)
    parser.add_argument("--reference-macro-f1", type=float, default=None)
    parser.add_argument("--reference-balanced-accuracy", type=float, default=None)
    parser.add_argument("--blend-hunch-weight", type=float, default=None)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    frame = development_rows(args.manifest)
    feature_file = np.load(args.feature_cache, allow_pickle=False)
    if feature_file["image_ids"].tolist() != frame.image_id.tolist():
        raise ValueError("DINOv2 feature cache does not align with development manifest")
    features = feature_file["features"].astype(np.float32)
    with sqlite3.connect(args.hunch_cache) as db:
        cache = {key: blob for key, blob in db.execute("SELECT image_id, vector FROM vectors")}
    if set(frame.image_id) - set(cache):
        raise ValueError("Hunch embedding cache does not cover the development pool")
    hunch = np.stack([np.load(io.BytesIO(cache[key]), allow_pickle=False) for key in frame.image_id])
    soft = make_hunch_targets(frame, hunch, args.teacher_folds, args.seed, args.teacher_c)
    labels = frame.class_label.to_numpy()
    counts = np.array([(labels == c).sum() for c in OFFICIAL_CLASSES], dtype=np.float64)
    cw = (counts.sum() / (len(OFFICIAL_CLASSES) * np.maximum(counts, 1))) ** args.class_weight_strength
    cw = torch.tensor(cw / cw.mean(), dtype=torch.float32, device="cuda" if torch.cuda.is_available() else "cpu")
    if features.ndim == 3:
        scaler = StandardScaler().fit(features.reshape(-1, features.shape[-1]))
        x = scaler.transform(features.reshape(-1, features.shape[-1])).reshape(features.shape).astype(np.float32)
    else:
        scaler = StandardScaler().fit(features)
        x = scaler.transform(features).astype(np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hunch_blend_weight = args.blend_hunch_weight if args.blend_hunch_weight is not None else (0.5 if features.ndim == 3 else 0.58)
    baseline = train_head(x, labels, soft, args.baseline_epochs, cw, device, args.seed, 0.001, 0.0)
    hunch_model = train_head(x, labels, soft, args.hunch_epochs, cw, device, args.seed, 0.001, args.distill_weight)
    torch.save(baseline.state_dict(), args.output / "baseline_head.pt")
    torch.save(hunch_model.state_dict(), args.output / "hunch_assisted_head.pt")
    joblib.dump(scaler, args.output / "feature_scaler.joblib")
    metadata = {
        "status": "full-development research refit; no validation performance measured",
        "scope": "train plus validation development rows only; original holdout excluded",
        "images": len(frame),
        "lesion_groups": int(frame.lesion_id.nunique()),
        "device": str(device),
        "image_encoder": "timm vit_small_patch14_dinov2.lvd142m; frozen Apache-2.0 ImageNet-pretrained DINOv2-S/14",
        "preprocessing": {"resize": [518, 518], "interpolation": "bicubic", "normalization_mean": [0.485, 0.456, 0.406], "normalization_std": [0.229, 0.224, 0.225]},
        "head": "LayerNorm(384)-Linear(384,256)-GELU-Dropout(0.30)-Linear(256,7)",
        "multiview_random_flip_training": features.ndim == 3,
        "inference_views": ["original", "horizontal_flip", "vertical_flip", "horizontal_and_vertical_flip"] if features.ndim == 3 else ["original"],
        "classes": OFFICIAL_CLASSES,
        "student_training": {"batch_size": 128, "learning_rate": 0.001, "weight_decay": 0.01, "class_weight_strength": args.class_weight_strength, "baseline_epochs": args.baseline_epochs, "hunch_epochs": args.hunch_epochs, "hunch_distillation_weight": args.distill_weight, "distillation": "soft target KL from local EmbeddingGemma Hunch logistic teacher"},
        "hunch_teacher": {"head": "StandardScaler + class-balanced LogisticRegression", "C": args.teacher_c, "targets": f"{args.teacher_folds}-fold StratifiedGroupKFold out-of-fold predictions by lesion"},
        "inference_blend": {"baseline_weight": 1 - hunch_blend_weight, "hunch_assisted_weight": hunch_blend_weight, "cross_fitted_class_bias": "mean fold-held calibration bias is added by inference script"},
        "cross_fitted_reference": {"macro_f1": args.reference_macro_f1, "balanced_accuracy": args.reference_balanced_accuracy, "metric": "fold-held epoch selection, adaptive Hunch blend, and second-level cross-fitted calibration; development OOF estimate only" if args.reference_macro_f1 is not None else "no cross-fitted reference supplied"},
    }
    (args.output / "model_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("saved full-development DINOv2 + Hunch research models", flush=True)


if __name__ == "__main__":
    main()

