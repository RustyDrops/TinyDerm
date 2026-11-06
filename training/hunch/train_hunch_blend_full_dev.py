"""Train ImageNet-started full-development baseline and Hunch blend members."""
import argparse
import hashlib
import io
import json
import random
import sqlite3
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
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
        self.base, self.probabilities = base, probabilities

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


def ordered_probabilities(model, matrix):
    prob = model.predict_proba(matrix)
    return prob[:, [list(model.classes_).index(c) for c in OFFICIAL_CLASSES]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=31415)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=288)
    parser.add_argument("--teacher-c", type=float, default=0.001)
    parser.add_argument("--distill-weight", type=float, default=0.35)
    parser.add_argument("--class-weight-strength", type=float, default=0.5)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--backbone-lr", type=float, default=1.5e-4)
    parser.add_argument("--fine-tune-head-lr", type=float, default=3e-4)
    parser.add_argument("--fine-tune-backbone-lr", type=float, default=5e-5)
    parser.add_argument("--head-phase-epochs", type=int, default=8)
    parser.add_argument("--fine-tune-epochs", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        existing = {path.name for path in args.output.iterdir()}
        if existing != {"embeddings.sqlite"}:
            raise FileExistsError(f"Output directory must contain only embeddings.sqlite before training: {args.output}")
    else:
        args.output.mkdir(parents=True)
    frame = development_rows(args.manifest)
    train = frame.split.eq("train").to_numpy()
    val = frame.split.eq("val").to_numpy()
    labels = frame.class_label.to_numpy()
    database = sqlite3.connect(args.cache)
    vectors = {key: blob for key, blob in database.execute("SELECT image_id, vector FROM vectors")}
    database.close()
    if set(frame.image_id) - set(vectors):
        raise ValueError("Hunch embedding cache does not cover development rows")
    x = np.stack([np.load(io.BytesIO(vectors[key]), allow_pickle=False) for key in frame.image_id])

    train_rows = np.flatnonzero(train)
    targets = np.zeros((len(frame), len(OFFICIAL_CLASSES)), dtype=np.float32)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed)
    for fold, (fit_idx, held_idx) in enumerate(splitter.split(x[train], labels[train], frame.loc[train, "lesion_id"]), start=1):
        teacher_fold = make_pipeline(StandardScaler(), LogisticRegression(C=args.teacher_c, class_weight="balanced", max_iter=2000, random_state=args.seed))
        teacher_fold.fit(x[train][fit_idx], labels[train][fit_idx])
        targets[train_rows[held_idx]] = ordered_probabilities(teacher_fold, x[train][held_idx])
        print(f"grouped Hunch teacher fold {fold}/5 complete", flush=True)
    teacher = make_pipeline(StandardScaler(), LogisticRegression(C=args.teacher_c, class_weight="balanced", max_iter=2000, random_state=args.seed))
    teacher.fit(x[train], labels[train])
    targets[val] = ordered_probabilities(teacher, x[val])
    joblib.dump(teacher, args.output / "hunch_teacher.joblib")
    mapping = dict(zip(frame.image_id.astype(str), targets))
    dataset = WithTeacher(DermDataset(frame, "dev", get_train_transforms(args.image_size)), mapping)
    counts = np.array([(frame.class_label == c).sum() for c in OFFICIAL_CLASSES], dtype=np.float64)
    class_weights = (counts.sum() / (len(OFFICIAL_CLASSES) * np.maximum(counts, 1))) ** args.class_weight_strength
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_weights = torch.tensor(class_weights / class_weights.mean(), dtype=torch.float32, device=device)
    history = {}
    for arm in ("baseline", "hunch_assisted"):
        seed_all(args.seed)
        model = TinyDermModel(pretrained=True).to(device)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        model_parameters = model.model.named_parameters()
        backbone = [p for n, p in model_parameters if "classifier" not in n]
        head = [p for n, p in model.model.named_parameters() if "classifier" in n]
        arm_history = []
        epochs = [(args.head_phase_epochs, args.backbone_lr, args.head_lr), (args.fine_tune_epochs, args.fine_tune_backbone_lr, args.fine_tune_head_lr)]
        epoch_number = 0
        for phase_index, (phase_epochs, backbone_lr, head_lr) in enumerate(epochs):
            if phase_epochs <= 0:
                continue
            if phase_index == 0:
                for parameter in backbone:
                    parameter.requires_grad = False
                for parameter in head:
                    parameter.requires_grad = True
                parameter_groups = [{"params": head, "lr": head_lr}]
            else:
                for parameter in model.parameters():
                    parameter.requires_grad = True
                parameter_groups = [{"params": backbone, "lr": backbone_lr}, {"params": head, "lr": head_lr}]
            optimizer = torch.optim.AdamW(parameter_groups, weight_decay=0.01)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=phase_epochs, eta_min=1e-6)
            for _ in range(phase_epochs):
                epoch_number += 1
                model.train()
                if phase_index == 0:
                    for module in model.modules():
                        if isinstance(module, nn.modules.batchnorm._BatchNorm):
                            module.eval()
                total_loss = 0.0
                for images, target, soft in loader:
                    images, target, soft = images.to(device), target.to(device), soft.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(images)
                    hard = F.cross_entropy(logits, target, weight=class_weights)
                    if arm == "hunch_assisted":
                        distill = F.kl_div(F.log_softmax(logits, dim=1), soft, reduction="batchmean")
                        loss = (1 - args.distill_weight) * hard + args.distill_weight * distill
                    else:
                        loss = hard
                    loss.backward()
                    optimizer.step()
                    total_loss += float(loss.item()) * len(target)
                scheduler.step()
                arm_history.append({"epoch": epoch_number, "phase": "head" if phase_index == 0 else "fine_tune", "training_loss": total_loss / len(dataset)})
                print(f"{arm} epoch {epoch_number}: training loss {arm_history[-1]['training_loss']:.4f}", flush=True)
                if epoch_number in {args.head_phase_epochs, args.head_phase_epochs + args.fine_tune_epochs}:
                    torch.save(model.state_dict(), args.output / f"{arm}_epoch{epoch_number}.pt")
            del optimizer, scheduler
        history[arm] = arm_history

    configuration = {
        "scope": "train plus validation; no performance metrics computed",
        "initialization": "cached ImageNet-pretrained timm MobileNetV3-Small",
        "device": str(device),
        "train_images": int(train.sum()),
        "validation_images_in_training_pool": int(val.sum()),
        "seed": args.seed,
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "teacher": {"head": "standardized class-balanced logistic regression", "c": args.teacher_c, "train_targets": "five-fold lesion-grouped OOF", "validation_targets": "teacher fit on training split"},
        "student": {"distill_weight": args.distill_weight, "class_weight_strength": args.class_weight_strength, "head_phase_epochs": args.head_phase_epochs, "fine_tune_epochs": args.fine_tune_epochs, "head_lr": args.head_lr, "backbone_lr": args.backbone_lr, "fine_tune_head_lr": args.fine_tune_head_lr, "fine_tune_backbone_lr": args.fine_tune_backbone_lr},
        "history": history,
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "embedding_revision": "914f7f89142e33e77833254d9c9b90c3cef7303b",
        "hunch_revision": "6541bd157e11a5e7d7434360e99ea81f40a8c181",
    }
    (args.output / "settings.json").write_text(json.dumps(configuration, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

