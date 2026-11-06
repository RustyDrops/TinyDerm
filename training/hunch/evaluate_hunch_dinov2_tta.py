"""Evaluate saved fine-tuned DINOv2 fold models with four flip views."""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
os.environ.setdefault("HF_HOME", str(Path(__file__).parent / "model-cache"))
import torch
from sklearn.metrics import balanced_accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader

from hunch_experiment import development_rows
from tinyderm.dataset import OFFICIAL_CLASSES
from crossvalidate_hunch_dinov2_finetune import DinoStudent, LabeledImages
from tinyderm.augment import get_eval_transforms


def metric(y, probability):
    prediction = np.asarray(OFFICIAL_CLASSES)[probability.argmax(axis=1)]
    return {
        "macro_f1": float(f1_score(y, prediction, labels=OFFICIAL_CLASSES, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "per_class": classification_report(y, prediction, labels=OFFICIAL_CLASSES, output_dict=True, zero_division=0),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=52711)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--unfreeze-blocks", type=int, default=4)
    parser.add_argument("--epoch", type=int, default=3)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)

    frame = development_rows(args.manifest)
    labels = frame.class_label.to_numpy()
    groups = frame.lesion_id.to_numpy()
    splits = list(StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed).split(np.zeros(len(frame)), labels, groups))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arms = ("baseline", "hunch_assisted")
    oof = {arm: np.zeros((len(frame), len(OFFICIAL_CLASSES)), dtype=np.float32) for arm in arms}
    started = time.monotonic()

    for fold, (_, held_idx) in enumerate(splits, start=1):
        held = frame.iloc[held_idx].reset_index(drop=True)
        loader = DataLoader(
            LabeledImages(held, get_eval_transforms(args.image_size)),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
        )
        fold_outputs = {}
        for arm in arms:
            checkpoint = args.checkpoints / f"fold{fold}_{arm}_epoch{args.epoch}.pt"
            model = DinoStudent(args.unfreeze_blocks).to(device)
            model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
            model.eval()
            chunks = []
            with torch.inference_mode():
                for images, _ in loader:
                    images = images.to(device, non_blocking=True)
                    views = torch.stack((images, torch.flip(images, [3]), torch.flip(images, [2]), torch.flip(images, [2, 3])), dim=1)
                    batch, view_count, channels, height, width = views.shape
                    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                        logits = model(views.reshape(batch * view_count, channels, height, width))
                    probability = torch.softmax(logits.float(), dim=1).reshape(batch, view_count, -1).mean(dim=1)
                    chunks.append(probability.cpu().numpy())
            probability = np.concatenate(chunks)
            oof[arm][held_idx] = probability
            fold_outputs[arm] = probability
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        np.savez_compressed(args.output / f"fold{fold}_tta_predictions.npz", held_indices=held_idx, baseline=fold_outputs["baseline"], hunch_assisted=fold_outputs["hunch_assisted"])
        print(f"completed four-view fold {fold}/{args.folds}", flush=True)

    summary = {
        "scope": f"{args.folds}-fold lesion-grouped four-flip OOF on expanded development data; original holdout excluded",
        "images": len(frame),
        "lesion_groups": int(frame.lesion_id.nunique()),
        "device": str(device),
        "settings": vars(args) | {"manifest": str(args.manifest), "checkpoints": str(args.checkpoints), "output": str(args.output), "views": ["identity", "horizontal_flip", "vertical_flip", "horizontal_vertical_flip"]},
        "aggregate_oof": {arm: metric(labels, probability) for arm, probability in oof.items()},
        "elapsed_seconds": time.monotonic() - started,
    }
    (args.output / "evaluation.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    for arm, result in summary["aggregate_oof"].items():
        print(f"aggregate {arm}: macro F1 {result['macro_f1']:.4f}; balanced accuracy {result['balanced_accuracy']:.4f}", flush=True)


if __name__ == "__main__":
    main()

