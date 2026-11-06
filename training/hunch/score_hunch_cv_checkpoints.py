"""Score saved fold checkpoints from the grouped Hunch CV experiment."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader

from hunch_experiment import development_rows
from tinyderm.augment import get_eval_transforms
from tinyderm.dataset import DermDataset, OFFICIAL_CLASSES
from tinyderm.model import TinyDermModel


def metric_report(y_true, probabilities):
    predicted = np.asarray(OFFICIAL_CLASSES)[probabilities.argmax(axis=1)]
    return {
        "macro_f1": float(f1_score(y_true, predicted, labels=OFFICIAL_CLASSES, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "per_class": classification_report(y_true, predicted, labels=OFFICIAL_CLASSES, output_dict=True, zero_division=0),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--checkpoints", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=31415)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=288)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    frame = development_rows(args.manifest)
    labels = frame.class_label.to_numpy()
    groups = frame.lesion_id.to_numpy()
    splitter = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    splits = list(splitter.split(np.zeros(len(frame)), labels, groups))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    oof_plain = {arm: np.empty((len(frame), len(OFFICIAL_CLASSES)), dtype=np.float32) for arm in ("baseline", "hunch_assisted")}
    oof_tta = {arm: np.empty((len(frame), len(OFFICIAL_CLASSES)), dtype=np.float32) for arm in ("baseline", "hunch_assisted")}
    per_fold = []
    for fold, (model_train_idx, held_idx) in enumerate(splits, start=1):
        held = frame.iloc[held_idx].copy().reset_index(drop=True)
        held["split"] = "val"
        loader = DataLoader(DermDataset(held, "val", get_eval_transforms(args.image_size)), batch_size=args.batch_size, shuffle=False, num_workers=0)
        fold_report = {"fold": fold, "heldout_images": len(held), "arms": {}}
        fold_probs = {}
        for arm in oof_plain:
            path = args.checkpoints / f"fold{fold}_{arm}_epoch{args.epochs}.pt"
            model = TinyDermModel(pretrained=False).to(device)
            model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
            model.eval()
            plain_chunks, tta_chunks = [], []
            with torch.inference_mode():
                for images, _, _, _ in loader:
                    images = images.to(device)
                    views = [images, torch.flip(images, [3]), torch.flip(images, [2]), torch.flip(images, [2, 3])]
                    view_probabilities = [torch.softmax(model(view), dim=1) for view in views]
                    plain_chunks.append(view_probabilities[0].cpu().numpy())
                    tta_chunks.append(torch.stack(view_probabilities).mean(dim=0).cpu().numpy())
            plain = np.concatenate(plain_chunks)
            tta = np.concatenate(tta_chunks)
            oof_plain[arm][held_idx] = plain
            oof_tta[arm][held_idx] = tta
            fold_probs[arm] = {"plain": plain, "tta": tta}
            fold_report["arms"][arm] = {"plain": metric_report(labels[held_idx], plain), "four_view_flip_tta": metric_report(labels[held_idx], tta)}
        per_fold.append(fold_report)
        np.savez_compressed(args.output / f"fold{fold}_oof_predictions.npz", held_indices=held_idx, baseline=fold_probs["baseline"]["tta"], hunch_assisted=fold_probs["hunch_assisted"]["tta"], baseline_plain=fold_probs["baseline"]["plain"], hunch_assisted_plain=fold_probs["hunch_assisted"]["plain"])
        print(f"scored held-out fold {fold}/{args.folds}", flush=True)
    report = {
        "scope": "5-fold lesion-grouped out-of-fold estimate across train plus validation; consumed holdout excluded",
        "initialization": "ImageNet-pretrained MobileNetV3-Small",
        "epoch": args.epochs,
        "device": str(device),
        "aggregate_oof": {
            arm: {"plain": metric_report(labels, oof_plain[arm]), "four_view_flip_tta": metric_report(labels, oof_tta[arm])}
            for arm in oof_plain
        },
        "folds_detail": per_fold,
    }
    (args.output / "crossvalidation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for arm, modes in report["aggregate_oof"].items():
        for mode, metric in modes.items():
            print(f"{arm} {mode}: macro F1 {metric['macro_f1']:.4f}; balanced accuracy {metric['balanced_accuracy']:.4f}", flush=True)


if __name__ == "__main__":
    main()

