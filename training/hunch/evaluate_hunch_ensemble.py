"""Evaluate seed ensembles and flip averaging on development validation only."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader

from hunch_experiment import development_rows
from tinyderm.augment import get_eval_transforms
from tinyderm.dataset import DermDataset, OFFICIAL_CLASSES
from tinyderm.model import TinyDermModel


def scores(y_true, probabilities):
    y_pred = np.asarray(OFFICIAL_CLASSES)[probabilities.argmax(axis=1)]
    return {
        "macro_f1": float(f1_score(y_true, y_pred, labels=OFFICIAL_CLASSES, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "per_class": classification_report(y_true, y_pred, labels=OFFICIAL_CLASSES, output_dict=True, zero_division=0),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkpoint", action="append", required=True, help="Repeat as LABEL=PATH")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    frame = development_rows(args.manifest)
    val = frame[frame.split == "val"].copy().reset_index(drop=True)
    dataset = DermDataset(val, "val", get_eval_transforms(288))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = []
    for item in args.checkpoint:
        label, separator, path = item.partition("=")
        if not separator:
            raise ValueError("Checkpoints must be LABEL=PATH")
        model = TinyDermModel(pretrained=False).to(device)
        model.load_state_dict(torch.load(Path(path), map_location=device, weights_only=True))
        model.eval()
        models.append((label, model))

    actual = []
    per_model_plain = {label: [] for label, _ in models}
    per_model_tta = {label: [] for label, _ in models}
    ensemble_plain = []
    ensemble_tta = []
    with torch.inference_mode():
        for images, labels, _, _ in loader:
            images = images.to(device)
            actual.extend(np.asarray([OFFICIAL_CLASSES[i] for i in labels.numpy()]))
            batch_plain, batch_tta = [], []
            views = [images, torch.flip(images, [3]), torch.flip(images, [2]), torch.flip(images, [2, 3])]
            for name, model in models:
                plain = torch.softmax(model(images), dim=1)
                tta = torch.stack([torch.softmax(model(view), dim=1) for view in views]).mean(dim=0)
                plain_np, tta_np = plain.cpu().numpy(), tta.cpu().numpy()
                per_model_plain[name].append(plain_np)
                per_model_tta[name].append(tta_np)
                batch_plain.append(plain)
                batch_tta.append(tta)
            ensemble_plain.append(torch.stack(batch_plain).mean(dim=0).cpu().numpy())
            ensemble_tta.append(torch.stack(batch_tta).mean(dim=0).cpu().numpy())

    report = {
        "scope": "reused development validation only; not an independent test",
        "device": str(device),
        "validation_images": len(actual),
        "ensemble": {},
        "individual": {},
    }
    for label in per_model_plain:
        plain = np.concatenate(per_model_plain[label])
        tta = np.concatenate(per_model_tta[label])
        report["individual"][label] = {"plain": scores(actual, plain), "four_view_flip_tta": scores(actual, tta)}
    report["ensemble"]["plain"] = scores(actual, np.concatenate(ensemble_plain))
    report["ensemble"]["four_view_flip_tta"] = scores(actual, np.concatenate(ensemble_tta))
    report["checkpoints"] = [{"label": label, "path": str(path)} for label, path in (x.split("=", 1) for x in args.checkpoint)]
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for mode, metric in report["ensemble"].items():
        print(f"ensemble {mode}: macro F1 {metric['macro_f1']:.4f}; balanced accuracy {metric['balanced_accuracy']:.4f}", flush=True)
    for label, variants in report["individual"].items():
        print(f"{label} plain {variants['plain']['macro_f1']:.4f}/{variants['plain']['balanced_accuracy']:.4f}; TTA {variants['four_view_flip_tta']['macro_f1']:.4f}/{variants['four_view_flip_tta']['balanced_accuracy']:.4f}", flush=True)


if __name__ == "__main__":
    main()

