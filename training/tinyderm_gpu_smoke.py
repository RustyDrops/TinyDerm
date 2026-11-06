"""Tiny development-only smoke run through the original TinyDerm trainer."""
from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(os.environ.get("TINYDERM_ROOT", Path(__file__).resolve().parents[1]))
MANIFEST = Path(os.environ.get("TINYDERM_MANIFEST", PROJECT_ROOT / "data" / "splits" / "split_manifest.csv"))
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import torch
import yaml
from tinyderm.experiment_db import ExperimentDB
from tinyderm.train import get_device, train_student_model


def main() -> int:
    if not torch.cuda.is_available():
        print(json.dumps({"cuda_available": False, "device": "cpu", "campaign_approved": False}))
        return 2
    frame = pd.read_csv(MANIFEST)
    # Only the fixed development train/validation partitions enter this smoke run.
    frame = frame[frame["split"].isin(["train", "val"])].copy()
    frame = frame.groupby(["split", "class_label"], group_keys=False).head(3).copy()
    if not set(frame["split"]) == {"train", "val"}:
        raise RuntimeError("Smoke subset did not contain both train and validation partitions")
    config = yaml.safe_load((DOWNLOADS / "config" / "default.yaml").read_text(encoding="utf-8"))
    config["training"].update({"head_epochs": 1, "finetune_max_epochs": 1,
                               "early_stopping_patience": 1, "batch_size": 8, "num_workers": 0})
    config["model"].update({"pretrained": False, "image_size": 128})
    run_id = "cuda-smoke-" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = ROOT / "tinyderm_research" / "controller" / "gpu_smoke"
    db = ExperimentDB(output / "experiments.sqlite")
    db.create_run(run_id, seed=73, config=config,
                  hardware={"python": sys.version, "torch": torch.__version__,
                            "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(0)})
    device = get_device()
    model, metrics = train_student_model(frame, config, db, run_id, output_dir=output / "runs")
    parameter_device = str(next(model.parameters()).device)
    result = {"run_id": run_id, "pid": os.getpid(), "device": str(device),
              "parameter_device": parameter_device, "cuda_device": torch.cuda.get_device_name(0),
              "cuda_allocated_bytes": torch.cuda.memory_allocated(device),
              "train_rows": int((frame["split"] == "train").sum()),
              "validation_rows": int((frame["split"] == "val").sum()),
              "holdout_rows_loaded": 0, "smoke_metrics": metrics}
    if device.type != "cuda" or parameter_device != "cuda:0" or result["cuda_allocated_bytes"] <= 0:
        raise RuntimeError("The actual TinyDerm trainer did not train on CUDA")
    print(json.dumps(result, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

