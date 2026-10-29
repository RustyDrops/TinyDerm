"""Comprehensive test suite for TinyDerm Lab verifying Section 17.2 requirements."""

import os
from pathlib import Path
import numpy as np
import onnxruntime as ort
import pandas as pd
import pytest
import torch

from tinyderm.dataset import DermDataset, OFFICIAL_CLASSES
from tinyderm.experiment_db import ExperimentDB
from tinyderm.export import export_model_artifacts
from tinyderm.model import TinyDermModel
from tinyderm.split import build_and_lock_splits, verify_split_integrity


@pytest.fixture
def dummy_metadata(tmp_path):
    """Creates a controlled synthetic dermatology dataset for tests."""
    records = []
    # 20 lesions with varying numbers of images
    for i in range(20):
        lesion_id = f"L_{i:03d}"
        cls = OFFICIAL_CLASSES[i % len(OFFICIAL_CLASSES)]
        num_imgs = (i % 3) + 1
        for j in range(num_imgs):
            img_id = f"IMG_{i:03d}_{j}"
            img_path = tmp_path / f"{img_id}.jpg"
            # Create a blank 224x224 image
            from PIL import Image
            Image.new("RGB", (224, 224), color=(100, 150, 200)).save(img_path)

            records.append({
                "image_id": img_id,
                "lesion_id": lesion_id,
                "class_label": cls,
                "source_path": str(img_path),
            })
    return pd.DataFrame(records), tmp_path


def test_split_leakage_and_integrity(dummy_metadata, tmp_path):
    """Test 1: Split leakage test — zero lesion_id crosses splits."""
    df, img_dir = dummy_metadata
    splits_dir = tmp_path / "splits"

    manifest_df, lock_data = build_and_lock_splits(df, images_dir=img_dir, splits_dir=splits_dir, seed=42)

    valid, msg = verify_split_integrity(manifest_df)
    assert valid, msg

    train_lesions = set(manifest_df[manifest_df["split"] == "train"]["lesion_id"])
    val_lesions = set(manifest_df[manifest_df["split"] == "val"]["lesion_id"])
    holdout_lesions = set(manifest_df[manifest_df["split"] == "holdout"]["lesion_id"])

    assert len(train_lesions.intersection(val_lesions)) == 0, "Train and Val lesions overlap!"
    assert len(train_lesions.intersection(holdout_lesions)) == 0, "Train and Holdout lesions overlap!"
    assert len(val_lesions.intersection(holdout_lesions)) == 0, "Val and Holdout lesions overlap!"


def test_holdout_protection(dummy_metadata, tmp_path):
    """Test 2: Holdout protection test — loader cannot access holdout without permission."""
    df, img_dir = dummy_metadata
    splits_dir = tmp_path / "splits"
    manifest_df, _ = build_and_lock_splits(df, images_dir=img_dir, splits_dir=splits_dir, seed=42)

    # Attempting to load holdout without permission must raise PermissionError
    with pytest.raises(PermissionError):
        DermDataset(manifest_df, split="holdout", allow_holdout=False)

    # Explicit permission allowed
    ds = DermDataset(manifest_df, split="holdout", allow_holdout=True)
    assert len(ds) > 0


def test_label_and_dataset_sanity(dummy_metadata):
    """Test 3: Label test — only official classes are present."""
    df, _ = dummy_metadata
    unique_classes = set(df["class_label"].unique())
    assert unique_classes.issubset(set(OFFICIAL_CLASSES))
    assert len(unique_classes) == 7


def test_reproducibility(dummy_metadata, tmp_path):
    """Test 4: Reproducibility test — same seed produces identical split manifest hash."""
    df, img_dir = dummy_metadata
    s1_dir = tmp_path / "s1"
    s2_dir = tmp_path / "s2"

    _, lock1 = build_and_lock_splits(df, images_dir=img_dir, splits_dir=s1_dir, seed=42)
    _, lock2 = build_and_lock_splits(df, images_dir=img_dir, splits_dir=s2_dir, seed=42)

    assert lock1["split_hash"] == lock2["split_hash"]


def test_model_smoke_forward_backward():
    """Test 5: Model smoke test — one complete forward and backward pass."""
    model = TinyDermModel("mobilenetv3_small_100", pretrained=False, num_classes=7)
    x = torch.randn(2, 3, 224, 224)
    target = torch.tensor([0, 1])

    out = model(x)
    assert out.shape == (2, 7)

    loss = torch.nn.functional.cross_entropy(out, target)
    loss.backward()

    # Check gradients exist
    grad_found = any(p.grad is not None for p in model.parameters())
    assert grad_found


def test_export_and_onnx_smoke(tmp_path):
    """Test 6: Export smoke test — ONNX produces 7 class outputs."""
    model = TinyDermModel("mobilenetv3_small_100", pretrained=False, num_classes=7)
    artifacts = export_model_artifacts(model, output_dir=tmp_path)

    onnx_path = artifacts["onnx_fp32"]["path"]
    assert Path(onnx_path).exists()

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    inp_name = session.get_inputs()[0].name
    dummy_inp = np.random.randn(1, 3, 224, 224).astype(np.float32)

    outputs = session.run(None, {inp_name: dummy_inp})
    assert len(outputs) == 1
    assert outputs[0].shape == (1, 7)


def test_prediction_parity(tmp_path):
    """Test 7: Prediction parity test — PyTorch and ONNX outputs match closely."""
    model = TinyDermModel("mobilenetv3_small_100", pretrained=False, num_classes=7)
    model.eval()
    artifacts = export_model_artifacts(model, output_dir=tmp_path)

    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        pt_out = model(dummy_input).numpy()

    session = ort.InferenceSession(artifacts["onnx_fp32"]["path"], providers=["CPUExecutionProvider"])
    onnx_out = session.run(None, {session.get_inputs()[0].name: dummy_input.numpy()})[0]

    np.testing.assert_allclose(pt_out, onnx_out, rtol=1e-3, atol=1e-4)


def test_database_persistence(tmp_path):
    """Test 8: Database test — training events and predictions persist correctly."""
    db_file = tmp_path / "test.sqlite"
    db = ExperimentDB(db_file)

    run_id = "test_run_001"
    db.create_run(run_id, seed=42, config={"test": True}, hardware={"cpu": True})
    db.log_event(run_id, "TEST", "SETUP", "Test event message")
    db.log_predictions(
        run_id,
        split="val",
        predictions_list=[{
            "image_id": "TEST_001",
            "lesion_id": "L_001",
            "true_class": "NV",
            "predicted_class": "NV",
            "probability_true": 0.95,
            "probability_predicted": 0.95,
            "confidence": 0.95,
            "entropy": 0.1,
            "correct": True,
            "probabilities": {"NV": 0.95, "MEL": 0.05},
        }],
    )

    events = db.get_events(run_id)
    assert len(events) >= 1
    preds = db.get_predictions(run_id, split="val")
    assert len(preds) == 1
    assert preds[0]["image_id"] == "TEST_001"
