"""Model export module for TinyDerm Lab.

Exports:
- PyTorch Checkpoint (.pt)
- TorchScript (.pt)
- ONNX FP32 (.onnx)
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Tuple
import torch
import torch.nn as nn

from tinyderm.provenance import compute_file_hash

logger = logging.getLogger(__name__)


def export_model_artifacts(
    model: nn.Module,
    output_dir: str | Path,
    image_size: int = 224,
) -> Dict[str, Any]:
    """Exports PyTorch checkpoint, TorchScript, and ONNX FP32 models."""
    output_dir = Path(output_dir)
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    original_device = next(model.parameters()).device
    model.eval()
    cpu_model = model.cpu()
    dummy_input = torch.randn(1, 3, image_size, image_size, dtype=torch.float32)

    artifacts = {}

    # 1. PyTorch checkpoint
    pt_path = models_dir / "mobilenetv3_student_fp32.pt"
    torch.save(cpu_model.state_dict(), pt_path)
    artifacts["pytorch_checkpoint"] = {
        "path": str(pt_path),
        "size_mb": round(pt_path.stat().st_size / (1024 * 1024), 2),
        "sha256": compute_file_hash(pt_path),
    }

    # 2. TorchScript
    ts_path = models_dir / "mobilenetv3_student_torchscript.pt"
    try:
        scripted_model = torch.jit.trace(cpu_model, dummy_input)
        scripted_model.save(str(ts_path))
        artifacts["torchscript"] = {
            "path": str(ts_path),
            "size_mb": round(ts_path.stat().st_size / (1024 * 1024), 2),
            "sha256": compute_file_hash(ts_path),
        }
    except Exception as e:
        logger.warning(f"TorchScript tracing failed: {e}")

    # 3. ONNX FP32
    onnx_path = models_dir / "mobilenetv3_student_fp32.onnx"
    torch.onnx.export(
        cpu_model,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
    )
    artifacts["onnx_fp32"] = {
        "path": str(onnx_path),
        "size_mb": round(onnx_path.stat().st_size / (1024 * 1024), 2),
        "sha256": compute_file_hash(onnx_path),
    }

    # Restore model device
    model.to(original_device)

    # Save manifest
    manifest_path = output_dir / "model_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(artifacts, f, indent=2)

    return artifacts
