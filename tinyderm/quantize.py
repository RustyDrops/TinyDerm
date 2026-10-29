"""Model quantization module for TinyDerm Lab.

Quantizes ONNX FP32 to INT8 using development data only.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional
import onnx
from onnxruntime.quantization import QuantType, quantize_dynamic

from tinyderm.provenance import compute_file_hash

logger = logging.getLogger(__name__)


def quantize_onnx_model(
    onnx_fp32_path: str | Path,
    output_dir: str | Path,
) -> Optional[Dict[str, Any]]:
    """Applies dynamic INT8 quantization to ONNX model."""
    onnx_fp32_path = Path(onnx_fp32_path)
    output_dir = Path(output_dir)
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    int8_path = models_dir / "mobilenetv3_student_int8.onnx"

    try:
        quantize_dynamic(
            model_input=str(onnx_fp32_path),
            model_output=str(int8_path),
            weight_type=QuantType.QInt8,
        )

        size_mb = round(int8_path.stat().st_size / (1024 * 1024), 2)
        sha256 = compute_file_hash(int8_path)

        logger.info(f"Quantized ONNX model generated at {int8_path} ({size_mb} MB)")
        return {
            "path": str(int8_path),
            "size_mb": size_mb,
            "sha256": sha256,
        }
    except Exception as e:
        logger.error(f"Quantization failed: {e}")
        return None
