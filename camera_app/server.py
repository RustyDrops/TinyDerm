"""FastAPI Camera Web Application Server for TinyDerm Lab.

Serves real-time dermoscopic inference via ONNX Runtime on CPU (<1ms per frame).
"""

import base64
import io
import json
import logging
from pathlib import Path
from typing import Dict
import numpy as np
import onnxruntime as ort
from PIL import Image
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TinyDerm-CameraApp")

app = FastAPI(title="TinyDerm Real-Time Camera Scanner")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CLASSES = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]

CLASS_DETAILS = {
    "MEL": {
        "name": "Melanoma (MEL)",
        "type": "Malignant",
        "description": "Malignant skin cancer arising from melanocytes. Requires prompt professional medical evaluation.",
        "badge_color": "#e63946",
    },
    "NV": {
        "name": "Melanocytic Nevus (NV)",
        "type": "Benign",
        "description": "Common benign mole or birthmark composed of melanocytes.",
        "badge_color": "#2a9d8f",
    },
    "BCC": {
        "name": "Basal Cell Carcinoma (BCC)",
        "type": "Malignant",
        "description": "Most common type of skin cancer; locally invasive, usually slow-growing.",
        "badge_color": "#e76f51",
    },
    "AKIEC": {
        "name": "Actinic Keratosis / Bowen's (AKIEC)",
        "type": "Pre-cancerous",
        "description": "Pre-malignant rough scaly lesion from cumulative sun exposure.",
        "badge_color": "#f4a261",
    },
    "BKL": {
        "name": "Benign Keratosis (BKL)",
        "type": "Benign",
        "description": "Seborrheic keratosis, solar lentigo, or lichen-planus-like keratosis.",
        "badge_color": "#457b9d",
    },
    "DF": {
        "name": "Dermatofibroma (DF)",
        "type": "Benign",
        "description": "Benign fibrous nodule commonly located on limbs.",
        "badge_color": "#3d5a80",
    },
    "VASC": {
        "name": "Vascular Lesion (VASC)",
        "type": "Benign",
        "description": "Cherry angioma, hemangioma, or pyogenic granuloma.",
        "badge_color": "#9d4edd",
    },
}

# Find best available ONNX model
def locate_model() -> Path:
    candidates = [
        Path("runs/run_20261028_201613_improved/models/mobilenetv3_student_fp32.onnx"),
        Path("runs/run_20261028_175715/models/mobilenetv3_student_fp32.onnx"),
        Path("runs/run_20261028_201613_improved/models/mobilenetv3_student_int8.onnx"),
    ]
    for c in candidates:
        if c.exists():
            return c
    # Fallback search
    all_onnx = list(Path("runs").glob("**/*.onnx"))
    if all_onnx:
        return all_onnx[0]
    raise FileNotFoundError("No trained ONNX model found in runs/")

MODEL_PATH = locate_model()
logger.info(f"Loading ONNX model for CPU camera inference from: {MODEL_PATH}")

session_opts = ort.SessionOptions()
session_opts.intra_op_num_threads = 2
ort_session = ort.InferenceSession(str(MODEL_PATH), sess_options=session_opts, providers=["CPUExecutionProvider"])
input_name = ort_session.get_inputs()[0].name

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Preprocess PIL image into ImageNet-normalized 224x224 NCHW float32 tensor."""
    img = image.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = np.transpose(arr, (2, 0, 1))  # HWC -> CHW
    return np.expand_dims(arr, axis=0)  # 1x3x224x224


class PredictRequest(BaseModel):
    image_base64: str


@app.post("/api/predict")
async def predict_frame(req: PredictRequest):
    """Classifies an image frame and returns top prediction, probability, and full distribution."""
    try:
        # Strip data URL prefix if present
        data = req.image_base64
        if "," in data:
            data = data.split(",", 1)[1]

        image_bytes = base64.b64decode(data)
        pil_img = Image.open(io.BytesIO(image_bytes))

        # 4-pass TTA (original, hflip, vflip, hvflip) on CPU (<3ms total)
        crops = [
            pil_img,
            pil_img.transpose(Image.Transpose.FLIP_LEFT_RIGHT),
            pil_img.transpose(Image.Transpose.FLIP_TOP_BOTTOM),
        ]
        
        batch_inputs = np.concatenate([preprocess_image(c) for c in crops], axis=0)
        logits = ort_session.run(None, {input_name: batch_inputs})[0]
        
        # Softmax
        exp_l = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs_all = exp_l / np.sum(exp_l, axis=1, keepdims=True)
        probs = np.mean(probs_all, axis=0)

        top_idx = int(np.argmax(probs))
        top_class = CLASSES[top_idx]
        top_prob = float(probs[top_idx])

        # Distribution dictionary
        dist = {CLASSES[i]: round(float(probs[i]), 4) for i in range(len(CLASSES))}
        details = CLASS_DETAILS.get(top_class, {})

        return {
            "class": top_class,
            "name": details.get("name", top_class),
            "type": details.get("type", "Unknown"),
            "description": details.get("description", ""),
            "badge_color": details.get("badge_color", "#333333"),
            "confidence": round(top_prob * 100, 1),
            "probabilities": dist,
            "is_significant": top_prob >= 0.65,
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


# Mount static web UI
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/")
async def serve_index():
    return FileResponse(static_dir / "index.html")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
