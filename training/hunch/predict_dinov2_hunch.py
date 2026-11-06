"""Run the full-development DINOv2 + Hunch research candidate on one image."""
import argparse
import json
import os
from pathlib import Path

import joblib
import numpy as np
os.environ.setdefault("HF_HOME", str(Path(__file__).parent / "model-cache"))
import timm
import torch
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

from crossvalidate_hunch_dinov2 import Head
from tinyderm.dataset import OFFICIAL_CLASSES


def load_candidate(candidate_dir: Path, device: torch.device):
    config = json.loads((candidate_dir / "model_config.json").read_text(encoding="utf-8"))
    encoder = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True, num_classes=0, dynamic_img_size=True).to(device).eval()
    baseline = Head(encoder.num_features, len(OFFICIAL_CLASSES))
    assisted = Head(encoder.num_features, len(OFFICIAL_CLASSES))
    baseline.load_state_dict(torch.load(candidate_dir / "baseline_head.pt", map_location="cpu", weights_only=True))
    assisted.load_state_dict(torch.load(candidate_dir / "hunch_assisted_head.pt", map_location="cpu", weights_only=True))
    baseline.to(device).eval()
    assisted.to(device).eval()
    scaler = joblib.load(candidate_dir / "feature_scaler.joblib")
    preprocess = T.Compose([
        T.Resize((518, 518), interpolation=InterpolationMode.BICUBIC, antialias=True),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    return encoder, baseline, assisted, scaler, preprocess, config


def predict(image_path: Path, candidate_dir: Path, device: torch.device):
    encoder, baseline, assisted, scaler, preprocess, config = load_candidate(candidate_dir, device)
    with Image.open(image_path) as image:
        image = image.convert("RGB")
    x = preprocess(image).unsqueeze(0).to(device)
    with torch.inference_mode():
        view_names = config.get("inference_views", ["original"])
        views = [x]
        if len(view_names) == 4:
            views = [x, torch.flip(x, [3]), torch.flip(x, [2]), torch.flip(x, [2, 3])]
        features = torch.stack([encoder(view).float()[0] for view in views]).cpu().numpy()
        scaled = torch.from_numpy(scaler.transform(features).astype(np.float32)).to(device)
        baseline_prob = torch.softmax(baseline(scaled), dim=1).mean(dim=0, keepdim=True)
        assisted_prob = torch.softmax(assisted(scaled), dim=1).mean(dim=0, keepdim=True)
        blend = config["inference_blend"]
        probability = blend["baseline_weight"] * baseline_prob + blend["hunch_assisted_weight"] * assisted_prob
        bias = torch.tensor(blend.get("class_bias", [0.0] * len(OFFICIAL_CLASSES)), dtype=probability.dtype, device=device)
        log_probability = torch.log(probability.clamp_min(1e-8)) + bias
        probability = torch.softmax(log_probability, dim=1)[0].cpu().numpy()
    ranking = np.argsort(probability)[::-1]
    return [{"class": OFFICIAL_CLASSES[i], "probability": float(probability[i])} for i in ranking]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--candidate", type=Path, default=Path(__file__).parent / "dinov2-hunch-full-dev-candidate")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    result = predict(args.image, args.candidate, torch.device(args.device))
    print(json.dumps({"classes": result, "status": "research candidate; not clinically validated"}, indent=2))


if __name__ == "__main__":
    main()

