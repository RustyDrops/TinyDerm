"""PyTorch Dataset implementation for TinyDerm Lab.

Includes programmatic protection to prevent accidental holdout access.
"""

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

OFFICIAL_CLASSES = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
CLASS_TO_IDX = {c: i for i, c in enumerate(OFFICIAL_CLASSES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(OFFICIAL_CLASSES)}


class DermDataset(Dataset):
    """Dataset for dermatology lesion classification."""

    def __init__(
        self,
        manifest_df: pd.DataFrame,
        split: str,
        transform: Optional[Callable] = None,
        allow_holdout: bool = False,
    ):
        if split == "holdout" and not allow_holdout:
            raise PermissionError(
                "Holdout dataset access blocked! Sealed 10% holdout cannot be loaded without explicit authorization."
            )

        if split == "dev":
            self.df = manifest_df[manifest_df["split"].isin(["train", "val"])].reset_index(drop=True)
        else:
            self.df = manifest_df[manifest_df["split"] == split].reset_index(drop=True)

        self.split = split
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str, str]:
        row = self.df.iloc[idx]
        image_path = row["source_path"]

        try:
            with Image.open(image_path) as img:
                img_rgb = img.convert("RGB")
        except Exception as e:
            # Fallback placeholder if image corrupted
            img_rgb = Image.new("RGB", (224, 224), (128, 128, 128))

        if self.transform is not None:
            tensor_img = self.transform(img_rgb)
        else:
            tensor_img = torch.zeros((3, 224, 224), dtype=torch.float32)

        class_name = row["class_label"]
        class_idx = CLASS_TO_IDX.get(class_name, 0)
        image_id = str(row["image_id"])
        lesion_id = str(row["lesion_id"])

        return tensor_img, class_idx, image_id, lesion_id

    def get_class_counts(self) -> Dict[str, int]:
        return self.df["class_label"].value_counts().to_dict()

    def get_class_weights(self) -> torch.Tensor:
        """Calculate smoothed inverse-frequency class weights for focal loss."""
        counts = [self.df[self.df["class_label"] == c].shape[0] for c in OFFICIAL_CLASSES]
        total = sum(counts)
        num_classes = len(OFFICIAL_CLASSES)
        weights = []
        for count in counts:
            if count > 0:
                # Square root smoothed inverse frequency
                w = (total / (count * num_classes)) ** 0.5
                weights.append(w)
            else:
                weights.append(1.0)
        weights_t = torch.tensor(weights, dtype=torch.float32)
        # Normalize so mean is 1.0
        return weights_t / weights_t.mean()
