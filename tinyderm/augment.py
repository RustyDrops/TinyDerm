"""Data augmentation module for TinyDerm Lab.

Preserves medically relevant color cues while providing standard dermoscopic transforms:
- Horizontal / Vertical Flip
- Modest rotation (+/- 20 degrees)
- Modest crop/scale (0.85 to 1.0)
- Restrained brightness/contrast variation (no wild hue alterations)
- Deterministic transforms for validation and sealed holdout
"""

import torchvision.transforms as T
from PIL import Image

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(image_size: int = 224) -> T.Compose:
    """Training transformations with restrained medical augmentations."""
    return T.Compose([
        T.RandomResizedCrop(image_size, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomRotation(degrees=20),
        T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.0),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_eval_transforms(image_size: int = 224) -> T.Compose:
    """Deterministic transformations for validation and final holdout."""
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
