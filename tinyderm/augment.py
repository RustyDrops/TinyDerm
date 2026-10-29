"""Data augmentation module for TinyDerm Lab.

Preserves medically relevant color cues while providing enhanced dermoscopic transforms:
- Horizontal / Vertical Flip
- Dermoscopic rotation invariance (+/- 45 degrees)
- Modest crop/scale (0.80 to 1.0)
- Restrained color jitter (brightness/contrast, zero hue distortion)
- Random erasing / cutout to prevent overfitting on hair or skin markings
- Deterministic transforms for validation and sealed holdout
"""

import torchvision.transforms as T
from PIL import Image

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(image_size: int = 224) -> T.Compose:
    """Enhanced training transformations with dermoscopic rotational invariance."""
    return T.Compose([
        T.RandomResizedCrop(image_size, scale=(0.80, 1.0), ratio=(0.9, 1.1)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomRotation(degrees=45),
        T.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08, hue=0.0),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        T.RandomErasing(p=0.25, scale=(0.02, 0.15), value="random"),
    ])


def get_eval_transforms(image_size: int = 224) -> T.Compose:
    """Deterministic transformations for validation and final holdout."""
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
