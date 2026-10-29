"""Split and Holdout integrity module for TinyDerm Lab.

Enforces strict lesion-grouped stratification:
- 90% Development (sub-split into ~81% Train / ~9% Val)
- 10% Sealed Final Holdout
- Manifest: data/splits/split_manifest.csv
- SHA256: data/splits/split_manifest.sha256
- Lock: data/splits/holdout_lock.json
"""

import datetime
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from tinyderm.provenance import compute_file_hash

logger = logging.getLogger(__name__)

OFFICIAL_CLASSES = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]


def evaluate_split_balance(df: pd.DataFrame, split_col: str = "split") -> float:
    """Compute sum of squared relative deviations in class distribution across splits."""
    overall_dist = df["class_label"].value_counts(normalize=True)
    score = 0.0
    for split_val in df[split_col].unique():
        split_dist = df[df[split_col] == split_val]["class_label"].value_counts(normalize=True)
        for c in OFFICIAL_CLASSES:
            expected = overall_dist.get(c, 0.0)
            actual = split_dist.get(c, 0.0)
            score += ((actual - expected) / (expected + 1e-6)) ** 2
    return float(score)


def create_lesion_grouped_splits(
    metadata_df: pd.DataFrame,
    images_dir: Path,
    seed: int = 42,
    max_seed_search: int = 50,
) -> pd.DataFrame:
    """Generates leakage-safe 81% train / 9% val / 10% holdout grouped strictly by lesion_id."""
    df = metadata_df.copy()
    if "source_path" not in df.columns:
        df["source_path"] = df["image_id"].apply(
            lambda img_id: str((images_dir / f"{img_id}.jpg").resolve())
        )

    # Determine dominant class per lesion
    lesion_summary = (
        df.groupby("lesion_id")
        .agg(
            primary_class=("class_label", lambda s: s.mode()[0] if not s.empty else "NV"),
            image_count=("image_id", "count"),
        )
        .reset_index()
    )

    best_seed = seed
    best_score = float("inf")
    best_manifest: Optional[pd.DataFrame] = None

    # Search deterministic seeds starting from default
    candidate_seeds = [seed] + [seed + i for i in range(1, max_seed_search)]

    for s in candidate_seeds:
        rng = np.random.RandomState(s)
        holdout_lesions = set()
        val_lesions = set()
        train_lesions = set()

        # Stratify by primary class
        for c in OFFICIAL_CLASSES:
            c_lesions = lesion_summary[lesion_summary["primary_class"] == c].copy()
            # Shuffle deterministically
            shuffled_ids = np.array(c_lesions["lesion_id"].tolist())
            rng.shuffle(shuffled_ids)

            n_total = len(shuffled_ids)
            n_holdout = max(1, int(round(n_total * 0.10))) if n_total >= 10 else (1 if n_total >= 3 else 0)
            n_val = max(1, int(round(n_total * 0.09))) if n_total >= 10 else (1 if n_total >= 5 else 0)

            c_holdout = set(shuffled_ids[:n_holdout])
            c_val = set(shuffled_ids[n_holdout : n_holdout + n_val])
            c_train = set(shuffled_ids[n_holdout + n_val :])

            holdout_lesions.update(c_holdout)
            val_lesions.update(c_val)
            train_lesions.update(c_train)

        def assign_split(lesion_id: str) -> str:
            if lesion_id in holdout_lesions:
                return "holdout"
            elif lesion_id in val_lesions:
                return "val"
            else:
                return "train"

        candidate_df = df.copy()
        candidate_df["split"] = candidate_df["lesion_id"].apply(assign_split)

        score = evaluate_split_balance(candidate_df)
        if score < best_score:
            best_score = score
            best_seed = s
            best_manifest = candidate_df

    logger.info(f"Selected seed {best_seed} for lesion-grouped split (balance score: {best_score:.4f})")
    assert best_manifest is not None, "Failed to create split manifest"
    return best_manifest


def verify_split_integrity(manifest_df: pd.DataFrame) -> Tuple[bool, str]:
    """Strictly verifies that no lesion_id appears in more than one split."""
    splits = manifest_df["split"].unique()
    lesion_sets = {s: set(manifest_df[manifest_df["split"] == s]["lesion_id"]) for s in splits}

    for s1 in splits:
        for s2 in splits:
            if s1 < s2:
                overlap = lesion_sets[s1].intersection(lesion_sets[s2])
                if overlap:
                    return False, f"DATA LEAKAGE DETECTED: {len(overlap)} lesions overlap between {s1} and {s2}!"

    # Verify all expected columns
    req_cols = ["image_id", "lesion_id", "class_label", "split", "source_path"]
    for col in req_cols:
        if col not in manifest_df.columns:
            return False, f"Missing required column in manifest: {col}"

    return True, "Split integrity verified: zero lesion leakage across splits."


def build_and_lock_splits(
    metadata_df: pd.DataFrame,
    images_dir: str | Path,
    splits_dir: str | Path = "data/splits",
    seed: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Builds split manifest, computes SHA256, and creates holdout_lock.json."""
    splits_dir = Path(splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)
    images_dir = Path(images_dir)

    manifest_csv = splits_dir / "split_manifest.csv"
    manifest_sha = splits_dir / "split_manifest.sha256"
    lock_file = splits_dir / "holdout_lock.json"

    # Reuse if already exists and valid
    if manifest_csv.exists() and lock_file.exists():
        existing_manifest = pd.read_csv(manifest_csv)
        valid, msg = verify_split_integrity(existing_manifest)
        if valid:
            current_hash = compute_file_hash(manifest_csv)
            with open(lock_file, "r", encoding="utf-8") as f:
                lock_data = json.load(f)
            if lock_data.get("split_hash") == current_hash:
                logger.info("Existing valid split manifest and holdout lock loaded.")
                return existing_manifest, lock_data

    # Generate splits
    manifest_df = create_lesion_grouped_splits(metadata_df, images_dir=images_dir, seed=seed)
    valid, msg = verify_split_integrity(manifest_df)
    if not valid:
        raise RuntimeError(msg)

    # Save manifest
    manifest_df.to_csv(manifest_csv, index=False)
    split_hash = compute_file_hash(manifest_csv)

    with open(manifest_sha, "w", encoding="utf-8") as f:
        f.write(f"{split_hash}  split_manifest.csv\n")

    holdout_subset = manifest_df[manifest_df["split"] == "holdout"]
    lock_data = {
        "split_hash": split_hash,
        "holdout_image_count": int(len(holdout_subset)),
        "holdout_lesion_count": int(holdout_subset["lesion_id"].nunique()),
        "class_counts": holdout_subset["class_label"].value_counts().to_dict(),
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "consumed": False,
        "seed_used": seed,
    }

    with open(lock_file, "w", encoding="utf-8") as f:
        json.dump(lock_data, f, indent=2)

    return manifest_df, lock_data


def mark_holdout_consumed(splits_dir: str | Path = "data/splits") -> Dict[str, Any]:
    """Marks holdout_lock.json as consumed=True after the single final evaluation."""
    lock_file = Path(splits_dir) / "holdout_lock.json"
    with open(lock_file, "r", encoding="utf-8") as f:
        lock_data = json.load(f)
    lock_data["consumed"] = True
    lock_data["consumed_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with open(lock_file, "w", encoding="utf-8") as f:
        json.dump(lock_data, f, indent=2)
    return lock_data
