"""Dataset acquisition module for TinyDerm Lab.

Handles the official 3-tier acquisition order:
1. Official ISIC CLI filtered to collection 66
2. Official ISIC Archive API for collection 66
3. Official ISIC 2018 Task 3 archive (S3 direct)
"""

import io
import json
import logging
import os
import shutil
import subprocess
import time
import zipfile
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple
import pandas as pd
import requests

from tinyderm.provenance import record_provenance

logger = logging.getLogger(__name__)

OFFICIAL_CLASSES = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
GROUNDTRUTH_URL = "https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task3_Training_GroundTruth.zip"
INPUT_IMAGES_URL = "https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task3_Training_Input.zip"


def download_file_with_progress(
    url: str,
    output_path: Path,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    chunk_size: int = 1024 * 1024,
) -> Path:
    """Download a file with progress updates."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".part")

    with requests.get(url, stream=True, timeout=30) as r:
        r.raise_for_status()
        total_size = int(r.headers.get("content-length", 0))
        downloaded = 0
        t0 = time.time()

        with open(temp_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if progress_callback and total_size > 0:
                        pct = downloaded / total_size
                        elapsed = max(time.time() - t0, 0.001)
                        speed_mb = (downloaded / (1024 * 1024)) / elapsed
                        msg = f"Downloaded {downloaded / (1024*1024):.1f}/{total_size / (1024*1024):.1f} MB ({speed_mb:.1f} MB/s)"
                        progress_callback(pct, msg)

    if output_path.exists():
        output_path.unlink()
    temp_path.rename(output_path)
    return output_path


def acquire_groundtruth(raw_dir: Path) -> pd.DataFrame:
    """Acquire the official ISIC 2018 Task 3 Ground Truth CSV."""
    gt_file = raw_dir / "groundtruth.csv"
    if gt_file.exists():
        df = pd.read_csv(gt_file)
        if set(OFFICIAL_CLASSES).issubset(df.columns):
            return df

    logger.info("Downloading official ISIC 2018 Task 3 Ground Truth...")
    r = requests.get(GROUNDTRUTH_URL, timeout=30)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        for name in zf.namelist():
            if name.endswith(".csv"):
                df = pd.read_csv(zf.open(name))
                df.to_csv(gt_file, index=False)
                return df
    raise RuntimeError("Could not extract ground truth CSV from challenge archive")


def acquire_metadata_via_cli(raw_dir: Path) -> Optional[pd.DataFrame]:
    """Attempt to download collection 66 metadata via ISIC CLI."""
    meta_path = raw_dir / "metadata_cli.csv"
    if meta_path.exists():
        return pd.read_csv(meta_path)

    # Check for isic executable
    isic_bin = shutil.which("isic")
    if not isic_bin:
        exe_parent = Path(sys.executable).parent
        candidate = exe_parent / "isic.exe"
        if candidate.exists():
            isic_bin = str(candidate)
        else:
            venv_isic = Path(os.environ.get("VIRTUAL_ENV", "")) / "Scripts" / "isic.exe"
            if venv_isic.exists():
                isic_bin = str(venv_isic)

    if not isic_bin:
        logger.warning("isic CLI binary not found in PATH or venv.")
        return None

    try:
        cmd = [isic_bin, "metadata", "download", "-c", "66", "-l", "0", "-o", str(meta_path)]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if res.returncode == 0 and meta_path.exists():
            return pd.read_csv(meta_path)
    except Exception as e:
        logger.warning(f"ISIC CLI metadata download failed: {e}")
    return None


def acquire_dataset(
    raw_dir: str | Path = "data/raw/isic2018",
    collection_id: int = 66,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Acquires the ISIC 2018 Task 3 / HAM10000 dataset following the 3-tier order."""
    raw_dir = Path(raw_dir)
    images_dir = raw_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    unified_metadata_path = raw_dir / "metadata.csv"

    source_used = "official_isic_cli_and_challenge_archive"

    # Step 1: Acquire ground truth
    if progress_callback:
        progress_callback(0.05, "Acquiring official ground truth labels...")
    gt_df = acquire_groundtruth(raw_dir)

    # Step 2: Acquire metadata containing lesion IDs
    if progress_callback:
        progress_callback(0.10, "Acquiring collection 66 metadata with lesion IDs...")
    cli_meta = acquire_metadata_via_cli(raw_dir)
    if cli_meta is None:
        # Check if pre-downloaded collection 66 metadata exists
        fallback_meta = raw_dir.parent / "metadata_collection66.csv"
        if fallback_meta.exists():
            cli_meta = pd.read_csv(fallback_meta)
            source_used = "isic_api_collection_66"
        else:
            # REST API fallback
            r = requests.get(f"https://api.isic-archive.com/api/v2/images/search/?collections={collection_id}&limit=10015", timeout=30)
            if r.status_code == 200:
                results = r.json().get("results", [])
                rows = []
                for item in results:
                    clinical = item.get("metadata", {}).get("clinical", {})
                    rows.append({
                        "isic_id": item.get("isic_id"),
                        "lesion_id": clinical.get("lesion_id"),
                        "attribution": item.get("attribution"),
                        "copyright_license": item.get("copyright_license"),
                    })
                cli_meta = pd.DataFrame(rows)
                source_used = "isic_api_v2_rest"

    # Step 3: Unify ground truth and metadata
    # Map one-hot vector to single class_label
    def get_class_label(row):
        for c in OFFICIAL_CLASSES:
            if row.get(c, 0.0) == 1.0:
                return c
        return "UNKNOWN"

    gt_df["class_label"] = gt_df.apply(get_class_label, axis=1)

    if cli_meta is not None and "isic_id" in cli_meta.columns:
        unified_df = pd.merge(
            gt_df,
            cli_meta[["isic_id", "lesion_id"]],
            left_on="image",
            right_on="isic_id",
            how="left",
        )
        if "isic_id" in unified_df.columns:
            unified_df = unified_df.drop(columns=["isic_id"])
    else:
        unified_df = gt_df.copy()
        unified_df["lesion_id"] = unified_df["image"]

    # Fill any null lesion_id with deterministic image id
    unified_df["lesion_id"] = unified_df["lesion_id"].fillna(unified_df["image"])
    unified_df = unified_df.rename(columns={"image": "image_id"})
    unified_df.to_csv(unified_metadata_path, index=False)

    # Step 4: Acquire images
    # Check existing count
    existing_images = list(images_dir.glob("*.jpg"))
    if len(existing_images) >= 10015:
        if progress_callback:
            progress_callback(1.0, f"Found all {len(existing_images)} images verified locally.")
    else:
        # Download challenge input archive
        zip_path = raw_dir / "ISIC2018_Task3_Training_Input.zip"
        if not zip_path.exists() or zip_path.stat().st_size < 1000000:
            if progress_callback:
                progress_callback(0.15, "Downloading official ISIC 2018 Task 3 Training Input archive (2.77 GB)...")
            download_file_with_progress(
                INPUT_IMAGES_URL,
                zip_path,
                progress_callback=lambda pct, msg: progress_callback(0.15 + 0.70 * pct, msg) if progress_callback else None,
            )

        # Extract archive
        if progress_callback:
            progress_callback(0.88, "Extracting images to data/raw/isic2018/images/...")
        with zipfile.ZipFile(zip_path) as zf:
            namelist = [n for n in zf.namelist() if n.lower().endswith(".jpg")]
            total_imgs = len(namelist)
            for idx, name in enumerate(namelist):
                filename = Path(name).name
                dest = images_dir / filename
                if not dest.exists():
                    with open(dest, "wb") as f_out:
                        f_out.write(zf.read(name))
                if idx % 1000 == 0 and progress_callback:
                    progress_callback(0.88 + 0.11 * (idx / max(total_imgs, 1)), f"Extracted {idx}/{total_imgs} images")

    # Step 5: Provenance
    provenance = record_provenance(
        output_path="data/provenance.json",
        source=source_used,
        collection_id=collection_id,
        file_count=len(list(images_dir.glob("*.jpg"))),
        license_info="CC-BY-NC research dataset (ISIC / HAM10000)",
        metadata_files=[str(unified_metadata_path)],
        raw_dir=raw_dir,
        additional_info={
            "classes": OFFICIAL_CLASSES,
            "unique_lesions": int(unified_df["lesion_id"].nunique()),
            "total_records": len(unified_df),
        },
    )

    if progress_callback:
        progress_callback(1.0, f"Acquisition complete: {len(unified_df)} records, {provenance['file_count']} images.")

    return unified_df, provenance
