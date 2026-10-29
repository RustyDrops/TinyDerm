"""Provenance tracking module for TinyDerm Lab."""

import datetime
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


def compute_file_hash(path: str | Path, algorithm: str = "sha256") -> str:
    """Compute hash of a file efficiently."""
    hasher = hashlib.new(algorithm)
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def record_provenance(
    output_path: str | Path,
    source: str,
    collection_id: int | str,
    file_count: int,
    license_info: str,
    metadata_files: List[str],
    raw_dir: str | Path,
    additional_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Record provenance data and write to data/provenance.json."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metadata_hashes = {}
    for mf in metadata_files:
        mf_path = Path(mf)
        if mf_path.exists():
            metadata_hashes[mf_path.name] = compute_file_hash(mf_path)

    provenance_data = {
        "dataset_name": "ISIC 2018 Task 3 / HAM10000",
        "source": source,
        "collection_id": collection_id,
        "download_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "file_count": file_count,
        "license": license_info,
        "metadata_files_used": [str(Path(f).name) for f in metadata_files],
        "metadata_hashes": metadata_hashes,
        "provenance_version": "1.0",
        "notes": "ISIC 2018 Challenge Task 3 official training set. Seven diagnostic categories: MEL, NV, BCC, AKIEC, BKL, DF, VASC.",
    }

    if additional_info:
        provenance_data.update(additional_info)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(provenance_data, f, indent=2)

    return provenance_data


def load_provenance(path: str | Path) -> Optional[Dict[str, Any]]:
    """Load existing provenance information if available."""
    p = Path(path)
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)
