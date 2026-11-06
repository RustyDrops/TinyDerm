"""Development-only CSV and embedding-cache readers for campaign training."""
from __future__ import annotations

import csv
from contextlib import closing
import hashlib
import io
import json
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd

CLASSES = ("MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC")
REQUIRED = {"split", "lesion_id", "image_id", "source_path", "class_label"}


def development_rows(manifest: str | Path) -> pd.DataFrame:
    """Read train/validation rows while ignoring every other holdout field."""
    rows: list[dict[str, str]] = []
    with Path(manifest).open(newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None or len(header) != len(set(header)) or not REQUIRED.issubset(header):
            raise ValueError("Manifest is missing required columns or has duplicate headers")
        split_index = header.index("split")
        for values in reader:
            if len(values) <= split_index:
                raise ValueError("Manifest row has no split field")
            split = values[split_index]
            if split == "holdout":
                # Do not access labels, identities, image paths or metadata on this row.
                continue
            if len(values) != len(header):
                raise ValueError("Malformed development manifest row")
            if split not in ("train", "val"):
                raise ValueError("Unexpected non-holdout split")
            row = dict(zip(header, values))
            if any(not row[name].strip() for name in REQUIRED):
                raise ValueError("Development row has a missing required value")
            if row["class_label"] not in CLASSES:
                raise ValueError("Development row has an unknown class")
            rows.append(row)
    frame = pd.DataFrame(rows, columns=header)
    if frame.empty:
        raise ValueError("No development rows found")
    if frame["image_id"].duplicated().any():
        raise ValueError("Duplicate development image IDs")
    if frame.groupby("lesion_id")["split"].nunique().max() != 1:
        raise ValueError("Development lesion groups cross train/validation partitions")
    return frame.reset_index(drop=True)


def development_identity_fingerprint(frame: pd.DataFrame) -> str:
    """Fingerprint only train/validation identities, labels, and split assignments."""
    digest = hashlib.sha256()
    records = ((str(row.image_id), str(row.lesion_id), str(row.class_label), str(row.split))
               for row in frame.itertuples(index=False))
    for record in sorted(records):
        digest.update((json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8"))
    return digest.hexdigest()


def load_hunch(frame: pd.DataFrame, cache_path: str | Path) -> np.ndarray:
    """Fetch and hash-check only requested development vectors from read-only SQLite."""
    cache = Path(cache_path).resolve()
    if not cache.is_file():
        raise FileNotFoundError(cache)
    ids = frame["image_id"].astype(str).tolist()
    paths = {str(row.image_id): Path(row.source_path) for row in frame.itertuples(index=False)}
    vectors: dict[str, np.ndarray] = {}
    uri = cache.as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as db:
        columns = {r[1] for r in db.execute("PRAGMA table_info(vectors)")}
        if not {"image_id", "image_hash", "vector"}.issubset(columns):
            raise ValueError("Embedding cache lacks image IDs, SHA-256 hashes, or vectors")
        for start in range(0, len(ids), 500):
            batch_ids = ids[start:start + 500]
            marks = ",".join("?" for _ in batch_ids)
            for image_id, expected_hash, blob in db.execute(
                    f"SELECT image_id, image_hash, vector FROM vectors WHERE image_id IN ({marks})", batch_ids):
                image_path = paths[str(image_id)]
                digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
                if digest != expected_hash:
                    raise ValueError(f"Embedding provenance hash mismatch for development image {image_id}")
                vector = np.asarray(np.load(io.BytesIO(blob), allow_pickle=False), dtype=np.float32).reshape(-1)
                if not np.isfinite(vector).all():
                    raise ValueError(f"Non-finite development embedding for {image_id}")
                vectors[str(image_id)] = vector
    absent = set(ids) - set(vectors)
    if absent:
        raise ValueError(f"Embedding cache is missing {len(absent)} development image IDs")
    dimensions = {len(v) for v in vectors.values()}
    if len(dimensions) != 1:
        raise ValueError("Development embeddings have inconsistent dimensions")
    return np.stack([vectors[image_id] for image_id in ids])

