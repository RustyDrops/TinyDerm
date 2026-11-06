"""Append missing Hunch/EmbeddingGemma image embeddings to a resumable SQLite cache."""
import argparse
import hashlib
import io
import os
import site
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

os.environ.setdefault("HF_HOME", str(Path(__file__).parent / "hunch-model-cache"))
from hunch_experiment import development_rows


def encode_npy(vector):
    buffer = io.BytesIO()
    np.save(buffer, vector.astype(np.float32), allow_pickle=False)
    return buffer.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--seed-cache", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--tokens", type=int, default=70)
    parser.add_argument("--commit-every", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not args.cache.exists():
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy2(args.seed_cache, args.cache)
    frame = development_rows(args.manifest)
    hunch_site = Path(__file__).parent / "hunch-upstream" / ".venv" / "Lib" / "site-packages"
    if hunch_site.is_dir():
        site.addsitedir(str(hunch_site))
    from eg2cpu import Model
    model = Model()
    connection = sqlite3.connect(args.cache)
    connection.execute("CREATE TABLE IF NOT EXISTS vectors (image_id TEXT PRIMARY KEY, image_hash TEXT NOT NULL, vector BLOB NOT NULL)")
    existing = {row[0] for row in connection.execute("SELECT image_id FROM vectors")}
    missing = frame[~frame.image_id.isin(existing)].reset_index(drop=True)
    print(f"Hunch needs to encode {len(missing)} of {len(frame)} development images", flush=True)
    started = time.monotonic()
    def encode(row):
        image_bytes = Path(row.source_path).read_bytes()
        t0 = time.perf_counter()
        vector, _ = model.embed_image(image_bytes, max_soft_tokens=args.tokens)
        vector = np.asarray(vector, dtype=np.float32).reshape(-1)
        if vector.shape != (768,) or not np.isfinite(vector).all():
            raise ValueError(f"Invalid Hunch embedding for {row.image_id}: {vector.shape}")
        return str(row.image_id), hashlib.sha256(image_bytes).hexdigest(), vector, (time.perf_counter() - t0) * 1000

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        done = 0
        for start in range(0, len(missing), args.commit_every):
            rows = list(missing.iloc[start:start + args.commit_every].itertuples(index=False))
            for image_id, image_hash, vector, elapsed_ms in pool.map(encode, rows):
                connection.execute("INSERT OR REPLACE INTO vectors(image_id,image_hash,vector) VALUES(?,?,?)", (
                    image_id, image_hash, encode_npy(vector)
                ))
                done += 1
            connection.commit()
            elapsed = time.monotonic() - started
            print(f"encoded {done}/{len(missing)}; last Hunch pass {elapsed_ms:.1f} ms; elapsed {elapsed:.1f}s", flush=True)
    connection.commit()
    total = connection.execute("SELECT COUNT(*) FROM vectors").fetchone()[0]
    connection.close()
    print(f"complete: {total} cached Hunch embeddings in {time.monotonic()-started:.1f}s", flush=True)


if __name__ == "__main__":
    main()

