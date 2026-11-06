"""Inspect grouped development CV identities without loading image pixels or holdout rows."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys

from sklearn.model_selection import StratifiedGroupKFold
from tinyderm_preflight import audit_manifest


def identity_hash(rows):
    digest = hashlib.sha256()
    for row in sorted(rows):
        digest.update((json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8"))
    return digest.hexdigest()


def audit(path, folds=3, seed=52711):
    base = audit_manifest(path)
    if base["errors"]:
        raise ValueError("Manifest diagnostics failed; correct errors before fold analysis")
    rows = []
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["split"] in ("train", "val"):
                rows.append((r["image_id"], r["lesion_id"], r["class_label"]))
    labels = [r[2] for r in rows]
    groups = [r[1] for r in rows]
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    results = []
    seen = set()
    for number, (train, val) in enumerate(splitter.split(rows, labels, groups), 1):
        train_groups = {rows[i][1] for i in train}
        val_groups = {rows[i][1] for i in val}
        intersection = len(train_groups & val_groups)
        overlap_indices = len(seen & set(val))
        seen.update(val)
        results.append({"fold": number, "train_count": len(train), "validation_count": len(val),
                        "group_overlap": intersection, "duplicate_validation_indices": overlap_indices,
                        "validation_class_counts": dict(Counter(rows[i][2] for i in val)),
                        "train_identity_sha256": identity_hash(rows[i] for i in train),
                        "validation_identity_sha256": identity_hash(rows[i] for i in val)})
    return {"scope": "development-only grouped CV; no images opened", "seed": seed, "fold_count": folds,
            "development_identity_sha256": base["development_identity_fingerprint_sha256"],
            "folds": results, "validation_coverage": len(seen), "development_images": len(rows),
            "caveats": ["Group IDs are not proof of patient independence or absence of near duplicates.",
                        "Fingerprints do not include preprocessing, code, or image bytes.",
                        "The training cache and teacher fitting were not inspected per fold."],
            "training_authorized": False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("manifest", type=Path)
    p.add_argument("--folds", type=int, default=3)
    p.add_argument("--seed", type=int, default=52711)
    args = p.parse_args()
    try:
        report = audit(args.manifest, args.folds, args.seed)
        print(json.dumps(report, indent=2))
        return int(report["validation_coverage"] != report["development_images"] or
                   any(f["group_overlap"] or f["duplicate_validation_indices"] for f in report["folds"]))
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

