"""Read-only development-manifest diagnostics. Does not run or authorize training."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

CLASSES = ("MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC")
REQUIRED = {"split", "image_id", "lesion_id", "class_label", "source_path"}


def audit_manifest(path, check_paths=False, cache=None):
    counts, by_split, bad = Counter(), Counter(), Counter()
    missing_label_vectors = 0
    ids, groups, dev_rows = set(), {}, []
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or not REQUIRED.issubset(reader.fieldnames):
            raise ValueError("Missing required manifest columns")
        one_hot = all(c in reader.fieldnames for c in CLASSES)
        for row in reader:
            split = row["split"]
            if split == "holdout":
                # Do not inspect identities, labels, paths, or images in this partition.
                continue
            if split not in ("train", "val"):
                bad["unknown_split"] += 1
                continue
            by_split[split] += 1
            image_id, group, label = (row[x].strip() for x in ("image_id", "lesion_id", "class_label"))
            if not image_id or not group or not label or not row["source_path"].strip():
                bad["missing_fields"] += 1
            if image_id in ids:
                bad["duplicate_image_ids"] += 1
            ids.add(image_id)
            if label not in CLASSES:
                bad["unknown_label"] += 1
            counts[label] += 1
            if group in groups and groups[group] != split:
                bad["cross_split_groups"] += 1
            groups[group] = split
            if one_hot:
                if all(not row[c].strip() for c in CLASSES):
                    missing_label_vectors += 1
                else:
                    try:
                        hot = [float(row[c]) for c in CLASSES]
                        if any(v not in (0.0, 1.0) for v in hot) or sum(hot) != 1 or label not in CLASSES or hot[CLASSES.index(label)] != 1:
                            bad["label_vector_mismatch"] += 1
                    except (ValueError, TypeError):
                        bad["label_vector_mismatch"] += 1
            if check_paths and not Path(row["source_path"]).is_file():
                bad["missing_image_files"] += 1
            dev_rows.append((image_id, group, label, split))
    digest = hashlib.sha256()
    for row in sorted(dev_rows):
        digest.update((json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8"))
    cache_result = None
    if cache is not None:
        # URI mode=ro avoids creating a missing SQLite database.
        with sqlite3.connect(Path(cache).resolve().as_uri() + "?mode=ro", uri=True) as db:
            cache_ids = {str(x[0]) for x in db.execute("SELECT image_id FROM vectors")}
        cache_result = {"development_missing": len(ids - cache_ids),
                        "validation_entries": sum(row[0] in cache_ids for row in dev_rows if row[3] == "val"),
                        "note": "Membership only; does not validate vector provenance or fold-specific training isolation."}
    warnings = []
    if not by_split["val"]:
        warnings.append("No fixed development-validation rows; grouped CV may exist but is not verified by this check.")
    if not one_hot:
        warnings.append("One-hot label columns absent; mapping consistency not checked.")
    if missing_label_vectors:
        warnings.append(f"One-hot columns are blank in {missing_label_vectors} development rows; their class_label values were checked for allowed names but not independently validated.")
    if not check_paths:
        warnings.append("Image file existence not checked.")
    warnings.append("Patient overlap, image contents, transforms, cache provenance and fold-level leakage are not checked.")
    return {"scope": "development rows only; holdout identities, labels, paths and images excluded",
            "development_identity_fingerprint_sha256": digest.hexdigest(),
            "fingerprint_definition": "SHA-256 of sorted JSON lines [image_id, lesion_id, class_label, split]",
            "development_images": len(dev_rows), "development_groups": len(groups),
            "split_counts": dict(by_split), "class_counts": dict(counts),
            "errors": dict(bad), "cache": cache_result, "warnings": warnings,
            "training_authorized": False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("manifest", type=Path)
    p.add_argument("--check-paths", action="store_true", help="Check development image paths locally")
    p.add_argument("--cache", type=Path, help="Read-only Hunch SQLite membership check")
    args = p.parse_args()
    try:
        report = audit_manifest(args.manifest, args.check_paths, args.cache)
    except (OSError, ValueError, sqlite3.Error) as exc:
        print(f"Preflight failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2))
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())

