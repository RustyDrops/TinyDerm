"""Extract non-overlapping ISIC 2019 images and build a development-only manifest."""
import argparse
import json
import re
import zipfile
from pathlib import Path

import pandas as pd

from hunch_experiment import development_rows
from tinyderm.dataset import OFFICIAL_CLASSES


def normalize_image_id(value):
    return re.sub(r"_downsampled$", "", str(value))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--groundtruth", required=True, type=Path)
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--output-manifest", required=True, type=Path)
    parser.add_argument("--output-report", required=True, type=Path)
    args = parser.parse_args()
    if args.output_manifest.exists() or args.output_report.exists():
        raise FileExistsError("Refusing to overwrite an existing expanded manifest or report")

    original = pd.read_csv(args.manifest)
    # Include the sealed holdout's IDs and lesion IDs in the exclusion set.
    excluded_ids = {normalize_image_id(x) for x in original.image_id}
    excluded_lesions = set(original.lesion_id.astype(str))
    metadata = pd.read_csv(args.metadata).set_index("image")
    labels = pd.read_csv(args.groundtruth).set_index("image")
    source_to_class = {"MEL": "MEL", "NV": "NV", "BCC": "BCC", "AK": "AKIEC", "BKL": "BKL", "DF": "DF", "VASC": "VASC"}
    label_columns = list(source_to_class)
    if not set(label_columns).issubset(labels.columns):
        raise ValueError(f"Unexpected ISIC 2019 columns: {labels.columns.tolist()}")

    additions = []
    skipped = {"duplicate_id": 0, "duplicate_lesion": 0, "excluded_class": 0, "missing_image": 0}
    args.image_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.archive) as archive:
        entries = {Path(info.filename).stem: info for info in archive.infolist() if Path(info.filename).suffix.lower() in {".jpg", ".jpeg", ".png"}}
        for image_id, row in labels.iterrows():
            image_id = str(image_id)
            if normalize_image_id(image_id) in excluded_ids:
                skipped["duplicate_id"] += 1
                continue
            active = row[label_columns]
            if float(active.max()) != 1.0:
                skipped["excluded_class"] += 1
                continue
            diagnosis = source_to_class[str(active.idxmax())]
            info = entries.get(image_id)
            if info is None:
                skipped["missing_image"] += 1
                continue
            meta = metadata.loc[image_id]
            raw_lesion = meta.get("lesion_id")
            lesion_id = str(raw_lesion) if pd.notna(raw_lesion) else f"ISIC2019:{image_id}"
            if lesion_id in excluded_lesions:
                skipped["duplicate_lesion"] += 1
                continue
            target = args.image_dir / Path(info.filename).name
            if not target.exists():
                target.write_bytes(archive.read(info))
            additions.append({
                "image_id": image_id,
                "lesion_id": lesion_id,
                "class_label": diagnosis,
                "split": "train",
                "source_path": str(target.resolve()),
                "dataset_source": "ISIC2019_nonoverlap",
            })

    # All original development rows participate in the new grouped folds.
    original.loc[original.split.isin(["train", "val"]), "split"] = "train"
    if "dataset_source" not in original:
        original["dataset_source"] = "ISIC2018_development"
    else:
        original["dataset_source"] = original["dataset_source"].fillna("ISIC2018_development")
    combined = pd.concat([original, pd.DataFrame(additions)], ignore_index=True, sort=False)
    # Check group/split consistency after merging, before any training can read the file.
    dev = combined[combined.split.isin(["train", "val"])].copy()
    if dev.groupby("lesion_id").split.nunique().max() != 1:
        raise ValueError("Expanded development manifest has lesions crossing splits")
    if dev.image_id.duplicated().any():
        raise ValueError("Expanded development manifest has duplicate image IDs")
    if not set(dev.class_label).issubset(OFFICIAL_CLASSES):
        raise ValueError("Expanded manifest contains a class outside the TinyDerm label set")
    combined.to_csv(args.output_manifest, index=False)
    report = {
        "scope": "ISIC 2019 expansion; original holdout rows preserved and excluded from training",
        "original_manifest_rows": len(original),
        "original_ids_excluded_from_isic2019": len(excluded_ids),
        "added_images": len(additions),
        "added_lesion_groups": len({row["lesion_id"] for row in additions}),
        "added_class_counts": pd.Series([row["class_label"] for row in additions]).value_counts().to_dict(),
        "skipped": skipped,
        "total_development_images": len(dev),
        "total_development_lesion_groups": int(dev.lesion_id.nunique()),
        "merged_class_counts": dev.class_label.value_counts().to_dict(),
        "isic2019_citation": "International Skin Imaging Collaboration. SIIM-ISIC 2019 Challenge Dataset. https://doi.org/10.34970/2019-ds01. CC BY-NC 4.0.",
    }
    args.output_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    # Re-open through the production research manifest filter for a final structural check.
    checked = development_rows(args.output_manifest)
    if len(checked) != len(dev):
        raise ValueError("Expanded manifest filter unexpectedly dropped development images")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

