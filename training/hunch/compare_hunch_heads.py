"""Compare train-only classifier heads over cached Hunch embeddings."""
import argparse
import io
import json
import sqlite3
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, classification_report, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression

from hunch_experiment import development_rows
from tinyderm.dataset import OFFICIAL_CLASSES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    frame = development_rows(args.manifest)
    db = sqlite3.connect(args.cache)
    rows = {key: value for key, value in db.execute("SELECT image_id, vector FROM vectors")}
    db.close()
    if set(frame.image_id) - set(rows):
        raise ValueError("Hunch embedding cache does not cover development rows")
    x = np.stack([np.load(io.BytesIO(rows[key]), allow_pickle=False) for key in frame.image_id])
    y = frame.class_label.to_numpy()
    train = (frame.split == "train").to_numpy()
    val = (frame.split == "val").to_numpy()
    candidates = {
        "logistic_c_0.0001": make_pipeline(StandardScaler(), LogisticRegression(C=0.0001, class_weight="balanced", max_iter=2000, random_state=42)),
        "logistic_c_0.001": make_pipeline(StandardScaler(), LogisticRegression(C=0.001, class_weight="balanced", max_iter=2000, random_state=42)),
        "logistic_c_0.01": make_pipeline(StandardScaler(), LogisticRegression(C=0.01, class_weight="balanced", max_iter=2000, random_state=42)),
        "logistic_c_0.1": make_pipeline(StandardScaler(), LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000, random_state=42)),
        "linear_svc_c_0.01": make_pipeline(StandardScaler(), LinearSVC(C=0.01, class_weight="balanced", random_state=42, max_iter=10000)),
        "linear_svc_c_0.1": make_pipeline(StandardScaler(), LinearSVC(C=0.1, class_weight="balanced", random_state=42, max_iter=10000)),
        "rbf_svc_c_0.1": make_pipeline(StandardScaler(), SVC(C=0.1, kernel="rbf", class_weight="balanced", probability=False)),
        "rbf_svc_c_1": make_pipeline(StandardScaler(), SVC(C=1, kernel="rbf", class_weight="balanced", probability=False)),
        "cosine_knn_15": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=15, weights="distance", metric="cosine", n_jobs=-1)),
    }
    results = {"scope": "development validation only; teacher fit uses training labels only", "n_train": int(train.sum()), "n_validation": int(val.sum()), "embedding_dimensions": int(x.shape[1]), "heads": {}}
    for name, estimator in candidates.items():
        print(f"fitting {name}", flush=True)
        estimator.fit(x[train], y[train])
        pred = estimator.predict(x[val])
        results["heads"][name] = {
            "macro_f1": float(f1_score(y[val], pred, labels=OFFICIAL_CLASSES, average="macro", zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y[val], pred)),
            "per_class": classification_report(y[val], pred, labels=OFFICIAL_CLASSES, output_dict=True, zero_division=0),
        }
    (args.output / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    for name, metric in results["heads"].items():
        print(f"{name}: macro F1 {metric['macro_f1']:.4f}; balanced accuracy {metric['balanced_accuracy']:.4f}", flush=True)


if __name__ == "__main__":
    main()

