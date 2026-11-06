"""Research-only Hunch/EG2 embedding baseline. Never reads holdout images.

Requires the upstream hunch and eg2cpu packages and local EG2 weights. Run with
--manifest pointing to the original split manifest and --output to a new directory.
This is a separate embedding classifier, not distillation into MobileNet.
"""
import argparse
import json
import site
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, classification_report, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from tinyderm.dataset import OFFICIAL_CLASSES


def development_rows(manifest):
    frame = pd.read_csv(manifest)
    required = {'split', 'lesion_id', 'image_id', 'source_path', 'class_label'}
    if not required.issubset(frame.columns) or frame[list(required)].isna().any().any():
        raise ValueError('Manifest has missing columns or values')
    if not set(frame['split']).issubset({'train', 'val', 'holdout'}):
        raise ValueError('Unexpected split')
    if not set(frame['class_label']).issubset(OFFICIAL_CLASSES):
        raise ValueError('Unexpected class')
    if frame.groupby('lesion_id')['split'].nunique().max() != 1:
        raise ValueError('Lesions cross split boundaries')
    if frame['image_id'].duplicated().any():
        raise ValueError('Duplicate image IDs')
    return frame[frame['split'].isin(['train', 'val'])].copy().reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--tokens', default=70, type=int)
    args = parser.parse_args()
    frame = development_rows(args.manifest)
    hunch_site = Path(__file__).parent / 'hunch-upstream' / '.venv' / 'Lib' / 'site-packages'
    if hunch_site.is_dir(): site.addsitedir(str(hunch_site))
    from hunch.live import Eyes
    eyes = Eyes(tokens=args.tokens)
    # Fail if the output exists: never silently overwrite an experiment.
    args.output.mkdir(parents=True, exist_ok=False)
    vectors, elapsed = [], []
    for index, row in frame.iterrows():
        vector, ms = eyes.see(Path(row.source_path).read_bytes())
        vector = np.asarray(vector, dtype=np.float32).reshape(-1)
        if not np.isfinite(vector).all():
            raise ValueError(f'Non-finite embedding for {row.image_id}')
        vectors.append(vector)
        elapsed.append(float(ms))
        if (index + 1) % 100 == 0:
            print(f'Encoded {index + 1}/{len(frame)}', flush=True)
    x = np.stack(vectors)
    train = (frame['split'] == 'train').to_numpy()
    val = (frame['split'] == 'val').to_numpy()
    if not train.any() or not val.any() or set(frame.loc[train, 'class_label']) != set(OFFICIAL_CLASSES):
        raise ValueError('Training and validation must exist and training must contain every class')
    classifier = make_pipeline(StandardScaler(), LogisticRegression(class_weight='balanced', max_iter=2000, random_state=42))
    classifier.fit(x[train], frame.loc[train, 'class_label'])
    predicted = classifier.predict(x[val])
    truth = frame.loc[val, 'class_label']
    report = {'scope': 'development validation only; research use', 'train_images': int(train.sum()), 'validation_images': int(val.sum()), 'embedding_dimensions': int(x.shape[1]), 'tokens': args.tokens, 'embedding_ms_median': float(np.median(elapsed)), 'macro_f1': float(f1_score(truth, predicted, labels=OFFICIAL_CLASSES, average='macro', zero_division=0)), 'balanced_accuracy': float(balanced_accuracy_score(truth, predicted)), 'classes': classification_report(truth, predicted, labels=OFFICIAL_CLASSES, output_dict=True, zero_division=0)}
    (args.output / 'validation_metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    joblib.dump(classifier, args.output / 'classifier.joblib')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

