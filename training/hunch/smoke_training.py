from pathlib import Path
import os
import pandas as pd
import yaml
from tinyderm.train import train_student_model
from tinyderm.experiment_db import ExperimentDB
m = pd.read_csv(Path(os.environ.get("TINYDERM_MANIFEST", Path(__file__).resolve().parents[2] / "data" / "splits" / "split_manifest.csv")))
m = m[m['split'].isin(['train','val'])].copy()
m = m.groupby(['split', 'class_label'], group_keys=False).head(8).copy()
config = yaml.safe_load(Path('config/default.yaml').read_text())
config['training'].update({'head_epochs': 1, 'finetune_max_epochs': 1, 'batch_size': 32, 'num_workers': 0})
config['model']['pretrained'] = False
config['model']['image_size'] = 128
# A short mechanics smoke test, not a quality comparison.
db = ExperimentDB('smoke_retrain/experiments.sqlite')
db.create_run('smoke',seed=42,config=config,hardware={})
model, metrics = train_student_model(m,config,db,'smoke',output_dir='smoke_retrain')
print('Validation macro F1:', metrics['macro_f1'], flush=True)
print('Selected epochs:',metrics['selected_head_epochs'],metrics['selected_finetune_epochs'],flush=True)
model, _ = train_student_model(m, config, db, 'smoke', output_dir='smoke_retrain', retrain_on_dev=True, retrain_head_epochs=metrics['selected_head_epochs'], retrain_finetune_epochs=metrics['selected_finetune_epochs'])
print('Full-development retrain completed using fixed epoch counts.', flush=True)

