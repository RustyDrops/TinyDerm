"""Compare a train-only Hunch/EG2 teacher with an unassisted MobileNet baseline.

Research use only. The consumed holdout is never loaded. Outputs are development
validation results and must not be interpreted as an independent test.
"""
import argparse
import hashlib
import io
import json
import random
import sqlite3
import site
import time
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import balanced_accuracy_score, classification_report, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.svm import SVC
from torch.utils.data import DataLoader, Dataset

from hunch_experiment import development_rows
from tinyderm.augment import get_eval_transforms, get_train_transforms
from tinyderm.dataset import DermDataset, OFFICIAL_CLASSES
from tinyderm.model import TinyDermModel


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def embeddings(frame, output, tokens):
    hunch_site = Path(__file__).parent / 'hunch-upstream' / '.venv' / 'Lib' / 'site-packages'
    if hunch_site.is_dir(): site.addsitedir(str(hunch_site))
    from hunch.live import Eyes
    from eg2cpu import Model
    eyes = Eyes(model=Model(), tokens=tokens)
    db = sqlite3.connect(output / 'embeddings.sqlite')
    db.execute('CREATE TABLE IF NOT EXISTS vectors (image_id TEXT PRIMARY KEY, image_hash TEXT NOT NULL, vector BLOB NOT NULL)')
    vectors = []
    for i, row in frame.iterrows():
        raw = Path(row.source_path).read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        saved = db.execute('SELECT image_hash, vector FROM vectors WHERE image_id=?', (row.image_id,)).fetchone()
        if saved:
            if saved[0] != digest: raise ValueError(f'Image changed: {row.image_id}')
            vector = np.load(io.BytesIO(saved[1]), allow_pickle=False)
        else:
            vector, _ = eyes.see(raw)
            vector = np.asarray(vector, dtype=np.float32).reshape(-1)
            if not np.isfinite(vector).all(): raise ValueError(f'Invalid embedding: {row.image_id}')
            buf = io.BytesIO(); np.save(buf, vector, allow_pickle=False)
            db.execute('INSERT INTO vectors VALUES (?,?,?)', (row.image_id, digest, buf.getvalue()))
            db.commit()
        vectors.append(vector)
        if (i + 1) % 100 == 0: print(f'Hunch embeddings {i+1}/{len(frame)}', flush=True)
    db.close()
    return np.stack(vectors)


class WithTeacher(Dataset):
    def __init__(self, base, probabilities):
        self.base, self.probabilities = base, probabilities
    def __len__(self): return len(self.base)
    def __getitem__(self, idx):
        image, label, image_id, _ = self.base[idx]
        return image, label, torch.tensor(self.probabilities[image_id], dtype=torch.float32)


def validation(model, loader, device):
    model.eval(); actual=[]; predicted=[]
    with torch.no_grad():
        for image, target, _ in loader:
            pred = model(image.to(device)).argmax(dim=1).cpu().numpy()
            actual.extend(target.numpy()); predicted.extend(pred)
    return {'macro_f1': float(f1_score(actual, predicted, labels=list(range(7)), average='macro', zero_division=0)),
            'balanced_accuracy': float(balanced_accuracy_score(actual, predicted)),
            'per_class': classification_report(actual, predicted, labels=list(range(7)), target_names=OFFICIAL_CLASSES, output_dict=True, zero_division=0)}


def make_teacher_head(kind, c, seed):
    if kind == 'logistic':
        classifier = LogisticRegression(C=c, class_weight='balanced', max_iter=2000, random_state=seed)
    elif kind == 'rbf_svc':
        svc = SVC(C=c, kernel='rbf', class_weight='balanced', random_state=seed, cache_size=1024)
        classifier = CalibratedClassifierCV(estimator=svc, method='sigmoid', cv=5)
    else:
        raise ValueError(f'Unsupported Hunch teacher head: {kind}')
    return make_pipeline(StandardScaler(), classifier)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--tokens', type=int, default=70)
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--image-size', type=int, default=288)
    ap.add_argument('--teacher-c', type=float, default=1.0)
    ap.add_argument('--teacher-head', choices=['logistic', 'rbf_svc'], default='logistic')
    ap.add_argument('--distill-weight', type=float, default=0.5, help='Fraction of assisted loss assigned to Hunch KL targets')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--learning-rate', type=float, default=5e-5)
    ap.add_argument('--class-weight-strength', type=float, default=0.0, help='Power for smoothed inverse-frequency CE weights; 0 disables weighting')
    ap.add_argument('--retrain-on-dev', action='store_true', help='After validation comparison, retrain the assisted model on train+validation for a fixed number of epochs; report no validation score')
    ap.add_argument('--limit-per-class', type=int, default=0, help='Optional development-only pilot; 0 means all.')
    args = ap.parse_args()
    frame = development_rows(args.manifest)
    if args.limit_per_class:
        frame = frame.groupby(['split','class_label'], group_keys=False).head(args.limit_per_class).reset_index(drop=True)
    if set(frame.loc[frame['split']=='train', 'class_label']) != set(OFFICIAL_CLASSES): raise ValueError('All seven classes required in training')
    args.output.mkdir(parents=True, exist_ok=True)
    if not 0.0 <= args.distill_weight <= 1.0: raise ValueError('distill-weight must be between 0 and 1')
    if args.class_weight_strength < 0: raise ValueError('class-weight-strength must be nonnegative')
    settings = {'manifest_sha256': hashlib.sha256(args.manifest.read_bytes()).hexdigest(), 'checkpoint_sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(), 'tokens': args.tokens, 'limit_per_class': args.limit_per_class, 'epochs': args.epochs, 'batch_size': args.batch_size, 'image_size': args.image_size, 'teacher_head': args.teacher_head, 'teacher_c': args.teacher_c, 'distill_weight': args.distill_weight, 'seed': args.seed, 'learning_rate': args.learning_rate, 'class_weight_strength': args.class_weight_strength, 'encoder_revision': '914f7f89142e33e77833254d9c9b90c3cef7303b', 'hunch_revision': '6541bd157e11a5e7d7434360e99ea81f40a8c181'}
    settings_file = args.output / 'settings.json'
    if settings_file.exists() and json.loads(settings_file.read_text()) != settings: raise ValueError('Existing output has different settings')
    settings_file.write_text(json.dumps(settings, indent=2))
    start = time.monotonic(); x = embeddings(frame, args.output, args.tokens)
    train = (frame['split']=='train').to_numpy(); val = (frame['split']=='val').to_numpy()
    class_counts = np.array([(frame.loc[train, 'class_label'] == c).sum() for c in OFFICIAL_CLASSES], dtype=np.float64)
    ce_weights = np.power(class_counts.sum() / (len(OFFICIAL_CLASSES) * np.maximum(class_counts, 1)), args.class_weight_strength)
    ce_weights = torch.tensor(ce_weights / ce_weights.mean(), dtype=torch.float32, device='cuda' if torch.cuda.is_available() else 'cpu')
    teacher = make_teacher_head(args.teacher_head, args.teacher_c, args.seed)
    labels = frame['class_label'].to_numpy()
    train_rows = np.flatnonzero(train)
    oof = np.zeros((len(frame), len(OFFICIAL_CLASSES)), dtype=np.float32)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed)
    for fold, (fit_idx, held_idx) in enumerate(splitter.split(x[train], labels[train], frame.loc[train, 'lesion_id']), start=1):
        fold_model = make_teacher_head(args.teacher_head, args.teacher_c, args.seed)
        fold_model.fit(x[train][fit_idx], labels[train][fit_idx])
        fold_prob = fold_model.predict_proba(x[train][held_idx])
        oof[train_rows[held_idx]] = fold_prob[:, [list(fold_model.classes_).index(c) for c in OFFICIAL_CLASSES]]
        print(f'Hunch grouped teacher fold {fold}/5 complete', flush=True)
    teacher.fit(x[train], labels[train])
    val_prob = teacher.predict_proba(x[val])[:, [list(teacher.classes_).index(c) for c in OFFICIAL_CLASSES]]
    oof[val] = val_prob
    val_predictions = np.array(OFFICIAL_CLASSES)[np.argmax(val_prob, axis=1)]; truth=frame.loc[val,'class_label']
    teacher_metrics = {'macro_f1': float(f1_score(truth,val_predictions,labels=OFFICIAL_CLASSES,average='macro',zero_division=0)), 'balanced_accuracy': float(balanced_accuracy_score(truth,val_predictions))}
    joblib.dump(teacher, args.output/'hunch_teacher.joblib')
    mapping = dict(zip(frame.image_id, oof))
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results={'scope':'development validation only; not an independent test', 'settings':settings, 'train_images':int(train.sum()), 'val_images':int(val.sum()), 'embedding_dimensions':int(x.shape[1]), 'hunch_teacher':teacher_metrics, 'arms':{}}
    for assisted in (False, True):
        seed_all(args.seed)
        model=TinyDermModel(pretrained=False).to(device)
        model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
        training=WithTeacher(DermDataset(frame,'train',get_train_transforms(args.image_size)),mapping)
        validating=WithTeacher(DermDataset(frame,'val',get_eval_transforms(args.image_size)),mapping)
        loader=DataLoader(training,batch_size=args.batch_size,shuffle=True,num_workers=0)
        valid_loader=DataLoader(validating,batch_size=args.batch_size,shuffle=False,num_workers=0)
        optimizer=torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=0.01)
        arm='hunch_assisted' if assisted else 'baseline'
        history=[]
        for epoch in range(1,args.epochs+1):
            model.train(); loss_sum=0
            for image,target,soft in loader:
                image,target,soft=image.to(device),target.to(device),soft.to(device)
                optimizer.zero_grad(set_to_none=True); logits=model(image)
                hard=F.cross_entropy(logits,target,weight=ce_weights)
                # Teacher probabilities are learned only from train labels and Hunch embeddings.
                loss=(1-args.distill_weight)*hard + args.distill_weight*F.kl_div(F.log_softmax(logits,dim=1),soft,reduction='batchmean') if assisted else hard
                loss.backward(); optimizer.step(); loss_sum += float(loss.item())*len(target)
            metrics=validation(model,valid_loader,device); metrics.update(epoch=epoch,train_loss=loss_sum/len(training)); history.append(metrics)
            torch.save(model.state_dict(),args.output/f'{arm}_epoch{epoch}.pt')
            print(f"{arm} epoch {epoch}: macro F1 {metrics['macro_f1']:.4f}; balanced accuracy {metrics['balanced_accuracy']:.4f}",flush=True)
        results['arms'][arm]=history
        (args.output/'results.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    if args.retrain_on_dev:
        seed_all(args.seed)
        model=TinyDermModel(pretrained=False).to(device)
        model.load_state_dict(torch.load(args.checkpoint,map_location=device,weights_only=True))
        dev_base=DermDataset(frame,'dev',get_train_transforms(args.image_size))
        dev_set=WithTeacher(dev_base,mapping)
        dev_loader=DataLoader(dev_set,batch_size=args.batch_size,shuffle=True,num_workers=0)
        optimizer=torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=0.01)
        retrain_history=[]
        for epoch in range(1,args.epochs+1):
            model.train(); total_loss=0.0
            for image,target,soft in dev_loader:
                image,target,soft=image.to(device),target.to(device),soft.to(device)
                optimizer.zero_grad(set_to_none=True); logits=model(image)
                hard=F.cross_entropy(logits,target,weight=ce_weights)
                distill=F.kl_div(F.log_softmax(logits,dim=1),soft,reduction='batchmean')
                loss=(1-args.distill_weight)*hard+args.distill_weight*distill
                loss.backward(); optimizer.step(); total_loss += float(loss.item())*len(target)
            model_path=args.output/f'hunch_assisted_full_dev_epoch{epoch}.pt'
            torch.save(model.state_dict(),model_path)
            retrain_history.append({'epoch':epoch,'training_loss':total_loss/len(dev_set),'checkpoint':model_path.name})
            print(f"full-development assisted retrain epoch {epoch} training loss: {retrain_history[-1]['training_loss']:.4f}",flush=True)
        results['full_development_retrain']={'scope':'train+validation only; no validation metric computed','epochs':retrain_history,'settings':settings}
        (args.output/'results.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    results['elapsed_seconds']=time.monotonic()-start
    (args.output/'results.json').write_text(json.dumps(results,indent=2),encoding='utf-8')


if __name__=='__main__': main()

