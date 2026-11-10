"""Run: python -m coffee.train --output outputs"""
import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .features import audio_features, image_features
from .fusion import ROASTS


def discover(root, classes, suffix):
    rows = []
    for split in ['train', 'val', 'test']:
        for label, name in enumerate(classes):
            for p in sorted((root / split / name).glob('*')):
                if p.suffix.lower() in suffix:
                    rows.append({'path': str(p), 'label': label, 'split': split})
    if not rows:
        raise ValueError(f'No dataset at {root}')
    return rows


def deduplicate(rows):
    # Keep held-out data; remove duplicate copies from lower-priority splits.
    seen = {}
    unique = []
    priority = {'test': 0, 'val': 1, 'train': 2}
    for row in sorted(rows, key=lambda r: priority[r['split']]):
        digest = hashlib.sha256(Path(row['path']).read_bytes()).hexdigest()
        if digest in seen:
            if seen[digest] != row['label']:
                raise ValueError('Identical file has conflicting labels')
            continue
        seen[digest] = row['label']
        unique.append(row)
    return unique


def image_rows(root):
    rows = deduplicate(discover(root, ROASTS, {'.png', '.jpg', '.jpeg'}))
    # Reserve 15% of deduplicated total from the remaining train pool.
    train = [r for r in rows if r['split'] == 'train']
    _, val_indices = train_test_split(np.arange(len(train)), test_size=.15*len(rows)/len(train),
                                     stratify=[r['label'] for r in train], random_state=42)
    for i in val_indices:
        train[i]['split'] = 'val'
    return rows


def audit(rows):
    hashes = {}
    groups = {s: set() for s in ['train', 'val', 'test']}
    duplicates = 0
    for row in rows:
        digest = hashlib.sha256(Path(row['path']).read_bytes()).hexdigest()
        if digest in hashes and hashes[digest] != row['split']:
            duplicates += 1
        hashes[digest] = row['split']
        match = re.match(r'([^_]+)_', Path(row['path']).stem)
        if match:
            groups[row['split']].add(match[1])
    return {'counts': dict(Counter(r['split'] for r in rows)),
            'cross_split_identical_files': duplicates,
            'train_test_trial_overlap': sorted(groups['train'] & groups['test'])}


def run_modality(rows, feature_fn, names, output, name):
    output.mkdir(parents=True, exist_ok=True)
    (output / f'{name}_manifest.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    features = np.stack([feature_fn(r['path']) for r in rows])
    if not np.isfinite(features).all():
        raise ValueError('Non-finite features')
    labels = np.array([r['label'] for r in rows])
    masks = {s: np.array([r['split'] == s for r in rows]) for s in ['train', 'val', 'test']}
    for s, mask in masks.items():
        if set(labels[mask]) != set(range(len(names))):
            raise ValueError(f'{name}/{s}: missing classes')
    models = {'svm': make_pipeline(StandardScaler(), SVC(probability=True, random_state=42)),
              'rf': RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1)}
    val_scores = {}
    for key, model in models.items():
        model.fit(features[masks['train']], labels[masks['train']])
        val_scores[key] = accuracy_score(labels[masks['val']], model.predict(features[masks['val']]))
    selected = max(val_scores, key=val_scores.get)
    model = models[selected]
    # Test is used once, only after selecting by validation accuracy.
    pred = model.predict(features[masks['test']])
    report = {'audit': audit(rows), 'selected': selected, 'validation_accuracy': val_scores,
              'test_accuracy': accuracy_score(labels[masks['test']], pred),
              'test_report': classification_report(labels[masks['test']], pred, target_names=names, output_dict=True, zero_division=0),
              'confusion_matrix': confusion_matrix(labels[masks['test']], pred).tolist()}
    joblib.dump({'model': model, 'classes': names, 'validation_accuracy': val_scores[selected]}, output / f'{name}.joblib')
    np.savez_compressed(output / f'{name}_test_predictions.npz', labels=labels[masks['test']],
                        probabilities=model.predict_proba(features[masks['test']]))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', type=Path, default=Path('gpiosenka_coffee-bean-dataset-resized-224-x-224'))
    parser.add_argument('--audio', type=Path, default=Path('data/audio-lfs/datasets'))
    parser.add_argument('--output', type=Path, default=Path('outputs'))
    parser.add_argument('--cnn', action='store_true', help='Also compare frozen ImageNet MobileNetV3 features; requires torch/torchvision')
    args = parser.parse_args()
    results = {}
    for name, rows, fn, classes in [
        ('image', image_rows(args.images), image_features, ROASTS),
        ('audio', deduplicate(discover(args.audio, ['0', '1', '2'], {'.wav'})), audio_features, ['Background', 'First Crack', 'Second Crack'])]:
        print(f'{name}: extracting {len(rows)} files', flush=True)
        results[name] = run_modality(rows, fn, classes, args.output, name)
        print(f'{name}: test accuracy={results[name]["test_accuracy"]:.4f}', flush=True)
    results['fusion'] = 'Not evaluated: no paired ground truth. Validation weights are heuristic across different tasks.'
    if args.cnn:
        from .deep import MobileNetFeatures
        results['image_cnn'] = run_modality(image_rows(args.images), MobileNetFeatures(), ROASTS, args.output, 'image_cnn')
    (args.output / 'metrics.json').write_text(json.dumps(results, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
