"""
5-fold cross-validation splits using dataset2/combined (original+augmented train)
Object-level stratified splits across all folds
"""
import os
import glob
import json
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

import sys
import os as _os

DATA_ROOT = r'C:\TA_SDSS\dataset2'
CLASSES = {'AGN': 0, 'SNIa': 1, 'pSNIa': 1, 'Variable': 2}
NUM_CLASSES = 3

# Create outputs directory if it doesn't exist
_OUTPUT_DIR = r'C:\TA_SDSS\outputs'
_os.makedirs(_OUTPUT_DIR, exist_ok=True)

DEFAULT_CV_SPLITS_PATH = _os.path.join(_OUTPUT_DIR, 'data_splits_cv_v3.json')


def _rel(path):
    return os.path.join(os.path.basename(os.path.dirname(path)), os.path.basename(path))


def get_cv_splits_v3(n_splits=5, test_size=0.15, seed=123,
                     splits_path=DEFAULT_CV_SPLITS_PATH, force_recompute=False):
    """
    Returns dict with keys: 'folds', 'test'

    folds: list of n_splits dicts, each with keys 'train_orig', 'train_aug', 'val'
      - train_orig: list of relative paths from original/ (augmented)
      - train_aug: list of relative paths from augmented/ (augmented)
      - val: list of relative paths from original/ (validation)
    test: list of relative paths from original/ (held-out test set)

    All paths use the same relative format so they can be resolved to any variant
    Stratification is by merged class (pSNIa -> SNIa)
    """
    if os.path.exists(splits_path) and not force_recompute:
        with open(splits_path, 'r') as fh:
            splits = json.load(fh)
        print(f"Loaded existing CV splits from {splits_path}")
        return splits

    pattern = os.path.join(DATA_ROOT, 'original', '*', '*.pickle')
    files = sorted(glob.glob(pattern))
    if len(files) == 0:
        raise FileNotFoundError(f"No pickles found at {pattern}")

    labels = [CLASSES[os.path.basename(os.path.dirname(f))] for f in files]

    # First split: test set
    trainval_files, test_files, trainval_labels, _ = train_test_split(
        files, labels, test_size=test_size, stratify=labels, random_state=seed)

    # 5-fold CV on the trainval set
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(trainval_files, trainval_labels)):
        train_files = [trainval_files[i] for i in train_idx]
        val_files = [trainval_files[i] for i in val_idx]

        fold_data = {
            'train_orig': [_rel(f) for f in train_files],
            'train_aug': [_rel(f) for f in train_files],  # same files, different variant
            'val': [_rel(f) for f in val_files],
        }
        folds.append(fold_data)

    test_rel = [_rel(f) for f in test_files]

    splits = {'folds': folds, 'test': test_rel}

    # Save splits
    os.makedirs(os.path.dirname(splits_path), exist_ok=True)
    with open(splits_path, 'w') as fh:
        json.dump(splits, fh, indent=2)

    print(f"Computed and saved CV splits to {splits_path}")
    print(f"  n_folds: {n_splits}")
    print(f"  test set: {len(test_rel)} objects")
    for i, fold in enumerate(folds):
        print(f"  fold {i}: train {len(fold['train_orig'])}, val {len(fold['val'])}")

    return splits


def resolve_cv_fold(fold_dict, variant='combined'):
    """
    Resolve a fold dict (with relative paths) to absolute paths

    variant='combined' uses train_orig + train_aug for training,
    'original' uses just train_orig, 'augmented' uses just train_aug
    """
    assert variant in ('combined', 'original', 'augmented'), f"Unknown variant: {variant}"

    variant_dir = os.path.join(DATA_ROOT, 'original') if variant in ('combined', 'original') else os.path.join(DATA_ROOT, 'augmented')

    train_files = [os.path.join(variant_dir, r) for r in fold_dict['train_orig']]

    if variant == 'combined':
        aug_dir = os.path.join(DATA_ROOT, 'augmented')
        train_aug = [os.path.join(aug_dir, r) for r in fold_dict['train_aug']]
        train_files = train_files + train_aug

    val_files = [os.path.join(variant_dir, r) for r in fold_dict['val']]

    # Check all files exist
    for f in train_files + val_files:
        if not os.path.exists(f):
            raise FileNotFoundError(f"Split file missing: {f}")

    return {'train': train_files, 'val': val_files}


def resolve_test_set():
    splits = get_cv_splits_v3()
    variant_dir = os.path.join(DATA_ROOT, 'original')
    test_files = [os.path.join(variant_dir, r) for r in splits['test']]

    for f in test_files:
        if not os.path.exists(f):
            raise FileNotFoundError(f"Test set file missing: {f}")

    return test_files
