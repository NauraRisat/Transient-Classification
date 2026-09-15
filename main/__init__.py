from .model import ConvBERTTime, ConvBERTClassificationTime, MulticlassClassification
from .dataset import SDSSDatasetTime, FocalLoss, collate_fn, setup_seed, AUX_DIM
from .splits_cv import get_cv_splits_v3, resolve_cv_fold, resolve_test_set
from .train_cv import train_all_folds, evaluate_on_test

# Constants
CLASSES = {'AGN': 0, 'SNIa': 1, 'pSNIa': 1, 'Variable': 2}
CLASS_LIST = ['AGN', 'SNIa', 'Variable']
NUM_CLASSES = 3
CONFIG_SIMPLE = {
    'hidden': [0, 4, 9, 9, 33],  # [unused, C, H, W, L_chunked]
    'layers': 1,
    'attn_heads': 2,
    'dropout': 0.75,
}

__all__ = [
    'ConvBERTTime',
    'ConvBERTClassificationTime',
    'MulticlassClassification',
    'SDSSDatasetTime',
    'FocalLoss',
    'collate_fn',
    'setup_seed',
    'AUX_DIM',
    'get_cv_splits_v3',
    'resolve_cv_fold',
    'resolve_test_set',
    'train_all_folds',
    'evaluate_on_test',
    'CLASSES',
    'CLASS_LIST',
    'NUM_CLASSES',
    'CONFIG_SIMPLE',
]
