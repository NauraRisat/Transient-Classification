"""
5-fold CV training infrastructure
Handles model building, fold training, and result aggregation for dashboard3
"""
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

from .model import ConvBERTTime, ConvBERTClassificationTime
from .dataset import SDSSDatasetTime, FocalLoss, setup_seed, collate_fn, AUX_DIM
from .splits_cv import get_cv_splits_v3, resolve_cv_fold, resolve_test_set


def build_model(config=None, num_classes=3):
    if config is None:
        config = {'hidden': [0, 4, 9, 9, 33], 'layers': 1, 'attn_heads': 2, 'dropout': 0.75}
    sbert = ConvBERTTime(
        hidden=config['hidden'],
        n_layers=config['layers'],
        attn_heads=config['attn_heads'],
        dropout=config['dropout'],
        cnn_stack=3
    )
    return ConvBERTClassificationTime(sbert, num_classes)


def compute_class_weights(file_list, classes=None, num_classes=3):
    if classes is None:
        classes = {'AGN': 0, 'SNIa': 1, 'pSNIa': 1, 'Variable': 2}
    # Compute Focal Loss weights (inverse class frequency)
    labels = []
    for f in file_list:
        try:
            import pickle
            with open(f, 'rb') as fh:
                d = pickle.load(fh)
            label_str = d.get('label', 'Unknown')
            labels.append(classes.get(label_str, 0))
        except:
            pass

    labels = np.array(labels)
    class_counts = np.bincount(labels, minlength=num_classes)
    total = class_counts.sum()
    weights = torch.tensor([total / (c + 1e-6) for c in class_counts], dtype=torch.float32)
    return weights


def train_one_epoch(model, train_loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in train_loader:
        optimizer.zero_grad()

        bert_input = batch['bert_input'].float().to(device)
        bands = batch['bands'].long().to(device)
        time_norm = batch['time_norm'].float().to(device)
        aux_feats = batch['aux_feats'].float().to(device)
        labels = batch['class_label'].long().to(device)

        logits = model(bert_input, None, bands, time_norm, aux=aux_feats)
        loss = loss_fn(logits, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def validate(model, val_loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            bert_input = batch['bert_input'].float().to(device)
            bands = batch['bands'].long().to(device)
            time_norm = batch['time_norm'].float().to(device)
            aux_feats = batch['aux_feats'].float().to(device)
            labels = batch['class_label'].long().to(device)

            logits = model(bert_input, None, bands, time_norm, aux=aux_feats)
            loss = loss_fn(logits, labels)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    avg_loss = total_loss / len(val_loader)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    accuracy = (all_preds == all_labels).mean()

    return avg_loss, accuracy, all_preds, all_labels


def train_fold(fold_idx, fold_data, device, config=None, max_epochs=30,
               early_stopping_patience=5, batch_size=16, classes=None, num_classes=3):
    if config is None:
        config = {'hidden': [0, 4, 9, 9, 33], 'layers': 1, 'attn_heads': 2, 'dropout': 0.75}
    if classes is None:
        classes = {'AGN': 0, 'SNIa': 1, 'pSNIa': 1, 'Variable': 2}

    print(f"\n--- Fold {fold_idx + 1} / 5 ---")

    # Build datasets and loaders
    fold_resolved = resolve_cv_fold(fold_data, variant='combined')
    train_dataset = SDSSDatasetTime(
        file_list=fold_resolved['train'],
        classes=classes,
        seq_len=99,
        is_train=True,
        mask_prob=0.1
    )
    val_dataset = SDSSDatasetTime(
        file_list=fold_resolved['val'],
        classes=classes,
        seq_len=99,
        is_train=False,
        mask_prob=0.0
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=0)

    # Build model and optimizer
    model = build_model(config, num_classes).to(device)
    weights = compute_class_weights(fold_resolved['train'], classes, num_classes)
    weights = weights.to(device)
    loss_fn = FocalLoss(alpha=weights, gamma=2.0)

    optimizer = AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4, amsgrad=True)

    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    pbar = tqdm(range(max_epochs), desc=f"Fold {fold_idx + 1}")
    for epoch in pbar:
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss, val_acc, _, _ = validate(model, val_loader, loss_fn, device)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        pbar.set_postfix({'train_loss': f'{train_loss:.4f}', 'val_loss': f'{val_loss:.4f}',
                          'val_acc': f'{val_acc:.4f}'})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                pbar.close()
                print(f"Early stopping at epoch {epoch + 1}")
                break

    # Restore best model and evaluate on validation
    model.load_state_dict(best_model_state)
    _, _, val_preds, val_labels = validate(model, val_loader, loss_fn, device)

    return {
        'model': model,
        'history': history,
        'val_preds': val_preds,
        'val_labels': val_labels,
        'best_val_loss': best_val_loss,
    }


def train_all_folds(device=None, config=None, max_epochs=30,
                    early_stopping_patience=5, batch_size=16, n_splits=5,
                    classes=None, num_classes=3):
    # Train all 5 folds and return results
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if config is None:
        config = {'hidden': [0, 4, 9, 9, 33], 'layers': 1, 'attn_heads': 2, 'dropout': 0.75}
    if classes is None:
        classes = {'AGN': 0, 'SNIa': 1, 'pSNIa': 1, 'Variable': 2}

    setup_seed(123)

    splits = get_cv_splits_v3(n_splits=n_splits)
    folds_results = []

    for fold_idx, fold_data in enumerate(splits['folds']):
        result = train_fold(fold_idx, fold_data, device, config, max_epochs,
                           early_stopping_patience, batch_size, classes, num_classes)
        folds_results.append(result)

    return folds_results, splits


def evaluate_on_test(folds_results, splits, device=None, classes=None, num_classes=3):
    # Evaluate all fold models on the held-out test set
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if classes is None:
        classes = {'AGN': 0, 'SNIa': 1, 'pSNIa': 1, 'Variable': 2}

    test_files = resolve_test_set()
    test_dataset = SDSSDatasetTime(
        file_list=test_files,
        classes=classes,
        seq_len=99,
        is_train=False,
        mask_prob=0.0
    )
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False,
                             collate_fn=collate_fn, num_workers=0)

    all_preds_per_fold = []
    all_probs_per_fold = []

    for fold_idx, fold_result in enumerate(folds_results):
        model = fold_result['model'].to(device)
        model.eval()
        preds = []
        probs = []

        with torch.no_grad():
            for batch in test_loader:
                bert_input = batch['bert_input'].float().to(device)
                bands = batch['bands'].long().to(device)
                time_norm = batch['time_norm'].float().to(device)
                aux_feats = batch['aux_feats'].float().to(device)

                logits = model(bert_input, None, bands, time_norm, aux=aux_feats)
                pred = torch.argmax(logits, dim=1).cpu().numpy()
                prob = torch.softmax(logits, dim=1).cpu().numpy()

                preds.append(pred)
                probs.append(prob)

        all_preds_per_fold.append(np.concatenate(preds))
        all_probs_per_fold.append(np.concatenate(probs))

    # Get true labels from test set
    test_labels = np.array([test_dataset[i]['class_label'].item() for i in range(len(test_dataset))])

    return {
        'test_labels': test_labels,
        'preds_per_fold': all_preds_per_fold,
        'probs_per_fold': all_probs_per_fold,
    }