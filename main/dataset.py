import glob
import pickle
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.signal import find_peaks
from einops import rearrange


AUX_DIM = 17
CNN_STACK = 3


# Set random seeds for reproducibility across PyTorch, NumPy, and Python
def setup_seed(seed=123):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# Filter out None samples before collating a batch
def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    return torch.utils.data.dataloader.default_collate(batch)


# Cross-entropy loss with focal modulation and label smoothing for imbalanced data
class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=None, gamma=2, label_smoothing=0.05):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        ce_loss = torch.nn.functional.cross_entropy(
            inputs, targets, reduction='none', weight=self.alpha,
            label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss


# Calculate lag-1 autocorrelation for a 1D sequence
def _lag1_autocorr(x):
    # 0.0 if too short or degenerate
    if len(x) < 3:
        return 0.0
    x0 = x[:-1] - x[:-1].mean()
    x1 = x[1:] - x[1:].mean()
    denom = np.sqrt((x0 ** 2).sum() * (x1 ** 2).sum())
    if denom <= 0:
        return 0.0
    return float((x0 * x1).sum() / denom)


# Extract 17 scale-free variability statistics from raw light-curve data
def compute_aux_features(images, filters):
    flux = images.sum(axis=(1, 2)).astype(np.float64)
    eps = 1e-8
    feats = np.zeros(AUX_DIM, dtype=np.float32)

    for i, band_id in enumerate(range(1, 6)):
        band_flux = flux[filters == band_id]
        if len(band_flux) >= 2:
            feats[i] = np.arcsinh(band_flux.std() / (abs(band_flux.mean()) + eps))
        feats[5 + i] = _lag1_autocorr(band_flux)

    feats[10] = _lag1_autocorr(flux)
    mean = flux.mean()
    std = flux.std()
    if std > 0:
        feats[11] = np.arcsinh(((flux - mean) ** 3).mean() / (std ** 3 + eps))
    feats[12] = np.arcsinh((flux.max() - flux.min()) / (abs(mean) + eps))

    if len(flux) >= 4:
        sorted_flux = np.sort(flux)
        noise_proxy = sorted_flux[: max(1, len(flux) // 4)].std()
    else:
        noise_proxy = std
    feats[13] = np.arcsinh(max(0.0, std ** 2 - noise_proxy ** 2) / (mean ** 2 + eps))

    if std > 0:
        feats[14] = np.arcsinh(((flux - mean) ** 4).mean() / (std ** 4 + eps) - 3)

    if std > 0 and len(flux) >= 5:
        _, props = find_peaks(flux, prominence=0.5 * std)
        prominences = props['prominences']
        feats[15] = np.arcsinh(len(prominences))
        if len(prominences) >= 2:
            top2 = np.sort(prominences)[-2:]
            feats[16] = float(top2[0] / (top2[1] + eps))

    return feats


# PyTorch Dataset for loading and processing SDSS image time-series data with pickle format {'images', 'filter', 'label', 'mjd'}
class SDSSDatasetTime(Dataset):
    # Initialize data file paths, sequence settings, and augmentation parameters
    def __init__(self, file_list=None, classes=None, seq_len=99, is_train=False, mask_prob=0.1):
        self.seq_len = seq_len
        self.classes = classes
        self.data_files = list(file_list) if file_list is not None else []
        self.TS_num = len(self.data_files)
        self.is_train = is_train
        self.mask_prob = mask_prob

    # Return total number of samples in dataset
    def __len__(self):
        return self.TS_num

    # Load, augment, subsample, and format a single astronomical sample into PyTorch tensors
    def __getitem__(self, item):
        with open(self.data_files[item], "rb") as fh:
            data_object = pickle.load(fh)

        images = data_object['images']
        label = data_object['label']
        mjd = np.asarray(data_object['mjd'], dtype=np.float64)
        total_len = images.shape[0]

        aux_feats = compute_aux_features(images, data_object['filter'])

        if self.is_train:
            k = random.randint(0, 3)
            if k:
                images = np.rot90(images, k, axes=(1, 2))
            if random.random() < 0.5:
                images = images[:, ::-1, :]
            if random.random() < 0.5:
                images = images[:, :, ::-1]
            images = np.ascontiguousarray(images)

        if total_len > self.seq_len:
            if label in ('SNIa', 'pSNIa'):
                flux_per_frame = np.sum(images, axis=(1, 2))
                peak_idx = int(np.argmax(flux_per_frame))
                if self.is_train:
                    shift = random.randint(-2, 2)
                    peak_idx = max(0, min(total_len - 1, peak_idx + shift))
                start_ind = max(0, peak_idx - self.seq_len // 2)
                if start_ind + self.seq_len > total_len:
                    start_ind = total_len - self.seq_len
                indices = np.arange(start_ind, start_ind + self.seq_len)
            else:
                base_indices = np.linspace(0, total_len - 1, self.seq_len)
                if self.is_train:
                    jitter = np.random.randint(-2, 3, size=self.seq_len)
                    indices = np.clip(np.round(base_indices + jitter), 0, total_len - 1).astype(int)
                    indices = np.sort(indices)
                else:
                    indices = np.round(base_indices).astype(int)
            ts_length = self.seq_len
        else:
            indices = np.arange(total_len)
            ts_length = total_len

        window_mjd = mjd[indices]
        t_span = window_mjd.max() - window_mjd.min()
        if t_span > 0:
            t_norm_window = (window_mjd - window_mjd.min()) / t_span
        else:
            t_norm_window = np.zeros_like(window_mjd)
        time_norm = np.zeros(self.seq_len, dtype=np.float32)
        time_norm[:ts_length] = t_norm_window.astype(np.float32)

        banded_seq = np.zeros((self.seq_len, 1, 32, 32))
        banded_seq[:ts_length, 0] = images[indices]

        seq_label = np.array(self.classes[label], dtype=int)
        bands = np.zeros(self.seq_len)
        bands[:ts_length] = data_object['filter'][indices]

        banded_seq = rearrange(banded_seq, 'l n h w -> n h w l',
                               l=self.seq_len, n=1, h=32, w=32)

        non_zero = banded_seq[banded_seq != 0]
        if len(non_zero) > 0:
            mean_val = np.mean(non_zero)
            std_val = np.std(non_zero)
            if std_val == 0:
                std_val = 1.0
            banded_seq = np.where(banded_seq != 0, (banded_seq - mean_val) / std_val, 0)

        if self.is_train:
            noise = np.random.normal(0, 0.05, banded_seq.shape)
            banded_seq = np.where(banded_seq != 0, banded_seq + noise, 0)

            if self.mask_prob > 0:
                frame_mask = np.random.rand(self.seq_len) < self.mask_prob
                banded_seq[..., frame_mask] = 0

        output = {
            "bert_input": banded_seq,
            "class_label": seq_label,
            "bands": bands,
            "time_norm": time_norm,
            "aux_feats": aux_feats,
            "original_idx": item,
        }
        return {key: torch.from_numpy(np.asarray(value)) if not isinstance(value, int) else value
                for key, value in output.items()}