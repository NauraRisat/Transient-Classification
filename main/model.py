import numpy as np
import torch
import torch.nn as nn
from einops import rearrange

from .attention import TransformerBlock
from .feature_generator import feature_generator
from .dataset import AUX_DIM


# Continuous-time sinusoidal positional encoding using normalized MJD timestamp values
class TimePositionalEncoding(nn.Module):
    # Initialise frequency scale factors for continuous-time embedding
    def __init__(self, configs):
        super().__init__()
        self.num_hidden = configs[1]
        self.seq = configs[4]
        inv_freq = torch.tensor(
            [1.0 / np.power(10000, 2 * (j // 2) / self.num_hidden)
             for j in range(self.num_hidden)], dtype=torch.float32)
        self.register_buffer('inv_freq', inv_freq)

    # Compute time-based sine and cosine embeddings and add them to input spatial features
    def forward(self, x, t_chunk):
        """
        :param x: (B, num_hidden, H, W, L) chunked features
        :param t_chunk: (B, L) normalized chunk times in [0, 1]
        """
        angles = t_chunk[:, None, :] * (self.seq - 1) * self.inv_freq[None, :, None]
        pe = torch.empty_like(angles)
        pe[:, 0::2, :] = torch.sin(angles[:, 0::2, :])
        pe[:, 1::2, :] = torch.cos(angles[:, 1::2, :])
        return x + pe[:, :, None, None, :]


# Core ConvBERT backbone with time-aware positional encoding and spatio-temporal layers
class ConvBERTTime(nn.Module):
    # Initialise filter embeddings, feature extractors, and transformer blocks.
    def __init__(self, hidden, n_layers, attn_heads, dropout=0.1, cnn_stack=3):
        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads
        self.feed_forward_hidden = hidden[1] * 2
        self.cnn_stack = cnn_stack

        self.embedding_band = nn.Embedding(10, 1024, padding_idx=0)
        self.embedding = TimePositionalEncoding(configs=hidden)
        self.feature_embedding = feature_generator(hidden)
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, self.feed_forward_hidden, dropout)
             for _ in range(n_layers)])

    # Calculate average normalized timestamp for valid frames inside each chunk
    def chunk_times(self, time_norm, band):
        b, l_raw = time_norm.shape
        n_chunks = l_raw // self.cnn_stack
        usable = n_chunks * self.cnn_stack
        t = time_norm[:, :usable].reshape(b, n_chunks, self.cnn_stack)
        valid = (band[:, :usable].reshape(b, n_chunks, self.cnn_stack) != 0).float()
        return (t * valid).sum(-1) / valid.sum(-1).clamp(min=1.0)

    # Process input tensors through band embedding, CNN feature generation, continuous time encoding, and transformer blocks
    def forward(self, x, mask, band, time_norm):
        b, n, h, w, l = x.shape
        curr_band = self.embedding_band(band)
        curr_band = rearrange(torch.unsqueeze(curr_band, 2), 'b l n (h w) -> b n h w l',
                              b=b, n=1, h=h, w=w, l=l)
        x = torch.cat((x, curr_band), 1)

        gen_img = []
        for i in range(l // self.cnn_stack):
            gen_img.append(torch.squeeze(
                self.feature_embedding(x[..., i * self.cnn_stack:(i + 1) * self.cnn_stack]),
                dim=-1))
        x = torch.stack(gen_img, dim=-1)

        t_chunk = self.chunk_times(time_norm.float(), band)
        x = self.embedding(x, t_chunk)

        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)
        return x


# Wrapper module linking ConvBERT backbone with classification header
class ConvBERTClassificationTime(nn.Module):
    # Initialise classification wrapper around ConvBERT backbone
    def __init__(self, sbert: ConvBERTTime, num_classes):
        super().__init__()
        self.sbert = sbert
        self.cnn_stack = sbert.cnn_stack
        self.classification = MulticlassClassification(self.sbert.hidden, num_classes)

    # Execute ConvBERT backbone, construct chunk validity masks, and pass features to classifier
    def forward(self, x, mask, band, time_norm, aux=None, return_attn=False):
        seq_out = self.sbert(x, mask, band, time_norm)

        b, l_raw = band.shape
        n_chunks = seq_out.shape[-1]
        cs = self.cnn_stack
        usable = (l_raw // cs) * cs
        valid_mask = None
        if usable > 0:
            band_chunks = band[:, :usable].reshape(b, l_raw // cs, cs)
            valid_mask = (band_chunks != 0).any(dim=-1)
            if valid_mask.shape[1] != n_chunks:
                valid_mask = None

        return self.classification(seq_out, valid_mask=valid_mask, aux=aux,
                                   return_attn=return_attn)


# Classification head combining masked temporal pooling (attention, std, diff) and auxiliary statistics
class MulticlassClassification(nn.Module):
    # Initialise temporal pooling layers, auxiliary feature MLP, and linear classification head
    def __init__(self, hidden, num_classes):
        super().__init__()
        feat_dim = hidden[1] * hidden[2] * hidden[3]

        self.temporal_attn = nn.Linear(feat_dim, 1)
        self.pool_norm = nn.LayerNorm(feat_dim * 3)
        self.linear1 = nn.Linear(feat_dim * 3, 128)

        self.aux_embed = nn.Sequential(
            nn.Linear(AUX_DIM, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
        )

        self.linear3 = nn.Linear(128 + 32, num_classes)
        self.dropout = nn.Dropout(p=0.4)
        self.activation = nn.ReLU(inplace=True)

    # Perform masked temporal pooling, combine with auxiliary features, and compute output logits
    def forward(self, x, valid_mask=None, aux=None, return_attn=False):
        b, c, h, w, l = x.shape
        x_seq = x.permute(0, 4, 1, 2, 3).reshape(b, l, c * h * w)

        attn_logits = self.temporal_attn(x_seq).squeeze(-1)
        if valid_mask is not None:
            attn_logits = attn_logits.masked_fill(~valid_mask, float('-inf'))

        attn_weights = torch.softmax(attn_logits, dim=1)
        pooled_attn = torch.bmm(attn_weights.unsqueeze(1), x_seq).squeeze(1)

        if valid_mask is not None:
            mask_f = valid_mask.unsqueeze(-1).float()
            count = mask_f.sum(dim=1).clamp(min=1.0)
            mean = (x_seq * mask_f).sum(dim=1) / count
            var = ((x_seq - mean.unsqueeze(1)) ** 2 * mask_f).sum(dim=1) / count
            pooled_std = torch.sqrt(var + 1e-6)

            diffs = (x_seq[:, 1:, :] - x_seq[:, :-1, :]).abs()
            pair_mask = (valid_mask[:, 1:] & valid_mask[:, :-1]).unsqueeze(-1).float()
            pair_count = pair_mask.sum(dim=1).clamp(min=1.0)
            pooled_diff = (diffs * pair_mask).sum(dim=1) / pair_count
        else:
            pooled_std = x_seq.std(dim=1)
            pooled_diff = (x_seq[:, 1:, :] - x_seq[:, :-1, :]).abs().mean(dim=1)

        pooled = torch.cat([pooled_attn, pooled_std, pooled_diff], dim=1)
        pooled = self.pool_norm(pooled)

        if aux is None:
            aux = torch.zeros(b, AUX_DIM, device=x.device, dtype=pooled.dtype)
        aux_feat = self.aux_embed(aux)

        out = self.dropout(self.activation(self.linear1(self.dropout(pooled))))
        out = self.linear3(torch.cat([out, aux_feat], dim=1))

        if return_attn:
            return out, attn_weights
        return out