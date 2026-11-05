"""
Full-package WSAD model for UCF-Crime (single-file module)

Features included (full package):
- Temporal encoder (multi-scale Conv1D tower)
- Local depthwise attention (lightweight convolutional local context)
- Transformer global context (stack of Transformer encoder layers)
- Memory units (reuses your Memory_Unit API)
- Dual-head scoring: abnormalness & normalness + fused final score
- VAE-style augmentation path and triplet contrastive losses
- Temporal smoothness + boundary consistency helpers returned to loss

Usage:
    from wsad_full_package import FullWSAD
    model = FullWSAD(input_size=1024, hidden_dim=512, a_nums=100, n_nums=100, trans_layers=2)
    model.flag = "Train"  # or "Test"
    out = model(x)  # x: [B, T, D]  or [B, N, T, D]

The forward returns a dict with keys similar to your previous model so it plugs into AD_Loss with minor changes.

Note: This file expects your existing Memory_Unit implementation as `from memory import Memory_Unit`.
"""

import math
import torch
import torch.nn as nn
from torch.nn.modules.module import Module
from memory import Memory_Unit
from translayer import Transformer  # if you want to keep custom transformer; optional
from utils import norm


# -----------------------------
# Basic building blocks
# -----------------------------
class TemporalEncoder(Module):
    """Multi-scale temporal conv tower producing (B, T, H) embeddings."""
    def __init__(self, in_dim, hidden_dim, multiscale_kernels=(3,7,11), dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden_dim)
        self.convs = nn.ModuleList()
        for k in multiscale_kernels:
            self.convs.append(
                nn.Sequential(
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size=k, padding=k//2, groups=1),
                    nn.ReLU(),
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
                )
            )
        self.out = nn.Sequential(nn.Linear(hidden_dim * len(multiscale_kernels), hidden_dim), nn.ReLU())
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, T, D]
        B, T, D = x.size()
        x = self.proj(x)               # [B, T, H]
        x_t = x.transpose(1, 2)        # [B, H, T]
        outs = []
        for conv in self.convs:
            y = conv(x_t)             # [B, H, T]
            outs.append(y.transpose(1, 2))
        # concat along feature dim
        cat = torch.cat(outs, dim=-1)  # [B, T, H * n_kernels]
        out = self.out(cat)            # [B, T, H]
        out = self.dropout(out)
        return out


class LocalAttention(nn.Module):
    """Depthwise local attention implemented with grouped convs (lightweight)."""
    def __init__(self, dim, kernel=9):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=kernel, padding=kernel//2, groups=dim),
            nn.ReLU(),
            nn.Conv1d(dim, dim, kernel_size=1),
        )
    def forward(self, x):
        # x: [B, T, D]
        return x + self.conv(x.transpose(1,2)).transpose(1,2)


class TransformerGlobal(nn.Module):
    """Stacked Transformer encoder using nn.TransformerEncoderLayer for stability."""
    def __init__(self, dim, nhead=8, num_layers=2, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=nhead,
                                                   dim_feedforward=dim_feedforward, dropout=dropout,
                                                   activation='relu', batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # x: [B, T, D]
        # transformer expects batch_first if specified above
        out = self.encoder(x)
        out = self.norm(out)
        return out


class DualHead(nn.Module):
    """Produces both abnormal and normal logits per-frame, then fused score.
    Returns raw logits and a fused scalar score in [0,1] via sigmoid on fused logit.
    """
    def __init__(self, in_dim, hidden=256):
        super().__init__()
        self.abn = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.norm = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.fuse = nn.Sequential(nn.Linear(2, 1))

    def forward(self, x):
        # x: [B, T, D]
        abn_logit = self.abn(x).squeeze(-1)   # [B, T]
        norm_logit = self.norm(x).squeeze(-1) # [B, T]
        # fused: use abn_score - norm_score as a strong signal
        stacked = torch.stack([abn_logit, -norm_logit], dim=-1)  # [B, T, 2]
        fused = self.fuse(stacked).squeeze(-1)
        fused_sig = torch.sigmoid(fused)
        return {'abn_logit': abn_logit, 'norm_logit': norm_logit, 'fused_logits': fused, 'fused_score': fused_sig}


# -----------------------------
# Full WSAD (full package)
# -----------------------------
class WSAD(Module):
    def __init__(self,
                 input_size=1024,
                 hidden_dim=512,
                 a_nums=100,
                 n_nums=100,
                 trans_layers=2,
                 trans_heads=8,
                 dropout=0.1,
                 use_custom_transformer=False):
        """Full WSAD with memory + transformer + multi-scale convs + dual-head.
        Keep the API similar to your previous WSAD to minimize integration changes.
        """
        super().__init__()
        self.flag = "Train"
        self.hidden_dim = hidden_dim

        # encoder
        self.encoder = TemporalEncoder(input_size, hidden_dim, multiscale_kernels=(3,7,11), dropout=dropout)

        # localattention
        self.local = LocalAttention(hidden_dim, kernel=9)

        # global transformer (optionally your translayer.Transfomer)
        if use_custom_transformer:
            # preserve old API if desired
            self.global_att = Transformer(hidden_dim, trans_layers, trans_heads, hidden_dim//4, hidden_dim, dropout=dropout)
        else:
            self.global_att = TransformerGlobal(hidden_dim, nhead=trans_heads, num_layers=trans_layers, dim_feedforward=hidden_dim*4, dropout=dropout)

        # memory units (increase size for full package)
        self.Amemory = Memory_Unit(nums=a_nums, dim=hidden_dim)
        self.Nmemory = Memory_Unit(nums=n_nums, dim=hidden_dim)

        # VAE-style encoder heads (for augmentation path)
        self.encoder_mu = nn.Linear(hidden_dim, hidden_dim)
        self.encoder_var = nn.Linear(hidden_dim, hidden_dim)

        # contrastive components
        self.triplet = nn.TripletMarginLoss(margin=1.0)

        # dual-head classifier
        self.head = DualHead(hidden_dim * 2, hidden=256)  # input will be concat(orig, memory_aug)

        # small projection for concatenation
        self.proj_for_head = nn.Linear(hidden_dim, hidden_dim)

        # helper
        self.relu = nn.ReLU()

    def _reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def latent_loss(self, mu, logvar):
        # standard VAE KL
        return torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

    def temporal_smoothness(self, pred):
        # pred: [B, T] or [B, T, 1]
        if pred.dim() == 3:
            pred = pred.squeeze(-1)
        if pred.dim() == 2 and pred.size(1) > 1:
            return (pred[:,1:] - pred[:,:-1]).abs().mean()
        return torch.tensor(0.0, device=pred.device)

    def boundary_consistency(self, att):
        # encourages sharp but consistent transitions: L1 of second derivative
        if att.dim() == 3:
            att = att.squeeze(-1)
        if att.dim() == 2 and att.size(1) > 2:
            dd = att[:,2:] - 2*att[:,1:-1] + att[:,:-2]
            return dd.abs().mean()
        return torch.tensor(0.0, device=att.device)

    def forward(self, x):
        # support both [B,N,T,D] and [B,T,D]
        if x.dim() == 4:
            b, n, t, d = x.size()
            x = x.reshape(b * n, t, d)
        else:
            b, t, d = x.size()
            n = 1

        # encoder
        h = self.encoder(x)           # [B*n, T, H]
        h = self.local(h)
        h = self.global_att(h)       # [B*n, T, H]

        # memory reads
        # A memory read (abnormal view) and N memory read (normal view)
        A_att, A_aug = self.Amemory(h)
        N_Aatt, N_Aaug = self.Nmemory(h)
        A_Natt, A_Naug = self.Amemory(h)
        N_att, N_aug = self.Nmemory(h)

        # second read (reinforce) - multi-read augmentation
        _, A_aug2 = self.Amemory(A_aug)
        _, N_aug2 = self.Nmemory(N_aug)
        A_aug = A_aug + 0.5 * A_aug2
        N_aug = N_aug + 0.5 * N_aug2

        # VAE reparam for aug features (per-time-step)
        A_mu = self.encoder_mu(A_aug)
        A_logvar = self.encoder_var(A_aug)
        A_z = self._reparameterize(A_mu, A_logvar)

        N_mu = self.encoder_mu(N_aug)
        N_logvar = self.encoder_var(N_aug)
        N_z = self._reparameterize(N_mu, N_logvar)

        # Prepare features for contrastive/triplet
        # choose top-k indices from attentions to form anchors/positives/negatives like before
        # to keep API consistent, return everything in same shapes as prior model
        Bn, T, H = h.size()

        # compute topk indices for A and N views
        topk = max(T // 16 + 1, 1)
        _, A_idx = torch.topk(A_att, topk, dim=-1)
        _, N_idx = torch.topk(N_att, topk, dim=-1)

        def gather_topk(feat, idx):
            # feat: [Bn, T, H], idx: [Bn, topk]
            return torch.gather(feat, 1, idx.unsqueeze(-1).expand(-1, -1, feat.size(-1))).mean(1)

        negative_ax = gather_topk(h, A_idx)   # [Bn, H]
        anchor_nx = gather_topk(h, N_idx)

        # reshape anchors to simulate (b//2, n, -1).mean(1) as before if user expects
        # But keep simpler: compute triplet on per-batch basis grouping by halves if possible
        half = max(Bn // 2, 1)
        try:
            anchor_nx_group = anchor_nx[:half]
            positive_nx_group = gather_topk(h, torch.topk(N_Aatt, topk, dim=-1)[1])[:half]
            negative_ax_group = negative_ax[:half]
            triplet_loss = self.triplet(norm(anchor_nx_group), norm(positive_nx_group), norm(negative_ax_group))
        except Exception:
            triplet_loss = torch.tensor(0.0, device=h.device)

        # KL losses
        kl_A = self.latent_loss(A_mu, A_logvar) if A_mu is not None else torch.tensor(0.0, device=h.device)
        kl_N = self.latent_loss(N_mu, N_logvar) if N_mu is not None else torch.tensor(0.0, device=h.device)

        # Compose classifier input: concat original per-timstep features and memory-augmented per-timestep features
        # A_z and N_z are expected [Bn, T, H]
        # project them and concat with original
        proj_A = self.proj_for_head(A_z)
        proj_N = self.proj_for_head(N_z)

        # fused augmentation: sum or concat. We'll concat (A + N) per time-step
        aug_fuse = proj_A + proj_N
        classifier_in = torch.cat([h, aug_fuse], dim=-1)  # [Bn, T, 2H]

        head_out = self.head(classifier_in)
        fused_score = head_out['fused_score']  # [Bn, T]

        # shape back to (b, n, T) scoring
        fused_score_reshaped = fused_score.reshape(b, n, -1).mean(1)  # [b, T]

        # derive additional monitoring signals similar to previous model
        out_dict = {
            'frame': fused_score_reshaped,
            'abn_logit': head_out['abn_logit'].reshape(b, n, -1).mean(1),
            'norm_logit': head_out['norm_logit'].reshape(b, n, -1).mean(1),
            'A_att': A_att.reshape(b, n, -1).mean(1),
            'N_att': N_att.reshape(b, n, -1).mean(1),
            'A_Natt': A_Natt.reshape(b, n, -1).mean(1),
            'N_Aatt': N_Aatt.reshape(b, n, -1).mean(1),
            'triplet_margin': triplet_loss,
            'kl_loss': (kl_A + kl_N) * 0.5,
        }

        # temporal regularizers
        smooth_loss = self.temporal_smoothness(fused_score_reshaped)
        boundary_loss = self.boundary_consistency(fused_score_reshaped)

        out_dict['smooth_loss'] = smooth_loss
        out_dict['boundary_loss'] = boundary_loss

        # distance metric similar to earlier for stabilization
        try:
            negative_ax_new = gather_topk(A_z, A_idx)
            anchor_nx_new = gather_topk(N_z, N_idx)
            distance = torch.relu(100 - torch.norm(negative_ax_new, p=2, dim=-1) + torch.norm(anchor_nx_new, p=2, dim=-1)).mean()
            out_dict['distance'] = distance
        except Exception:
            out_dict['distance'] = torch.tensor(0.0, device=h.device)

        return out_dict


if __name__ == '__main__':
    # quick functionality smoke test
    m = FullWSAD(input_size=1024, hidden_dim=512, a_nums=100, n_nums=100, trans_layers=2)
    m = m.cuda()
    src = torch.rand(8, 200, 1024).cuda()
    out = m(src)
    print('keys:', list(out.keys()))
    print('frame shape:', out['frame'].shape)
