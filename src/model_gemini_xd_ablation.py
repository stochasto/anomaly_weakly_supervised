import torch
import torch.nn as nn
from torch.nn.modules.module import Module
from memory_resume_cgpt import Memory_Unit
from translayer_resume_cgpt import Transformer
from utils import norm
import torch.nn.functional as F


class Temporal(Module):
    def __init__(self, input_size, out_size):
        super(Temporal, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=out_size, kernel_size=3,
                      stride=1, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv_1(x)
        x = x.permute(0, 2, 1)
        return x


class LocalAttention(nn.Module):
    """Lightweight depthwise-local attention via grouped convolutions."""
    def __init__(self, dim, kernel=9):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=kernel, padding=kernel // 2, groups=dim),
            nn.ReLU(),
            nn.Conv1d(dim, dim, kernel_size=1)
        )

    def forward(self, x):
        return x + self.conv(x.transpose(1, 2)).transpose(1, 2)


class ADCLS_head(Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim)
        )

    def forward(self, x):
        return self.mlp(x)


class WSAD(Module):
    """ 
    config dict can contain:
        'use_local_att' : bool (default True)
        'use_transformer' : bool (default True)
        'use_memory' : bool (default True)
        'use_vae' : bool (default True)
    """
    def __init__(self, input_size, flag, a_nums, n_nums, config=None):
        super().__init__()
        self.flag = flag
        self.a_nums = a_nums
        self.n_nums = n_nums

        # default config
        if config is None:
            config = {}
        self.use_local_att = config.get('use_local_att', True)
        self.use_transformer = config.get('use_transformer', True)
        self.use_memory = config.get('use_memory', True)
        self.use_vae = config.get('use_vae', True)

        # Core modules
        self.embedding = Temporal(input_size, 512)

        if self.use_local_att:
            self.localatt = LocalAttention(512)
        else:
            self.localatt = nn.Identity()

        if self.use_transformer:
            self.selfatt = Transformer(512, 2, 4, 128, 512, dropout=0.25)
        else:
            self.selfatt = nn.Identity()

        # Learnable attention temperature
        self.attn_temp = nn.Parameter(torch.tensor(0.5))

        self.triplet = nn.TripletMarginLoss(margin=0.3, swap=True)

        # Classification head: 1024 if memory is used (512 + 512), else 512
        if self.use_memory:
            self.cls_head = ADCLS_head(1024, 1)
        else:
            self.cls_head = ADCLS_head(512, 1)

        if self.use_memory:
            self.Amemory = Memory_Unit(nums=a_nums, dim=512)
            self.Nmemory = Memory_Unit(nums=n_nums, dim=512)

            if self.use_vae:
                self.encoder_mu = nn.Sequential(nn.Linear(512, 512))
                self.encoder_var = nn.Sequential(nn.Linear(512, 512))
            else:
                # No VAE: just use mu for deterministic augmentation
                self.encoder_mu = nn.Sequential(nn.Linear(512, 512))
                self.encoder_var = None   # will not be used
        else:
            # No memory blocks – nothing extra to create
            pass

        self.relu = nn.ReLU()   # not used, harmless

    def _reparameterize(self, mu, logvar):
        if self.use_vae and logvar is not None:
            std = torch.exp(0.5 * logvar)
            epsilon = torch.randn_like(std)
            return mu + epsilon * std
        else:
            return mu

    def latent_loss(self, mu, var):
        if self.use_vae and var is not None:
            return torch.mean(-0.5 * torch.sum(1 + var - mu ** 2 - var.exp(), dim=1))
        else:
            return torch.tensor(0.0, device=mu.device)

    def temporal_smoothness(self, pred):
        if pred.dim() == 2 and pred.size(1) > 1:
            return (pred[:, 1:] - pred[:, :-1]).abs().mean()
        return torch.tensor(0.0, device=pred.device)

    def attn_pool(self, feats, scores):
        temp = torch.clamp(self.attn_temp, min=0.1)
        weights = torch.softmax(scores / temp, dim=1)
        pooled = (feats * weights.unsqueeze(-1)).sum(dim=1)
        return pooled

    def forward(self, x):
        if len(x.size()) == 4:
            b, n, t, d = x.size()
            x = x.reshape(b * n, t, d)
        else:
            b, t, d = x.size()
            n = 1

        x = self.embedding(x)
        x = self.localatt(x)
        x = self.selfatt(x)

        feat_magnitudes = torch.norm(x, p=2, dim=-1)   # [B, T]

        if self.flag == "Train":
            if not self.use_memory:
                # Simple MIL mode: no memory, just classify and return MIL loss
                pre_att = self.cls_head(x).reshape((b, n, -1)).mean(1)
                return {"frame": pre_att}   # loss will use SimpleMIL

            N_x = x[: b * n // 2]                      # normal videos
            A_x = x[b * n // 2 :]                      # abnormal videos

            # ---------- Memory reads ----------
            A_att, A_aug = self.Amemory(A_x)
            N_Aatt, N_Aaug = self.Nmemory(A_x)
            A_Natt, A_Naug = self.Amemory(N_x)
            N_att, N_aug = self.Nmemory(N_x)

            A_Natt_raw = A_Natt.clone()

            # Gating
            A_Natt = A_Natt * (1.0 - N_att)

            # ---------- Triplet ----------
            anchor   = self.attn_pool(N_x, N_att).reshape(b // 2, n, -1).mean(1)
            positive = self.attn_pool(A_x, N_Aatt).reshape(b // 2, n, -1).mean(1)
            negative = self.attn_pool(A_x, A_att).reshape(b // 2, n, -1).mean(1)

            anchor   = norm(anchor)
            positive = norm(positive)
            negative = norm(negative)
            triplet_margin_loss = self.triplet(anchor, positive, negative)

            # ---------- VAE ----------
            if self.use_vae:
                N_aug_mu = self.encoder_mu(N_aug)
                N_aug_var = self.encoder_var(N_aug)
                N_aug_new = 0.8 * N_aug_mu + 0.2 * self._reparameterize(N_aug_mu, N_aug_var)
                kl_loss = self.latent_loss(N_aug_mu, N_aug_var)
            else:
                # Deterministic encoding
                N_aug_new = self.encoder_mu(N_aug)
                kl_loss = torch.tensor(0.0, device=x.device)

            A_aug_new = self.encoder_mu(A_aug)
            A_Naug = self.encoder_mu(A_Naug)
            N_Aaug = self.encoder_mu(N_Aaug)

            # ---------- Distance loss ----------
            abnormal_topk = self.attn_pool(A_aug_new, A_att).reshape(b // 2, n, -1).mean(1)
            normal_topk   = self.attn_pool(N_aug_new, N_att).reshape(b // 2, n, -1).mean(1)
            distance_loss = torch.relu(
                100.0 - abnormal_topk.pow(2).sum(1) + normal_topk.pow(2).sum(1)
            ).mean()

            # ---------- Confusion penalties ----------
            A_N_weights = torch.softmax(A_Natt_raw / 0.5, dim=1)
            A_N_mean = (A_Natt_raw * A_N_weights).sum(dim=1)
            conf_loss = torch.relu(A_N_mean - 0.1).mean()
            A_N_mse = F.mse_loss(A_Natt_raw, torch.zeros_like(A_Natt_raw))

            # ---------- Gated fusion ----------
            gate_N   = N_att.unsqueeze(-1)
            gate_A_N = A_Natt.unsqueeze(-1)
            gate_A   = A_att.unsqueeze(-1)
            gate_N_A = N_Aatt.unsqueeze(-1)

            N_aug_final   = N_aug_new * gate_N
            A_Naug_final  = A_Naug   * gate_A_N
            A_aug_final   = A_aug_new * gate_A
            N_Aaug_final  = N_Aaug   * gate_N_A

            x = torch.cat((x, torch.cat([N_aug_final + A_Naug_final,
                                          A_aug_final + N_Aaug_final], dim=0)), dim=-1)
            pre_att = self.cls_head(x).reshape((b, n, -1)).mean(1)

            # ---------- Feature magnitude loss ----------
            k_mag = max(t // 16 + 1, 1)
            N_mag = feat_magnitudes[: b * n // 2]
            A_mag = feat_magnitudes[b * n // 2 :]
            topk_A_mag = torch.topk(A_mag, k_mag, dim=-1)[0].mean(-1).mean()
            topk_N_mag = torch.topk(N_mag, k_mag, dim=-1)[0].mean(-1).mean()
            feat_mag_loss = torch.relu(100.0 - (topk_A_mag - topk_N_mag))
            mag_reg = 0.001 * (topk_A_mag ** 2 + topk_N_mag ** 2)
            feat_mag_loss = feat_mag_loss + mag_reg

            # ---------- Temporal smoothness ----------
            smooth_loss = self.temporal_smoothness(pre_att)

            # ---------- Orthogonality ----------
            ma = F.normalize(self.Amemory.memory_block, p=2, dim=-1)
            mn = F.normalize(self.Nmemory.memory_block, p=2, dim=-1)
            o_margin = 0.01
            ortho_matrix = torch.mm(ma, mn.t())
            ortho_loss = torch.relu(ortho_matrix.abs() - o_margin).mean() + 1.2 * ortho_matrix.abs().max()

            return {
                "frame": pre_att,
                "triplet_margin": triplet_margin_loss,
                "kl_loss": kl_loss,
                "distance": distance_loss,
                "smooth_loss": smooth_loss,
                "feat_mag_loss": feat_mag_loss,
                "A_att": A_att,
                "N_att": N_att,
                "A_Natt": A_Natt,
                "N_Aatt": N_Aatt,
                "conf_loss": conf_loss,
                "A_N_mse": A_N_mse,
                "A_Natt_raw": A_Natt_raw,
                "N_att_raw": N_att,
                "ortho_loss": ortho_loss
            }

        else:   # Inference
            if not self.use_memory:
                pre_att = self.cls_head(x).reshape((b, n, -1)).mean(1)
                return {"frame": pre_att}

            b, t, d = x.size()
            n = 1

            A_att, A_aug = self.Amemory(x)
            N_att, N_aug = self.Nmemory(x)

            A_aug = self.encoder_mu(A_aug)
            N_aug = self.encoder_mu(N_aug)

            gate_A = A_att.unsqueeze(-1)
            gate_N = N_att.unsqueeze(-1)

            fused_features = torch.cat([x, (A_aug * gate_A) + (N_aug * gate_N)], dim=-1)
            pre_att = self.cls_head(fused_features)
            pre_att = pre_att.reshape((b, n, -1)).mean(1)
            return {"frame": pre_att}