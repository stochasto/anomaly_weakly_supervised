import torch
import torch.nn as nn
from torch.nn.modules.module import Module
import torch.nn.functional as F
from memory_resume_cgpt import Memory_Unit      # soft attention, acceptable
from translayer_resume_cgpt import Transformer
from utils import norm


class Temporal(Module):
    def __init__(self, input_size, out_size):
        super().__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv1d(input_size, out_size, 3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv_1(x)
        x = x.permute(0, 2, 1)
        return x


class LocalAttention(nn.Module):
    def __init__(self, dim, kernel=9):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(dim, dim, kernel, padding=kernel // 2, groups=dim),
            nn.ReLU(),
            nn.Conv1d(dim, dim, 1),
        )

    def forward(self, x):
        return x + self.conv(x.transpose(1, 2)).transpose(1, 2)


class ADCLS_head(Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(), nn.Linear(128, out_dim)
        )

    def forward(self, x):
        return self.mlp(x)


class WSAD(Module):
    def __init__(self, input_size, flag, a_nums, n_nums):
        super().__init__()
        self.flag = flag
        self.a_nums = a_nums
        self.n_nums = n_nums

        # Core feature extractors
        self.embedding = Temporal(input_size, 512)
        self.localatt = LocalAttention(512)
        self.selfatt = Transformer(512, 2, 4, 128, 512, dropout=0.25)

        # Learnable attention temperature (starting at proven baseline)
        self.attn_temp = nn.Parameter(torch.tensor(0.5))

        self.triplet = nn.TripletMarginLoss(margin=0.3)
        self.cls_head = ADCLS_head(1024, 1)       # 512 (x) + 512 (aug) = 1024

        self.Amemory = Memory_Unit(nums=a_nums, dim=512)
        self.Nmemory = Memory_Unit(nums=n_nums, dim=512)

        self.encoder_mu = nn.Sequential(nn.Linear(512, 512))
        self.encoder_var = nn.Sequential(nn.Linear(512, 512))

    def _reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        return mu + epsilon * std

    def latent_loss(self, mu, var):
        return torch.mean(-0.5 * torch.sum(1 + var - mu ** 2 - var.exp(), dim=1))

    def temporal_smoothness(self, pred):
        if pred.dim() == 2 and pred.size(1) > 1:
            return (pred[:, 1:] - pred[:, :-1]).abs().mean()
        return torch.tensor(0.0, device=pred.device)

    def attn_pool(self, feats, scores):
        temp = torch.clamp(self.attn_temp, min=0.05)
        weights = torch.softmax(scores / temp, dim=1)       # [B, T]
        pooled = (feats * weights.unsqueeze(-1)).sum(dim=1) # [B, D]
        return pooled

    def forward(self, x):
        if len(x.size()) == 4:
            b, n, t, d = x.size()
            x = x.reshape(b * n, t, d)
        else:
            b, t, d = x.size()
            n = 1

        # Feature encoding
        x = self.embedding(x)
        x = self.localatt(x)
        x = self.selfatt(x)

        feat_magnitudes = torch.norm(x, p=2, dim=-1)   # [B, T]

        if self.flag == "Train":
            N_x = x[: b * n // 2]     # normal videos
            A_x = x[b * n // 2 :]     # abnormal videos

            # ---------- Memory reads ----------
            A_att, A_aug = self.Amemory(A_x)
            N_Aatt, N_Aaug = self.Nmemory(A_x)
            A_Natt, A_Naug = self.Amemory(N_x)
            N_att, N_aug = self.Nmemory(N_x)

            # Raw copy BEFORE gating (for sparsity/stubborn losses)
            A_Natt_raw = A_Natt.clone()

            # Reciprocal gating – gently reduce false alarms
            A_Natt = A_Natt * (1.0 - torch.sigmoid(N_att))

            # ---------- Triplet (corrected) ----------
            anchor   = self.attn_pool(N_x, N_att).reshape(b // 2, n, -1).mean(1)
            positive = self.attn_pool(A_x, N_Aatt).reshape(b // 2, n, -1).mean(1)
            negative = self.attn_pool(A_x, A_att).reshape(b // 2, n, -1).mean(1)   # crime, not false‑alarm

            anchor   = norm(anchor)
            positive = norm(positive)
            negative = norm(negative)
            triplet_margin_loss = self.triplet(anchor, positive, negative)

            # ---------- VAE encoding (blended, proven stable) ----------
            N_aug_mu = self.encoder_mu(N_aug)
            N_aug_var = self.encoder_var(N_aug)
            N_aug_new = 0.8 * N_aug_mu + 0.2 * self._reparameterize(N_aug_mu, N_aug_var)
            A_aug_new = self.encoder_mu(A_aug)
            kl_loss = self.latent_loss(N_aug_mu, N_aug_var)

            A_Naug = self.encoder_mu(A_Naug)
            N_Aaug = self.encoder_mu(N_Aaug)

            # ---------- Distance loss (margin 100) ----------
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

            # ---------- Feature fusion ----------
            x = torch.cat((x, torch.cat([N_aug_new + A_Naug,
                                         A_aug_new + N_Aaug], dim=0)), dim=-1)
            pre_att = self.cls_head(x).reshape((b, n, -1)).mean(1)

            # ---------- Feature magnitude loss (optional) ----------
            k_mag = max(t // 16 + 1, 1)
            N_mag = feat_magnitudes[: b * n // 2]
            A_mag = feat_magnitudes[b * n // 2 :]
            topk_A_mag = torch.topk(A_mag, k_mag, dim=-1)[0].mean(-1).mean()
            topk_N_mag = torch.topk(N_mag, k_mag, dim=-1)[0].mean(-1).mean()
            feat_mag_loss = torch.relu(100.0 - (topk_A_mag - topk_N_mag))
            feat_mag_loss = feat_mag_loss + 0.001 * (topk_A_mag ** 2 + topk_N_mag ** 2)

            smooth_loss = self.temporal_smoothness(pre_att)

            # ---------- Orthogonality ----------
            ma = F.normalize(self.Amemory.memory_block, p=2, dim=-1)
            mn = F.normalize(self.Nmemory.memory_block, p=2, dim=-1)
            ortho_loss = torch.mm(ma, mn.t()).pow(2).mean()

            # ---------- Return (raw temporal attentions!) ----------
            return {
                "frame": pre_att,
                "triplet_margin": triplet_margin_loss,
                "kl_loss": kl_loss,
                "distance": distance_loss,
                "smooth_loss": smooth_loss,
                "feat_mag_loss": feat_mag_loss,

                # RAW temporal tensors [b, T] for top‑k BCE + noose loss
                "A_att": A_att,
                "N_att": N_att,
                "A_Natt": A_Natt,           # gated version (for BCE A_Nloss)
                "N_Aatt": N_Aatt,

                "conf_loss": conf_loss,
                "A_N_mse": A_N_mse,
                "A_Natt_raw": A_Natt_raw,   # before gating – for sparsity/stubborn
                "N_att_raw": N_att,
                "ortho_loss": ortho_loss,
            }

        else:
            # Inference path
            _, A_aug = self.Amemory(x)
            _, N_aug = self.Nmemory(x)

            A_aug = self.encoder_mu(A_aug)
            N_aug = self.encoder_mu(N_aug)

            x = torch.cat([x, A_aug + N_aug], dim=-1)
            pre_att = self.cls_head(x).reshape((b, n, -1)).mean(1)
            return {"frame": pre_att}