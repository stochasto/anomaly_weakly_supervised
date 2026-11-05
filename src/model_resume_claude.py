import torch
import torch.nn as nn
from torch.nn.modules.module import Module
from memory_resume_claude import Memory_Unit  # Use fixed memory unit
#from PATCHED_Memory_Unit import Memory_Unit
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
        # x: [B, T, D]
        x = x.permute(0, 2, 1)
        x = self.conv_1(x)
        x = x.permute(0, 2, 1)
        return x


class LocalAttention(nn.Module):
    """Lightweight depthwise-local attention implemented via grouped convolutions.
    Preserves shape: input [B, T, D] -> output [B, T, D]
    """
    def __init__(self, dim, kernel=9):
        super().__init__()
        self.conv = nn.Sequential(
            # depthwise (groups=dim) for lightweight local interaction
            nn.Conv1d(dim, dim, kernel_size=kernel, padding=kernel // 2, groups=dim),
            nn.ReLU(),
            nn.Conv1d(dim, dim, kernel_size=1)
        )

    def forward(self, x):
        # x: [B, T, D] -> convert to [B, D, T] for conv1d
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
    def __init__(self, input_size, flag, a_nums, n_nums):
        super().__init__()
        self.flag = flag
        self.a_nums = a_nums
        self.n_nums = n_nums

        # core modules
        self.embedding = Temporal(input_size, 512)

        # NEW: local attention prior to transformer
        self.localatt = LocalAttention(512)

        # ✅ FIXED: Increased triplet margin from 0.3 to 0.5
        # Larger margin pushes positive (same-class abnormal) further from anchor
        # Better separation for anomaly detection
        self.triplet = nn.TripletMarginLoss(margin=0.5)
        
        self.cls_head = ADCLS_head(1024, 1)

        self.Amemory = Memory_Unit(nums=a_nums, dim=512)
        self.Nmemory = Memory_Unit(nums=n_nums, dim=512)

        self.selfatt = Transformer(512, 2, 4, 128, 512, dropout=0.25)

        self.encoder_mu = nn.Sequential(nn.Linear(512, 512))
        self.encoder_var = nn.Sequential(nn.Linear(512, 512))
        self.relu = nn.ReLU()

    def _reparameterize(self, mu, logvar):
        """
        ✅ FIXED: Corrected reparameterization formula
        std = exp(0.5 * logvar) instead of exp(logvar).sqrt()
        Ensures stable variance scaling
        """
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        return mu + epsilon * std

    def latent_loss(self, mu, var):
        """KL divergence loss for VAE regularization"""
        kl_loss = torch.mean(-0.5 * torch.sum(1 + var - mu ** 2 - var.exp(), dim=1))
        return kl_loss

    # NEW: temporal smoothness loss to reduce frame jitter
    def temporal_smoothness(self, pred):
        """
        Encourages smooth predictions across consecutive frames.
        Reduces jitter and improves temporal consistency.
        """
        if pred.dim() == 2 and pred.size(1) > 1:
            return (pred[:, 1:] - pred[:, :-1]).abs().mean()
        return torch.tensor(0.0, device=pred.device)

    def forward(self, x):
        # Accepts either [B, N, T, D] or [B, T, D]
        if len(x.size()) == 4:
            b, n, t, d = x.size()
            x = x.reshape(b * n, t, d)
        else:
            b, t, d = x.size()
            n = 1

        # embedding + local attention + transformer
        x = self.embedding(x)
        x = self.localatt(x)   # NEW: local attention for fine-grained interaction
        x = self.selfatt(x)    # Global transformer attention
        
        # Feature magnitudes (L2 norm per snippet) for RTFM loss
        feat_magnitudes = torch.norm(x, p=2, dim=-1)   # shape [B*n, T]

        if self.flag == "Train":
            # split into normal / abnormal parts (assumes balanced concatenation)
            # Input is balanced: [normal_samples, abnormal_samples]
            N_x = x[: b * n // 2]
            A_x = x[b * n // 2 :]

            # Memory bank reads: match query samples against normal/abnormal prototypes
            A_att, A_aug = self.Amemory(A_x)        # Abnormal queries vs abnormal memory
            N_Aatt, N_Aaug = self.Nmemory(A_x)      # Abnormal queries vs normal memory
            A_Natt, A_Naug = self.Amemory(N_x)      # Normal queries vs abnormal memory
            N_att, N_aug = self.Nmemory(N_x)        # Normal queries vs normal memory

            # ----------  Soft-attention triplet (differentiable)  ----------
            # ✅ FIXED: Use stochastic sampling for normal augmentation
            # Anchor: weighted representation of abnormal video (weighted by A_att)
            w_anc = F.softmax(A_att, dim=-1).unsqueeze(-1)  # [B/2, T, 1]
            anchor = (A_x * w_anc).sum(1)  # [B/2, D]
            
            # Positive: roll anchor by 1 (different abnormal video from same batch)
            positive = torch.roll(anchor, shifts=1, dims=0)  # [B/2, D]
            
            # Negative: normal video representation
            w_neg = F.softmax(N_Aatt, dim=-1).unsqueeze(-1)  # [B/2, T, 1]
            negative = (N_x * w_neg).sum(1)  # [B/2, D]
            
            # Normalize embeddings for metric learning
            anchor   = norm(anchor)
            positive = norm(positive)
            negative = norm(negative)
            
            triplet_margin_loss = self.triplet(anchor, positive, negative)
            # ----------------------------------------------------------------

            # ✅ FIXED: Restore stochastic VAE sampling for normal branch
            # Sample from the posterior distribution instead of just using mean
            N_aug_mu = self.encoder_mu(N_aug)
            N_aug_var = self.encoder_var(N_aug)
            N_aug_new = self._reparameterize(N_aug_mu, N_aug_var)  # ← STOCHASTIC
            
            # Deterministic encoding for abnormal (they're already well-separated)
            A_aug_new = self.encoder_mu(A_aug)

            # KL regularization from VAE
            kl_loss = self.latent_loss(N_aug_mu, N_aug_var)

            # Cross-augmentations (abnormal queries augmented by normal memory, etc.)
            A_Naug = self.encoder_mu(A_Naug)
            N_Aaug = self.encoder_mu(N_Aaug)

            # Classifier input: concat original x + memory-augmented features
            x = torch.cat((x, (torch.cat([N_aug_new + A_Naug,
                                          A_aug_new + N_Aaug], dim=0))), dim=-1)
            pre_att = self.cls_head(x).reshape((b, n, -1)).mean(1)

            # Temporal smoothness loss (encourages smooth frame-level predictions)
            smooth_loss = self.temporal_smoothness(pre_att)
            
            # ---- Feature Magnitude Loss (RTFM-style) ----
            # Encourages abnormal videos to have higher magnitudes than normal
            k_mag = max(t // 32 + 1, 1)  # ✅ FIXED: More aggressive selection (t//32 instead of t//16)
            N_mag = feat_magnitudes[: b * n // 2]    # normal videos
            A_mag = feat_magnitudes[b * n // 2 :]    # abnormal videos
            topk_A_mag = torch.topk(A_mag, k_mag, dim=-1)[0].mean(-1).mean()
            topk_N_mag = torch.topk(N_mag, k_mag, dim=-1)[0].mean(-1).mean()
            margin = 0.3
            feat_mag_loss = torch.relu(margin - (topk_A_mag - topk_N_mag))
            
            # Prevent feature norms from growing without bound
            mag_reg = 0.001 * (topk_A_mag ** 2 + topk_N_mag ** 2)
            feat_mag_loss = feat_mag_loss + mag_reg
            # -------------------------------------------

            return {
                "frame": pre_att,
                "triplet_margin": triplet_margin_loss,
                "kl_loss": kl_loss,
                "distance": torch.tensor(0.0, device=x.device),   # not used
                "smooth_loss": smooth_loss,
                "feat_mag_loss": feat_mag_loss,
                "A_att": A_att.reshape((b // 2, n, -1)).mean(1),
                "N_att": N_att.reshape((b // 2, n, -1)).mean(1),
                "A_Natt": A_Natt.reshape((b // 2, n, -1)).mean(1),
                "N_Aatt": N_Aatt.reshape((b // 2, n, -1)).mean(1),
            }
        else:
            # Inference path
            _, A_aug = self.Amemory(x)
            _, N_aug = self.Nmemory(x)

            A_aug = self.encoder_mu(A_aug)
            N_aug = self.encoder_mu(N_aug)

            x = torch.cat([x, A_aug + N_aug], dim=-1)
            pre_att = self.cls_head(x)
            pre_att = pre_att.reshape((b, n, -1)).mean(1)
            return {"frame": pre_att}


if __name__ == "__main__":
    # Quick smoke test
    m = WSAD(input_size=1024, flag="Train", a_nums=60, n_nums=60).cuda()
    src = torch.rand(100, 32, 1024).cuda()
    out = m(src)["frame"]
    print(f"✓ Output shape: {out.size()}")
    print(f"✓ Model initialized correctly!")
