import torch
import torch.nn as nn
from torch.nn.modules.module import Module
from memory_ref_resume import Memory_Unit
#from PATCHED_Memory_Unit import Memory_Unit
from translayer_ref import Transformer
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
        # keep original head (sigmoid) to avoid changing score scaling; simple to swap to softmax if desired
        #self.mlp = nn.Sequential(nn.Linear(in_dim, 128), nn.ReLU(), nn.Linear(128, out_dim), nn.Sigmoid())
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

        self.triplet = nn.TripletMarginLoss(margin=0.5)
        #self.triplet = nn.TripletMarginLoss(margin=1)
        self.cls_head = ADCLS_head(1024, 1)

        self.Amemory = Memory_Unit(nums=a_nums, dim=512)
        self.Nmemory = Memory_Unit(nums=n_nums, dim=512)

        self.selfatt = Transformer(512, 2, 4, 128, 512, dropout=0.25)

        self.encoder_mu = nn.Sequential(nn.Linear(512, 512))
        self.encoder_var = nn.Sequential(nn.Linear(512, 512))
        self.relu = nn.ReLU()

    #def _reparameterize(self, mu, logvar):
    #    std = torch.exp(logvar).sqrt()
    #    epsilon = torch.randn_like(std)
    #    return mu + epsilon * std
    
    def _reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        return mu + epsilon * std

    def latent_loss(self, mu, var):
        kl_loss = torch.mean(-0.5 * torch.sum(1 + var - mu ** 2 - var.exp(), dim=1))
        return kl_loss

    # NEW: temporal smoothness loss to reduce frame jitter
    def temporal_smoothness(self, pred):
        # pred: [B, T] or [B, 1] handled outside
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
        x = self.localatt(x)   # NEW
        x = self.selfatt(x)
        
        # Feature magnitudes (L2 norm per snippet) for RTFM loss
        feat_magnitudes = torch.norm(x, p=2, dim=-1)   # shape [B*n, T]

        #if self.flag == "Train":
        if self.flag == "Train":
            # split into normal / abnormal parts (assumes balanced concatenation)
            N_x = x[: b * n // 2]
            A_x = x[b * n // 2 :]

            # First memory read
            A_att, A_aug = self.Amemory(A_x)
            N_Aatt, N_Aaug = self.Nmemory(A_x)
            A_Natt, A_Naug = self.Amemory(N_x)
            N_att, N_aug = self.Nmemory(N_x)

            # Second memory read (augment features)
            _, A_aug2 = self.Amemory(A_aug)
            _, N_aug2 = self.Nmemory(N_aug)
            A_aug = A_aug + 0.1 * A_aug2
            N_aug = N_aug + 0.1 * N_aug2

            # ----------  Soft-attention triplet (differentiable)  ----------
            # ----------  Correct triplet (anchor ≠ positive) ----------
            w_anc = F.softmax(A_att, dim=-1).unsqueeze(-1)
            anchor = (A_x * w_anc).sum(1)
            
            # positive = representation of a different abnormal video
            positive = torch.roll(anchor, shifts=1, dims=0)
            
            w_neg = F.softmax(N_Aatt, dim=-1).unsqueeze(-1)
            negative = (N_x * w_neg).sum(1)
            
            anchor   = norm(anchor)
            positive = norm(positive)
            negative = norm(negative)
            
            triplet_margin_loss = self.triplet(anchor, positive, negative)
# ------------------------------------------------------------
            # ----------------------------------------------------------------

            # VAE-style encoding + reparam (normal branch only)
            N_aug_mu = self.encoder_mu(N_aug)
            N_aug_var = self.encoder_var(N_aug)
            N_aug_new = self._reparameterize(N_aug_mu, N_aug_var)

            A_aug_new = self.encoder_mu(A_aug)      # no reparam for abnormal

            kl_loss = self.latent_loss(N_aug_mu, N_aug_var)

            A_Naug = self.encoder_mu(A_Naug)
            N_Aaug = self.encoder_mu(N_Aaug)

            # Classifier input: concat original x + memory‑augmented features
            x = torch.cat((x, (torch.cat([N_aug_new + A_Naug,
                                          A_aug_new + N_Aaug], dim=0))), dim=-1)
            pre_att = self.cls_head(x).reshape((b, n, -1)).mean(1)

            # Temporal smoothness (optional – can be removed if you don't use it)
            smooth_loss = self.temporal_smoothness(pre_att)
            
            # ---- Feature Magnitude Loss (RTFM-style) ----
            k_mag = max(t // 16 + 1, 1)
            N_mag = feat_magnitudes[: b * n // 2]    # normal videos
            A_mag = feat_magnitudes[b * n // 2 :]    # abnormal videos
            topk_A_mag = torch.topk(A_mag, k_mag, dim=-1)[0].mean(-1).mean()
            topk_N_mag = torch.topk(N_mag, k_mag, dim=-1)[0].mean(-1).mean()
            margin = 1.0
            feat_mag_loss = torch.relu(margin - (topk_A_mag - topk_N_mag))
            
            # Prevent feature norms from growing without bound (stabilises VAE)
            mag_reg = 0.001 * (topk_A_mag ** 2 + topk_N_mag ** 2)
            feat_mag_loss = feat_mag_loss + mag_reg
            # -------------------------------------------

            return {
                "frame": pre_att,
                "triplet_margin": triplet_margin_loss,
                "kl_loss": kl_loss,
                "distance": torch.tensor(0.0, device=x.device),   # not used anymore
                "smooth_loss": smooth_loss,
                "feat_mag_loss": feat_mag_loss,
                "A_att": A_att.reshape((b // 2, n, -1)).mean(1),
                "N_att": N_att.reshape((b // 2, n, -1)).mean(1),
                "A_Natt": A_Natt.reshape((b // 2, n, -1)).mean(1),
                "N_Aatt": N_Aatt.reshape((b // 2, n, -1)).mean(1),
            }
        else:
            # Inference path: keep second read to reinforce test-time augmentations
            _, A_aug = self.Amemory(x)
            _, N_aug = self.Nmemory(x)

            _, A_aug2 = self.Amemory(A_aug)
            _, N_aug2 = self.Nmemory(N_aug)
            A_aug = A_aug + 0.1 * A_aug2
            N_aug = N_aug + 0.1 * N_aug2

            A_aug = self.encoder_mu(A_aug)
            N_aug = self.encoder_mu(N_aug)

            x = torch.cat([x, A_aug + N_aug], dim=-1)

            pre_att = self.cls_head(x).reshape((b, n, -1)).mean(1)
            return {"frame": pre_att}


if __name__ == "__main__":
    # quick smoke-test
    m = WSAD(input_size=1024, flag="Train", a_nums=60, n_nums=60).cuda()
    src = torch.rand(100, 32, 1024).cuda()
    out = m(src)["frame"]
    print(out.size())
