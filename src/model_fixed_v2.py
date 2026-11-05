import torch
import torch.nn as nn
from torch.nn.modules.module import Module
from memory import Memory_Unit
from translayer import Transformer
from utils import norm


# -------------------------
# Temporal Conv
# -------------------------
class Temporal(Module):
    def __init__(self, input_size, out_size):
        super().__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv1d(input_size, out_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_size),
            nn.ReLU(),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv_1(x)
        x = x.permute(0, 2, 1)
        return x


# -------------------------
# Local Attention
# -------------------------
class LocalAttention(nn.Module):
    def __init__(self, dim, kernel=9):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=kernel, padding=kernel // 2, groups=dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Conv1d(dim, dim, kernel_size=1),
            nn.BatchNorm1d(dim)
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        return x + self.dropout(self.conv(x.transpose(1, 2))).transpose(1, 2)


# -------------------------
# Classifier
# -------------------------
class ADCLS_head(Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, out_dim)
        )

    def forward(self, x):
        return self.mlp(x)


# -------------------------
# WSAD Model
# -------------------------
class WSAD(Module):
    def __init__(self, input_size, flag, a_nums, n_nums):
        super().__init__()

        self.flag = flag
        self.a_nums = a_nums
        self.n_nums = n_nums

        self.embedding = Temporal(input_size, 512)
        self.localatt = LocalAttention(512)
        self.selfatt = Transformer(512, 2, 8, 128, 512, dropout=0.2)
        self.cls_head = ADCLS_head(1024, 1)
        self.triplet = nn.TripletMarginLoss(margin=1.0, p=2)

        self.Amemory = Memory_Unit(nums=a_nums, dim=512)
        self.Nmemory = Memory_Unit(nums=n_nums, dim=512)

        self.encoder_mu = nn.Linear(512, 512)
        self.encoder_var = nn.Linear(512, 512)

    # -------------------------
    # VAE helpers
    # -------------------------
    def _reparameterize(self, mu, logvar):
        std = torch.exp(logvar).sqrt()
        eps = torch.randn_like(std)
        return mu + eps * std

    def latent_loss(self, mu, var):
        return torch.mean(-0.5 * torch.sum(1 + var - mu**2 - var.exp(), dim=1))

    def temporal_smoothness(self, pred):
        if pred.size(1) > 1:
            diff = pred[:, 1:] - pred[:, :-1]
            return (diff ** 2).mean()
        return torch.tensor(0.0, device=pred.device)

    # -------------------------
    # Forward
    # -------------------------
    def forward(self, x, labels=None):

        if len(x.size()) == 4:
            b, n, t, d = x.size()
            x = x.reshape(b * n, t, d)
        else:
            b, t, d = x.size()
            n = 1

        x = self.embedding(x)
        x = self.localatt(x)
        x = self.selfatt(x)

        # -------------------------
        # TRAIN
        # -------------------------
        if self.flag == "Train":

            if labels is not None:
                if labels.dim() == 1:
                    labels = labels.view(-1)
                    if n > 1:
                        labels = labels.repeat_interleave(n)
                
                anomaly_mask = labels == 1
                normal_mask = labels == 0
                
                A_x = x[anomaly_mask]
                N_x = x[normal_mask]
            else:
                half = x.size(0) // 2
                N_x = x[:half]
                A_x = x[half:]

            # Safety: handle empty batches
            if N_x.size(0) == 0 or A_x.size(0) == 0:
                pre_att_all = self.cls_head(torch.cat([x, x], dim=-1)).reshape((b, n, -1))
                pre_att = pre_att_all.mean(1).squeeze(-1)

                return {
                    "frame": pre_att,
                    "triplet_margin": torch.tensor(0.0, device=x.device),
                    "kl_loss": torch.tensor(0.0, device=x.device),
                    "distance": torch.tensor(0.0, device=x.device),
                    "smooth_loss": torch.tensor(0.0, device=x.device),
                }

            # ✓ FIX 1: CORRECT MEMORY READS
            # Anomaly samples should read ANOMALY memory strongly
            A_att, A_aug = self.Amemory(A_x)        # ✓ CORRECT: A reads A
            N_Aatt, N_Aaug = self.Nmemory(A_x)      # ✓ CORRECT: A reads N (for contrast)

            A_Natt, A_Naug = self.Amemory(N_x)      # ✓ CORRECT: N reads A (for contrast)
            N_att, N_aug = self.Nmemory(N_x)        # ✓ CORRECT: N reads N

            # Second read for refinement
            _, A_aug2 = self.Amemory(A_aug)
            _, N_aug2 = self.Nmemory(N_aug)

            A_aug = A_aug + 0.5 * A_aug2
            N_aug = N_aug + 0.5 * N_aug2

            # Top-k selection
            k_A = max(1, min(A_x.size(1) // 16, A_x.size(1)))
            k_N = max(1, min(N_x.size(1) // 16, N_x.size(1)))

            # ✓ FIX 2: CORRECT TRIPLET SELECTION
            # For triplet loss: anchor=normal, positive=normal-like anomaly, negative=anomaly
            _, A_idx = torch.topk(A_att, k_A, dim=-1)     # Most anomalous frames in anomaly videos
            _, N_idx = torch.topk(N_att, k_N, dim=-1)     # Most normal frames in normal videos  
            _, P_idx = torch.topk(N_Aatt, k_A, dim=-1)    # Most normal-like frames in anomaly videos

            def gather(x, idx):
                if x.size(0) == 0 or idx.size(0) == 0:
                    return torch.zeros(1, x.size(-1), device=x.device)
                return torch.gather(x, 1, idx.unsqueeze(-1).expand(-1, -1, x.size(-1))).mean(1)

            # ✓ CRITICAL: Correct triplet assignment
            # Anchor: normal samples (should be close to positive, far from negative)
            # Positive: normal-like anomaly frames (should be close to anchor)
            # Negative: clearly anomalous frames (should be far from anchor)
            anchor = gather(N_x, N_idx)       # Normal videos, normal frames
            positive = gather(A_x, P_idx)     # Anomaly videos, normal-like frames
            negative = gather(A_x, A_idx)     # Anomaly videos, anomalous frames

            min_len = min(anchor.size(0), positive.size(0), negative.size(0))
            
            if min_len > 0:
                anchor = anchor[:min_len]
                positive = positive[:min_len]
                negative = negative[:min_len]
                
                triplet_loss = self.triplet(
                    norm(anchor),
                    norm(positive),
                    norm(negative)
                )
            else:
                triplet_loss = torch.tensor(0.0, device=x.device)

            # VAE for normal samples
            mu = self.encoder_mu(N_aug)
            var = self.encoder_var(N_aug)
            N_aug_new = self._reparameterize(mu, var)

            kl_loss = self.latent_loss(mu, var)

            # Distance: anomaly features should be far from normal
            A_aug_new = self.encoder_mu(A_aug)
            dist = torch.relu(
                100 - torch.norm(A_aug_new.mean(0), dim=-1) + torch.norm(N_aug_new.mean(0), dim=-1)
            )

            # ✓ FIX 3: CORRECT FEATURE COMBINATION
            # Combine: [original features | memory-augmented features]
            # For anomaly samples: combine with normal-augmented (N_Aaug)
            # For normal samples: combine with anomaly-augmented (A_Naug)
            
            # Stack: [N_x with A_Naug, A_x with N_Aaug]
            N_combined = torch.cat([N_x, N_aug_new + A_Naug], dim=-1)
            A_combined = torch.cat([A_x, A_aug_new + N_Aaug], dim=-1)
            
            combined = torch.cat([N_combined, A_combined], dim=0)

            # Classification
            pre_att_all = self.cls_head(combined).reshape((b, n, -1))
            pre_att = pre_att_all.mean(1).squeeze(-1)

            smooth_loss = self.temporal_smoothness(pre_att)

            return {
                "frame": pre_att,
                "triplet_margin": triplet_loss,
                "kl_loss": kl_loss,
                "distance": dist,
                "smooth_loss": smooth_loss,
            }

        # -------------------------
        # TEST
        # -------------------------
        else:
            # Read from both memories
            _, A_aug = self.Amemory(x)
            _, N_aug = self.Nmemory(x)

            # Refinement
            _, A_aug2 = self.Amemory(A_aug)
            _, N_aug2 = self.Nmemory(N_aug)

            A_aug = A_aug + 0.5 * A_aug2
            N_aug = N_aug + 0.5 * N_aug2

            # Encode
            A_aug = self.encoder_mu(A_aug)
            N_aug = self.encoder_mu(N_aug)

            # ✓ FIX 4: CORRECT TEST FEATURE COMBINATION
            # Combine original features with memory augmentation
            x_combined = torch.cat([x, A_aug + N_aug], dim=-1)

            pre_att = self.cls_head(x_combined).reshape((b, n, -1)).mean(1).squeeze(-1)

            return {"frame": pre_att}
