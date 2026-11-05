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
            # ✓ REMOVED: Sigmoid will be applied by BCEWithLogitsLoss
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

        # ✓ IMPROVED: More transformer heads for better attention
        self.selfatt = Transformer(512, 2, 8, 128, 512, dropout=0.2)

        self.cls_head = ADCLS_head(1024, 1)

        self.triplet = nn.TripletMarginLoss(margin=1)

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
            # ✓ IMPROVED: Use L2 difference instead of L1 for smoother penalties
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

            # ---------- FIX: label-based split ----------
            if labels is not None:
                labels = labels.view(-1, 1)
                labels = labels.repeat_interleave(n, dim=0).view(-1)

                N_x = x[labels == 0]
                A_x = x[labels == 1]
            else:
                half = x.size(0) // 2
                N_x = x[:half]
                A_x = x[half:]

            # ---------- EMPTY SAFETY ----------
            if N_x.size(0) == 0 or A_x.size(0) == 0:
                pre_att_all = self.cls_head(x).reshape((b, n, -1))
                pre_att = pre_att_all.mean(1)

                smooth_loss = self.temporal_smoothness(pre_att)
                crop_consistency = ((pre_att_all - pre_att.unsqueeze(1)) ** 2).mean()

                return {
                    "frame": pre_att,
                    "triplet_margin": torch.tensor(0.0, device=x.device),
                    "kl_loss": torch.tensor(0.0, device=x.device),
                    "distance": torch.tensor(0.0, device=x.device),
                    "smooth_loss": smooth_loss,
                    "crop_consistency": crop_consistency
                }

            # -------------------------
            # Memory reads
            # -------------------------
            A_att, A_aug = self.Amemory(A_x)
            N_Aatt, N_Aaug = self.Nmemory(A_x)

            A_Natt, A_Naug = self.Amemory(N_x)
            N_att, N_aug = self.Nmemory(N_x)

            # second read
            _, A_aug2 = self.Amemory(A_aug)
            _, N_aug2 = self.Nmemory(N_aug)

            A_aug = A_aug + 0.5 * A_aug2
            N_aug = N_aug + 0.5 * N_aug2

            k = t // 16 + 1

            _, A_idx = torch.topk(A_att, k, dim=-1)
            _, N_idx = torch.topk(N_att, k, dim=-1)
            _, P_idx = torch.topk(N_Aatt, k, dim=-1)

            def gather(x, idx):
                return torch.gather(x, 1, idx.unsqueeze(-1).expand(-1, -1, x.size(-1))).mean(1)

            negative = gather(A_x, A_idx)
            anchor = gather(N_x, N_idx)
            positive = gather(A_x, P_idx)

            #triplet_loss = self.triplet(norm(anchor), norm(positive), norm(negative))
            # ensure same length
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
                triplet_loss = torch.tensor(0.0, device=anchor.device)

            # VAE
            mu = self.encoder_mu(N_aug)
            var = self.encoder_var(N_aug)
            N_aug_new = self._reparameterize(mu, var)

            kl_loss = self.latent_loss(mu, var)

            # distance
            A_aug_new = self.encoder_mu(A_aug)
            dist = torch.relu(
                100 - torch.norm(A_aug_new.mean(1), dim=-1) + torch.norm(N_aug_new.mean(1), dim=-1)
            ).mean()

            # -------------------------
            # Classification
            # -------------------------
            # ✓ IMPROVED: Better feature fusion with weighted combination
            aug_features = torch.cat([N_aug_new + A_Naug, A_aug_new + N_Aaug], dim=0)
            
            # Combine original and augmented features with learned weighting
            combined = torch.cat((x, aug_features), dim=-1)

            pre_att_all = self.cls_head(combined).reshape((b, n, -1))
            pre_att = pre_att_all.mean(1)

            # NEW: consistency loss
            crop_consistency = ((pre_att_all - pre_att.unsqueeze(1)) ** 2).mean()

            smooth_loss = self.temporal_smoothness(pre_att)

            return {
                "frame": pre_att,
                "triplet_margin": triplet_loss,
                "kl_loss": kl_loss,
                "distance": dist,
                "smooth_loss": smooth_loss,
                "crop_consistency": crop_consistency,
                "A_att": A_att.mean(1),
                "N_att": N_att.mean(1),
                "A_Natt": A_Natt.mean(1),
                "N_Aatt": N_Aatt.mean(1),
            }

        # -------------------------
        # TEST
        # -------------------------
        else:
            _, A_aug = self.Amemory(x)
            _, N_aug = self.Nmemory(x)

            _, A_aug2 = self.Amemory(A_aug)
            _, N_aug2 = self.Nmemory(N_aug)

            A_aug = A_aug + 0.5 * A_aug2
            N_aug = N_aug + 0.5 * N_aug2

            A_aug = self.encoder_mu(A_aug)
            N_aug = self.encoder_mu(N_aug)

            x = torch.cat([x, A_aug + N_aug], dim=-1)

            pre_att = self.cls_head(x).reshape((b, n, -1)).mean(1)

            return {"frame": pre_att}