import torch
import torch.nn as nn
from torch.nn.modules.module import Module

#from memory_taml import Memory_Unit
from claude_memory_taml_10crop import Memory_Unit
from translayer import Transformer


# =====================================================
# BASIC BLOCKS
# =====================================================
class Temporal(Module):
    def __init__(self, input_size, out_size):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_size, out_size, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        # x: (B, T, D)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        return x


class LocalAttention(nn.Module):
    def __init__(self, dim, kernel=9):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(dim, dim, kernel, padding=kernel // 2, groups=dim),
            nn.ReLU(),
            nn.Conv1d(dim, dim, 1)
        )

    def forward(self, x):
        return x + self.conv(x.transpose(1, 2)).transpose(1, 2)


class ADCLS_head(Module):
    def __init__(self, in_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.mlp(x)


# =====================================================
# WSAD MODEL (FINAL CLEAN VERSION)
# =====================================================
class WSAD(Module):
    def __init__(self, input_size, a_nums, n_nums):
        super().__init__()

        # -------------------------
        # BACKBONE
        # -------------------------
        self.embedding = Temporal(input_size, 512)
        self.localatt = LocalAttention(512)
        self.selfatt = Transformer(512, 2, 4, 128, 512, dropout=0.5)

        # -------------------------
        # MEMORY
        # -------------------------
        self.Amemory = Memory_Unit(nums=a_nums, dim=512)
        self.Nmemory = Memory_Unit(nums=n_nums, dim=512)

        # -------------------------
        # LOSSES
        # -------------------------
        self.triplet = nn.TripletMarginLoss(margin=0.5)

        # -------------------------
        # CLASSIFIER
        # -------------------------
        self.cls_head = ADCLS_head(1024)

        self.flag = "Train"

    # =================================================
    # UTILITIES
    # =================================================
    def _prepare(self, x, y=None):
        """
        Handles (B, 10, T, D) → (B*10, T, D)
        """
        if x.dim() == 4:
            b, k, t, d = x.shape
            x = x.view(b * k, t, d)
            if y is not None:
                y = y.repeat_interleave(k)
        return x, y

    def encode(self, x):
        """
        Safe encode for both 3D and 4D inputs
        """
        #if x.dim() == 4:
        #    b, k, t, d = x.shape
        #    x = x.view(b * k, t, d)

        x = self.embedding(x)
        x = self.localatt(x)
        x = self.selfatt(x)

        return x

    def split_by_label(self, x, y):
        y = y.view(-1)
        return x[y == 0], x[y == 1]

    def compute_smooth_loss(self, pred):
        diff = pred[:, 1:] - pred[:, :-1]
        diff = torch.clamp(diff, -10, 10)
        return (diff ** 2).mean()

    # =================================================
    # TEST MEMORY (FIXED)
    # =================================================

    def _build_test_memory(self, x_feat):
        self.Amemory.reset()
        self.Nmemory.reset()

        score = x_feat.mean(dim=-1)  # (K, T)
        thresh = score.mean()

        for i in range(x_feat.size(0)):
            feat_i = x_feat[i]  # (T, D)

            A_mask = score[i] > thresh
            N_mask = score[i] <= thresh

            if A_mask.sum() > 0:
                A_x = feat_i[A_mask].unsqueeze(0)  # (1, Ta, D)
                self.Amemory.write(A_x.detach())

            if N_mask.sum() > 0:
                N_x = feat_i[N_mask].unsqueeze(0)  # (1, Tn, D)
                self.Nmemory.write(N_x.detach())

    # =================================================
    # FORWARD
    # =================================================
    def forward(self, x, y=None):

        # =================================================
        # TEST MODE
        # =================================================
        if self.flag != "Train":

            if x.dim() == 4:
                B, K, T, D = x.shape
                assert B == 1, "Test expects batch size 1"
                x = x.view(K, T, D)

            x_feat = self.encode(x)  # (10, T, D)

            # Build memory from same video
            self._build_test_memory(x_feat)

            A_att, A_aug = self.Amemory(x_feat)
            N_att, N_aug = self.Nmemory(x_feat)

            feat = torch.cat([x_feat, (A_aug + N_aug) / 2], dim=-1)

            pred = self.cls_head(feat).squeeze(-1)  # (10, T)

            # average crops
            pred = pred.mean(dim=0, keepdim=True)  # (1, T)

            return {"frame": pred}

        # =================================================
        # TRAIN MODE
        # =================================================
        x, y = self._prepare(x, y)

        x_feat = self.encode(x)

        # reset per forward (prevents task leakage)
        self.Amemory.reset()
        self.Nmemory.reset()

        N_x, A_x = self.split_by_label(x_feat, y)

        if len(N_x) > 0:
            self.Nmemory.write(N_x.detach())

        if len(A_x) > 0:
            self.Amemory.write(A_x.detach())

        # -------------------------
        # MEMORY READ
        # -------------------------
        A_att, A_aug = self.Amemory(x_feat)
        N_att, N_aug = self.Nmemory(x_feat)

        A_Natt, _ = self.Amemory(N_aug)
        N_Aatt, _ = self.Nmemory(A_aug)

        # -------------------------
        # CLASSIFIER
        # -------------------------
        feat = torch.cat([x_feat, A_aug], dim=-1)
        pred = self.cls_head(feat).squeeze(-1)

        # =================================================
        # LOSSES
        # =================================================
        device = x.device

        triplet_loss = torch.tensor(0.0, device=device)
        distance = torch.tensor(0.0, device=device)

        #if len(N_x) > 0 and len(A_x) > 1:
        if len(N_x) > 0 and len(A_x) > 0:
            # temporal pooling
            A_embed = A_x.mean(dim=1)   # (Na, D)
            N_embed = N_x.mean(dim=1)   # (Nn, D)

            # match sizes
            k = min(A_embed.size(0), N_embed.size(0))

            A_embed = A_embed[:k]
            N_embed = N_embed[:k]

            anchor = A_embed
            #positive = A_embed[torch.randperm(A_embed.size(0))]
            #positive = A_embed.roll(1, dims=0)
            if A_embed.size(0) > 1:
                positive = A_embed.roll(1, dims=0)
            else:
                positive = A_embed
            negative = N_embed

            triplet_loss = self.triplet(anchor, positive, negative)

            distance = torch.norm(anchor.mean(0) - negative.mean(0), p=2)
        else:
            triplet_loss = torch.tensor(0.0, device=x.device)
            distance = torch.tensor(0.0, device=x.device)


        # -------------------------
        # STABLE KL
        # -------------------------
        #eps = 1e-6
        #ratio = (A_att + eps) / (N_att + eps)
        #ratio = torch.clamp(ratio, 0.1, 10.0)
        #kl_loss = torch.mean(A_att * torch.log(ratio))
        
        kl_loss = torch.mean((A_att - N_att) ** 2)

        # -------------------------
        # SMOOTH
        # -------------------------
        smooth_loss = self.compute_smooth_loss(pred)

        return {
            "frame": pred,
            "triplet_margin": triplet_loss,
            "A_att": A_att,
            "N_att": N_att,
            "A_Natt": A_Natt,
            "N_Aatt": N_Aatt,
            "kl_loss": kl_loss,
            "distance": distance,
            "smooth_loss": smooth_loss
        }