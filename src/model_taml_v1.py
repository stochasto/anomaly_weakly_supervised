import torch
import torch.nn as nn
from torch.nn.modules.module import Module

from memory_taml import Memory_Unit
from translayer import Transformer
from utils import norm


# -------------------------
# Basic blocks
# -------------------------
class Temporal(Module):
    def __init__(self, input_size, out_size):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_size, out_size, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
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
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim)
        )

    def forward(self, x):
        return self.mlp(x)


# -------------------------
# WSAD MODEL (FIXED)
# -------------------------
class WSAD(Module):
    def __init__(self, input_size, flag, a_nums, n_nums):
        super().__init__()

        self.flag = flag

        # backbone
        self.embedding = Temporal(input_size, 512)
        self.localatt = LocalAttention(512)
        self.selfatt = Transformer(512, 2, 4, 128, 512, dropout=0.5)

        # prototypes
        self.A_proto = nn.Parameter(torch.randn(1, 512) * 0.1)
        self.N_proto = nn.Parameter(torch.randn(1, 512) * 0.1)

        # memory
        self.Amemory = Memory_Unit(nums=a_nums, dim=512)
        self.Nmemory = Memory_Unit(nums=n_nums, dim=512)

        # losses
        self.triplet = nn.TripletMarginLoss(margin=0.5)

        # VAE
        self.encoder_mu = nn.Linear(512, 512)
        self.encoder_var = nn.Linear(512, 512)

        # classifier
        self.cls_head = ADCLS_head(1024, 1)

        # inner lr
        self.inner_lr = nn.Parameter(torch.tensor(-2.0))

    # -------------------------
    # utilities
    # -------------------------
    def reset_memory(self):
        self.Amemory.reset()
        self.Nmemory.reset()

    def get_inner_lr(self):
        return torch.sigmoid(self.inner_lr) * 0.002

    def _reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def latent_loss(self, mu, logvar):
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        return kl.mean()

    def temporal_smoothness(self, pred):
        if pred.size(1) > 1:
            return (pred[:, 1:] - pred[:, :-1]).abs().mean()
        return torch.tensor(0.0, device=pred.device)
    
    def split_by_label(self, x, y):
        """
        x: (B, T, D) or (B*10, T, D)
        y: (B,) or (B*10,)
        """
        y = y.view(-1)
        N_x = x[y == 0]
        A_x = x[y == 1]
        return N_x, A_x


    # -------------------------
    # MEMORY BUILD (FIXED)
    # -------------------------
    @torch.no_grad()
    def build_memory_from_support(self, support_x, support_y):

        # support_x: (B, 10, T, D)
        b, n, t, d = support_x.shape

        support_x = support_x.view(b * n, t, d)
        support_y = support_y.repeat_interleave(n)

        x = self.embedding(support_x)
        x = self.localatt(x)
        x = self.selfatt(x)
        
        x = torch.nn.functional.normalize(x, dim=-1)  

        N_x, A_x = self.split_by_label(x, support_y)

        if len(N_x) == 0 or len(A_x) == 0:
            return  # safety

        T = x.size(1)

        A_proto = self.A_proto.unsqueeze(1).expand(-1, T, -1)
        N_proto = self.N_proto.unsqueeze(1).expand(-1, T, -1)

        A_mem = torch.cat([A_proto, A_x.detach()], dim=0)
        N_mem = torch.cat([N_proto, N_x.detach()], dim=0)

        A_mem = torch.nn.functional.normalize(A_mem, dim=-1)
        N_mem = torch.nn.functional.normalize(N_mem, dim=-1)

        self.Amemory.write(A_mem)
        self.Nmemory.write(N_mem)
    # -------------------------
    # FORWARD
    # -------------------------
    def forward(self, x, y=None):
        # x: (B, 10, T, D)
        b, n, t, d = x.shape
    
        x = x.view(b * n, t, d)
    
        x = self.embedding(x)
        x = self.localatt(x)
        x = self.selfatt(x)
    
        # -------------------------
        # TRAIN MODE
        # -------------------------
        if self.flag == "Train":
        
            if y is None:
                raise ValueError("labels required for training")
    
            y = y.repeat_interleave(n)
    
            N_x, A_x = self.split_by_label(x, y)
    
            # memory reads
            A_att, A_aug = self.Amemory(A_x)
            N_Aatt, _ = self.Nmemory(A_x)
            _, N_aug = self.Nmemory(N_x)
            A_Natt, _ = self.Amemory(N_x)
    
            # stabilize
            A_aug = A_aug.detach()
            N_aug = N_aug.detach()
    
            # -------------------------
            # HARD TRIPLET (SAFE)
            # -------------------------
            pos_idx = torch.argmax(A_att, dim=-1)
            positive = A_x[torch.arange(A_x.size(0)), pos_idx]
    
            k_anchor = max(t // 16 + 1, 1)
            anchor = torch.topk(A_att, k_anchor, dim=-1)[0].mean(-1)
    
            neg_idx = torch.argmax(N_Aatt, dim=-1)
            negative = N_x[torch.arange(N_x.size(0)), neg_idx]
    
            #anchor = norm(anchor)
            #positive = norm(positive)
            #negative = norm(negative)
            anchor = anchor.unsqueeze(-1) if anchor.dim() == 1 else anchor
            positive = positive if positive.dim() == 2 else positive.unsqueeze(-1)
            negative = negative if negative.dim() == 2 else negative.unsqueeze(-1)
    
            triplet_loss = self.triplet(anchor, positive, negative)
    
            # -------------------------
            # VAE
            # -------------------------
            N_mu = self.encoder_mu(N_aug)
            N_logvar = torch.log(1e-6 + torch.nn.functional.softplus(self.encoder_var(N_aug)))
    
            A_mu = self.encoder_mu(A_aug)
            A_logvar = torch.log(1e-6 + torch.nn.functional.softplus(self.encoder_var(A_aug)))
    
            N_latent = self._reparameterize(N_mu, N_logvar)
            A_latent = self._reparameterize(A_mu, A_logvar)
    
            kl_loss = self.latent_loss(N_mu, N_logvar) + self.latent_loss(A_mu, A_logvar)
    
            feat = torch.cat([x, torch.cat([N_latent, A_latent], dim=0)], dim=-1)
    
            pred = self.cls_head(feat).squeeze(-1)
    
            smooth_loss = self.temporal_smoothness(pred)
    
            return {
                "frame": pred,
                "triplet_margin": triplet_loss,
                "kl_loss": kl_loss,
                "smooth_loss": smooth_loss,
                "A_att": A_att,
                "N_Aatt": N_Aatt,
            }
        else:
            _, A_aug = self.Amemory(x)
            _, N_aug = self.Nmemory(x)

            A_aug = self.encoder_mu(A_aug)
            N_aug = self.encoder_mu(N_aug)

            feat = torch.cat([x, A_aug + N_aug], dim=-1)
            pred = self.cls_head(feat).reshape((b, n, -1)).mean(1)

            return {"frame": pred}

if __name__ == "__main__":
    m = WSAD(1024, "Train", 60, 60).cuda()
    x = torch.rand(32, 32, 1024).cuda()
    y = torch.randint(0, 2, (32,)).cuda()

    out = m(x, y)["frame"]
    print(out.shape)