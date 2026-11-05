import torch
import torch.nn as nn
from torch.nn.modules.module import Module
from memory_taml import Memory_Unit
from translayer import Transformer
from utils import norm


class Temporal(Module):
    def __init__(self, input_size, out_size):
        super().__init__()
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
    """
    🔴 FIXED: Removed Sigmoid from head
    
    Reason: Loss function uses BCEWithLogitsLoss which applies Sigmoid internally.
    Having Sigmoid here causes double-sigmoid (sigmoid applied twice), breaking gradients.
    
    Change: outputs raw logits → BCEWithLogitsLoss → single Sigmoid applied
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim)
            # NO Sigmoid - BCEWithLogitsLoss handles sigmoid internally
        )

    def forward(self, x):
        return self.mlp(x)


# -------------------------
# WSAD MODEL (FINAL - FIXED)
# -------------------------
class WSAD(Module):
    def __init__(self, input_size, flag, a_nums, n_nums):
        super().__init__()

        self.flag = flag

        # backbone
        self.embedding = Temporal(input_size, 512)
        self.localatt = LocalAttention(512)
        self.selfatt = Transformer(512, 2, 4, 128, 512, dropout=0.5)

        # memory
        self.Amemory = Memory_Unit(nums=a_nums, dim=512)
        self.Nmemory = Memory_Unit(nums=n_nums, dim=512)

        # losses
        # 🔴 FIXED: Increased triplet margin for better separation
        self.triplet = nn.TripletMarginLoss(margin=0.5)  # was 0.3, now 0.5

        # VAE
        self.encoder_mu = nn.Linear(512, 512)
        self.encoder_var = nn.Linear(512, 512)

        # classifier
        self.cls_head = ADCLS_head(1024, 1)

    # -------------------------
    # TAML MEMORY FUNCTIONS
    # -------------------------
    def reset_memory(self):
        """Reset episodic memory for new task"""
        self.Amemory.reset()
        self.Nmemory.reset()

    @torch.no_grad()
    def build_memory_from_support(self, support_x):
        """
        Build memory using support set
        
        Args:
            support_x: [B, T, D] or [B, N, T, D]
        """

        if support_x.dim() == 4:
            b, n, t, d = support_x.size()
            support_x = support_x.reshape(b * n, t, d)
        else:
            b, t, d = support_x.size()

        x = self.embedding(support_x)
        x = self.localatt(x)
        x = self.selfatt(x)

        # IMPORTANT: balanced split
        half = x.size(0) // 2
        N_x = x[:half]
        A_x = x[half:]

        self.Amemory.write(A_x.detach())
        self.Nmemory.write(N_x.detach())

    # -------------------------
    # helpers
    # -------------------------
    def _reparameterize(self, mu, logvar):
        """
        Reparameterization trick for VAE sampling
        
        Args:
            mu: mean of latent distribution
            logvar: log-variance (NOT variance)
        """
        std = torch.exp(0.5 * logvar)   # numerically stable
        eps = torch.randn_like(std)
        return mu + eps * std

    def latent_loss(self, mu, logvar):
        """
        KL divergence loss for VAE
        
        KL(N(mu, sigma^2) || N(0, I)) = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        
        Args:
            mu: latent mean [B, D]
            logvar: latent log-variance [B, D]
            
        Returns:
            scalar KL loss
        """
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        return kl.mean()

    def temporal_smoothness(self, pred):
        """
        Encourage smooth predictions over time
        
        Args:
            pred: [B, T]
            
        Returns:
            scalar smoothness loss
        """
        if pred.size(1) > 1:
            return (pred[:, 1:] - pred[:, :-1]).abs().mean()
        return torch.tensor(0.0, device=pred.device)

    # -------------------------
    # FORWARD
    # -------------------------
    def forward(self, x):

        if x.dim() == 4:
            b, n, t, d = x.size()
            x = x.reshape(b * n, t, d)
        else:
            b, t, d = x.size()
            n = 1

        # backbone
        x = self.embedding(x)
        x = self.localatt(x)
        x = self.selfatt(x)

        # -------------------------
        # TRAIN MODE
        # -------------------------
        if self.flag == "Train":

            half = x.size(0) // 2
            N_x = x[:half]
            A_x = x[half:]

            # memory read
            A_att, A_aug = self.Amemory(A_x)
            N_Aatt, N_Aaug = self.Nmemory(A_x)
            A_Natt, A_Naug = self.Amemory(N_x)
            N_att, N_aug = self.Nmemory(N_x)

            # second read (stable version)
            _, A_aug2 = self.Amemory(A_aug.detach())
            _, N_aug2 = self.Nmemory(N_aug.detach())

            A_aug = A_aug + 0.1 * A_aug2
            N_aug = N_aug + 0.1 * N_aug2

            # indices
            k = t // 16 + 1

            # -------------------------
            # 🔴 FIXED: HARD TRIPLET SAMPLING (improved clarity)
            # -------------------------
            # Triplet loss: minimize d(anchor, positive) and maximize d(anchor, negative)
            
            # 1. Positive: hardest anomaly frame (highest anomaly attention)
            pos_idx = torch.argmax(A_att, dim=-1)  # [B]
            positive = A_x[torch.arange(A_x.size(0)), pos_idx]  # [B, D]
            
            # 2. Anchor: stable anomaly representative (mean of top-k)
            k_anchor = max(t // 16 + 1, 1)
            topk_vals, topk_idx = torch.topk(A_att, k_anchor, dim=-1)  # [B, k]
            anchor = torch.gather(
                A_x, 1, 
                topk_idx.unsqueeze(2).expand(-1, -1, A_x.size(-1))
            ).mean(1)  # [B, D]
            
            # 3. Negative: hardest confusing normal frame
            #    (normal sample that most activates anomaly memory)
            neg_idx = torch.argmax(N_Aatt, dim=-1)  # [B]
            negative = N_x[torch.arange(N_x.size(0)), neg_idx]  # [B, D]
            
            # 4. Normalize embeddings for L2 distance
            anchor   = norm(anchor)
            positive = norm(positive)
            negative = norm(negative)
            
            # 5. Triplet margin loss
            triplet_loss = self.triplet(anchor, positive, negative)

            # -------------------------
            # 🔴 FIXED: VAE (both branches with proper reparameterization)
            # -------------------------

            # --- Normal branch (N_aug) ---
            N_mu = self.encoder_mu(N_aug)
            N_raw_var = self.encoder_var(N_aug)
            
            # Stable log-variance: apply softplus to ensure positive
            N_logvar = torch.log(1e-6 + torch.nn.functional.softplus(N_raw_var))
            
            # Reparameterization
            N_latent = self._reparameterize(N_mu, N_logvar)

            # --- Anomaly branch (A_aug) ---
            A_mu = self.encoder_mu(A_aug)
            A_raw_var = self.encoder_var(A_aug)
            
            # Same parameterization (CRITICAL for consistency)
            A_logvar = torch.log(1e-6 + torch.nn.functional.softplus(A_raw_var))
            
            # Reparameterization
            A_latent = self._reparameterize(A_mu, A_logvar)

            # --- KL loss (both branches) ---
            # Dimension-normalized to prevent KL from collapsing
            kl_loss = (
                self.latent_loss(N_mu, N_logvar) +
                self.latent_loss(A_mu, A_logvar)
            ) * 0.5  # Average both branches
            
            # classifier
            feat = torch.cat([x, torch.cat([N_latent, A_latent], dim=0)], dim=-1)
            pred = self.cls_head(feat).reshape((b, n, -1)).mean(1)

            smooth_loss = self.temporal_smoothness(pred)

            return {
                "frame": pred,
                "triplet_margin": triplet_loss,
                "kl_loss": kl_loss,
                "distance": torch.tensor(0.0, device=pred.device),
                "smooth_loss": smooth_loss,
                "A_att": A_att,
                "N_att": N_att,
                "A_Natt": A_Natt,
                "N_Aatt": N_Aatt,
            }

        # -------------------------
        # TEST MODE
        # -------------------------
        else:
            _, A_aug = self.Amemory(x)
            _, N_aug = self.Nmemory(x)

            A_aug = self.encoder_mu(A_aug)
            N_aug = self.encoder_mu(N_aug)

            feat = torch.cat([x, A_aug + N_aug], dim=-1)
            pred = self.cls_head(feat).reshape((b, n, -1)).mean(1)

            return {"frame": pred}


if __name__ == "__main__":
    # quick smoke-test
    m = WSAD(input_size=1024, flag="Train", a_nums=60, n_nums=60).cuda()
    src = torch.rand(100, 32, 1024).cuda()
    out = m(src)["frame"]
    print(out.size())
