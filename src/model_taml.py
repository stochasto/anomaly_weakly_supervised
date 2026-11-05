import torch
import torch.nn as nn
from torch.nn.modules.module import Module
from memory_taml import Memory_Unit
#from PATCHED_Memory_Unit import Memory_Unit
from translayer import Transformer
from utils import norm
import torch.nn.functional as F



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
        # [B, T, D] -> [B, D, T] -> conv -> [B, T, D]
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
            #nn.Sigmoid()
        )

    def forward(self, x):
        return self.mlp(x)


# -------------------------
# WSAD MODEL (FINAL)
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
        #self.triplet = nn.TripletMarginLoss(margin=1)
        self.triplet = nn.TripletMarginLoss(margin=0.5)

        # VAE
        self.encoder_mu = nn.Linear(512, 512)
        self.encoder_var = nn.Linear(512, 512)

        # classifier
        self.cls_head = ADCLS_head(1024, 1)

    # -------------------------
    # TAML MEMORY FUNCTIONS
    # -------------------------
    def reset_memory(self):
        self.Amemory.reset()
        self.Nmemory.reset()

    @torch.no_grad()
    def build_memory_from_support(self, support_x):
        """
        Build memory using support set
        """
        #self.eval()

        if support_x.dim() == 4:
            b, n, t, d = support_x.size()
            support_x = support_x.reshape(b * n, t, d)
        else:
            b, t, d = support_x.size()

        x = self.embedding(support_x)
        x = self.localatt(x)
        x = self.selfatt(x)
        # added norm

        # IMPORTANT: balanced split
        half = x.size(0) // 2
        N_x = x[:half]
        A_x = x[half:]

        self.Amemory.write(A_x.detach())
        self.Nmemory.write(N_x.detach())

        #self.train()

    # -------------------------
    # helpers
    # -------------------------
    #def _reparameterize(self, mu, logvar):
    #    std = torch.exp(logvar).sqrt()
    #    eps = torch.randn_like(std)
    #    return mu + eps * std
    
    def _reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)   # numerically stable
        eps = torch.randn_like(std)
        return mu + eps * std

    #def latent_loss(self, mu, var):
    #    return torch.mean(-0.5 * torch.sum(1 + var - mu**2 - var.exp(), dim=1))
    #def latent_loss(self, mu, logvar):
    #    return torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    
    def latent_loss(self, mu, logvar):
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        return kl.mean()

    def temporal_smoothness(self, pred):
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

            # older triplet
            '''_, A_idx = torch.topk(A_att, k, dim=-1)
            _, N_idx = torch.topk(N_att, k, dim=-1)
            _, P_idx = torch.topk(N_Aatt, k, dim=-1)

            # embeddings
            negative = torch.gather(A_x, 1, A_idx.unsqueeze(2).expand(-1, -1, x.size(-1))).mean(1)
            anchor = torch.gather(N_x, 1, N_idx.unsqueeze(2).expand(-1, -1, x.size(-1))).mean(1)
            positive = torch.gather(A_x, 1, P_idx.unsqueeze(2).expand(-1, -1, x.size(-1))).mean(1)
            # Norm triplet
            triplet_loss = self.triplet(norm(anchor), norm(positive), norm(negative))
            # abs triplet
            #triplet_loss = self.triplet(anchor, positive, negative)'''
            
            #New triplet
            # -------------------------
            # HARD TRIPLET SAMPLING
            # -------------------------
            # 1. Hard positive (strong anomaly region)
            '''pos_idx = torch.argmax(A_att, dim=-1)  # [B]
            positive = A_x[torch.arange(A_x.size(0)), pos_idx]
            # 2. Anchor (stable anomaly representation)
            # use mean of top-k to stabilize anchor
            k_anchor = max(t // 16 + 1, 1)
            topk_vals, topk_idx = torch.topk(A_att, k_anchor, dim=-1)
            anchor = torch.gather(A_x,1,topk_idx.unsqueeze(2).expand(-1, -1, x.size(-1))).mean(1)
            # 3. Hard negative (most confusing normal region)
            # use N_Aatt: normal attention when querying anomaly memory
            neg_idx = torch.argmax(N_Aatt, dim=-1)
            negative = N_x[torch.arange(N_x.size(0)), neg_idx]
            # 4. Normalize embeddings (important)
            anchor   = norm(anchor)
            positive = norm(positive)
            negative = norm(negative)
            # 5. Triplet loss
            triplet_loss = self.triplet(anchor, positive, negative)'''
            
            # Soft-attention triplet (differentiable)
            w_pos = F.softmax(A_att, dim=-1).unsqueeze(-1)      # (B, T, 1)
            positive = (A_x * w_pos).sum(1)                      # (B, D)

            # Anchor: still from anomaly memory response to anomaly input
            w_anc = F.softmax(A_att, dim=-1).unsqueeze(-1)
            anchor  = (A_x * w_anc).sum(1)

            # Negative: most confusing normal region (soft)
            w_neg = F.softmax(N_Aatt, dim=-1).unsqueeze(-1)
            negative = (N_x * w_neg).sum(1)

            # Normalize embeddings (your norm function is fine)
            anchor   = norm(anchor)
            positive = norm(positive)
            negative = norm(negative)
            triplet_loss = self.triplet(anchor, positive, negative)


            # VAE
            # -------------------------
            # VAE (clean + consistent)
            # -------------------------

            # --- N branch ---
            N_mu = self.encoder_mu(N_aug)
            N_raw_var = self.encoder_var(N_aug)

            # stable log-variance
            N_logvar = torch.log(1e-6 + torch.nn.functional.softplus(N_raw_var))

            # reparameterization
            N_latent = self._reparameterize(N_mu, N_logvar)

            # --- A branch ---
            A_mu = self.encoder_mu(A_aug)
            A_raw_var = self.encoder_var(A_aug)

            # same parameterization (IMPORTANT: must match N branch)
            A_logvar = torch.log(1e-6 + torch.nn.functional.softplus(A_raw_var))

            # reparameterization
            A_latent = self._reparameterize(A_mu, A_logvar)

            # --- KL (dimension-normalized) ---
            kl_loss = (
                self.latent_loss(N_mu, N_logvar) +
                self.latent_loss(A_mu, A_logvar)
                ) * 0.5
            
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
