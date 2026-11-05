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
    """
    Classification head for anomaly detection
    Outputs raw logits (no sigmoid) for BCEWithLogitsLoss
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim)
            # NO sigmoid - BCEWithLogitsLoss applies it internally
        )

    def forward(self, x):
        return self.mlp(x)


# -------------------------
# WSAD MODEL (10-CROP FIXED)
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
        self.triplet = nn.TripletMarginLoss(margin=0.5)

        # VAE
        self.encoder_mu = nn.Linear(512, 512)
        self.encoder_var = nn.Linear(512, 512)

        # classifier
        self.cls_head = ADCLS_head(1024, 1)

        # learnable inner LR
        self.inner_lr = nn.Parameter(torch.tensor(-2.0))

    # -------------------------
    # utilities
    # -------------------------
    def reset_memory(self):
        """Reset episodic memory for new task"""
        self.Amemory.reset()
        self.Nmemory.reset()

    def get_inner_lr(self):
        """Get learnable inner loop learning rate with sigmoid gating"""
        return torch.sigmoid(self.inner_lr) * 0.002

    def _reparameterize(self, mu, logvar):
        """Reparameterization trick for VAE"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def latent_loss(self, mu, logvar):
        """KL divergence loss for VAE"""
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        return kl.mean()

    def temporal_smoothness(self, pred):
        """Temporal smoothness regularization"""
        if pred.size(1) > 1:
            return (pred[:, 1:] - pred[:, :-1]).abs().mean()
        return torch.tensor(0.0, device=pred.device)
    
    def split_by_label(self, x, y):
        """
        Split features by label (normal=0, abnormal=1)
        
        Args:
            x: (B*n, T, D) flattened features
            y: (B*n,) flattened labels
        """
        y = y.view(-1)
        N_x = x[y == 0]
        A_x = x[y == 1]
        return N_x, A_x

    # -------------------------
    # MEMORY BUILD (FIXED FOR 10-CROP)
    # -------------------------
    @torch.no_grad()
    def build_memory_from_support(self, support_x, support_y):
        """
        Build episodic memory from support set
        
        ✅ FIX: Properly handle 10-crop format
        
        Args:
            support_x: (B, 10, T, D) - 10 crops of B videos
            support_y: (B,) - labels for B videos
        """
        
        # Reshape to (B*10, T, D) to process all crops
        b, n, t, d = support_x.shape
        support_x_flat = support_x.view(b * n, t, d)
        
        # Expand labels to match flattened crops
        support_y_flat = support_y.repeat_interleave(n)
        
        # Backbone forward
        x = self.embedding(support_x_flat)
        x = self.localatt(x)
        x = self.selfatt(x)
        
        # ✅ FIX: Don't normalize here (normalize in memory.write)
        # x = torch.nn.functional.normalize(x, dim=-1)
        
        # Split by label
        N_x, A_x = self.split_by_label(x, support_y_flat)
        
        # Write to memory (write() does normalization)
        if len(N_x) > 0:
            self.Nmemory.write(N_x.detach())
        if len(A_x) > 0:
            self.Amemory.write(A_x.detach())

    # -------------------------
    # FORWARD
    # -------------------------
    def forward(self, x, y=None):
        """
        Forward pass with 10-crop support
        
        Args:
            x: (B, 10, T, D) or (B*10, T, D)
            y: (B,) labels (required in train, optional in test)
        """
        
        # ✅ FIX: Properly detect and handle 10-crop format
        if x.dim() == 4:
            b, n, t, d = x.shape
            is_crops = True
            x_flat = x.view(b * n, t, d)
        else:
            x_flat = x
            b, t, d = x_flat.shape
            is_crops = False
            n = 1
        
        # Backbone forward
        x_feat = self.embedding(x_flat)
        x_feat = self.localatt(x_feat)
        x_feat = self.selfatt(x_feat)
        
        # -------------------------
        # TRAIN MODE
        # -------------------------
        if self.flag == "Train":
            
            if y is None:
                raise ValueError("Training requires labels (y)")
            
            # ✅ FIX: Expand labels to match flattened crops
            if is_crops:
                y_flat = y.repeat_interleave(n)
            else:
                y_flat = y.view(-1)
            
            # Split by label
            N_x, A_x = self.split_by_label(x_feat, y_flat)
            
            # Safety check
            if len(N_x) == 0 or len(A_x) == 0:
                raise RuntimeError(f"Empty split: N={len(N_x)}, A={len(A_x)}")
            
            # Memory reads
            A_att, A_aug = self.Amemory(A_x)  # [M, T], [M, T, 512]
            N_Aatt, _ = self.Nmemory(A_x)     # [M, T]
            _, N_aug = self.Nmemory(N_x)      # [N, T, 512]
            A_Natt, _ = self.Amemory(N_x)     # [N, T]
            
            # Detach augmentations for stability
            A_aug = A_aug.detach()
            N_aug = N_aug.detach()
            
            # ✅ FIX: Proper hard triplet sampling with correct shapes
            # Get positive: peak anomaly frame
            pos_idx = torch.argmax(A_att, dim=-1)  # [M]
            positive = A_x[torch.arange(A_x.size(0)), pos_idx]  # [M, 512]
            
            # Get anchor: mean of top-k anomaly frames
            k_anchor = max(t // 16 + 1, 1)
            ## Old anchor block
            #topk_vals, topk_idx = torch.topk(A_att, k_anchor, dim=-1)  # [M, k]
            
            ## Gather top-k features and average
            #anchor_gathered = torch.gather(
            #    A_x.unsqueeze(1).expand(-1, A_att.size(1), -1),
            #    1,
            #    topk_idx.unsqueeze(2).expand(-1, -1, A_x.size(-1))
            #)  # [M, k, 512]
            #anchor = anchor_gathered.mean(dim=1)  # [M, 512]
            ## Old block
            
            ## New anchor block ###############################################3V
            # # top-k indices over time dimension
            topk_vals, topk_idx = torch.topk(A_att, k_anchor, dim=1)  # [M, k]
            # gather features along temporal dimension
            anchor_gathered = torch.gather(A_x,  # [M, T, 512]
                                           dim=1,
                                           index=topk_idx.unsqueeze(-1).expand(-1, -1, A_x.size(-1))  # [M, k, 512]
                                           )
            anchor = anchor_gathered.mean(dim=1)  # [M, 512]
            ## New anchor block ###############################################3V
            
            # Get negative: peak false-positive (normal that triggers anomaly memory)
            # ✅ FIX: Resample negative to match anchor batch size
            neg_idx = torch.argmax(N_Aatt, dim=-1)  # [M]
            
            # Ensure N_x has enough samples
            if len(N_x) < len(A_x):
                # Oversample negatives to match anchor batch
                sample_idx = torch.randint(0, len(N_x), (len(A_x),), device=N_x.device)
                negative = N_x[sample_idx]  # [M, 512]
            else:
                # Use first len(anchor) negatives
                negative = N_x[:len(A_x)]  # [M, 512]
            
            # Normalize for L2 distance
            anchor = torch.nn.functional.normalize(anchor, dim=-1)
            positive = torch.nn.functional.normalize(positive, dim=-1)
            negative = torch.nn.functional.normalize(negative, dim=-1)
            
            # ✅ FIX: Ensure all shapes are (B, 512)
            assert anchor.dim() == 2 and anchor.size(1) == 512
            assert positive.dim() == 2 and positive.size(1) == 512
            assert negative.dim() == 2 and negative.size(1) == 512
            
            triplet_loss = self.triplet(anchor, positive, negative)
            
            # -------------------------
            # VAE
            # -------------------------
            N_mu = self.encoder_mu(N_aug)
            N_logvar = torch.log(
                1e-6 + torch.nn.functional.softplus(self.encoder_var(N_aug))
            )
            
            A_mu = self.encoder_mu(A_aug)
            A_logvar = torch.log(
                1e-6 + torch.nn.functional.softplus(self.encoder_var(A_aug))
            )
            
            N_latent = self._reparameterize(N_mu, N_logvar)
            A_latent = self._reparameterize(A_mu, A_logvar)
            
            kl_loss = (
                self.latent_loss(N_mu, N_logvar) + 
                self.latent_loss(A_mu, A_logvar)
            ) * 0.5
            
            # ✅ FIX: Pad latents to match x_feat
            # Currently: N_latent + A_latent have size < x_feat (which is b*n)
            # Need to reconstruct full batch
            
            # Create full latent tensor
            full_latent = torch.zeros_like(x_feat)
            full_latent[y_flat == 0] = N_latent
            full_latent[y_flat == 1] = A_latent
            
            feat = torch.cat([x_feat, full_latent], dim=-1)  # [B*n, T, 1024]
            
            pred = self.cls_head(feat).squeeze(-1)  # [B*n, T]
            
            smooth_loss = self.temporal_smoothness(pred)
            
            return {
                "frame": pred,
                "triplet_margin": triplet_loss,
                "kl_loss": kl_loss,
                "smooth_loss": smooth_loss,
                "A_att": A_att,
                "N_Aatt": N_Aatt,
            }
        
        # -------------------------
        # TEST MODE
        # -------------------------
        else:
            
            # In test, use all features (can't split by label)
            _, A_aug = self.Amemory(x_feat)
            _, N_aug = self.Nmemory(x_feat)
            
            A_aug = self.encoder_mu(A_aug)
            N_aug = self.encoder_mu(N_aug)
            
            feat = torch.cat([x_feat, A_aug + N_aug], dim=-1)
            pred = self.cls_head(feat).squeeze(-1)  # [B*n, T]
            
            # ✅ FIX: Properly reshape if 10-crop
            if is_crops:
                pred = pred.view(b, n, -1).mean(dim=1)  # [B, T]
            
            return {"frame": pred}


if __name__ == "__main__":
    m = WSAD(1024, "Train", 60, 60).cuda()
    x = torch.rand(4, 10, 32, 1024).cuda()
    y = torch.randint(0, 2, (4,)).cuda()

    out = m(x, y)["frame"]
    print(f"Train output shape: {out.shape}")  # Should be [40, 32]
    
    m.flag = "Test"
    out = m(x)["frame"]
    print(f"Test output shape: {out.shape}")  # Should be [4, 32]
