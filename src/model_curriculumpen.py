import torch
import torch.nn as nn
from torch.nn.modules.module import Module
#from memory_resume_cgpt import Memory_Unit
#from translayer_resume_cgpt import Transformer
from memory_curriculumpen import Memory_Unit
from translayer_curriculumpen import Transformer
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
        x = x.permute(0, 2, 1)
        x = self.conv_1(x)
        x = x.permute(0, 2, 1)
        return x


class LocalAttention(nn.Module):
    """Lightweight depthwise-local attention via grouped convolutions."""
    def __init__(self, dim, kernel=9):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=kernel, padding=kernel // 2, groups=dim),
            nn.ReLU(),
            nn.Conv1d(dim, dim, kernel_size=1)
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



class PrototypeExpansionNetwork(nn.Module):
    def __init__(self, feat_dim, proto_dim, hidden_dim=256, n_heads=4, n_expand=8):
        super().__init__()
        self.n_expand = n_expand

        self.feat_proj = nn.Linear(feat_dim, hidden_dim)
        self.proto_proj = nn.Linear(proto_dim, hidden_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            batch_first=True,
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, proto_dim)

    def forward(self, support_feats, base_protos):
        """
        Args:
            support_feats: [B, feat_dim]
            base_protos:   [P, proto_dim]
        Returns:
            expanded:      [n_expand, proto_dim]
        """
        s = self.feat_proj(support_feats).unsqueeze(0)   # [1, B, hidden]
        p = self.proto_proj(base_protos).unsqueeze(0)    # [1, P, hidden]

        attn_out, _ = self.cross_attn(query=s, key=p, value=p)
        attn_out = self.norm1(attn_out + s)

        pooled = attn_out.mean(dim=1)                    # [1, hidden]
        pooled = self.norm2(pooled)

        expanded = self.out_proj(pooled.repeat(self.n_expand, 1))
        return expanded

class WSAD(Module):
    def __init__(self, input_size, flag, a_nums, n_nums):
        super().__init__()
        self.flag = flag
        self.a_nums = a_nums
        self.n_nums = n_nums

        # Core modules
        self.embedding = Temporal(input_size, 512)
        self.localatt = LocalAttention(512)

        # Learnable attention temperature (initialised at baseline 0.5)
        self.attn_temp = nn.Parameter(torch.tensor(0.5))

        self.triplet = nn.TripletMarginLoss(margin=0.3, swap=True)
        self.cls_head = ADCLS_head(1024, 1)          # 1024-dim input (512 + 512)

        self.Amemory = Memory_Unit(nums=a_nums, dim=512)
        self.Nmemory = Memory_Unit(nums=n_nums, dim=512)
        # PEN
        self.use_pen = True
        self.pen_n_expand = 8
        self.pen_hidden = 256
        self.pen_type = "class_mean"   # start with baseline; change to "learned" later
        if self.pen_type == "learned":
            self.expansion_net = PrototypeExpansionNetwork(feat_dim=512,proto_dim=512,hidden_dim=self.pen_hidden,n_heads=4,n_expand=self.pen_n_expand)
        else:
            self.expansion_net = None   # class-mean baseline

        self.selfatt = Transformer(512, 2, 4, 128, 512, dropout=0.25)

        self.encoder_mu = nn.Sequential(nn.Linear(512, 512))
        self.encoder_var = nn.Sequential(nn.Linear(512, 512))
        self.relu = nn.ReLU()   # not used, harmless
        #self.register_buffer('running_norm', torch.tensor(50.0))  # reasonable initial guess

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
        """
        feats:  [B, T, D]
        scores: [B, T]
        returns pooled [B, D]
        """
        temp = torch.clamp(self.attn_temp, min=0.1)
        weights = torch.softmax(scores / temp, dim=1)          # [B, T]
        pooled = (feats * weights.unsqueeze(-1)).sum(dim=1)   # [B, D]
        return pooled
    
    def _encode(self, x):
        x = self.embedding(x)
        x = self.localatt(x)
        x = self.selfatt(x)
        return x

    def forward(self, x, support_set=None, support_labels=None, use_pen=False):
        if len(x.size()) == 4:
            b, n, t, d = x.size()
            x = x.reshape(b * n, t, d)
        else:
            b, t, d = x.size()
            n = 1

        x = self._encode(x)
        
        extra_A = None
        extra_N = None
        pen_capacity = torch.tensor(0.0, device=x.device)
        pen_diversity = torch.tensor(0.0, device=x.device)

        if use_pen and self.flag == "Train":
            if support_set is not None:
                support_feats = self._encode(support_set)       # [B_sup, T, 512]
            else:
                support_feats = x

            support_pooled = support_feats.mean(dim=1)           # [B_sup, 512]

            if support_labels is None:
                raise ValueError("PEN requires support_labels in train mode")

            abnormal_mask = support_labels == 1
            normal_mask   = support_labels == 0
            abnormal_support = support_pooled[abnormal_mask]
            normal_support   = support_pooled[normal_mask]

            if abnormal_support.size(0) > 0:
                if self.pen_type == "class_mean":
                    extra_A = abnormal_support.mean(dim=0, keepdim=True).repeat(self.pen_n_expand, 1)
                else:
                    extra_A = self.expansion_net(abnormal_support, self.Amemory.memory_block)

            if normal_support.size(0) > 0:
                if self.pen_type == "class_mean":
                    extra_N = normal_support.mean(dim=0, keepdim=True).repeat(self.pen_n_expand, 1)
                else:
                    extra_N = self.expansion_net(normal_support, self.Nmemory.memory_block)

            if extra_A is not None and extra_N is not None:
                pen_capacity = extra_A.pow(2).mean() + extra_N.pow(2).mean()

                def pairwise_abs_cosine(p):
                    if p.size(0) < 2:
                        return torch.tensor(0.0, device=p.device)
                    p = F.normalize(p, p=2, dim=1)
                    sim = p @ p.t()
                    mask = torch.eye(p.size(0), device=p.device).bool()
                    return sim[~mask].abs().mean()

                pen_diversity = pairwise_abs_cosine(extra_A) + pairwise_abs_cosine(extra_N)

        feat_magnitudes = torch.norm(x, p=2, dim=-1)   # [B, T]

        if self.flag == "Train":
            N_x = x[: b * n // 2]                      # normal videos
            A_x = x[b * n // 2 :]                      # abnormal videos

            A_memory = self.Amemory.memory_block
            N_memory = self.Nmemory.memory_block
            if use_pen and extra_A is not None and extra_N is not None:
                A_memory = torch.cat([A_memory, extra_A], dim=0)
                N_memory = torch.cat([N_memory, extra_N], dim=0)
            A_att, A_aug = self.Amemory(A_x, external_memory=A_memory)
            N_Aatt, N_Aaug = self.Nmemory(A_x, external_memory=N_memory)
            A_Natt, A_Naug = self.Amemory(N_x, external_memory=A_memory)
            N_att, N_aug = self.Nmemory(N_x, external_memory=N_memory)
            # Raw copy BEFORE gating (used by sparsity/stubborn losses)
            A_Natt_raw = A_Natt.clone()

            # Reciprocal gating (normal memory suppresses abnormal on normal videos)
            #A_Natt = A_Natt * (1.0 - torch.sigmoid(N_att))
            # If Normal memory is active, it should AGGRESSIVELY kill Abnormal memory
            A_Natt = A_Natt * (1.0 - N_att)

            # ---------- Triplet (fixed version) ----------
            anchor = self.attn_pool(N_x, N_att).reshape(b // 2, n, -1).mean(1)
            positive = self.attn_pool(A_x, N_Aatt).reshape(b // 2, n, -1).mean(1)
            negative = self.attn_pool(A_x, A_att).reshape(b // 2, n, -1).mean(1)

            anchor   = norm(anchor)
            positive = norm(positive)
            negative = norm(negative)
            triplet_margin_loss = self.triplet(anchor, positive, negative)

            # ---------- VAE encoding ----------
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

            # ---------- Direct memory confusion penalties ----------
            A_N_weights = torch.softmax(A_Natt_raw / 0.5, dim=1)
            A_N_mean = (A_Natt_raw * A_N_weights).sum(dim=1)
            conf_loss = torch.relu(A_N_mean - 0.1).mean()
            A_N_mse = F.mse_loss(A_Natt_raw, torch.zeros_like(A_Natt_raw))
            
            # ---------- Feature fusion (addition) ----------
            # ---------- Gated Inhibition Feature Fusion ----------
            # Use the memory attention scores directly as gates (already ∈ [0,1])
            gate_N   = N_att.unsqueeze(-1)      # [b, T, 1]
            gate_A_N = A_Natt.unsqueeze(-1)     # gated version is fine; already suppressed
            gate_A   = A_att.unsqueeze(-1)
            gate_N_A = N_Aatt.unsqueeze(-1)

            # Apply gates to the memory augmentations
            N_aug_final   = N_aug_new * gate_N
            A_Naug_final  = A_Naug   * gate_A_N
            A_aug_final   = A_aug_new * gate_A
            N_Aaug_final  = N_Aaug   * gate_N_A

            # Concatenate with the original features
            x = torch.cat((x, torch.cat([N_aug_final + A_Naug_final,
                                          A_aug_final + N_Aaug_final], dim=0)), dim=-1)
            
            #x = torch.cat((x, (torch.cat([N_aug_new + A_Naug,
            #                              A_aug_new + N_Aaug], dim=0))), dim=-1)
            pre_att = self.cls_head(x).reshape((b, n, -1)).mean(1)

            # ---------- Feature magnitude loss (unchanged) ----------
            k_mag = max(t // 16 + 1, 1)
            N_mag = feat_magnitudes[: b * n // 2]
            A_mag = feat_magnitudes[b * n // 2 :]
            topk_A_mag = torch.topk(A_mag, k_mag, dim=-1)[0].mean(-1).mean()
            topk_N_mag = torch.topk(N_mag, k_mag, dim=-1)[0].mean(-1).mean()
            feat_mag_loss = torch.relu(100.0 - (topk_A_mag - topk_N_mag))
            mag_reg = 0.001 * (topk_A_mag ** 2 + topk_N_mag ** 2)
            feat_mag_loss = feat_mag_loss + mag_reg

            # Temporal smoothness (optional, weight usually 0)
            smooth_loss = self.temporal_smoothness(pre_att)

            # ---------- Memory orthogonality ----------
            #ortho_matrix = torch.mm(ma, mn.t())
            
            # UPGRADED: Penalize mean squared correlation PLUS the absolute max correlation
            #ortho_loss = ortho_matrix.pow(2).mean() + 0.2 * ortho_matrix.abs().max()
            
            #new
            # HINGE ORTHO: If absolute correlation > 0.1, punish it linearly.
            # This ensures a strong gradient even for minor drift.
            #o_margin = 0.2
            #ortho_loss = torch.relu(ortho_matrix.abs() - o_margin).mean() + 0.5 * ortho_matrix.abs().max()
            
            # If ortho is 0.0, the margin is too high (0.15/0.2 is too easy for 512-dim space)
            # Drop the margin to 0.01 to force CONSTANT pressure on memory separation.
            ma = F.normalize(self.Amemory.memory_block, p=2, dim=-1)
            mn = F.normalize(self.Nmemory.memory_block, p=2, dim=-1)
            o_margin = 0.01 

            ortho_matrix = torch.mm(ma, mn.t())
            ortho_loss = torch.relu(ortho_matrix.abs() - o_margin).mean() + 1.2 * ortho_matrix.abs().max()
            #ortho_loss = torch.relu(ortho_matrix.abs() - o_margin).mean() + 0.8 * torch.relu(ortho_matrix.abs().max() - 0.3)

            # ---------- Return dictionary ----------
            return {
                "frame": pre_att,
                "triplet_margin": triplet_margin_loss,
                "kl_loss": kl_loss,
                "distance": distance_loss,
                "smooth_loss": smooth_loss,
                "feat_mag_loss": feat_mag_loss,

                # Raw temporal attention (for top‑k BCE and noose losses)
                "A_att": A_att,              # [b, T]
                "N_att": N_att,              # [b, T]
                "A_Natt": A_Natt,            # gated [b, T]
                "N_Aatt": N_Aatt,            # [b, T]

                "conf_loss": conf_loss,
                "A_N_mse": A_N_mse,
                "A_Natt_raw": A_Natt_raw,    # raw, before gating [b, T]
                "N_att_raw": N_att,          # same as N_att [b, T]
                "ortho_loss": ortho_loss,
                "pen_capacity": pen_capacity,
                "pen_diversity": pen_diversity
            }

        else:
            # Inference path
            b, t, d = x.size()
            n = 1
    
            # Get memory reads
            A_att, A_aug = self.Amemory(x)
            N_att, N_aug = self.Nmemory(x)

            A_aug = self.encoder_mu(A_aug)
            N_aug = self.encoder_mu(N_aug)

            # GATING: Use the attention directly (ensure shape is [B, T, 1])
            gate_A = A_att.unsqueeze(-1)
            gate_N = N_att.unsqueeze(-1)

            # Combine: Original features + (Weighted Memory Augmentation)
            # This ensures the 512-dim features match
            fused_features = torch.cat([x, (A_aug * gate_A) + (N_aug * gate_N)], dim=-1)
    
            pre_att = self.cls_head(fused_features)
            pre_att = pre_att.reshape((b, n, -1)).mean(1)
            return {"frame": pre_att}
            # ---------- Inference path with Gated Inhibition ----------
            # Read memory **once** for each bank; keep both attention and augmentation
            #A_att_inf, A_aug = self.Amemory(x)
            #N_att_inf, N_aug = self.Nmemory(x)

            ## Apply encoder to the augmentations
            #A_aug = self.encoder_mu(A_aug)
            #N_aug = self.encoder_mu(N_aug)

            ## Gate the contributions using the raw attention scores (already ∈ [0,1])
            #gate_A_inf = A_att_inf.unsqueeze(-1)   # [B, T, 1]
            #gate_N_inf = N_att_inf.unsqueeze(-1)

            ## Concatenate original features with gated memory augmentation
            #x = torch.cat([x, A_aug * gate_A_inf + N_aug * gate_N_inf], dim=-1)

            #pre_att = self.cls_head(x).reshape((b, n, -1)).mean(1)
            #return {"frame": pre_att}