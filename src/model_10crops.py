import torch
import torch.nn as nn
from torch.nn.modules.module import Module
import numpy as np
import re

# Import your existing modules
try:
    from memory import Memory_Unit
    from translayer import Transformer
    from utils import norm
except ImportError:
    # Fallback implementations
    class Memory_Unit(nn.Module):
        def __init__(self, nums, dim):
            super().__init__()
            self.nums = nums
            self.dim = dim
            self.memory = nn.Parameter(torch.randn(nums, dim) * 0.1)
            self.scale = nn.Parameter(torch.ones(1))
            
        def forward(self, x):
            # x: [B, T, D]
            attn = torch.einsum('btd,nd->btn', x, self.memory) * self.scale
            attn = torch.softmax(attn, dim=-1)
            aug = torch.einsum('btn,nd->btd', attn, self.memory)
            return attn.mean(-1), aug
    
    class Transformer(nn.Module):
        def __init__(self, dim, num_layers, num_heads, ff_dim, out_dim, dropout=0.1):
            super().__init__()
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=dim, nhead=num_heads, dim_feedforward=ff_dim, 
                dropout=dropout, batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.proj = nn.Linear(dim, out_dim)
            
        def forward(self, x):
            x = self.transformer(x)
            return self.proj(x)
    
    def norm(x):
        return x / (x.norm(dim=-1, keepdim=True) + 1e-8)


class Temporal(Module):
    def __init__(self, input_size, out_size):
        super(Temporal, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=out_size, kernel_size=3,
                      stride=1, padding=1),
            nn.BatchNorm1d(out_size),
            nn.ReLU(),
        )

    def forward(self, x):
        # x: [B, T, D]
        x = x.permute(0, 2, 1)
        x = self.conv_1(x)
        x = x.permute(0, 2, 1)
        return x


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
        # x: [B, T, D]
        return x + self.dropout(self.conv(x.transpose(1, 2)).transpose(1, 2))


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
            nn.Linear(128, out_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.mlp(x)


class WSAD_10Crops(nn.Module):
    """
    WSAD model optimized for 10-crop I3D features with no averaging
    Preserves full spatial (10 crops) and temporal resolution
    """
    def __init__(self, input_size, flag, a_nums, n_nums):
        super().__init__()
        self.flag = flag
        self.a_nums = a_nums
        self.n_nums = n_nums
        
        # Feature extraction modules
        self.embedding = Temporal(input_size, 512)
        self.localatt = LocalAttention(512)
        self.selfatt = Transformer(512, 2, 8, 128, 512, dropout=0.2)
        
        # Memory modules
        self.Amemory = Memory_Unit(nums=a_nums, dim=512)
        self.Nmemory = Memory_Unit(nums=n_nums, dim=512)
        
        # Classifier
        self.cls_head = ADCLS_head(1024, 1)
        
        # Loss functions
        self.triplet = nn.TripletMarginLoss(margin=1.0, p=2)
        
        # VAE encoders
        self.encoder_mu = nn.Sequential(
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU()
        )
        self.encoder_var = nn.Sequential(
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU()
        )

    def _reparameterize(self, mu, logvar):
        """Reparameterization trick for VAE"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def latent_loss(self, mu, logvar):
        """KL divergence loss for VAE"""
        return torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

    def temporal_smoothness(self, pred):
        """Temporal smoothness loss to reduce frame jitter"""
        if pred.size(1) > 1:
            return ((pred[:, 1:] - pred[:, :-1]) ** 2).mean()
        return torch.tensor(0.0, device=pred.device)

    def forward(self, x, labels=None):
        """
        Args:
            x: [B, 10, T, D] - B videos, 10 crops, T frames, D=1024 features
            labels: [B] - video-level labels (0=normal, 1=anomaly)
        
        Returns:
            dict with keys:
                - frame: [B, T] frame-level anomaly scores
                - triplet_margin: triplet loss
                - kl_loss: VAE KL loss
                - distance: distance loss
                - smooth_loss: temporal smoothness loss
        """
        # Ensure input is float32
        if x.dtype != torch.float32:
            x = x.float()
        
        B, num_crops, T, D = x.shape
        
        # Process all crops in parallel by reshaping
        # [B*10, T, D]
        x_reshaped = x.view(B * num_crops, T, D)
        
        # Feature extraction (shared across all crops)
        x_reshaped = self.embedding(x_reshaped)
        x_reshaped = self.localatt(x_reshaped)
        x_reshaped = self.selfatt(x_reshaped)
        
        if self.flag == "Train" and labels is not None:
            # Expand labels to match crops
            labels_expanded = labels.repeat_interleave(num_crops)  # [B*10]
            
            # Split into anomaly and normal based on video labels
            anomaly_mask = labels_expanded == 1
            normal_mask = labels_expanded == 0
            
            # Check if we have both classes
            if anomaly_mask.sum() == 0 or normal_mask.sum() == 0:
                # Fallback: use all data with original labels
                x_combined = torch.cat([x_reshaped, x_reshaped], dim=-1)
                pre_att_all = self.cls_head(x_combined)
                pre_att = pre_att_all.view(B, num_crops, T).mean(1).squeeze(-1)
                return {
                    "frame": pre_att,
                    "triplet_margin": torch.tensor(0.0, device=x.device),
                    "kl_loss": torch.tensor(0.0, device=x.device),
                    "distance": torch.tensor(0.0, device=x.device),
                    "smooth_loss": torch.tensor(0.0, device=x.device),
                }
            
            A_x = x_reshaped[anomaly_mask]  # [num_anomaly_crops, T, D]
            N_x = x_reshaped[normal_mask]   # [num_normal_crops, T, D]
            
            # Memory reads
            # Anomaly memory with anomaly features
            A_att, A_aug = self.Amemory(A_x)
            # Normal memory with anomaly features (should be low)
            N_Aatt, N_Aaug = self.Nmemory(A_x)
            # Anomaly memory with normal features (should be low)
            A_Natt, A_Naug = self.Amemory(N_x)
            # Normal memory with normal features
            N_att, N_aug = self.Nmemory(N_x)
            
            # Second memory read for refinement
            _, A_aug2 = self.Amemory(A_aug)
            _, N_aug2 = self.Nmemory(N_aug)
            A_aug = A_aug + 0.5 * A_aug2
            N_aug = N_aug + 0.5 * N_aug2
            
            # Top-k frame selection
            k = max(T // 16 + 1, 1)
            
            # Reshape attentions to [num_videos, num_crops, T] and average over crops
            num_anomaly_videos = len(A_x) // num_crops
            num_normal_videos = len(N_x) // num_crops
            
            if num_anomaly_videos > 0 and num_normal_videos > 0:
                A_att_video = A_att.view(num_anomaly_videos, num_crops, T).mean(1)
                N_att_video = N_att.view(num_normal_videos, num_crops, T).mean(1)
                N_Aatt_video = N_Aatt.view(num_anomaly_videos, num_crops, T).mean(1)
                
                # Get top-k indices
                _, A_idx = torch.topk(A_att_video, min(k, A_att_video.size(1)), dim=-1)
                _, N_idx = torch.topk(N_att_video, min(k, N_att_video.size(1)), dim=-1)
                _, P_idx = torch.topk(N_Aatt_video, min(k, N_Aatt_video.size(1)), dim=-1)
                
                # Gather features for triplet loss
                def gather_features(x, idx, num_crops, num_videos):
                    # x: [num_crops*num_videos, T, D]
                    x_video = x.view(num_videos, num_crops, T, -1).mean(1)  # Average crops
                    k_idx = min(idx.size(1), x_video.size(1))
                    return torch.gather(x_video, 1, idx[:, :k_idx].unsqueeze(2).expand(-1, -1, x.size(-1))).mean(1)
                
                anchor = gather_features(N_x, N_idx, num_crops, num_normal_videos)
                positive = gather_features(A_x, P_idx, num_crops, num_anomaly_videos)
                negative = gather_features(A_x, A_idx, num_crops, num_anomaly_videos)
                
                # Triplet loss
                min_len = min(len(anchor), len(positive), len(negative))
                if min_len > 0:
                    triplet_loss = self.triplet(
                        norm(anchor[:min_len]),
                        norm(positive[:min_len]),
                        norm(negative[:min_len])
                    )
                else:
                    triplet_loss = torch.tensor(0.0, device=x.device)
                
                # Distance loss
                anchor_center = anchor.mean(0)
                negative_center = negative.mean(0)
                distance = torch.relu(100 - torch.norm(negative_center, p=2) + torch.norm(anchor_center, p=2))
            else:
                triplet_loss = torch.tensor(0.0, device=x.device)
                distance = torch.tensor(0.0, device=x.device)
            
            # VAE for normal samples
            N_aug_mu = self.encoder_mu(N_aug)
            N_aug_logvar = self.encoder_var(N_aug)
            N_aug_new = self._reparameterize(N_aug_mu, N_aug_logvar)
            kl_loss = self.latent_loss(N_aug_mu, N_aug_logvar)
            
            # Encode cross features
            A_Naug_enc = self.encoder_mu(A_Naug)
            N_Aaug_enc = self.encoder_mu(N_Aaug)
            A_aug_enc = self.encoder_mu(A_aug)
            
            # Combine original features with augmented features
            N_combined = torch.cat([N_x, N_aug_new + A_Naug_enc], dim=-1)
            A_combined = torch.cat([A_x, A_aug_enc + N_Aaug_enc], dim=-1)
            combined = torch.cat([N_combined, A_combined], dim=0)
            
            # Classification
            pre_att_all = self.cls_head(combined)  # [total_crops, T, 1]
            pre_att = pre_att_all.view(B, num_crops, T).mean(1).squeeze(-1)  # [B, T]
            
            # Smoothness loss
            smooth_loss = self.temporal_smoothness(pre_att)
            
            return {
                "frame": pre_att,
                "triplet_margin": triplet_loss,
                "kl_loss": kl_loss,
                "distance": distance,
                "smooth_loss": smooth_loss,
            }
        
        else:  # Test mode
            # For testing, combine features and average over crops
            x_combined = torch.cat([x_reshaped, x_reshaped], dim=-1)
            pre_att_all = self.cls_head(x_combined)
            pre_att = pre_att_all.view(B, num_crops, T).mean(1).squeeze(-1)
            return {"frame": pre_att}


if __name__ == "__main__":
    # Test the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WSAD_10Crops(input_size=1024, flag="Train", a_nums=200, n_nums=200).to(device)
    
    # Simulate input: 2 videos, 10 crops each, 300 frames, 1024 features
    x = torch.randn(2, 10, 300, 1024).to(device, dtype=torch.float32)
    labels = torch.tensor([1, 0]).to(device, dtype=torch.float32)
    
    with torch.cuda.amp.autocast(enabled=False):
        output = model(x, labels=labels)
    
    print(f"Input shape: {x.shape}")
    print(f"Output frame shape: {output['frame'].shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")