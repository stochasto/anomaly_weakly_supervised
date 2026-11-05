#!/usr/bin/env python3
"""
Improved WSAD training for 10-crop I3D features
Fixes for poor AUC: triplet balance, VAE reconstruction, cleaner feature fusion, temporal smoothing
"""

import os
import random
import argparse
from tqdm import tqdm
import numpy as np
from scipy.ndimage import gaussian_filter1d
import re
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from sklearn.metrics import roc_curve, auc, precision_recall_curve

import warnings
warnings.filterwarnings("ignore")


# ==================== UTILS ====================
def norm(x):
    """L2 normalization"""
    return x / (x.norm(dim=-1, keepdim=True) + 1e-8)


def apply_temporal_smoothing(pred, window=5, sigma=1.0):
    """Smooth predictions temporally (for test time)"""
    if pred.ndim == 1:
        return torch.from_numpy(gaussian_filter1d(pred.cpu().numpy(), sigma=sigma)).float()
    return pred


# ==================== MEMORY UNIT ====================
class Memory_Unit(nn.Module):
    def __init__(self, nums, dim):
        super().__init__()
        self.nums = nums
        self.dim = dim
        # Better initialization: orthogonal-like
        memory = torch.randn(nums, dim)
        memory = F.normalize(memory, p=2, dim=1)
        self.memory = nn.Parameter(memory)
        
        # Temperature for attention softmax
        self.temperature = nn.Parameter(torch.tensor(1.0))
        
    def forward(self, x):
        # x: [B, T, D]
        # Normalize for stable attention
        x_norm = norm(x)
        mem_norm = norm(self.memory)
        
        # Scaled dot product attention with temperature
        attn = torch.einsum('btd,nd->btn', x_norm, mem_norm) / self.temperature.clamp(min=0.1)
        attn = torch.softmax(attn, dim=-1)
        
        # Read from memory
        aug = torch.einsum('btn,nd->btd', attn, mem_norm)
        
        return attn.mean(-1), aug


# ==================== TRANSFORMER ====================
class Transformer(nn.Module):
    def __init__(self, dim, num_layers, num_heads, ff_dim, out_dim, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=num_heads, dim_feedforward=ff_dim, 
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.proj = nn.Linear(dim, out_dim)
        
    def forward(self, x):
        x = self.transformer(x)
        return self.proj(x)


# ==================== TEMPORAL CONV ====================
class Temporal(nn.Module):
    def __init__(self, input_size, out_size):
        super().__init__()
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


# ==================== LOCAL ATTENTION ====================
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
        return x + self.dropout(self.conv(x.transpose(1, 2)).transpose(1, 2))


# ==================== CLASSIFIER HEAD ====================
class ADCLS_head(nn.Module):
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


# ==================== VAE ENCODER/DECODER ====================
class VAE_Module(nn.Module):
    def __init__(self, input_dim, latent_dim=256):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
        )
        
        self.mu = nn.Linear(256, latent_dim)
        self.logvar = nn.Linear(256, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
        )
        
    def encode(self, x):
        h = self.encoder(x)
        mu = self.mu(h)
        logvar = self.logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
    
    def latent_loss(self, mu, logvar):
        """KL divergence"""
        return -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))


# ==================== MAIN MODEL ====================
class WSAD_10Crops(nn.Module):
    def __init__(self, input_size, flag, a_nums, n_nums):
        super().__init__()
        self.flag = flag
        self.a_nums = a_nums
        self.n_nums = n_nums
        
        self.embedding = Temporal(input_size, 512)
        self.localatt = LocalAttention(512)
        self.selfatt = Transformer(512, 2, 8, 128, 512, dropout=0.2)
        
        # Improved triplet loss with smaller margin
        self.triplet = nn.TripletMarginLoss(margin=0.5, p=2)
        
        # Separate classifier for clean features
        self.cls_head = ADCLS_head(512, 1)
        
        # Memory units
        self.Amemory = Memory_Unit(nums=a_nums, dim=512)
        self.Nmemory = Memory_Unit(nums=n_nums, dim=512)
        
        # VAE for augmentation with reconstruction loss
        self.vae = VAE_Module(input_dim=512, latent_dim=256)

    def temporal_smoothness(self, pred):
        """Encourage smooth predictions across time"""
        if pred.size(1) > 1:
            return ((pred[:, 1:] - pred[:, :-1]) ** 2).mean()
        return torch.tensor(0.0, device=pred.device)

    def forward(self, x, labels=None):
        B, num_crops, T, D = x.shape
        
        # Reshape to process all crops
        x_reshaped = x.view(B * num_crops, T, D)
        
        # Feature extraction
        x_reshaped = self.embedding(x_reshaped)
        x_reshaped = self.localatt(x_reshaped)
        x_reshaped = self.selfatt(x_reshaped)
        
        if self.flag == "Train" and labels is not None:
            labels_expanded = labels.repeat_interleave(num_crops)
            anomaly_mask = labels_expanded == 1
            normal_mask = labels_expanded == 0
            
            # Fallback for no anomalies/normals
            if anomaly_mask.sum() == 0 or normal_mask.sum() == 0:
                pre_att_all = self.cls_head(x_reshaped)
                pre_att = pre_att_all.view(B, num_crops, T).mean(1).squeeze(-1)
                return {
                    "frame": pre_att,
                    "triplet_margin": torch.tensor(0.0, device=x.device),
                    "kl_loss": torch.tensor(0.0, device=x.device),
                    "recon_loss": torch.tensor(0.0, device=x.device),
                    "distance": torch.tensor(0.0, device=x.device),
                    "smooth_loss": torch.tensor(0.0, device=x.device)
                }
            
            A_x = x_reshaped[anomaly_mask]
            N_x = x_reshaped[normal_mask]
            
            # ===== MEMORY AUGMENTATION =====
            A_att, A_aug = self.Amemory(A_x)
            N_Aatt, N_Aaug = self.Nmemory(A_x)
            A_Natt, A_Naug = self.Amemory(N_x)
            N_att, N_aug = self.Nmemory(N_x)
            
            # Second read for refinement
            _, A_aug2 = self.Amemory(A_aug)
            _, N_aug2 = self.Nmemory(N_aug)
            A_aug = A_aug + 0.3 * A_aug2  # Reduced blend weight
            N_aug = N_aug + 0.3 * N_aug2
            
            # ===== TRIPLET LOSS =====
            k = max(T // 16 + 1, 1)
            num_anomaly_videos = len(A_x) // num_crops
            num_normal_videos = len(N_x) // num_crops
            
            triplet_loss = torch.tensor(0.0, device=x.device)
            distance = torch.tensor(0.0, device=x.device)
            
            if num_anomaly_videos > 0 and num_normal_videos > 0:
                A_att_video = A_att.view(num_anomaly_videos, num_crops, T).mean(1)
                N_att_video = N_att.view(num_normal_videos, num_crops, T).mean(1)
                N_Aatt_video = N_Aatt.view(num_anomaly_videos, num_crops, T).mean(1)
                
                _, A_idx = torch.topk(A_att_video, min(k, A_att_video.size(1)), dim=-1)
                _, N_idx = torch.topk(N_att_video, min(k, N_att_video.size(1)), dim=-1)
                _, P_idx = torch.topk(N_Aatt_video, min(k, N_Aatt_video.size(1)), dim=-1)
                
                def gather_features(x, idx, nc, nv):
                    x_video = x.view(nv, nc, T, -1).mean(1)
                    k_idx = min(idx.size(1), x_video.size(1))
                    idx_expanded = idx[:, :k_idx].unsqueeze(-1).expand(-1, -1, x.size(-1))
                    gathered = torch.gather(x_video, 1, idx_expanded).mean(1)
                    return gathered
                
                try:
                    anchor = gather_features(N_x, N_idx, num_crops, num_normal_videos)
                    positive = gather_features(A_x, P_idx, num_crops, num_anomaly_videos)
                    negative = gather_features(A_x, A_idx, num_crops, num_anomaly_videos)
                    
                    min_len = min(len(anchor), len(positive), len(negative))
                    if min_len > 1:  # Need at least 2 samples for triplet
                        triplet_loss = self.triplet(
                            norm(anchor[:min_len]), 
                            norm(positive[:min_len]), 
                            norm(negative[:min_len])
                        )
                        
                        # Margin loss: push apart normal-anomaly centers
                        anchor_center = anchor.mean(0)
                        negative_center = negative.mean(0)
                        distance = torch.relu(1.0 - torch.norm(negative_center - anchor_center, p=2))
                except Exception as e:
                    print(f"[WARNING] Triplet loss error: {e}")
                    triplet_loss = torch.tensor(0.0, device=x.device)
            
            # ===== VAE WITH RECONSTRUCTION =====
            N_aug_recon, N_aug_mu, N_aug_logvar = self.vae(N_aug)
            kl_loss = self.vae.latent_loss(N_aug_mu, N_aug_logvar)
            
            # Reconstruction loss for normal augmentation
            recon_loss = F.mse_loss(N_aug_recon, N_aug)
            
            # Sample from latent for augmentation
            N_aug_latent = self.vae.reparameterize(N_aug_mu, N_aug_logvar)
            N_aug_new = self.vae.decode(N_aug_latent)
            
            # ===== CLEAN FEATURE CLASSIFICATION =====
            # Don't mix features; classify each stream separately
            N_pred = self.cls_head(N_x)
            A_pred = self.cls_head(A_x)
            N_aug_pred = self.cls_head(N_aug_new)
            
            # Average predictions
            N_combined_pred = (N_pred + N_aug_pred) / 2.0
            
            # Combine for batch
            pre_att_all = torch.cat([N_combined_pred, A_pred], dim=0)
            pre_att = pre_att_all.view(B, num_crops, T).mean(1).squeeze(-1)
            
            smooth_loss = self.temporal_smoothness(pre_att)
            
            return {
                "frame": pre_att,
                "triplet_margin": triplet_loss,
                "kl_loss": kl_loss,
                "recon_loss": recon_loss,
                "distance": distance,
                "smooth_loss": smooth_loss
            }
        else:
            # Test time: clean inference
            pre_att_all = self.cls_head(x_reshaped)
            pre_att = pre_att_all.view(B, num_crops, T).mean(1).squeeze(-1)
            return {"frame": pre_att}


# ==================== DATASET ====================
class UCF_Crime_10Crops(data.Dataset):
    def __init__(self, root_dir, mode, max_frames=300):
        self.mode = mode
        self.max_frames = max_frames
        self.root_dir = root_dir
        
        list_file = os.path.join('list', f'UCF_{self.mode}.list')
        
        if not os.path.exists(list_file):
            raise FileNotFoundError(f"Cannot find list file: {list_file}")
        
        # Group crops by video
        video_groups = defaultdict(lambda: {'crops': [None] * 10})
        
        with open(list_file, 'r') as f:
            for line in f:
                crop_path = line.strip()
                if not os.path.exists(crop_path):
                    continue
                
                # Extract base name
                base_name = re.sub(r'__\d+\.npy$', '.npy', crop_path)
                base_name = re.sub(r'\.npy$', '', base_name)
                
                # Get crop index
                crop_idx = 0
                match = re.search(r'__(\d+)\.npy', crop_path)
                if match:
                    crop_idx = int(match.group(1))
                
                if 0 <= crop_idx < 10:
                    video_groups[base_name]['crops'][crop_idx] = crop_path
        
        # Filter to videos with all 10 crops
        self.videos = []
        sorted_names = sorted(video_groups.keys())
        
        # UCF Crime: first 810 are anomaly for train, 140 for test
        num_anomaly = 810 if mode == 'Train' else 140
        
        for idx, base_name in enumerate(sorted_names):
            crops = video_groups[base_name]['crops']
            if all(c is not None for c in crops):
                label = 1 if idx < num_anomaly else 0
                self.videos.append((crops, label))
        
        print(f"[INFO] Loaded {len(self.videos)} {mode} videos (anomaly: {sum(1 for _,l in self.videos if l==1)})")
    
    def __len__(self):
        return len(self.videos)
    
    def __getitem__(self, idx):
        crop_paths, label = self.videos[idx]
        
        crops = []
        orig_lengths = []
        
        for crop_path in crop_paths:
            feat = np.load(crop_path).astype(np.float32)
            T = feat.shape[0]
            orig_lengths.append(min(T, self.max_frames))
            
            if T > self.max_frames:
                feat = feat[:self.max_frames]
            elif T < self.max_frames:
                pad = np.zeros((self.max_frames - T, feat.shape[1]), dtype=np.float32)
                feat = np.vstack([feat, pad])
            
            crops.append(feat)
        
        crops = np.stack(crops, axis=0)
        return torch.from_numpy(crops).float(), label, orig_lengths


# ==================== LOSS FUNCTION ====================
class ADLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # Use weighted BCE for class imbalance
        self.bce = nn.BCELoss(reduction='mean')
    
    def forward(self, result, labels, mask=None):
        att = result['frame']
        T = att.size(1)
        k = max(T // 16, 1)
        
        # Top-k MIL loss
        if mask is not None:
            att_masked = att.clone()
            att_masked[mask == 0] = -float('inf')
            scores = torch.topk(att_masked, min(k, T), dim=-1)[0].mean(-1)
        else:
            scores = torch.topk(att, min(k, T), dim=-1)[0].mean(-1)
        
        cls_loss = self.bce(scores, labels.float())
        
        # Aux losses with improved balancing
        triplet = result.get('triplet_margin', torch.tensor(0.0, device=att.device))
        kl = result.get('kl_loss', torch.tensor(0.0, device=att.device))
        recon = result.get('recon_loss', torch.tensor(0.0, device=att.device))
        smooth = result.get('smooth_loss', torch.tensor(0.0, device=att.device))
        distance = result.get('distance', torch.tensor(0.0, device=att.device))
        
        # Reduced weights for auxiliary losses
        total = cls_loss + 0.1 * triplet + 0.01 * kl + 0.05 * recon + 0.1 * smooth + 0.1 * distance
        
        return total, {
            'cls': cls_loss.item(),
            'triplet': triplet.item() if triplet.numel() == 1 else 0,
            'kl': kl.item() if kl.numel() == 1 else 0,
            'recon': recon.item() if recon.numel() == 1 else 0,
            'smooth': smooth.item() if smooth.numel() == 1 else 0,
        }


# ==================== TRAINING ====================
def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='/mnt/c/code/data/I3D')
    parser.add_argument('--gt_file', type=str, default='frame_label/gt_ucf_v2.npy')
    parser.add_argument('--support_k', type=int, default=4)
    parser.add_argument('--query_k', type=int, default=8)
    parser.add_argument('--meta_iters', type=int, default=1000)
    parser.add_argument('--warmup_steps', type=int, default=100)
    parser.add_argument('--inner_lr', type=float, default=1e-3)
    parser.add_argument('--meta_lr', type=float, default=1e-4)
    parser.add_argument('--max_frames', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--eval_interval', type=int, default=100)
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {device}")
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load datasets
    print("[INFO] Loading datasets...")
    train_dataset = UCF_Crime_10Crops(args.root_dir, 'Train', args.max_frames)
    test_dataset = UCF_Crime_10Crops(args.root_dir, 'Test', args.max_frames)
    
    # Create model
    model = WSAD_10Crops(input_size=1024, flag="Train", a_nums=200, n_nums=200).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.meta_lr, weight_decay=1e-4)
    criterion = ADLoss()
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.meta_iters)
    
    best_auc = 0
    step = 0
    
    print(f"[INFO] Starting training for {args.meta_iters} iterations...")
    
    for step in range(1, args.meta_iters + 1):
        model.train()
        model.flag = "Train"
        
        # Sample balanced batch: mix of anomalies and normals
        anomaly_indices = [i for i, (_, label, _) in enumerate(train_dataset.videos) if label == 1]
        normal_indices = [i for i, (_, label, _) in enumerate(train_dataset.videos) if label == 0]
        
        # Balanced sampling
        n_anomaly = min(args.batch_size // 2, len(anomaly_indices))
        n_normal = args.batch_size - n_anomaly
        
        sampled_indices = random.sample(anomaly_indices, min(n_anomaly, len(anomaly_indices)))
        sampled_indices += random.sample(normal_indices, min(n_normal, len(normal_indices)))
        
        batch = [train_dataset[i] for i in sampled_indices]
        
        # Collate
        x_list, y_list, len_list = zip(*batch)
        x = torch.stack(x_list, dim=0).to(device)
        y = torch.tensor(y_list, dtype=torch.float32).to(device)
        
        # Forward pass
        result = model(x, labels=y)
        loss, loss_dict = criterion(result, y)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        scheduler.step()
        
        if step % 10 == 0 and args.debug:
            msg = f"[Step {step}] Loss: {loss.item():.4f}"
            for k, v in loss_dict.items():
                msg += f" | {k}: {v:.4f}"
            print(msg)
        
        # Evaluation
        if step % args.eval_interval == 0:
            print(f"\n[Step {step}] Evaluating...")
            model.eval()
            model.flag = "Test"
            
            all_preds = []
            all_gts = []
            gt = np.load(args.gt_file)
            gt_idx = 0
            
            with torch.no_grad():
                for i in range(len(test_dataset)):
                    crops, _, orig_lengths = test_dataset[i]
                    crops = crops.unsqueeze(0).to(device)
                    
                    out = model(crops)['frame'].squeeze(0).cpu().numpy()
                    orig_len = min(orig_lengths[0], args.max_frames)
                    out = out[:orig_len]
                    
                    # Apply temporal smoothing
                    out = apply_temporal_smoothing(torch.from_numpy(out).float(), sigma=1.0).numpy()
                    
                    gt_seg = gt[gt_idx:gt_idx + len(out)]
                    if len(gt_seg) < len(out):
                        out = out[:len(gt_seg)]
                    
                    all_preds.append(out)
                    all_gts.append(gt_seg)
                    gt_idx += len(out)
            
            all_preds = np.concatenate(all_preds)
            all_gts = np.concatenate(all_gts)
            
            fpr, tpr, _ = roc_curve(all_gts, all_preds)
            auc_score = auc(fpr, tpr)
            
            print(f"[Step {step}] AUC: {auc_score:.4f} | Best: {best_auc:.4f}")
            
            if auc_score > best_auc:
                best_auc = auc_score
                torch.save(model.state_dict(), os.path.join(args.save_dir, f'best_model_{best_auc:.4f}.pth'))
                print(f"[INFO] New best model! AUC: {best_auc:.4f}")
    
    print(f"\n[FINISHED] Best AUC: {best_auc:.4f}")


if __name__ == "__main__":
    train()
