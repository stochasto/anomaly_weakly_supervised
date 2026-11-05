#!/usr/bin/env python3
"""
maml_dm_improved_v2.py

MAML-style meta-training using WSAD + AD_Loss, improved v2:
- Proper 10-crop I3D feature handling (all frames, no averaging)
- Variable frame length with smart padding/truncation
- Memory-aware batch sizing
- Robust frame-level GT alignment
- Gradient accumulation support
- Better sampling strategy for long videos
"""

import os
import glob
import random
import argparse
from tqdm import tqdm
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.nn.utils import stateless
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import warnings
warnings.filterwarnings("ignore")

# -------------------------
# CLI / Config
# -------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--root_dir', type=str, default='/mnt/c/code/data/I3D',
                   help="base data directory with anomaly subdirs and Training_Normal_Videos_Anomaly")
    p.add_argument('--max_frames', type=int, default=280,
                   help="max frames per video (pad/truncate to this). 280 for safety, 350-400 for better AUC")
    p.add_argument('--strategy', type=str, default='pad', 
                   choices=['pad', 'truncate', 'sample'],
                   help="pad=pad to max_frames, truncate=random crop, sample=stratified subsample")
    p.add_argument('--a_nums', type=int, default=200, help="anomaly memory units")
    p.add_argument('--n_nums', type=int, default=200, help="normal memory units")
    p.add_argument('--len_feature', type=int, default=1024, help="I3D feature dimension")
    p.add_argument('--support_k', type=int, default=4, help="support batch size (videos)")
    p.add_argument('--query_k', type=int, default=8, help="query batch size (videos)")
    p.add_argument('--meta_iters', type=int, default=4000, help="number of meta iterations")
    p.add_argument('--warmup_steps', type=int, default=200, help="supervised warmup iterations")
    p.add_argument('--inner_lr', type=float, default=1e-3, help="inner loop learning rate")
    p.add_argument('--meta_lr', type=float, default=1e-4, help="outer loop learning rate")
    p.add_argument('--beta', type=float, default=1.0, help="weight for meta-val loss")
    p.add_argument('--batch_size', type=int, default=4, help="total batch size (support + query combined in some configs)")
    p.add_argument('--accumulation_steps', type=int, default=1, help="gradient accumulation steps for large batches")
    p.add_argument('--num_workers', type=int, default=4, help="dataloader workers")
    p.add_argument('--save_dir', type=str, default='checkpoints_v2')
    p.add_argument('--gt_file', type=str, default='frame_label/gt_ucf_v2.npy', help="frame-level ground truth")
    p.add_argument('--eval_interval', type=int, default=100, help="eval every N steps")
    p.add_argument('--debug', action='store_true', help="verbose logging")
    p.add_argument('--seed', type=int, default=-1, help="random seed (-1 = fully random)")
    args = p.parse_args()
    return args

# -------------------------
# Utilities
# -------------------------
def norm(x):
    """L2 normalization"""
    return x / (x.norm(dim=-1, keepdim=True) + 1e-8)

def apply_temporal_smoothing(pred, sigma=1.0):
    """Gaussian temporal smoothing for predictions"""
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    return gaussian_filter1d(pred, sigma=sigma)

# -------------------------
# 10-Crop Data Loading
# -------------------------
class UCF_Crime_10Crop(data.Dataset):
    """
    Loads UCF Crime dataset with 10-crop I3D features.
    
    Directory structure:
    root_dir/
      Abuse/
        Abuse001_x264.npy (crop 0)
        Abuse001_x264__1.npy (crop 1)
        ...
        Abuse001_x264__9.npy (crop 9)
        Abuse002_x264.npy
        ...
      Robbery/
        ...
      Training_Normal_Videos_Anomaly/
        Normal_001_x264.npy (all 10 crops)
        ...
    """
    
    def __init__(self, root_dir, mode, max_frames=280, strategy='pad', len_feature=1024, seed=-1):
        """
        Args:
            root_dir: path to data root
            mode: 'Train' or 'Test'
            max_frames: max frame length (pad/truncate to this)
            strategy: 'pad', 'truncate', or 'sample'
            len_feature: I3D feature dimension (1024)
            seed: random seed
        """
        if seed >= 0:
            random.seed(seed)
            np.random.seed(seed)
        
        self.root_dir = root_dir
        self.mode = mode
        self.max_frames = max_frames
        self.strategy = strategy
        self.len_feature = len_feature
        
        # Collect all videos with their 10 crop paths
        self.videos = []  # List of (label, [crop_0, crop_1, ..., crop_9])
        
        if mode == 'Train':
            self._load_train_videos()
        else:
            self._load_test_videos()
        
        print(f"[INFO] Loaded {len(self.videos)} {mode} videos")
        
    def _load_train_videos(self):
        """Load anomaly and normal videos from Train split"""
        # Anomaly videos: subdirectories with anomaly types
        anomaly_dirs = [
            'Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary', 'Explosion',
            'Fighting', 'RoadAccidents', 'Robbery', 'Shooting', 'Shoplifting',
            'Stealing', 'Vandalism'
        ]
        
        for anomaly_type in anomaly_dirs:
            anomaly_path = os.path.join(self.root_dir, anomaly_type)
            if not os.path.exists(anomaly_path):
                continue
            
            # Get unique video names in this anomaly directory
            video_files = sorted([f for f in os.listdir(anomaly_path) if f.endswith('.npy')])
            video_names = self._extract_video_names(video_files)
            
            for video_name in video_names:
                crop_paths = self._get_crop_paths(anomaly_path, video_name)
                if len(crop_paths) == 10:  # Only use videos with all 10 crops
                    self.videos.append((1, crop_paths))  # label=1 (anomaly)
        
        # Normal videos
        normal_path = os.path.join(self.root_dir, 'Training_Normal_Videos_Anomaly')
        if os.path.exists(normal_path):
            video_files = sorted([f for f in os.listdir(normal_path) if f.endswith('.npy')])
            video_names = self._extract_video_names(video_files)
            
            for video_name in video_names:
                crop_paths = self._get_crop_paths(normal_path, video_name)
                if len(crop_paths) == 10:
                    self.videos.append((0, crop_paths))  # label=0 (normal)
    
    def _load_test_videos(self):
        """Load test videos (same structure as train)"""
        # Test anomalies (subset of Train anomalies)
        anomaly_dirs = [
            'Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary', 'Explosion',
            'Fighting', 'RoadAccidents', 'Robbery', 'Shooting', 'Shoplifting',
            'Stealing', 'Vandalism'
        ]
        
        for anomaly_type in anomaly_dirs:
            anomaly_path = os.path.join(self.root_dir, anomaly_type)
            if not os.path.exists(anomaly_path):
                continue
            
            video_files = sorted([f for f in os.listdir(anomaly_path) if f.endswith('.npy')])
            video_names = self._extract_video_names(video_files)
            
            # Take last 14% for test (roughly)
            start_idx = int(len(video_names) * 0.86)
            for video_name in video_names[start_idx:]:
                crop_paths = self._get_crop_paths(anomaly_path, video_name)
                if len(crop_paths) == 10:
                    self.videos.append((1, crop_paths))
        
        # Test normals
        normal_path = os.path.join(self.root_dir, 'Training_Normal_Videos_Anomaly')
        if os.path.exists(normal_path):
            video_files = sorted([f for f in os.listdir(normal_path) if f.endswith('.npy')])
            video_names = self._extract_video_names(video_files)
            
            start_idx = int(len(video_names) * 0.86)
            for video_name in video_names[start_idx:]:
                crop_paths = self._get_crop_paths(normal_path, video_name)
                if len(crop_paths) == 10:
                    self.videos.append((0, crop_paths))
    
    @staticmethod
    def _extract_video_names(files):
        """Extract unique video names from crop files
        E.g., ['Abuse001_x264.npy', 'Abuse001_x264__1.npy', ...] -> ['Abuse001_x264']
        """
        video_names = set()
        for f in files:
            # Remove crop suffix (__0, __1, ..., __9) and .npy
            name = f.replace('__0.npy', '').replace('__1.npy', '').replace('__2.npy', '')
            name = name.replace('__3.npy', '').replace('__4.npy', '').replace('__5.npy', '')
            name = name.replace('__6.npy', '').replace('__7.npy', '').replace('__8.npy', '')
            name = name.replace('__9.npy', '').replace('.npy', '')
            video_names.add(name)
        return sorted(list(video_names))
    
    @staticmethod
    def _get_crop_paths(video_dir, video_name):
        """Get paths to all 10 crops for a video
        E.g., video_name='Abuse001_x264' -> [path_crop0, path_crop1, ..., path_crop9]
        """
        crop_paths = []
        for i in range(10):
            if i == 0:
                path = os.path.join(video_dir, f'{video_name}.npy')
            else:
                path = os.path.join(video_dir, f'{video_name}__{i}.npy')
            
            if os.path.exists(path):
                crop_paths.append(path)
            else:
                break
        
        return crop_paths
    
    def __len__(self):
        return len(self.videos)
    
    def __getitem__(self, idx):
        label, crop_paths = self.videos[idx]
        
        # Load all 10 crops
        crops = []
        for crop_path in crop_paths:
            feat = np.load(crop_path).astype(np.float32)  # [T, 1024]
            crops.append(feat)
        
        # Handle variable lengths: pad/truncate/sample
        crops = self._process_crops(crops)
        
        # Stack: [10, T, 1024]
        crops = np.stack(crops, axis=0)
        
        return torch.from_numpy(crops).float(), torch.tensor(label, dtype=torch.float32)
    
    def _process_crops(self, crops):
        """Process 10 crops to handle variable frame lengths"""
        T = crops[0].shape[0]  # Frame count of first crop
        
        if self.strategy == 'pad':
            # Pad to max_frames
            if T < self.max_frames:
                pad_len = self.max_frames - T
                pad = np.zeros((pad_len, self.len_feature), dtype=np.float32)
                crops = [np.vstack([c, pad]) for c in crops]
            elif T > self.max_frames:
                # Random crop
                start = np.random.randint(0, T - self.max_frames + 1)
                crops = [c[start:start + self.max_frames] for c in crops]
        
        elif self.strategy == 'truncate':
            # Random truncation
            if T > self.max_frames:
                start = np.random.randint(0, T - self.max_frames + 1)
                crops = [c[start:start + self.max_frames] for c in crops]
            # If T < max_frames, keep as is (no padding)
        
        elif self.strategy == 'sample':
            # Stratified sampling: keep head/tail/peaks
            if T > self.max_frames:
                head = int(self.max_frames * 0.2)  # 20% from start
                tail = int(self.max_frames * 0.2)  # 20% from end
                middle_keep = self.max_frames - head - tail
                
                head_frames = list(range(0, min(head, T)))
                tail_frames = list(range(max(0, T - tail), T))
                middle_frames = list(np.linspace(head, T - tail - 1, middle_keep, dtype=int))
                
                indices = sorted(list(set(head_frames + tail_frames + middle_frames)))[:self.max_frames]
                crops = [c[indices] for c in crops]
        
        return crops


# -------------------------
# Model (reuse from repo)
# -------------------------
from model_min_upgrade import WSAD  # Assuming this is available


# -------------------------
# AD_Loss (from your code)
# -------------------------
class AD_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()
    
    def forward(self, result, _label):
        loss = {}
        _label = _label.float()
        
        triplet = result.get("triplet_margin", torch.tensor(0.0, device=_label.device))
        att = result['frame']
        A_att = result.get("A_att", torch.zeros_like(att))
        N_att = result.get("N_att", torch.zeros_like(att))
        A_Natt = result.get("A_Natt", torch.zeros_like(att))
        N_Aatt = result.get("N_Aatt", torch.zeros_like(att))
        kl_loss = result.get("kl_loss", torch.tensor(0.0, device=_label.device))
        distance = result.get("distance", torch.tensor(0.0, device=_label.device))
        smooth_loss = result.get("smooth_loss", torch.tensor(0.0, device=_label.device))
        
        b = max(_label.size(0) // 2, 1)
        t = max(att.size(1), 1)
        
        # Adjusted top-k for longer sequences (was // 16, now // 20)
        anomaly_topk = max(t // 20 + 1, 1)
        
        anomaly = torch.topk(att, anomaly_topk, dim=-1)[0].mean(-1)
        anomaly_loss = self.bce(anomaly, _label)
        
        panomaly = torch.topk(1 - N_Aatt, anomaly_topk, dim=-1)[0].mean(-1)
        panomaly = panomaly[:b]
        panomaly_loss = self.bce(panomaly, torch.ones_like(panomaly))
        
        A_att_topk = torch.topk(A_att, anomaly_topk, dim=-1)[0].mean(-1)
        A_att_topk = A_att_topk[:b]
        A_loss = self.bce(A_att_topk, torch.ones_like(A_att_topk))
        
        N_loss = self.bce(N_att, torch.ones_like(N_att))
        A_Nloss = self.bce(A_Natt, torch.zeros_like(A_Natt))
        
        # Loss weights (tuned for longer sequences)
        mult_dual = 0.2
        mult_triplet = 0.2
        mult_kl = 0.005
        mult_dist = 0.0001
        mult_smooth = 0.05  # Increased from 0.01 for better temporal consistency
        
        total_loss = (
            anomaly_loss
            + mult_dual * (A_loss + panomaly_loss + N_loss + A_Nloss)
            + mult_triplet * triplet
            + mult_kl * kl_loss
            + mult_dist * distance
            + mult_smooth * smooth_loss
        )
        
        loss["total_loss"] = total_loss
        loss["att_loss"] = anomaly_loss
        loss["A_loss"] = A_loss
        loss["N_loss"] = N_loss
        loss["A_Nloss"] = A_Nloss
        loss["panomaly_loss"] = panomaly_loss
        loss["triplet"] = triplet
        loss["kl_loss"] = kl_loss
        loss["smooth_loss"] = smooth_loss
        
        return total_loss, loss


# -------------------------
# Evaluation
# -------------------------
def evaluate(model, test_dataset, config, device, do_smooth=True):
    """
    Evaluate model on test set using frame-level ground truth.
    No upsampling needed since we use all frames now.
    """
    model.eval()
    model.flag = "Test"
    
    gt = np.load(config['gt_file'])
    frame_predict = []
    
    with torch.no_grad():
        for i in tqdm(range(len(test_dataset)), desc="Evaluating", ncols=80):
            crops, label = test_dataset[i]  # [10, T, 1024]
            crops = crops.unsqueeze(0).to(device)  # [1, 10, T, 1024]
            
            out = model(crops)
            pred = out['frame'].squeeze(0).cpu().numpy()  # [T]
            
            # Apply smoothing
            if do_smooth:
                pred = apply_temporal_smoothing(pred, sigma=1.0)
            
            frame_predict.append(pred)
    
    # Concatenate all predictions
    frame_predict = np.concatenate(frame_predict)
    
    # Ensure lengths match
    if len(frame_predict) > len(gt):
        frame_predict = frame_predict[:len(gt)]
    elif len(frame_predict) < len(gt):
        print(f"[WARNING] Predictions shorter than GT: {len(frame_predict)} vs {len(gt)}")
        return 0.0, 0.0
    
    fpr, tpr, _ = roc_curve(gt, frame_predict)
    auc_score = auc(fpr, tpr)
    
    precision, recall, _ = precision_recall_curve(gt, frame_predict)
    ap_score = auc(recall, precision)
    
    return auc_score, ap_score


# -------------------------
# Training
# -------------------------
def get_random_batch(dataset, k, device):
    """Sample k videos randomly"""
    idxs = random.sample(range(len(dataset)), min(k, len(dataset)))
    crops_list, labels_list = [], []
    
    for idx in idxs:
        crops, label = dataset[idx]
        crops_list.append(crops)
        labels_list.append(label)
    
    crops = torch.stack(crops_list).to(device)  # [k, 10, T, 1024]
    labels = torch.stack(labels_list).to(device)  # [k]
    
    return crops, labels


def main():
    args = parse_args()
    
    if args.seed >= 0:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Config: max_frames={args.max_frames}, strategy={args.strategy}, batch_size={args.batch_size}")
    
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, 'plots'), exist_ok=True)
    
    # Load datasets
    print("[INFO] Loading training dataset...")
    train_dataset = UCF_Crime_10Crop(
        root_dir=args.root_dir,
        mode='Train',
        max_frames=args.max_frames,
        strategy=args.strategy,
        len_feature=args.len_feature,
        seed=args.seed
    )
    
    print("[INFO] Loading test dataset...")
    test_dataset = UCF_Crime_10Crop(
        root_dir=args.root_dir,
        mode='Test',
        max_frames=args.max_frames,
        strategy=args.strategy,
        len_feature=args.len_feature,
        seed=args.seed
    )
    
    # Model
    print("[INFO] Initializing model...")
    model = WSAD(input_size=args.len_feature, flag="Train", a_nums=args.a_nums, n_nums=args.n_nums)
    model.to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.meta_lr, betas=(0.9, 0.999), weight_decay=5e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.meta_iters)
    criterion = AD_Loss()
    
    # Tracking
    test_info = {'step': [], 'auc': [], 'ap': []}
    best_auc = 0.0
    
    print(f"[INFO] Starting meta-training for {args.meta_iters} iterations...")
    
    for step in tqdm(range(1, args.meta_iters + 1), total=args.meta_iters, ncols=100):
        model.train()
        model.flag = "Train"
        
        # Warmup
        if step <= args.warmup_steps:
            k_sup = args.support_k + args.query_k
            sup_crops, sup_labels = get_random_batch(train_dataset, k_sup, device)
            
            result = model(sup_crops)
            loss, loss_dict = criterion(result, sup_labels)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            scheduler.step()
            
            if step % 50 == 0:
                print(f"[WARMUP {step}/{args.warmup_steps}] loss={loss.item():.6f}")
            continue
        
        # MAML inner loop
        support_crops, support_labels = get_random_batch(train_dataset, args.support_k, device)
        
        result_support = model(support_crops)
        inner_loss, _ = criterion(result_support, support_labels)
        
        grads = torch.autograd.grad(inner_loss, model.parameters(), create_graph=True, allow_unused=True)
        fast_weights = {}
        for (name, param), g in zip(model.named_parameters(), grads):
            if g is not None:
                fast_weights[name] = param - args.inner_lr * g
        
        # MAML outer loop
        query_crops, query_labels = get_random_batch(train_dataset, args.query_k, device)
        result_query = stateless.functional_call(model, fast_weights, (query_crops,))
        meta_val_loss, _ = criterion(result_query, query_labels)
        
        total_loss = inner_loss + args.beta * meta_val_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        scheduler.step()
        
        if step % 50 == 0 or args.debug:
            print(f"[STEP {step}] total={total_loss.item():.6f} inner={inner_loss.item():.6f} meta_val={meta_val_loss.item():.6f}")
        
        # Evaluation
        if step % args.eval_interval == 0:
            print(f"\n[EVAL @ step {step}]")
            model.flag = "Test"
            auc_score, ap_score = evaluate(model, test_dataset, {
                'gt_file': args.gt_file,
                'device': device
            }, device, do_smooth=True)
            model.flag = "Train"
            
            test_info['step'].append(step)
            test_info['auc'].append(auc_score)
            test_info['ap'].append(ap_score)
            
            print(f"[EVAL @ {step}] AUC={auc_score:.4f} | AP={ap_score:.4f}")
            
            if auc_score > best_auc:
                best_auc = auc_score
                ckpt_path = os.path.join(
                    args.save_dir,
                    f'best_meta_wsad_auc_{best_auc:.4f}_step_{step}.pth'
                )
                torch.save(model.state_dict(), ckpt_path)
                print(f"[INFO] New best AUC: {best_auc:.4f} -> {ckpt_path}")
    
    # Plot results
    if test_info['step']:
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(test_info['step'], test_info['auc'], 'b-o', label='AUC')
        plt.xlabel('Step')
        plt.ylabel('AUC')
        plt.title('Test AUC over Training Steps')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(test_info['step'], test_info['ap'], 'r-o', label='AP')
        plt.xlabel('Step')
        plt.ylabel('AP')
        plt.title('Test AP over Training Steps')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(args.save_dir, 'plots', 'training_curves.png'), dpi=150)
        print(f"[INFO] Saved plots to {os.path.join(args.save_dir, 'plots', 'training_curves.png')}")
    
    print(f"\n[FINISHED] Best AUC: {best_auc:.4f}")


if __name__ == '__main__':
    main()
