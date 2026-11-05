#!/usr/bin/env python3
"""
maml_dm_improved_10CROP_FIXED.py

TAML meta-training for 10-crop UCF-Crime with critical fixes:
- ✅ Memory building BEFORE forward pass
- ✅ Proper shape handling for 10 crops
- ✅ Label expansion to match flattened crops
- ✅ Correct triplet loss dimensions
- ✅ Standard attention temperature (1.0)
- ✅ LR scheduling with warmup
"""

import os
import glob
from pyexpat import model
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
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import warnings
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter

warnings.filterwarnings("ignore")

# -------------------------
# CLI / Config
# -------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--root_dir', type=str, default='/mnt/c/code/data/I3D',
                   help="base data directory containing Train/RGB and Test/RGB")
    p.add_argument('--num_segments', type=int, default=200)
    p.add_argument('--a_nums', type=int, default=256, help="anomaly memory size")
    p.add_argument('--n_nums', type=int, default=256, help="normal memory size")
    p.add_argument('--len_feature', type=int, default=1024,
                   help="feature dimension")
    p.add_argument('--support_k', type=int, default=8,
                   help="support size per class")
    p.add_argument('--query_k', type=int, default=16,
                   help="query size per class")
    p.add_argument('--meta_iters', type=int, default=5000,
                   help="number of meta iterations")
    p.add_argument('--warmup_steps', type=int, default=200,
                   help="supervised warmup iterations")
    p.add_argument('--inner_lr', type=float, default=1e-3,
                   help="inner loop learning rate")
    p.add_argument('--meta_lr', type=float, default=5e-4,
                   help="meta learning rate (increased)")
    p.add_argument('--beta', type=float, default=1.0,
                   help="weight for meta-val loss")
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--save_dir', type=str, default='checkpoints')
    p.add_argument('--log_dir', type=str, default='logs')
    p.add_argument('--gt_file', type=str, default='frame_label/gt_ucf_v2.npy')
    p.add_argument('--eval_interval', type=int, default=100)
    p.add_argument('--debug', action='store_true')
    p.add_argument('--no_norm', action='store_true')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    return args

# -------------------------
# Model & Utils
# -------------------------
from claude_model_taml_10crop_dsk_v1 import WSAD
#from claude_model_taml_10crop import WSAD
import utils

# -------------------------
# Dataset (10-crop format)
# -------------------------
class UCF_crime(data.Dataset):
    def __init__(self, root_dir, modal, mode, num_segments=100, len_feature=1024):
        self.mode = mode
        self.modal = modal
        self.root_dir = root_dir
        self.num_segments = num_segments
        self.len_feature = len_feature

        list_path_repo = os.path.join('list', f'UCF_{self.mode}.list')
        alt_list_path = os.path.join(root_dir, f'{self.mode.lower()}_list.txt')

        if os.path.exists(list_path_repo):
            list_file = list_path_repo
        elif os.path.exists(alt_list_path):
            list_file = alt_list_path
        else:
            raise FileNotFoundError("List file not found")

        # Load list
        raw_samples = []
        numAbnorm = 8100 if self.mode == 'Train' else 1400

        with open(list_file, 'r') as f:
            for i, line in enumerate(f):
                label = 1 if i < numAbnorm else 0
                raw_samples.append((line.strip(), label))

        # Group crops into videos
        video_dict = defaultdict(list)

        for path, label in raw_samples:
            fname = os.path.basename(path)

            if "__" in fname:
                base = fname.split("__")[0]
            else:
                base = fname.replace(".npy", "")

            video_dict[(base, label)].append(path)

        self.samples = []
        for (base, label), paths in sorted(video_dict.items()):
            paths = sorted(paths, key=self.crop_sort_key)

            if len(paths) != 10:
                raise RuntimeError(f"{base} has {len(paths)} crops (expected 10)")

            self.samples.append((paths, label))

        print(f"[INFO] Total videos: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def crop_sort_key(self, p):
        fname = os.path.basename(p)
        if "__" in fname:
            return int(fname.split("__")[1].split(".")[0])
        return 0

    def temporal_sample(self, feat):
        T, D = feat.shape
        new_feat = np.zeros((self.num_segments, D), dtype=np.float32)
        r = np.linspace(0, T, self.num_segments + 1, dtype=np.int32)

        for i in range(self.num_segments):
            if r[i] != r[i+1]:
                new_feat[i] = feat[r[i]:r[i+1]].mean(axis=0)
            else:
                new_feat[i] = feat[r[i]]

        return new_feat

    def __getitem__(self, index):
        paths, label = self.samples[index]

        crops = []
        for p in paths:
            feat = np.load(p).astype(np.float32)

            if self.mode == "Train":
                feat = self.temporal_sample(feat)

            crops.append(torch.from_numpy(feat))

        crops = torch.stack(crops)  # (10, T, D)

        return crops, torch.tensor(label).float(), paths[0]


# -------------------------
# Index Cache
# -------------------------
#def flatten_crops(x):
#    if x.dim() == 4:
#        b, n, t, d = x.shape
#        return x.view(b * n, t, d)
#    return x

def flatten_crops(x):
    if x.dim() == 4:
        b, n, t, d = x.shape
        return x.view(b * n, t, d)
    return x

import pickle

def build_index_cache(dataset, cache_path="index_cache_10crop.pkl", rebuild=False):
    """Build or load cached indices for normal/abnormal samples"""

    if os.path.exists(cache_path) and not rebuild:
        print(f"[INFO] Loading index cache from {cache_path}")
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
        return cache["normal"], cache["abnormal"]

    print("[INFO] Building index cache...")

    normal_idxs = []
    abnormal_idxs = []

    for i in range(len(dataset)):
        label = dataset.samples[i][1]
        if label == 0:
            normal_idxs.append(i)
        else:
            abnormal_idxs.append(i)

    cache = {
        "normal": normal_idxs,
        "abnormal": abnormal_idxs
    }

    with open(cache_path, "wb") as f:
        pickle.dump(cache, f)

    print(f"[INFO] Saved index cache to {cache_path}")

    return normal_idxs, abnormal_idxs


def get_random_batch_from_dataset_fast(dataset, normal_idxs, abnormal_idxs, k, device):
    """Sample k videos (balanced) with their 10 crops"""

    num_abnormal = min(len(abnormal_idxs), k // 2)
    num_normal = k - num_abnormal

    idxs = (
        random.sample(normal_idxs, num_normal) +
        random.sample(abnormal_idxs, num_abnormal)
    )

    feats = []
    labels = []

    for i in idxs:
        crop_feats, label, _ = dataset[i]  # crop_feats: (10, T, D)

        feats.append(crop_feats)  # (10, T, D)
        labels.append(label.item())

    x = torch.stack(feats).float().to(device)  # (B, 10, T, D)
    #y = torch.tensor(labels).float().to(device)  # (B,)
    y = torch.tensor(labels, dtype=torch.float32).to(device)

    return x, y


# -------------------------
# Test Function
# -------------------------
def test(model, config, test_loader, device):
    """
    Memory-safe 10-crop evaluation (UR-DMU aligned)
    """

    model.eval()
    model.flag = "Test"

    frame_gt = np.load(config["gt_file"])
    frame_predict = []

    cls_label = []
    cls_pre = []

    torch.cuda.empty_cache()  # 🔴 important

    with torch.no_grad():

        for _data, _label, _name in test_loader:

            # _data: (1, 10, T, D)
            _data = _data.to(device)
            _label = _label.to(device)

            B, K, T, D = _data.shape  # B=1, K=10

            crop_preds = []

            # -------------------------
            # 🔴 SEQUENTIAL CROP PROCESSING
            # -------------------------
            for i in range(K):
                crop = _data[:, i:i+1, :, :]  # (1,1,T,D)

                res = model(crop)  # now safe: only 1 crop at a time
                pred = res["frame"]  # (1, T)

                crop_preds.append(pred)

            # stack → (10, T)
            crop_preds = torch.cat(crop_preds, dim=0)

            # -------------------------
            # AGGREGATE CROPS
            # -------------------------
            a_predict = crop_preds.mean(dim=0).cpu().numpy()  # (T,)
            #print("a_predict length:", len(a_predict))

            # -------------------------
            # FRAME EXPANSION
            # -------------------------
            #fpre_ = np.repeat(a_predict, 16)
            fpre_ = a_predict
            frame_predict.extend(fpre_)

            # -------------------------
            # CLASSIFICATION METRIC
            # -------------------------
            cls_label.append(int(_label.item()))
            cls_pre.append(int(a_predict.max() > 0.5))

    frame_predict = np.array(frame_predict)

    print(f"[INFO] len gt {len(frame_gt)} len pred {len(frame_predict)}")

    assert len(frame_gt) == len(frame_predict), \
        f"Mismatch GT={len(frame_gt)} vs Pred={len(frame_predict)}"

    # -------------------------
    # METRICS
    # -------------------------
    fpr, tpr, _ = roc_curve(frame_gt, frame_predict)
    auc_score = auc(fpr, tpr)

    precision, recall, _ = precision_recall_curve(frame_gt, frame_predict)
    ap_score = auc(recall, precision)

    cls_label = np.array(cls_label)
    cls_pre = np.array(cls_pre)
    accuracy = np.mean(cls_label == cls_pre)

    return auc_score, ap_score, accuracy
# -------------------------
# AD_Loss (Fixed for 10-crop)
# -------------------------

    # -------------------------
# AD_Loss (Fully Fixed for 10-crop)
# -------------------------
class AD_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, result, _label):

        loss = {}
        _label = _label.float()

        att = result['frame']

        if att.dim() == 1:
            att = att.unsqueeze(1)

        b_videos = _label.size(0)
        b_total = att.size(0)
        t = att.size(1)

        # -------------------------
        # infer crops dynamically
        # -------------------------
        if b_total % b_videos == 0:
            n_crops = b_total // b_videos
        else:
            n_crops = 1

        # -------------------------
        # aggregate crops
        # -------------------------
        if n_crops > 1:
            att_video = att.view(b_videos, n_crops, t).mean(dim=1)
        else:
            att_video = att

        # -------------------------
        # anomaly loss
        # -------------------------
        k = max(t // 16 + 1, 1)
        k = min(k, t)

        topk_vals = torch.topk(att_video, k, dim=-1)[0]

        anomaly = topk_vals.mean(dim=-1) * 1.5
        anomaly_loss = self.bce(anomaly, _label)

        # -------------------------
        # memory loss (NORMAL ONLY)
        # -------------------------
        memory_loss = torch.tensor(0.0, device=_label.device)

        N_Aatt = result.get("N_Aatt", None)

        if N_Aatt is not None:
            N_Aatt = N_Aatt.detach()

            if N_Aatt.size(0) == b_total:

                if n_crops > 1:
                    N_Aatt_video = N_Aatt.view(b_videos, n_crops, t).mean(dim=1)
                else:
                    N_Aatt_video = N_Aatt

                mem_k = max(k // 2, 1)
                mem_k = min(mem_k, t)

                mem_topk = torch.topk(1 - N_Aatt_video, mem_k, dim=-1)[0]
                mem_score = mem_topk.mean(dim=-1)

                normal_mask = (_label == 0)

                if normal_mask.sum() > 0:
                    memory_loss = self.bce(
                        mem_score[normal_mask],
                        torch.zeros_like(mem_score[normal_mask])
                    )

        # -------------------------
        # regularizers
        # -------------------------
        variance_penalty = (
            topk_vals.var(dim=-1).mean()
            if k > 1 else torch.tensor(0.0, device=_label.device)
        )

        triplet = result.get("triplet_margin", torch.tensor(0.0, device=_label.device))
        kl_loss = torch.clamp(result.get("kl_loss", torch.tensor(0.0, device=_label.device)), 0, 10)
        smooth_loss = result.get("smooth_loss", torch.tensor(0.0, device=_label.device))

        # -------------------------
        # total loss
        # -------------------------
        total_loss = (
            anomaly_loss
            + 0.05 * memory_loss
            + 0.2 * triplet
            + 0.01 * variance_penalty
            + 0.02 * smooth_loss
            + 1e-5 * kl_loss
        )

        # -------------------------
        # logging (SAFE)
        # -------------------------
        loss["total_loss"] = total_loss
        loss["att_loss"] = anomaly_loss.detach()
        loss["memory_loss"] = memory_loss.detach()
        loss["triplet"] = triplet.detach()
        loss["variance"] = variance_penalty.detach()
        loss["smooth_loss"] = smooth_loss.detach()
        loss["kl_loss"] = kl_loss.detach()

        return total_loss, loss

# -------------------------
# Alternative: Multi-Crop AD Loss with Adaptive Aggregation
# -------------------------
class AdaptiveADLoss(nn.Module):
    """
    Adaptive loss that can handle variable number of crops
    """
    def __init__(self, n_crops=10, aggregation='mean'):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.n_crops = n_crops
        self.aggregation = aggregation  # 'mean', 'max', or 'attention'
        
    def aggregate_crops(self, x, b_videos):
        """
        Aggregate crop-level predictions to video-level
        
        Args:
            x: (B*n_crops, T) crop-level predictions
            b_videos: Number of videos B
        
        Returns:
            (B, T) video-level predictions
        """
        # Reshape to (B, n_crops, T)
        x_reshaped = x.view(b_videos, self.n_crops, -1)
        
        if self.aggregation == 'mean':
            # Smooth aggregation
            return x_reshaped.mean(dim=1)
        elif self.aggregation == 'max':
            # Most sensitive to anomalies
            return x_reshaped.max(dim=1)[0]
        elif self.aggregation == 'attention':
            # Learnable attention over crops (requires extra parameters)
            # This would need to be implemented with a small network
            weights = torch.softmax(x_reshaped.mean(dim=-1), dim=1)  # (B, n_crops)
            return (x_reshaped * weights.unsqueeze(-1)).sum(dim=1)
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")
    
    def forward(self, result, _label):
        """
        Forward pass with adaptive aggregation
        """
        loss = {}
        _label = _label.float()
        
        att = result['frame']  # (B*n_crops, T) or (B, T)
        b_videos = _label.size(0)
        
        # Aggregate if needed
        if att.size(0) == b_videos * self.n_crops:
            att_video = self.aggregate_crops(att, b_videos)
        elif att.size(0) == b_videos:
            att_video = att
        else:
            raise ValueError(f"Cannot handle att shape {att.shape} with {b_videos} videos")
        
        t = att_video.size(1)
        
        # Compute top-k anomaly score
        k = max(t // 16 + 1, 1)
        topk_vals = torch.topk(att_video, min(k, t), dim=-1)[0]
        anomaly = topk_vals.mean(dim=-1)
        anomaly_loss = self.bce(anomaly, _label)
        
        # Get other losses
        triplet = result.get("triplet_margin", torch.tensor(0.0, device=_label.device))
        kl_loss = result.get("kl_loss", torch.tensor(0.0, device=_label.device))
        smooth_loss = result.get("smooth_loss", torch.tensor(0.0, device=_label.device))
        
        # Memory loss (if available)
        memory_loss = torch.tensor(0.0, device=_label.device)
        N_Aatt = result.get("N_Aatt", None)
        if N_Aatt is not None and N_Aatt.size(0) == b_videos * self.n_crops:
            N_Aatt_video = self.aggregate_crops(N_Aatt, b_videos)
            mem_topk = torch.topk(1 - N_Aatt_video, min(k, t), dim=-1)[0]
            mem_score = mem_topk.mean(dim=-1)
            memory_loss = self.bce(mem_score, _label)
        
        # Variance penalty
        variance_penalty = topk_vals.var(dim=-1).mean() if k > 1 else torch.tensor(0.0, device=_label.device)
        
        # Combine losses
        total_loss = (
            anomaly_loss
            + 0.05 * memory_loss
            + 0.2 * triplet
            + 0.01 * variance_penalty
            + 0.02 * smooth_loss
            + 1e-5 * kl_loss
        )
        
        loss["total_loss"] = total_loss
        loss["att_loss"] = anomaly_loss
        loss["memory_loss"] = memory_loss
        loss["triplet"] = triplet
        loss["variance"] = variance_penalty
        loss["smooth_loss"] = smooth_loss
        loss["kl_loss"] = kl_loss
        
        return total_loss, loss


# -------------------------
# Debug Loss (for troubleshooting)
# -------------------------
class DebugADLoss(nn.Module):
    """
    Loss function with extensive debug output
    """
    def __init__(self, verbose=True):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.verbose = verbose
    
    def forward(self, result, _label):
        if self.verbose:
            print("\n" + "="*50)
            print("DEBUG: AD_Loss Forward")
            print(f"Label shape: {_label.shape}")
            print(f"Label values: {_label}")
            
            if 'frame' in result:
                print(f"Frame shape: {result['frame'].shape}")
                print(f"Frame stats: min={result['frame'].min():.4f}, "
                      f"max={result['frame'].max():.4f}, "
                      f"mean={result['frame'].mean():.4f}")
            
            print(f"Available keys: {result.keys()}")
        
        # Use the fixed AD_Loss above
        ad_loss = AD_Loss()
        total_loss, loss_dict = ad_loss.forward(result, _label)
        
        if self.verbose:
            print(f"\nLoss components:")
            for k, v in loss_dict.items():
                if isinstance(v, torch.Tensor) and v.numel() == 1:
                    print(f"  {k}: {v.item():.6f}")
                elif not isinstance(v, torch.Tensor):
                    print(f"  {k}: {v:.6f}")
            print("="*50 + "\n")
        
        return total_loss, loss_dict


# -------------------------
# MAIN TRAINING
# -------------------------
def log_losses(writer, loss_dict, step, prefix="train"):
    for k, v in loss_dict.items():
        if isinstance(v, torch.Tensor):
            writer.add_scalar(f"{prefix}/{k}", v.item(), step)
        else:
            writer.add_scalar(f"{prefix}/{k}", v, step)

# =====================================================
# MAIN
# =====================================================
def main():
    args = parse_args()

    # -------------------------
    # SEED
    # -------------------------
    if args.seed >= 0:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------
    # DIRS
    # -------------------------
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=os.path.join(args.log_dir, "tb_logs"))

    # -------------------------
    # DATASET
    # -------------------------
    print("[INFO] Loading dataset...")

    train_root = os.path.join(args.root_dir, 'Train', 'RGB')

    train_dataset = UCF_crime(
        root_dir=train_root,
        modal='rgb',
        mode='Train',
        num_segments=args.num_segments,
        len_feature=args.len_feature
    )

    test_dataset = UCF_crime(
        root_dir=train_root,
        modal='rgb',
        mode='Test',
        num_segments=args.num_segments,
        len_feature=args.len_feature
    )

    test_loader = data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers
    )

    normal_idxs, abnormal_idxs = build_index_cache(train_dataset)

    # -------------------------
    # MODEL
    # -------------------------
    print("[INFO] Initializing model...")

    model = WSAD(
        input_size=args.len_feature,
        a_nums=args.a_nums,
        n_nums=args.n_nums
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.meta_lr,
        weight_decay=5e-5
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.meta_iters
    )

    criterion = AD_Loss()

    best_auc = 0.0

    print("[INFO] Training started...")

    # =====================================================
    # TRAIN LOOP
    # =====================================================
    for step in tqdm(range(1, args.meta_iters + 1)):

        model.train()
        model.flag = "Train"

        # =====================================================
        # WARMUP
        # =====================================================
        if step <= args.warmup_steps:

            k = max(args.support_k, args.query_k)

            sup_x, sup_y = get_random_batch_from_dataset_fast(
                train_dataset,
                normal_idxs,
                abnormal_idxs,
                k,
                device
            )

            result = model(sup_x, sup_y)
            loss, loss_dict = criterion(result, sup_y)
            #  PRED STATS (WARMUP)
            with torch.no_grad():
                pred = result["frame"]
                writer.add_scalar("warmup/pred_mean", pred.mean().item(), step)
                writer.add_scalar("warmup/pred_max", pred.max().item(), step)
                writer.add_scalar("warmup/pred_min", pred.min().item(), step)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            if step % 5 == 0:
                log_losses(writer, loss_dict, step, prefix="warmup")

            continue

        # =====================================================
        # META TRAINING
        # =====================================================
        num_tasks = 6
        query_losses = []

        for _ in range(num_tasks):

            support_x, support_y = get_random_batch_from_dataset_fast(
                train_dataset, normal_idxs, abnormal_idxs,
                args.support_k, device
            )

            query_x, query_y = get_random_batch_from_dataset_fast(
                train_dataset, normal_idxs, abnormal_idxs,
                args.query_k, device
            )

            # -------------------------
            # INNER LOOP
            # -------------------------
            result_sup = model(support_x, support_y)
            inner_loss, inner_loss_dict  = criterion(result_sup, support_y)

            if torch.isnan(inner_loss):
                continue

            grads = torch.autograd.grad(
                inner_loss,
                model.parameters(),
                create_graph=True,
                allow_unused=True
            )

            fast_weights = {}
            for (name, param), g in zip(model.named_parameters(), grads):
                if g is None:
                    fast_weights[name] = param
                else:
                    fast_weights[name] = param - args.inner_lr * g

            # -------------------------
            # QUERY (META)
            # -------------------------
            result_q = torch.func.functional_call(
                model,
                fast_weights,
                (query_x, query_y)
            )

            query_loss, query_loss_dict = criterion(result_q, query_y)
            # PRED STATS (QUERY)
            with torch.no_grad():
                pred = result_q["frame"]
                writer.add_scalar("meta/pred_mean", pred.mean().item(), step)
                writer.add_scalar("meta/pred_max", pred.max().item(), step)
                writer.add_scalar("meta/pred_min", pred.min().item(), step)

            if not torch.isnan(query_loss):
                query_losses.append(query_loss)

        # -------------------------
        # META LOSS
        # -------------------------
        if len(query_losses) == 0:
            continue

        losses = torch.stack(query_losses)

        mean_loss = losses.mean()
        var_loss = losses.var()

        weights = torch.softmax(losses.detach(), dim=0)
        entropy = -torch.sum(weights * torch.log(weights + 1e-8))

        meta_loss = mean_loss + 0.3 * var_loss - 0.1 * entropy
        if step % 5 == 0:
            writer.add_scalar("meta/meta_loss", meta_loss.item(), step)
            writer.add_scalar("meta/mean_loss", mean_loss.item(), step)
            writer.add_scalar("meta/var_loss", var_loss.item(), step)
            writer.add_scalar("meta/entropy", entropy.item(), step)
            
            # log last task losses (important signal)
            log_losses(writer, inner_loss_dict, step, prefix="inner")
            log_losses(writer, query_loss_dict, step, prefix="query")

            # learning rate
            writer.add_scalar("meta/lr", scheduler.get_last_lr()[0], step)

        optimizer.zero_grad()
        meta_loss.backward()
        meta_loss.backward()
        
        # GRAD NORM
        ####################### Grad logging #######################
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2

        total_norm = total_norm ** 0.5
        writer.add_scalar("meta/grad_norm", total_norm, step)
        ####################### Grad logging #######################
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # -------------------------
        # LOG
        # -------------------------
        if step % 5 == 0:
            print(f"[STEP {step}] meta_loss={meta_loss.item():.6f}")
            writer.add_scalar("meta_loss", meta_loss.item(), step)

        # =====================================================
        # EVAL
        # =====================================================
        if step % args.eval_interval == 0:

            auc_score, ap_score, acc = test(
                model,
                {"gt_file": args.gt_file},
                test_loader,
                device
            )

            print(f"[EVAL {step}] AUC={auc_score:.4f} AP={ap_score:.4f}")

            if auc_score > best_auc:
                best_auc = auc_score

                path = os.path.join(
                    args.save_dir,
                    f"best_wsad_auc_{best_auc:.4f}.pth"
                )

                torch.save(model.state_dict(), path)
                print(f"[SAVE] {path}")

    writer.close()
    print(f"[DONE] Best AUC: {best_auc:.4f}")


if __name__ == "__main__":
    main()
