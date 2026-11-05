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
        labels.append(label)

    x = torch.stack(feats).float().to(device)  # (B, 10, T, D)
    y = torch.tensor(labels).float().to(device)  # (B,)

    return x, y


# -------------------------
# Test Function
# -------------------------
def test(model, config, test_loader, device):
    """Evaluate on test set"""

    model.eval()
    model.flag = "Test"

    frame_gt = np.load(config["gt_file"])
    frame_predict = None

    temp_predict = torch.zeros((0), device=device)

    cls_label = []
    cls_pre = []

    with torch.no_grad():

        for i in range(len(test_loader.dataset)):

            _data, _label, _name = next(iter(test_loader))
            _data = _data.to(device)
            _label = _label.to(device)

            res = model(_data)  # _data: (1, 10, T, D)
            a_predict = res["frame"]  # (1, T)

            temp_predict = torch.cat([temp_predict, a_predict], dim=0)

            if (i + 1) % 10 == 0:

                a_predict = temp_predict.mean(dim=0).cpu().numpy()

                fpre_ = np.repeat(a_predict, 16)

                if frame_predict is None:
                    frame_predict = fpre_
                else:
                    frame_predict = np.concatenate([frame_predict, fpre_])

                cls_label.append(int(_label.item()))
                cls_pre.append(int(a_predict.max() > 0.5))

                temp_predict = torch.zeros((0), device=device)

    print(f'[INFO] len gt {len(frame_gt)} len pred {len(frame_predict)}')

    assert len(frame_gt) == len(frame_predict), \
        f"Mismatch GT={len(frame_gt)} vs Pred={len(frame_predict)}"

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
class AD_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, result, _label):
        """
        ✅ FIX: Properly handle 10-crop aggregation
        
        Input shapes:
        - att: (B*10, T) from model forward
        - _label: (B,) original labels
        """
        loss = {}
        _label = _label.float()

        att = result['frame']  # (B*10, T)
        b_videos = _label.size(0)  # Original batch size B
        n_crops = 10

        # ✅ FIX: Defensive reshaping with validation
        b_total = att.size(0)
        
        if b_total != b_videos * n_crops:
            raise ValueError(
                f"Shape mismatch: att has {b_total} samples, "
                f"expected {b_videos} videos * {n_crops} crops = {b_videos * n_crops}"
            )

        # Reshape and aggregate crops per video
        att_video = att.view(b_videos, n_crops, -1).mean(dim=1)  # (B, T)

        t = att_video.size(1)

        # Get triplet loss (may be 0 if not computed in train mode)
        triplet = result.get("triplet_margin", torch.tensor(0.0, device=_label.device))
        kl_loss = result.get("kl_loss", torch.tensor(0.0, device=_label.device))
        smooth_loss = result.get("smooth_loss", torch.tensor(0.0, device=_label.device))

        # Anomaly topk
        k = max(t // 16 + 1, 1)
        topk_vals = torch.topk(att_video, k, dim=-1)[0]  # (B, k)
        anomaly = topk_vals.mean(dim=-1)  # (B,)

        anomaly_loss = self.bce(anomaly, _label)

        # Memory regularization
        N_Aatt = result.get("N_Aatt", torch.zeros_like(att))
        if N_Aatt.size(0) == b_total:
            N_Aatt_video = N_Aatt.view(b_videos, n_crops, -1).mean(dim=1)  # (B, T)
            mem_topk = torch.topk(1 - N_Aatt_video, k, dim=-1)[0]
            mem_score = mem_topk.mean(dim=-1)
            memory_loss = self.bce(mem_score, _label)
        else:
            memory_loss = torch.tensor(0.0, device=_label.device)

        # Variance penalty on attention
        variance_penalty = topk_vals.var(dim=-1).mean()

        # Final loss
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
# MAIN TRAINING
# -------------------------
def main():
    args = parse_args()

    if args.seed >= 0:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = {
        'num_segments': args.num_segments,
        'len_feature': args.len_feature,
        'device': device,
        'gt_file': args.gt_file
    }

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, 'plots'), exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    writer = SummaryWriter(log_dir=os.path.join(args.log_dir, "tb_logs"))

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

    # Build index cache
    normal_idxs, abnormal_idxs = build_index_cache(train_dataset)

    print("[INFO] Initializing model...")
    model = WSAD(
        input_size=args.len_feature,
        flag="Train",
        a_nums=args.a_nums,
        n_nums=args.n_nums
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.meta_lr,
        betas=(0.9, 0.999),
        weight_decay=5e-5
    )

    # ✅ FIX: LR scheduler with warmup
    from torch.optim.lr_scheduler import LinearLR, ChainedScheduler, CosineAnnealingLR

    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        total_iters=args.warmup_steps
    )

    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.meta_iters - args.warmup_steps,
        eta_min=1e-6
    )

    scheduler = ChainedScheduler([warmup_scheduler, cosine_scheduler])

    criterion = AD_Loss()

    test_info = {'step': [], 'auc': [], 'ap': [], 'ac': []}
    best_auc = 0.0

    print("[INFO] Training settings:")
    print(f"  device={device}")
    print(f"  support_k={args.support_k} query_k={args.query_k}")
    print(f"  inner_lr={args.inner_lr} meta_lr={args.meta_lr}")
    print(f"  a_nums={args.a_nums} n_nums={args.n_nums}")
    print(f"  warmup_steps={args.warmup_steps} meta_iters={args.meta_iters}")

    # -------------------------
    # TRAIN LOOP
    # -------------------------
    for step in tqdm(range(1, args.meta_iters + 1), total=args.meta_iters):

        model.train()

        # -------------------------
        # WARMUP
        # -------------------------
        if step <= args.warmup_steps:

            k_sup = max(args.support_k, args.query_k)

            sup_x, sup_y = get_random_batch_from_dataset_fast(
                train_dataset,
                normal_idxs,
                abnormal_idxs,
                k_sup,
                device
            )

            # ✅ FIX: Build memory BEFORE forward
            model.reset_memory()
            model.build_memory_from_support(sup_x, sup_y)

            result = model(sup_x, sup_y)
            loss_sup, _ = criterion(result, sup_y)

            optimizer.zero_grad()
            loss_sup.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            if step % 50 == 0:
                print(f"[WARMUP {step}] loss={loss_sup.item():.6f}")

            continue

        # -------------------------
        # META TRAINING
        # -------------------------
        num_tasks = 6
        query_losses = []

        for task_idx in range(num_tasks):

            bad_task = False

            support_x, support_y = get_random_batch_from_dataset_fast(
                train_dataset, normal_idxs, abnormal_idxs,
                args.support_k, device
            )

            query_x, query_y = get_random_batch_from_dataset_fast(
                train_dataset, normal_idxs, abnormal_idxs,
                args.query_k, device
            )

            # ✅ FIX: CRITICAL - build memory BEFORE forward
            model.reset_memory()
            model.build_memory_from_support(support_x, support_y)

            # Inner loop
            result_support = model(support_x, support_y)

            inner_loss, support_loss_dict = criterion(result_support, support_y)

            if torch.isnan(inner_loss) or torch.isinf(inner_loss):
                print(f"[WARN] NaN at step {step}, task {task_idx}")
                bad_task = True
                continue

            if step % 100 == 0 and args.debug:
                print(f"[STEP {step}][TASK {task_idx} SUPPORT]")
                for k, v in support_loss_dict.items():
                    print(f"  {k}: {v.item():.6f}")

            # Compute gradients
            grads = torch.autograd.grad(
                inner_loss,
                model.parameters(),
                create_graph=True,
                allow_unused=True
            )

            # Inner LR
            alpha = args.inner_lr if step < 100 else model.get_inner_lr()

            fast_weights = {}
            for (name, param), g in zip(model.named_parameters(), grads):
                if g is None:
                    fast_weights[name] = param
                else:
                    fast_weights[name] = param - alpha * g

            # Rebuild memory with fast weights
            model.reset_memory()
            _ = stateless.functional_call(
                model,
                fast_weights,
                (support_x, support_y)
            )

            # Query pass - NO LABELS (use full features)
            try:
                result_query = stateless.functional_call(
                    model,
                    fast_weights,
                    (query_x,)
                )
            except RuntimeError as e:
                if "labels required" in str(e):
                    # Need to handle test mode differently
                    model.flag = "Test"
                    result_query = stateless.functional_call(
                        model,
                        fast_weights,
                        (query_x,)
                    )
                    model.flag = "Train"
                else:
                    raise

            if "frame" in result_query:
                result_query["frame"] = torch.clamp(result_query["frame"], -8, 8)

            query_loss, query_loss_dict = criterion(result_query, query_y)

            if torch.isnan(query_loss) or torch.isinf(query_loss):
                print(f"[WARN] NaN query_loss at step {step}, task {task_idx}")
                continue

            query_losses.append(query_loss)

            if step % 100 == 0 and args.debug:
                print(f"[STEP {step}][TASK {task_idx} QUERY]")
                for k, v in query_loss_dict.items():
                    print(f"  {k}: {v.item():.6f}")

            if device.type == "cuda" and step % 10 == 0:
                torch.cuda.empty_cache()

        # -------------------------
        # META LOSS
        # -------------------------
        if len(query_losses) == 0:
            print(f"[CRITICAL] All tasks failed at step {step}")
            continue

        losses = torch.stack(query_losses)

        mean_loss = losses.mean()
        variance_penalty = losses.var(unbiased=False)

        weights = torch.softmax(losses.detach(), dim=0)
        entropy_bonus = -torch.sum(weights * torch.log(weights + 1e-8))

        lambda_var = 0.3
        lambda_ent = 0.1

        meta_loss = mean_loss + lambda_var * variance_penalty - lambda_ent * entropy_bonus

        if torch.isnan(meta_loss):
            print(f"[CRITICAL] NaN meta_loss at step {step}")
            exit(1)

        if step % 100 == 0:
            print(f"[STEP {step}] meta_loss={meta_loss.item():.6f}")
            writer.add_scalar("meta/meta_loss", meta_loss.item(), step)
            writer.add_scalar("meta/mean_loss", mean_loss.item(), step)
            writer.add_scalar("meta/variance", variance_penalty.item(), step)
            writer.add_scalar("meta/entropy", entropy_bonus.item(), step)

        optimizer.zero_grad()
        meta_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        # -------------------------
        # EVALUATION
        # -------------------------
        if step % args.eval_interval == 0:

            model.flag = "Test"

            auc_score, ap_score, acc = test(
                model,
                config,
                test_loader,
                device
            )

            model.flag = "Train"

            test_info["step"].append(step)
            test_info["auc"].append(auc_score)
            test_info["ap"].append(ap_score)
            test_info["ac"].append(acc)

            print(f"[EVAL @ {step}] AUC={auc_score:.4f} | AP={ap_score:.4f} | ACC={acc:.4f}")

            if auc_score > best_auc:
                best_auc = auc_score

                best_model_path = os.path.join(args.save_dir, 'best_meta_wsad')
                best_model_path += f"_sk_{args.support_k}_qk_{args.query_k}"
                best_model_path += f"_ilr_{args.inner_lr}_mlr_{args.meta_lr}"
                best_model_path += f"_segments_{args.num_segments}"
                best_model_path += f"_auc_{best_auc:.4f}.pth"

                torch.save(model.state_dict(), best_model_path)
                print(f"[INFO] New best AUC: {best_auc:.4f} saved to {best_model_path}")

    print(f"[INFO] Training complete. Best AUC: {best_auc:.4f}")
    writer.close()


if __name__ == '__main__':
    main()
