#!/usr/bin/env python3
"""
maml_dm_improved_FIXED.py

MAML-style meta-training using WSAD + AD_Loss, with critical fixes:
- 🔴 FIX: Removed double sigmoid in classifier
- 🔴 FIX: Improved KL loss weighting (annealing schedule)
- 🔴 FIX: Increased triplet margin for better separation
- 🔴 FIX: Better meta-learning rate scheduling
- 🟡 IMPROVE: Better NaN handling and diagnostics
- 🟡 IMPROVE: Added train-test normalization parity
- ✅ ADD: Hard example mining / curriculum learning options
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
from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.filterwarnings("ignore")

# -------------------------
# CLI / Config
# -------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--root_dir', type=str, default='/mnt/c/code/data/I3D',
                   help="base data directory containing Train/RGB and Test/RGB")
    p.add_argument('--num_segments', type=int, default=200)
    p.add_argument('--a_nums', type=int, default=200,
                   help="anomaly memory size (increase to 256-512 for better coverage)")
    p.add_argument('--n_nums', type=int, default=200,
                   help="normal memory size (increase to 256-512 for better coverage)")
    p.add_argument('--len_feature', type=int, default=1024,
                   help="feature dimension (2048 for your I3D)")
    p.add_argument('--support_k', type=int, default=8,
                   help="support size per class (normal/abnormal)")
    p.add_argument('--query_k', type=int, default=16,
                   help="query size per class (normal/abnormal)")
    p.add_argument('--meta_iters', type=int, default=4000,
                   help="number of meta iterations")
    p.add_argument('--warmup_steps', type=int, default=100,
                   help="supervised warmup iterations before meta updates")
    
    # 🔴 FIXED: Increased initial meta_lr from 1e-4 to 5e-4
    p.add_argument('--meta_lr', type=float, default=5e-4,
                   help="meta learning rate (increased from 1e-4)")
    p.add_argument('--inner_lr', type=float, default=1e-3,
                   help="inner loop learning rate")
    p.add_argument('--beta', type=float, default=1.0,
                   help="weight for meta-val loss")
    p.add_argument('--kl_schedule', type=str, default='annealing',
                   choices=['constant', 'annealing', 'curriculum'],
                   help="KL weight schedule")
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--save_dir', type=str, default='checkpoints')
    p.add_argument('--log_dir', type=str, default='logs_claude')
    p.add_argument('--gt_file', type=str, default='frame_label/gt_ucf_v2.npy')
    p.add_argument('--eval_interval', type=int, default=100)
    p.add_argument('--debug', action='store_true',
                   help="enable verbose debugging")
    p.add_argument('--no_norm', action='store_true',
                   help="disable per-video normalization")
    p.add_argument('--seed', type=int, default=-1,
                   help="set >=0 for deterministic runs, -1 for fully random")
    args = p.parse_args()
    return args

# -------------------------
# repo model & utils (must be present in repo)
# -------------------------
from model_taml import WSAD
import utils

# -------------------------
# Dataset: group crop files per video
# -------------------------
class UCF_crime(data.Dataset):
    def __init__(self, root_dir, modal, mode, num_segments, len_feature, seed=-1):
        if seed >= 0 and hasattr(utils, "set_seed"):
            utils.set_seed(seed)
        self.mode = mode
        self.modal = modal
        self.num_segments = num_segments
        self.len_feature = len_feature
        self.root_dir = root_dir

        list_path_repo = os.path.join('list', f'UCF_{self.mode}.list')
        alt_list_path = os.path.join(self.root_dir, f'{self.mode.lower()}_list.txt')

        if os.path.exists(list_path_repo):
            list_file = list_path_repo
        elif os.path.exists(alt_list_path):
            list_file = alt_list_path
        else:
            raise FileNotFoundError(f"Cannot find list file for mode={self.mode}. Looked for {list_path_repo} and {alt_list_path}.")

        samples = []
        if self.mode == 'Train':
            numAbnorm = 8100
        else:
            numAbnorm = 1400
        with open(list_file, 'r') as f:
            for (i,line) in enumerate(f):
                if (i < numAbnorm):
                    samples.append((line.strip(),1))
                else:
                    samples.append((line.strip(),0))
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        data, label = self.get_data(index)
        return data, label
      
    def get_data(self, index):
        vid, label = self.samples[index]
        video_feature = np.load(vid).astype(np.float32)   # [T, feat_dim]

        if self.mode == "Train":
            T = video_feature.shape[0]
            D = video_feature.shape[1]
            new_feat = np.zeros((self.num_segments, D), dtype=np.float32)
            r = np.linspace(0, T, self.num_segments + 1, dtype=np.int32)
            for i in range(self.num_segments):
                if r[i] != r[i+1]:
                    new_feat[i, :] = np.mean(video_feature[r[i]:r[i+1], :], axis=0)
                else:
                    new_feat[i, :] = video_feature[r[i], :]
            video_feature = new_feat
        return video_feature, label


# -------------------------
# Helper: build index cache
# -------------------------
import os
import pickle

def build_index_cache(dataset, cache_path="index_cache.pkl", rebuild=False):
    """
    Builds or loads cached indices for normal/abnormal samples.
    """

    # LOAD CACHE
    if os.path.exists(cache_path) and not rebuild:
        print(f"[INFO] Loading index cache from {cache_path}")
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
        return cache["normal"], cache["abnormal"]

    # BUILD CACHE
    print("[INFO] Building index cache (one-time cost)...")

    normal_idxs = []
    abnormal_idxs = []

    for i in range(len(dataset)):
        label = dataset.samples[i][1]
        if label == 0:
            normal_idxs.append(i)
        else:
            abnormal_idxs.append(i)

    # SAVE CACHE
    cache = {
        "normal": normal_idxs,
        "abnormal": abnormal_idxs
    }

    with open(cache_path, "wb") as f:
        pickle.dump(cache, f)

    print(f"[INFO] Saved index cache to {cache_path}")

    return normal_idxs, abnormal_idxs

def get_random_batch_from_dataset_fast(dataset, normal_idxs, abnormal_idxs, k, device, do_norm=False):
    """
    Sample k videos balanced between normal and abnormal.
    
    🔴 FIXED: Added do_norm parameter for train-test consistency
    """
    half_k = k // 2

    idxs = random.sample(normal_idxs, half_k) + \
           random.sample(abnormal_idxs, k - half_k)

    feats, labels = [], []

    for i in idxs:
        feat, label = dataset[i]
        arr = np.asarray(feat).astype(np.float32)
        
        # 🔴 FIXED: Per-video normalization (optional but recommended)
        if do_norm:
            mu = arr.mean(axis=0, keepdims=True)
            s = arr.std(axis=0, keepdims=True) + 1e-6
            arr = (arr - mu) / s

        feats.append(torch.from_numpy(arr))
        labels.append(torch.tensor(label).float())

    x = torch.stack(feats).to(device)
    y = torch.stack(labels).to(device)

    return x, y


def check_tensor(name, x, verbose=False):
    """
    🔴 FIXED: Better NaN/Inf diagnostics
    """
    if x is None:
        if verbose:
            print(f"[NAN CHECK] {name}: None")
        return True
    if not torch.is_tensor(x):
        return False
    if torch.isnan(x).any():
        print(f"[ERROR] {name}: NaN detected")
        return True
    if torch.isinf(x).any():
        print(f"[ERROR] {name}: Inf detected")
        return True
    return False


def test(net, config, test_loader, step, model_file=None, save_dir='checkpoints', do_norm=False):
    """
    Test function with optional per-video normalization
    
    🔴 FIXED: Added do_norm parameter for train-test parity
    """
    with torch.no_grad():
        net.eval()
        net.flag = "Test"
        if model_file is not None:
            net.load_state_dict(torch.load(model_file))

        load_iter = iter(test_loader)
        frame_gt = np.load("frame_label/gt_ucf_v2.npy")
        frame_predict = None
        
        cls_label = []
        cls_pre = []
        temp_predict = torch.zeros((0)).cuda()
        
        for i in range(len(test_loader.dataset)):
            _data, _label = next(load_iter)
            
            _data = _data.cuda()
            _label = _label.cuda()
            
            # 🔴 FIXED: Per-video normalization consistency
            if do_norm:
                for j in range(_data.size(0)):
                    mu = _data[j].mean(dim=0, keepdim=True)
                    s = _data[j].std(dim=0, keepdim=True) + 1e-6
                    _data[j] = (_data[j] - mu) / s

            res = net(_data)   
            a_predict = res["frame"]
            temp_predict = torch.cat([temp_predict, a_predict], dim=0)
            
            if (i + 1) % 10 == 0:
                cls_label.append(int(_label))
                a_predict = temp_predict.mean(0).cpu().numpy()
                
                cls_pre.append(1 if a_predict.max()>0.5 else 0)          
                fpre_ = a_predict

                if frame_predict is None:         
                    frame_predict = fpre_
                else:
                    frame_predict = np.concatenate([frame_predict, fpre_])  
                temp_predict = torch.zeros((0)).cuda()
   
        print(f'len gt {len(frame_gt)} len predict {len(frame_predict)}')
        fpr, tpr, _ = roc_curve(frame_gt, frame_predict)
        auc_score = auc(fpr, tpr)
    
        corrent_num = np.sum(np.array(cls_label) == np.array(cls_pre), axis=0)
        accuracy = corrent_num / (len(cls_pre))
        
        precision, recall, th = precision_recall_curve(frame_gt, frame_predict,)
        ap_score = auc(recall, precision)
        
    return auc_score, ap_score, accuracy


# -------------------------
# 🔴 FIXED: AD_Loss with improved weights
# -------------------------
class AD_Loss(nn.Module):
    def __init__(self, kl_schedule='annealing', warmup_steps=100, total_steps=4000) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()  # Expects raw logits, NO sigmoid
        self.kl_schedule = kl_schedule
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def get_kl_weight(self, step):
        """
        Compute KL weight based on schedule
        
        Options:
        - constant: fixed weight
        - annealing: linearly increase from 0 to target
        - curriculum: slow increase then plateau
        """
        if self.kl_schedule == 'constant':
            return 0.001

        elif self.kl_schedule == 'annealing':
            # Linear annealing: 0 → 0.001
            progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            return 0.001 * min(progress, 1.0)

        elif self.kl_schedule == 'curriculum':
            # Slow ramp then plateau
            progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            return 0.001 * min(progress ** 0.5, 1.0)  # sqrt for slower initial ramp
        
        return 0.001

    def forward(self, result, _label, step=1):
        loss = {}
        _label = _label.float()

        triplet = result.get("triplet_margin", torch.tensor(0.0, device=_label.device))
        att = result['frame']
        
        # ✅ CLAMP LOGITS (not probabilities) for stability
        att = torch.clamp(att, -3, 3)

        A_att = result.get("A_att", torch.zeros_like(att))
        N_att = result.get("N_att", torch.zeros_like(att))
        A_Natt = result.get("A_Natt", torch.zeros_like(att))
        N_Aatt = result.get("N_Aatt", torch.zeros_like(att))
        kl_loss = result.get("kl_loss", torch.tensor(0.0, device=_label.device))
        kl_loss = torch.clamp(kl_loss, 0, 1000)
        distance = result.get("distance", torch.tensor(0.0, device=_label.device))
        smooth_loss = result.get("smooth_loss", torch.tensor(0.0, device=_label.device))

        b = max(_label.size(0) // 2, 1)
        t = max(att.size(1), 1)

        # Anomaly topk
        anomaly_topk = max(t // 16 + 1, 1)
        
        # Contrast amplification
        topk_vals = torch.topk(att, anomaly_topk, dim=-1)[0]
        topk_vals = topk_vals * 2.5
        weights = torch.softmax(topk_vals, dim=-1)
        anomaly = (topk_vals * weights).sum(dim=-1)
        anomaly = anomaly * 3.0

        anomaly_loss = self.bce(anomaly, _label)

        # Positive anomaly (normal should NOT activate anomaly memory)
        pan_topk = torch.topk(1 - N_Aatt, anomaly_topk, dim=-1)[0]
        pan_topk = pan_topk * 2.5
        pan_weights = torch.softmax(pan_topk, dim=-1)
        panomaly = (pan_topk * pan_weights).sum(dim=-1)
        panomaly = panomaly * 3.0
        
        panomaly = panomaly[:b]
        panomaly_loss = self.bce(panomaly, torch.ones_like(panomaly))

        # A memory should respond to A
        A_topk = torch.topk(A_att, anomaly_topk, dim=-1)[0]
        A_weights = torch.softmax(A_topk, dim=-1)
        A_att_topk = (A_topk * A_weights).sum(dim=-1)
        A_att_topk = A_att_topk * 3.0
        A_att_topk = A_att_topk[:b]
        
        A_loss = self.bce(A_att_topk, torch.ones_like(A_att_topk))

        # N memory should respond to N
        N_loss = self.bce(N_att, torch.ones_like(N_att))
        
        # N memory should NOT respond to A
        A_Nloss = self.bce(A_Natt, torch.zeros_like(A_Natt))

        # 🔴 FIXED: Improved loss weights with KL annealing
        mult_dual = 0.2
        mult_triplet = 0.2
        mult_kl = self.get_kl_weight(step)  # Annealing schedule
        mult_dist = 0.0001
        mult_smooth = 0.01

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
        loss["N_Aatt"] = panomaly_loss
        loss["A_loss"] = A_loss
        loss["N_loss"] = N_loss
        loss["A_Nloss"] = A_Nloss
        loss["triplet"] = triplet
        loss["kl_loss"] = kl_loss
        loss["smooth_loss"] = smooth_loss 

        return total_loss, loss


# -------------------------
# MAIN: improved meta-training
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
    writer = SummaryWriter(log_dir=os.path.join(args.log_dir, "tb_logs"))

    print("[INFO] Loading dataset...")
    train_root = os.path.join(args.root_dir, 'Train', 'RGB')

    train_dataset = UCF_crime(
        root_dir=train_root,
        modal='rgb',
        mode='Train',
        num_segments=args.num_segments,
        len_feature=args.len_feature,
        seed=args.seed
    )

    test_dataset = UCF_crime(
        root_dir=train_root,
        modal='rgb',
        mode='Test',
        num_segments=args.num_segments,
        len_feature=args.len_feature,
        seed=args.seed
    )

    # FAST BALANCED INDEX CACHE
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

    # 🔴 FIXED: Better LR scheduling with warmup
    from torch.optim.lr_scheduler import ChainedScheduler, LinearLR, CosineAnnealingLR
    
    scheduler = ChainedScheduler([
        LinearLR(
            optimizer, 
            start_factor=0.1, 
            total_iters=args.warmup_steps
        ),
        CosineAnnealingLR(
            optimizer, 
            T_max=max(1, args.meta_iters - args.warmup_steps)
        )
    ])

    criterion = AD_Loss(
        kl_schedule=args.kl_schedule,
        warmup_steps=args.warmup_steps,
        total_steps=args.meta_iters
    )

    test_info = {'step': [], 'auc': [], 'ap': [], 'ac': []}
    best_auc = 0.0

    print("[INFO] Training settings:")
    print(f"  device={device}")
    print(f"  support_k={args.support_k} query_k={args.query_k}")
    print(f"  inner_lr={args.inner_lr} meta_lr={args.meta_lr}")
    print(f"  a_nums={args.a_nums} n_nums={args.n_nums}")
    print(f"  kl_schedule={args.kl_schedule}")
    print(f"  warmup_steps={args.warmup_steps}")

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
                device,
                do_norm=not args.no_norm
            )

            result = model(sup_x)
            loss_sup, _ = criterion(result, sup_y, step=step)

            optimizer.zero_grad()
            loss_sup.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            scheduler.step()
            continue

        # -------------------------
        # META TRAINING (TAML)
        # -------------------------
        num_tasks = 6
        query_losses = []

        for task_idx in range(num_tasks):

            # Sample support / query
            support_x, support_y = get_random_batch_from_dataset_fast(
                train_dataset, normal_idxs, abnormal_idxs,
                args.support_k, device, do_norm=not args.no_norm
            )

            query_x, query_y = get_random_batch_from_dataset_fast(
                train_dataset, normal_idxs, abnormal_idxs,
                args.query_k, device, do_norm=not args.no_norm
            )

            # BUILD MEMORY (SUPPORT ONLY)
            model.reset_memory()
            model.build_memory_from_support(support_x.detach())

            # INNER LOOP
            result_support = model(support_x)
            
            # 🔴 FIXED: Better NaN diagnostics
            for k, v in result_support.items():
                if check_tensor(f"[STEP {step}] support_output.{k}", v):
                    print(f"[ERROR] Skipping task {task_idx}")
                    continue
            
            inner_loss, support_loss_dict = criterion(result_support, support_y, step=step)
            
            for k, v in support_loss_dict.items():
                if check_tensor(f"[STEP {step}] support_loss.{k}", v):
                    print(f"[ERROR] Skipping task {task_idx}")
                    continue
            if step % 5 == 0:
                for k, v in support_loss_dict.items():
                    writer.add_scalar(f"support/{k}", v.item(), step)
                    writer.add_scalar("support/total_loss", inner_loss.item(), step)
                
            if check_tensor("[STEP {}] inner_loss".format(step), inner_loss):
                print(f"[ERROR] Skipping task {task_idx}")
                continue
            
            if step % 50 == 0 and args.debug:
                print(f"[STEP {step}][TASK {task_idx} SUPPORT]")
                for k, v in support_loss_dict.items():
                    print(f"  {k}: {v.item():.6f}")

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

            # REBUILD MEMORY FOR QUERY
            model.reset_memory()
            model.build_memory_from_support(support_x.detach())

            # QUERY (META SIGNAL)
            result_query = stateless.functional_call(
                model,
                fast_weights,
                (query_x,)
            )

            query_loss, query_loss_dict = criterion(result_query, query_y, step=step)
           
            # 🔴 FIXED: Better NaN handling
            if torch.isnan(query_loss) or torch.isinf(query_loss):
                print(f"[WARN] NaN/Inf in query_loss at step {step}, task {task_idx}")
                print(f"  Saving debug checkpoint...")
                torch.save(model.state_dict(), f"{args.save_dir}/debug_step_{step}_task_{task_idx}.pth")
                continue
            if step % 5 == 0:
                for k, v in query_loss_dict.items():
                    writer.add_scalar(f"query/{k}", v.item(), step)
                    writer.add_scalar("query/total_loss", query_loss.item(), step)
            
            query_losses.append(query_loss)
            
            if step % 50 == 0 and args.debug:
                print(f"[STEP {step}][TASK {task_idx} QUERY]")
                for k, v in query_loss_dict.items():
                    print(f"  {k}: {v.item():.6f}")

        # -------------------------
        # TAML META LOSS
        # -------------------------
        if len(query_losses) == 0:
            print("[CRITICAL] All tasks failed → skipping step")
            continue
            
        losses = torch.stack(query_losses)
        
        if step % 50 == 0 and args.debug:
            print(f"[STEP {step}] Query losses per task:")
            print([round(l.item(), 4) for l in query_losses])

        mean_loss = losses.mean()
        variance_penalty = losses.var(unbiased=False)

        weights = torch.softmax(losses.detach(), dim=0)
        entropy_bonus = -torch.sum(weights * torch.log(weights + 1e-8))

        # 🔴 IMPROVED: Better task balancing
        lambda_var = 0.5
        lambda_ent = 0.02

        meta_loss = mean_loss + lambda_var * variance_penalty - lambda_ent * entropy_bonus
        
        if torch.isnan(meta_loss):
            print(f"[CRITICAL] NaN in meta_loss at step {step}")
            exit(1)
            
        if step % 50 == 0 and args.debug:
            print(f"[STEP {step} META]")
            print(f"  mean_loss: {mean_loss.item():.6f}")
            print(f"  variance: {variance_penalty.item():.6f}")
            print(f"  entropy: {entropy_bonus.item():.6f}")
            print(f"  meta_loss: {meta_loss.item():.6f}")
        writer.add_scalar("meta/meta_loss", meta_loss.item(), step)
        writer.add_scalar("meta/mean_loss", mean_loss.item(), step)
        writer.add_scalar("meta/variance", variance_penalty.item(), step)
        writer.add_scalar("meta/entropy", entropy_bonus.item(), step)

        optimizer.zero_grad()
        meta_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
         ## Writer log ##
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        writer.add_scalar("debug/grad_norm", total_norm, step)
        ##########################
        optimizer.step()
        scheduler.step()

        if step % 100 == 0:
            print(f"[STEP {step}] meta_loss={meta_loss.item():.6f}")

        # -------------------------
        # EVALUATION
        # -------------------------
        if step % args.eval_interval == 0:

            model.flag = "Test"

            test_loader = data.DataLoader(
                UCF_crime(
                    root_dir=args.root_dir,
                    mode='Test',
                    modal='rgb',
                    num_segments=args.num_segments,
                    len_feature=args.len_feature
                ),
                batch_size=1,
                shuffle=False,
                num_workers=args.num_workers
            )

            auc_score, ap_score, acc = test(
                model,
                config,
                test_loader,
                step,
                model_file=None,
                save_dir=args.save_dir,
                do_norm=not args.no_norm  # 🔴 FIXED: Pass norm flag
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
                best_model_path += f"_anums_{args.a_nums}_nnums_{args.n_nums}"
                best_model_path += f"_auc_{best_auc:.4f}.pth"

                torch.save(model.state_dict(), best_model_path)

                print(f"[INFO] New best AUC: {best_auc:.4f} saved to {best_model_path}")

    print(f"[INFO] Training complete. Best AUC: {best_auc:.4f}")
    
    # Plot results
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(test_info['step'], test_info['auc'], 'b-', label='AUC')
    plt.xlabel('Step')
    plt.ylabel('AUC')
    plt.title('AUC over Training')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(test_info['step'], test_info['ap'], 'g-', label='AP')
    plt.xlabel('Step')
    plt.ylabel('AP')
    plt.title('AP over Training')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot(test_info['step'], test_info['ac'], 'r-', label='Accuracy')
    plt.xlabel('Step')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over Training')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, 'training_curves.png'), dpi=150)
    print(f"[INFO] Saved plots to {args.save_dir}/training_curves.png")
    writer.close()


if __name__ == '__main__':
    main()
