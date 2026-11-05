#!/usr/bin/env python3
"""
maml_dm_improved.py

MAML-style meta-training using WSAD + AD_Loss, improved:
- different k for support and query
- supervised warmup
- inner/outer LR tuning
- per-video normalization
- LR scheduler
- evaluation + plotting
- robust frame alignment
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
import matplotlib.pyplot as plt
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
    p.add_argument('--a_nums', type=int, default=200)
    p.add_argument('--n_nums', type=int, default=200)
    p.add_argument('--len_feature', type=int, default=1024,   # change to 1024 if using repo features
                   help="feature dimension (2048 for your I3D)")
    p.add_argument('--support_k', type=int, default=8, help="support size per class (normal/abnormal)")
    p.add_argument('--query_k', type=int, default=16, help="query size per class (normal/abnormal)")
    p.add_argument('--meta_iters', type=int, default=4000, help="number of meta iterations")
    p.add_argument('--warmup_steps', type=int, default=100, help="supervised warmup iterations before meta updates")
    p.add_argument('--inner_lr', type=float, default=1e-3)
    p.add_argument('--meta_lr', type=float, default=1e-4)
    p.add_argument('--beta', type=float, default=1.0, help="weight for meta-val loss")
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--save_dir', type=str, default='checkpoints')
    p.add_argument('--log_dir', type=str, default='logs')
    p.add_argument('--gt_file', type=str, default='frame_label/gt_ucf_v2.npy')
    p.add_argument('--eval_interval', type=int, default=100)
    p.add_argument('--debug', action='store_true')
    p.add_argument('--no_norm', action='store_true', help="disable per-video normalization")
    p.add_argument('--seed', type=int, default=-1, help="set >=0 for deterministic runs, -1 for fully random")
    args = p.parse_args()
    return args

# -------------------------
# repo model & utils (must be present in repo)
# -------------------------
#from model import WSAD
# minimal upgrade bundle
#from model_min_upgrade import WSAD
from model_taml import WSAD
#from model_ablation import WSAD
#from model_full import WSAD
#from model_dpsk import WSAD


import utils

# -------------------------
# Dataset: group crop files per video
# (adapted from your code)
# -------------------------
import os
import numpy as np
import torch
import torch.utils.data as data
from collections import defaultdict

#################################################################

from collections import defaultdict
import os
import numpy as np
import torch
import torch.utils.data as data

import os
import numpy as np
import torch
import torch.utils.data as data
from collections import defaultdict

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

        # -------------------------
        # Load list
        # -------------------------
        raw_samples = []
        numAbnorm = 8100 if self.mode == 'Train' else 1400

        with open(list_file, 'r') as f:
            for i, line in enumerate(f):
                label = 1 if i < numAbnorm else 0
                raw_samples.append((line.strip(), label))

        # -------------------------
        # Group crops into videos
        # -------------------------
        video_dict = defaultdict(list)

        for path, label in raw_samples:
            fname = os.path.basename(path)

            if "__" in fname:
                base = fname.split("__")[0]
            else:
                base = fname.replace(".npy", "")

            video_dict[(base, label)].append(path)

        self.samples = []
        for (base, label), paths in video_dict.items():
            paths = sorted(paths, key=self.crop_sort_key)

            if len(paths) != 10:
                raise RuntimeError(f"{base} has {len(paths)} crops (expected 10)")

            self.samples.append((paths, label))

        print(f"[INFO] Total videos: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    # -------------------------
    # crop ordering
    # -------------------------
    def crop_sort_key(self, p):
        fname = os.path.basename(p)
        if "__" in fname:
            return int(fname.split("__")[1].split(".")[0])
        return 0

    # -------------------------
    # ONLY USED FOR TRAINING (NOT TEST)
    # -------------------------
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

    # -------------------------
    # GET ITEM
    # -------------------------
    def __getitem__(self, index):
        paths, label = self.samples[index]

        crops = []
        for p in paths:
            feat = np.load(p).astype(np.float32)

            # -------------------------
            # TRAIN: segment
            # TEST: RAW
            # -------------------------
            if self.mode == "Train":
                feat = self.temporal_sample(feat)

            crops.append(torch.from_numpy(feat))

        crops = torch.stack(crops)  # (10, T, D)

        return crops, torch.tensor(label).float(), paths[0]

#################################################################

def taml_meta_loss(query_losses, lambda_var=0.1, lambda_ent=0.05):
    losses = torch.stack(query_losses)

    mean_loss = losses.mean()
    var_loss = losses.var()

    weights = torch.softmax(losses.detach(), dim=0)
    entropy = -torch.sum(weights * torch.log(weights + 1e-8))

    return mean_loss + lambda_var * var_loss - lambda_ent * entropy

# -------------------------
# Helper: random sampling by video (k videos)
# -------------------------

def check_tensor(name, x):
    if x is None:
        print(f"[NAN CHECK] {name}: None")
        return True
    if not torch.is_tensor(x):
        return False
    if torch.isnan(x).any():
        print(f"[NAN CHECK] {name}: NaN detected")
        return True
    if torch.isinf(x).any():
        print(f"[NAN CHECK] {name}: Inf detected")
        return True
    return False

import os
import pickle

##################################################################################
def build_index_cache(dataset, cache_path="index_cache_10crop.pkl", rebuild=False):
    """
    Builds or loads cached indices for normal/abnormal samples.
    """

    # -------------------------
    # LOAD CACHE
    # -------------------------
    if os.path.exists(cache_path) and not rebuild:
        print(f"[INFO] Loading index cache from {cache_path}")
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
        return cache["normal"], cache["abnormal"]

    # -------------------------
    # BUILD CACHE
    # -------------------------
    print("[INFO] Building index cache (one-time cost)...")

    normal_idxs = []
    abnormal_idxs = []

    for i in range(len(dataset)):
        label = dataset.samples[i][1]   # 🔴 FAST: avoids loading npy file
        if label == 0:
            normal_idxs.append(i)
        else:
            abnormal_idxs.append(i)

    # -------------------------
    # SAVE CACHE
    # -------------------------
    cache = {
        "normal": normal_idxs,
        "abnormal": abnormal_idxs
    }

    with open(cache_path, "wb") as f:
        pickle.dump(cache, f)

    print(f"[INFO] Saved index cache to {cache_path}")

    return normal_idxs, abnormal_idxs

import random
import torch

def get_random_batch_from_dataset_fast(dataset, normal_idxs, abnormal_idxs, k, device):

    num_abnormal = min(len(abnormal_idxs), k // 2)
    num_normal = k - num_abnormal

    idxs = (
        random.sample(normal_idxs, num_normal) +
        random.sample(abnormal_idxs, num_abnormal)
    )

    feats = []
    labels = []

    for i in idxs:
        crop_feats, label, _ = dataset[i]   # already Tensor

        feats.append(crop_feats)            # ✅ FIXED
        labels.append(label)

    x = torch.stack(feats).float().to(device)
    y = torch.tensor(labels).float().to(device)

    return x, y

######################################################################################

## test ##
## Ablation aware ADLoss
def test(model, config, test_loader, model_file=None):

    import torch
    import numpy as np
    from sklearn.metrics import roc_curve, auc, precision_recall_curve

    device = next(model.parameters()).device

    model.eval()
    model.flag = "Test"

    if model_file is not None:
        model.load_state_dict(torch.load(model_file, map_location=device))

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

            res = model(_data)
            a_predict = res["frame"]   # (10, T)

            # -------------------------
            # accumulate 10 crops
            # -------------------------
            temp_predict = torch.cat([temp_predict, a_predict], dim=0)

            if (i + 1) % 10 == 0:

                # -------------------------
                # aggregate crops
                # -------------------------
                a_predict = temp_predict.mean(dim=0).cpu().numpy()

                # -------------------------
                # UR-DMU style expansion
                # -------------------------
                fpre_ = np.repeat(a_predict, 16)

                # -------------------------
                # build full prediction stream
                # -------------------------
                if frame_predict is None:
                    frame_predict = fpre_
                else:
                    frame_predict = np.concatenate([frame_predict, fpre_])

                # -------------------------
                # video classification
                # -------------------------
                cls_label.append(int(_label.item()))
                cls_pre.append(int(a_predict.max() > 0.5))

                temp_predict = torch.zeros((0), device=device)

    # -------------------------
    # evaluation
    # -------------------------
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

    print("AUC:", auc_score)
    print("AP:", ap_score)
    print("ACC:", accuracy)

    return auc_score, ap_score, accuracy

# -------------------------
# AD_Loss (slightly adjusted weights for stronger localization)
# -------------------------
'''class AD_Loss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        #self.bce = nn.BCELoss()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, result, _label):
        loss = {}
        _label = _label.float()

        triplet = result.get("triplet_margin", torch.tensor(0.0, device=_label.device))
        att = result['frame']
        # att clip
        # --- PATCH: stabilize attention ---
        #att = att / 2.0
        #att = torch.clamp(att, -3, 3)

        A_att = result.get("A_att", torch.zeros_like(att))
        N_att = result.get("N_att", torch.zeros_like(att))
        A_Natt = result.get("A_Natt", torch.zeros_like(att))
        N_Aatt = result.get("N_Aatt", torch.zeros_like(att))
        kl_loss = result.get("kl_loss", torch.tensor(0.0, device=_label.device))
        kl_loss = torch.clamp(kl_loss, 0, 1000)
        distance = result.get("distance", torch.tensor(0.0, device=_label.device))

         # ✅ NEW: smooth loss
        smooth_loss = result.get("smooth_loss", torch.tensor(0.0, device=_label.device))

        b = max(_label.size(0) // 2, 1)
        t = max(att.size(1), 1)

        #orig
        anomaly_topk = max(t // 16 + 1, 1)
        #anomaly_topk = max(t // 24 + 1, 1)
        # --- PATCH: contrast amplification ---
        topk_vals = torch.topk(att, anomaly_topk, dim=-1)[0]
        # amplify differences (CRITICAL)
        #topk_vals = topk_vals * 2.5
        # softmax weighting (prevents flat gradients)
        #weights = torch.softmax(topk_vals, dim=-1)
        #anomaly = (topk_vals * weights).sum(dim=-1)
        
        anomaly = topk_vals.mean(dim=-1)
        #anomaly = anomaly*3.0

        #anomaly = torch.topk(att, anomaly_topk, dim=-1)[0].mean(-1)
        # FIX, clamping added
        #anomaly = torch.clamp(anomaly, -20, 20)
        anomaly_loss = self.bce(anomaly, _label)

        #panomaly = torch.topk(1 - N_Aatt, anomaly_topk, dim=-1)[0].mean(-1)
        pan_topk = torch.topk(1 - N_Aatt, anomaly_topk, dim=-1)[0]
        #pan_topk = pan_topk * 2.5
        pan_weights = torch.softmax(pan_topk, dim=-1)
        panomaly = (pan_topk * pan_weights).sum(dim=-1)
        #panomaly = panomaly * 3.0
        
        panomaly = panomaly[:b]
        panomaly_loss = self.bce(panomaly, torch.ones_like(panomaly))

        #A_att_topk = torch.topk(A_att, anomaly_topk, dim=-1)[0].mean(-1)
        A_topk = torch.topk(A_att, anomaly_topk, dim=-1)[0]
        A_weights = torch.softmax(A_topk, dim=-1)
        A_att_topk = (A_topk * A_weights).sum(dim=-1)
        #A_att_topk = A_att_topk * 3.0
        A_att_topk = A_att_topk[:b]
        
        A_loss = self.bce(A_att_topk, torch.ones_like(A_att_topk))

        N_loss = self.bce(N_att, torch.ones_like(N_att))
        A_Nloss = self.bce(A_Natt, torch.zeros_like(A_Natt))

        # stronger triplet/kl scaling to encourage sharper localization
        mult_dual = 0.2
        mult_triplet = 0.2
        #mult_kl = 5e-5
        mult_kl = 1e-5
        #mult_kl = min(1e-4, current_step / 1000 * 1e-4)
        mult_dist = 0.0001
        #mult_smooth = 0.01
        mult_smooth = 0.002
        total_loss = (
            anomaly_loss
            + mult_dual * (A_loss + panomaly_loss + N_loss + A_Nloss)
            + mult_triplet * triplet
            + mult_kl * kl_loss
            + mult_dist * distance
            + mult_smooth * smooth_loss
        )

        # gave 83.6% with num_seg 300, anum = nnum = 300
        #mult_dual = 0.2
        #mult_triplet = 0.2
        #mult_kl = 0.01
        #mult_dist = 0.0001
        #mult_smooth = 0.01
        #total_loss = (
        #    anomaly_loss
        #    + mult_dual * (A_loss + panomaly_loss + N_loss + A_Nloss)
        #    + mult_triplet * triplet
        #    + mult_kl * kl_loss
        #    + mult_dist * distance
        #    + mult_smooth * smooth_loss
        #)

        # orig loss
        #total_loss = (
        #    anomaly_loss
        #    + 0.1 * (A_loss + panomaly_loss + N_loss + A_Nloss)
        #    + 0.2 * triplet
        #    + 0.005 * kl_loss
        #    + 0.0001 * distance
        #)

        loss["total_loss"] = total_loss
        loss["att_loss"] = anomaly_loss
        loss["N_Aatt"] = panomaly_loss
        loss["A_loss"] = A_loss
        loss["N_loss"] = N_loss
        loss["A_Nloss"] = A_Nloss
        loss["triplet"] = triplet
        loss["kl_loss"] = kl_loss
        loss["smooth_loss"] = smooth_loss 

        return total_loss, loss'''

## Minimal loss
class AD_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, result, _label):
        loss = {}
        _label = _label.float()

        att = result['frame']
        triplet = result.get("triplet_margin", torch.tensor(0.0, device=_label.device))
        kl_loss = result.get("kl_loss", torch.tensor(0.0, device=_label.device))
        smooth_loss = result.get("smooth_loss", torch.tensor(0.0, device=_label.device))

        N_Aatt = result.get("N_Aatt", torch.zeros_like(att))

        # -------------------------------------------------
        # 🔥 FIX 1: CROP + VIDEO SHAPE ALIGNMENT
        # -------------------------------------------------
        b = _label.size(0)
        n = 10  # fixed 10-crop assumption

        # reshape: (B*n, T) → (B, n, T)
        att = att.view(b, n, -1)
        N_Aatt = N_Aatt.view(b, n, -1)

        # -------------------------------------------------
        # 🔥 FIX 2: CROPS → VIDEO AGGREGATION
        # -------------------------------------------------
        att = att.mean(dim=1)          # (B, T)
        N_Aatt = N_Aatt.mean(dim=1)    # (B, T)

        t = att.size(1)

        # -------------------------------------------------
        # PRIMARY ANOMALY LOSS
        # -------------------------------------------------
        k = max(t // 16 + 1, 1)

        topk_vals = torch.topk(att, k, dim=-1)[0]
        anomaly = topk_vals.mean(dim=-1)   # (B,)

        anomaly_loss = self.bce(anomaly, _label)

        # -------------------------------------------------
        # MEMORY LOSS
        # -------------------------------------------------
        mem_topk = torch.topk(1 - N_Aatt, k, dim=-1)[0]
        mem_score = mem_topk.mean(dim=-1)

        memory_loss = self.bce(mem_score, _label)

        # -------------------------------------------------
        # STABILITY TERM
        # -------------------------------------------------
        variance_penalty = topk_vals.var(dim=-1).mean()

        # -------------------------------------------------
        # FINAL LOSS
        # -------------------------------------------------
        total_loss = (
            anomaly_loss
            + 0.05 * memory_loss
            + 0.2 * triplet
            + 0.01 * variance_penalty
            + 0.02 * smooth_loss
            + 1e-5 * kl_loss
        )

        # logs
        loss["total_loss"] = total_loss
        loss["att_loss"] = anomaly_loss
        loss["memory_loss"] = memory_loss
        loss["triplet"] = triplet
        loss["variance"] = variance_penalty
        loss["smooth_loss"] = smooth_loss
        loss["kl_loss"] = kl_loss

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
        len_feature=args.len_feature
    )

    test_dataset = UCF_crime(
        root_dir=train_root,   # doesn't matter
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

    # FAST BALANCED INDEX CACHE
    normal_idxs, abnormal_idxs = build_index_cache(train_dataset)

    print("[INFO] Initializing model...")
    model = WSAD(
        input_size=args.len_feature,
        flag="Train",
        a_nums=args.a_nums,
        n_nums=args.n_nums
    ).to(device)

    # OPTIONAL BUT RECOMMENDED: freeze memory params
    #for name, param in model.named_parameters():
    #    if "memory" in name.lower():
    #        param.requires_grad = False

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.meta_lr,
        betas=(0.9, 0.999),
        weight_decay=5e-5
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.meta_iters, eta_min=1e-6)
    
    criterion = AD_Loss()

    test_info = {'step': [], 'auc': [], 'ap': [], 'ac': []}
    best_auc = 0.0

    print("[INFO] Training settings:")
    print(f" device={device} support_k={args.support_k} query_k={args.query_k} "
          f"inner_lr={args.inner_lr} meta_lr={args.meta_lr} beta={args.beta}")

    # -------------------------
    # TRAIN LOOP
    # -------------------------
    for step in tqdm(range(1, args.meta_iters + 1), total=args.meta_iters):

        model.train()

        # -------------------------
        # WARMUP (UNCHANGED LOGIC)
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
            print("sup_x shape ",sup_x.shape)

            result = model(sup_x, sup_y)
            loss_sup, _ = criterion(result, sup_y)

            optimizer.zero_grad()
            loss_sup.backward()
            optimizer.step()
            #scheduler.step()
            continue

        # -------------------------
        # META TRAINING (TAML)
        # -------------------------
        num_tasks = 6
        query_losses = []

        for _ in range(num_tasks):
            # -------------------------
            # SAMPLE SUPPORT / QUERY
            # -------------------------
             # ---- CHECK MODEL OUTPUT FIRST ----
            bad_task = False
            support_x, support_y = get_random_batch_from_dataset_fast(
                train_dataset, normal_idxs, abnormal_idxs,
                args.support_k, device
            )

            query_x, query_y = get_random_batch_from_dataset_fast(
                train_dataset, normal_idxs, abnormal_idxs,
                args.query_k, device
            )

            # -------------------------
            # BUILD MEMORY (SUPPORT ONLY)
            # -------------------------
            model.reset_memory()
            _ = model(support_x, support_y)
            #model.build_memory_from_support(support_x, support_y)

            # -------------------------
            # INNER LOOP
            # -------------------------
            result_support = model(support_x, support_y)
          
            for k, v in result_support.items():
                if check_tensor(f"support_output.{k}", v):
                    print("[STOP] NaN in model output BEFORE loss")
                    bad_task = True
                    break
            if bad_task:
                torch.cuda.empty_cache()
                continue
            
            inner_loss, support_loss_dict = criterion(result_support, support_y)
            
            # ---- CHECK LOSS DICT ----
            for k, v in support_loss_dict.items():
                if check_tensor(f"support_loss.{k}", v):
                    print(f"[STOP] NaN in support loss component: {k}")
                    bad_task = True
                    break
            if bad_task:
                torch.cuda.empty_cache()
                continue
                    
            if step % 5 == 0:
                for k, v in support_loss_dict.items():
                    writer.add_scalar(f"support/{k}", v.item(), step)
                    writer.add_scalar("support/total_loss", inner_loss.item(), step)
                
            if check_tensor("inner_loss", inner_loss):
                print("[STOP] inner_loss NaN/Inf")
                continue            

            grads = torch.autograd.grad(
                inner_loss,
                model.parameters(),
                create_graph=True,
                allow_unused=True
                )
            # -------------------------
            # FIX 4 — Freeze inner LR early
            # -------------------------
            if step < 100:
                alpha = torch.tensor(1e-3, device=device)
            else:
                alpha = model.get_inner_lr()
            # logging
            
            # Direct alpha
            #alpha = model.get_inner_lr()
            
            if step % 10 == 0:
                lr_val = alpha.item() if step < 100 else model.get_inner_lr().item()
                print(f"[LR] inner_lr = {lr_val:.6f}")
                writer.add_scalar("debug/inner_lr", lr_val, step)
            fast_weights = {}
            
            for (name, param), g in zip(model.named_parameters(), grads):
                if g is None:
                    fast_weights[name] = param
                    #print("NO GRAD:", name)
                else:
                    #fast_weights[name] = param - args.inner_lr * g
                    fast_weights[name] = param - alpha * g

            # -------------------------
            # REBUILD MEMORY (CRITICAL FIX)
            # -------------------------
            model.reset_memory()
             # Forward SUPPORT with FAST WEIGHTS to rebuild memory
            _ = stateless.functional_call(
                model,
                fast_weights,
                (support_x,support_y)
            )
            #model.build_memory_from_support(support_x.detach())

            # -------------------------
            # QUERY (META SIGNAL)
            # -------------------------
            result_query = stateless.functional_call(
                model,
                fast_weights,
                (query_x,)
            )
            # ---- PATCH: clamp attention logits ----
            if "frame" in result_query:
                result_query["frame"] = torch.clamp(result_query["frame"], -8, 8)

            query_loss, query_loss_dict  = criterion(result_query, query_y)
           
            #query_losses.append(query_loss)
            if torch.isnan(query_loss) or torch.isinf(query_loss):
                continue

            query_losses.append(query_loss)
            
            for k, v in query_loss_dict.items():
                if check_tensor(f"query_loss.{k}", v):
                    print(f"[STOP] NaN in query loss component: {k}")
                    bad_task = True
                    break
            if bad_task:
                torch.cuda.empty_cache()
                continue
            # CACHE CLEAR
            if device.type == "cuda" and step % 10 == 0:
                torch.cuda.empty_cache()
            
            if step % 5 == 0:
                for k, v in query_loss_dict.items():
                    writer.add_scalar(f"query/{k}", v.item(), step)
                    writer.add_scalar("query/total_loss", query_loss.item(), step)

        # -------------------------
        # TAML META LOSS
        # -------------------------
        if len(query_losses) == 0:
            print("[CRITICAL] All tasks failed → skipping step")
            continue
        losses = torch.stack(query_losses)

        mean_loss = losses.mean()
        variance_penalty = losses.var(unbiased=False)

        weights = torch.softmax(losses.detach(), dim=0)
        entropy_bonus = -torch.sum(weights * torch.log(weights + 1e-8))

        # Good, AUC 82.49
        #lambda_var = 0.1
        #lambda_ent = 0.05
        
        # another
        lambda_var = 0.3   
        lambda_ent = 0.1  

        meta_loss = mean_loss + lambda_var * variance_penalty - lambda_ent * entropy_bonus
        
        if torch.isnan(meta_loss):
            print("NaN detected in meta_loss")
            exit()
        if step % 5 == 0:
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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

        if step % 10 == 0:
            print(f"[STEP {step}] meta_loss={meta_loss.item():.6f}")

        # -------------------------
        # EVALUATION (UNCHANGED)
        # -------------------------
        if step % args.eval_interval == 0:

            model.flag = "Test"

            #test_loader = data.DataLoader(
            #    UCF_crime(
            #        root_dir=args.root_dir,
            #        mode='Test',
            #        modal='rgb',
            #        num_segments=args.num_segments,
            #        len_feature=args.len_feature
            #    ),
            #    batch_size=1,
            #    shuffle=False,
            #    num_workers=args.num_workers
            #)

            auc_score, ap_score, acc = test(
                model,
                config,
                test_loader,
                step,
                model_file=None,
                save_dir=args.save_dir
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
    writer.close()


if __name__ == '__main__':
    main()
