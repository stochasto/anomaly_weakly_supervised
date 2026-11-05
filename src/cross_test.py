#!/usr/bin/env python3
"""
cross_transfer.py

Zero‑shot + few‑shot cross‑dataset evaluation (UCF‑Crime ⇄ XD‑Violence).
Uses the EXACT official test functions from your training scripts.
Saves a JSON report with AUC, AP, ACC, and all run parameters.
"""

import os
import random
import json
import argparse
from datetime import datetime
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torch.nn.utils import stateless

# ------- UCF imports -------
from maml_resume_gem import UCF_crime, test_ucf_urdmu_style, AD_Loss as AD_Loss_UCF
#from model_gemini_v2 import WSAD as WSAD_UCF
from model_gemini_xd_v3 import WSAD as WSAD_UCF

# ------- XD imports -------
from maml_xd import XDVideoDataset, test_xd_frame, AD_Loss as AD_Loss_XD
from model_gemini_xd_v3 import WSAD as WSAD_XD

# ---------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--source', required=True, choices=['ucf', 'xd'])
    p.add_argument('--target', required=True, choices=['ucf', 'xd'])
    p.add_argument('--checkpoint', required=True, help='path to .pth checkpoint')
    p.add_argument('--support_k', type=int, default=32,
                   help='number of support videos (balanced normal/abnormal)')
    p.add_argument('--inner_lr', type=float, default=1e-3)
    p.add_argument('--adapt_steps', type=int, default=50)
    p.add_argument('--root_dir_ucf', default='/mnt/c/code/data/I3D')
    p.add_argument('--root_dir_xd', default='/mnt/c/code/ae_anomaly/UR-DMU/xdv')
    p.add_argument('--num_segments', type=int, default=300)
    p.add_argument('--len_feature', type=int, default=1024)
    p.add_argument('--a_nums', type=int, default=200)
    p.add_argument('--n_nums', type=int, default=200)
    p.add_argument('--seed', type=int, default=-1)
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--output_dir', default='cross_results')
    return p.parse_args()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ---------------------------------------------------------------
def build_support_loader(dataset, k):
    normal_idx = [i for i, (_, lbl) in enumerate(dataset.samples) if lbl == 0]
    abnormal_idx = [i for i, (_, lbl) in enumerate(dataset.samples) if lbl == 1]
    k_half = k // 2
    selected = random.sample(normal_idx, k_half) + random.sample(abnormal_idx, k_half)
    return DataLoader(Subset(dataset, selected), batch_size=1, shuffle=False), selected

# ---------------------------------------------------------------
def evaluate_with_official_fn(model, test_loader, device, target, fast_weights=None, config=None):
    """
    Call the exact official test function for the target dataset,
    exactly as evaluate_and_save does: set model.flag = "Test" first,
    optionally with adapted fast_weights loaded into the model.
    Returns (auc, ap, acc).
    """
    # If fast_weights provided, temporarily replace model state
    original_state = model.state_dict() if fast_weights is not None else None
    if fast_weights is not None:
        model.load_state_dict(fast_weights)

    # Set flag exactly as in evaluate_and_save
    model.flag = "Test"
    model.eval()

    if config is None:
        config = {
            'num_segments': 300,
            'len_feature': 1024,
            'device': device,
            'gt_file': ''
        }

    if target == 'ucf':
        auc, ap, acc = test_ucf_urdmu_style(model, config, test_loader, model_file=None)
    else:
        auc, ap, acc = test_xd_frame(model, config, test_loader, model_file=None)

    # Restore original state if needed
    if original_state is not None:
        model.load_state_dict(original_state)

    # We don't need to reset flag to "Train" because we aren't continuing training.
    # But if you want to mirror evaluate_and_save exactly, you can uncomment:
    # model.train()
    # model.flag = "Train"
    return auc, ap, acc

# ---------------------------------------------------------------
def inner_adapt(model, support_loader, inner_lr, steps, device, criterion):
    """MAML inner loop, exactly as in training."""
    # Stack all support videos into one balanced batch (normal first, abnormal second)
    feats, labels = [], []
    for x, y in support_loader:
        feats.append(x)
        labels.append(y)
    support_batch = torch.cat(feats, dim=0).to(device)   # (k, T, D)
    support_labels = torch.cat(labels, dim=0).to(device)  # (k,)

    original_flag = model.flag
    model.flag = "Train"
    model.train()

    weights = dict(model.named_parameters())
    for _ in range(steps):
        out = stateless.functional_call(model, weights, (support_batch,))
        loss, _ = criterion(out, support_labels.float())
        grads = torch.autograd.grad(loss, model.parameters(), create_graph=False)
        new_weights = {}
        for (n, p), g in zip(model.named_parameters(), grads):
            if ('encoder' in n or 'Amemory' in n or 'Nmemory' in n):
                new_weights[n] = p
                #new_weights[n] = p - inner_lr * g
            else:
                new_weights[n] = p - inner_lr * g
        weights = new_weights

    model.flag = original_flag
    return weights

# ---------------------------------------------------------------
def main():
    args = parse_args()
    if args.seed >= 0:
        set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Target datasets ----
    if args.target == 'ucf':
        target_test = UCF_crime(root_dir=args.root_dir_ucf, modal='rgb', mode='Test',
                                num_segments=args.num_segments, len_feature=args.len_feature)
        target_train = UCF_crime(root_dir=args.root_dir_ucf, modal='rgb', mode='Train',
                                 num_segments=args.num_segments, len_feature=args.len_feature)
    else:
        target_test = XDVideoDataset(root_dir=args.root_dir_xd, mode='Test', modal='rgb',
                                     num_segments=args.num_segments, len_feature=args.len_feature)
        target_train = XDVideoDataset(root_dir=args.root_dir_xd, mode='Train', modal='rgb',
                                      num_segments=args.num_segments, len_feature=args.len_feature)

    test_loader = DataLoader(target_test, batch_size=1, shuffle=False)

    # ---- Model & loss (from source domain) ----
    if args.source == 'ucf':
        model = WSAD_UCF(input_size=args.len_feature, flag='Test',
                         a_nums=args.a_nums, n_nums=args.n_nums)
        criterion = AD_Loss_UCF()
    else:
        model = WSAD_XD(input_size=args.len_feature, flag='Test',
                        a_nums=args.a_nums, n_nums=args.n_nums)
        criterion = AD_Loss_XD()
    model.to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)

    config = {
        'num_segments': args.num_segments,
        'len_feature': args.len_feature,
        'device': device,
        'gt_file': ''
    }

    # ---- Zero‑shot ----
    print(f"\n[INFO] Zero‑shot: {args.source.upper()} -> {args.target.upper()}")
    zs_auc, zs_ap, zs_acc = evaluate_with_official_fn(model, test_loader, device, args.target, config=config)
    print(f"Zero‑shot AUC: {zs_auc:.4f}  AP: {zs_ap:.4f}  ACC: {zs_acc:.4f}")

    # ---- Few‑shot (MAML adaptation) ----
    support_loader, support_indices = build_support_loader(target_train, args.support_k)
    print(f"[INFO] MAML adaptation on {len(support_indices)} support videos "
          f"(lr={args.inner_lr}, steps={args.adapt_steps})")
    fast_w = inner_adapt(model, support_loader, args.inner_lr, args.adapt_steps, device, criterion)
    fs_auc, fs_ap, fs_acc = evaluate_with_official_fn(model, test_loader, device, args.target,
                                                      fast_weights=fast_w, config=config)
    print(f"Few‑shot AUC: {fs_auc:.4f}  AP: {fs_ap:.4f}  ACC: {fs_acc:.4f}")

    # ---- Save report ----
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = {
        "timestamp": timestamp,
        "args": vars(args),
        "source": args.source,
        "target": args.target,
        "checkpoint": args.checkpoint,
        "support_indices": support_indices,
        "zero_shot": {"auc": zs_auc, "ap": zs_ap, "acc": zs_acc},
        "few_shot": {"auc": fs_auc, "ap": fs_ap, "acc": fs_acc}
    }
    fname = os.path.join(args.output_dir,
                         f"cross_{args.source}2{args.target}_{timestamp}.json")
    with open(fname, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport saved to {fname}")

    # ---- Quick summary ----
    print("\n" + "="*60)
    print(f"Cross‑transfer: {args.source.upper()} -> {args.target.upper()}")
    print(f"{'Metric':<10} {'Zero‑shot':<12} {'Few‑shot':<12}")
    print("-"*34)
    for name, zs, fs in [('AUC', zs_auc, fs_auc), ('AP', zs_ap, fs_ap), ('ACC', zs_acc, fs_acc)]:
        print(f"{name:<10} {zs:<12.4f} {fs:<12.4f}")
    print("="*60)

if __name__ == '__main__':
    main()