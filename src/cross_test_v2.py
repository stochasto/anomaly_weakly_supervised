#!/usr/bin/env python3
"""
cross_transfer.py

Zero‑shot + few‑shot cross‑dataset evaluation (UCF‑Crime ⇄ XD‑Violence).
Supports both MAML (inner‑loop adaptation) and ERM (standard fine‑tuning)
with identical support set, learning rate, steps, and frozen modules.
Saves a JSON report for each run.
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
from model_gemini_v2 import WSAD as WSAD_UCF

# ------- XD imports -------
from maml_xd import XDVideoDataset, test_xd_frame, AD_Loss as AD_Loss_XD
from model_gemini_xd_v3 import WSAD as WSAD_XD

# ---------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--source', required=True, choices=['ucf', 'xd'])
    p.add_argument('--target', required=True, choices=['ucf', 'xd'])
    p.add_argument('--checkpoint', required=True, help='path to .pth checkpoint')
    p.add_argument('--model_type', default='maml', choices=['maml', 'erm'],
                   help='maml = inner‑loop adaptation, erm = standard fine‑tuning')
    p.add_argument('--support_k', type=int, default=32)
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
    if seed >= 0:
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
def evaluate_with_official_fn(model, test_loader, device, target, config=None):
    """Call the official test function (mirrors evaluate_and_save)."""
    model.eval()
    model.flag = "Test"
    if config is None:
        config = {'num_segments': 300, 'len_feature': 1024, 'device': device, 'gt_file': ''}
    if target == 'ucf':
        auc, ap, acc = test_ucf_urdmu_style(model, config, test_loader, model_file=None)
    else:
        auc, ap, acc = test_xd_frame(model, config, test_loader, model_file=None)
    return auc, ap, acc

# ---------------------------------------------------------------
def inner_adapt_maml(model, support_loader, inner_lr, steps, device, criterion):
    """MAML inner loop – freezes encoder & memory, returns fast_weights."""
    feats, labels = [], []
    for x, y in support_loader:
        feats.append(x); labels.append(y)
    support_batch = torch.cat(feats, dim=0).to(device)
    support_labels = torch.cat(labels, dim=0).to(device)

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
            else:
                new_weights[n] = p - inner_lr * g
        weights = new_weights

    model.flag = original_flag
    return weights

def adapt_erm(model, support_loader, inner_lr, steps, device, criterion):
    """Standard fine‑tuning, freezing same modules as MAML for fair comparison."""
    # Freeze encoder and memory
    for n, p in model.named_parameters():
        if 'encoder' in n or 'Amemory' in n or 'Nmemory' in n:
            p.requires_grad = False
        else:
            p.requires_grad = True

    optim = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=inner_lr)
    model.train()
    model.flag = "Train"

    feats, labels = [], []
    for x, y in support_loader:
        feats.append(x); labels.append(y)
    support_batch = torch.cat(feats, dim=0).to(device)
    support_labels = torch.cat(labels, dim=0).to(device)

    for _ in range(steps):
        optim.zero_grad()
        out = model(support_batch)
        loss, _ = criterion(out, support_labels.float())
        loss.backward()
        optim.step()

    # The model itself is now adapted (in‑place). No need to return weights.
    # Restore requires_grad for safety (not strictly needed here)
    for p in model.parameters():
        p.requires_grad = True

# ---------------------------------------------------------------
def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    # Target datasets
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

    # Model & loss (source domain)
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

    config = {'num_segments': args.num_segments, 'len_feature': args.len_feature,
              'device': device, 'gt_file': ''}

    # ---- Zero‑shot ----
    print(f"\n[INFO] Zero‑shot ({args.model_type.upper()}): {args.source.upper()} -> {args.target.upper()}")
    zs_auc, zs_ap, zs_acc = evaluate_with_official_fn(model, test_loader, device, args.target, config=config)
    print(f"Zero‑shot AUC: {zs_auc:.4f}  AP: {zs_ap:.4f}  ACC: {zs_acc:.4f}")

    # ---- Few‑shot adaptation ----
    support_loader, support_indices = build_support_loader(target_train, args.support_k)
    print(f"[INFO] Adaptation ({args.model_type.upper()}) on {len(support_indices)} support videos "
          f"(lr={args.inner_lr}, steps={args.adapt_steps})")

    if args.model_type == 'maml':
        fast_w = inner_adapt_maml(model, support_loader, args.inner_lr, args.adapt_steps, device, criterion)
        # Evaluate with fast weights temporarily loaded
        original_state = model.state_dict()
        model.load_state_dict(fast_w)
        fs_auc, fs_ap, fs_acc = evaluate_with_official_fn(model, test_loader, device, args.target, config=config)
        model.load_state_dict(original_state)
    else:  # erm
        adapt_erm(model, support_loader, args.inner_lr, args.adapt_steps, device, criterion)
        fs_auc, fs_ap, fs_acc = evaluate_with_official_fn(model, test_loader, device, args.target, config=config)

    print(f"Few‑shot AUC: {fs_auc:.4f}  AP: {fs_ap:.4f}  ACC: {fs_acc:.4f}")

    # ---- Save report ----
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = {
        "timestamp": timestamp,
        "args": vars(args),
        "source": args.source, "target": args.target,
        "checkpoint": args.checkpoint,
        "support_indices": support_indices,
        "zero_shot": {"auc": zs_auc, "ap": zs_ap, "acc": zs_acc},
        "few_shot": {"auc": fs_auc, "ap": fs_ap, "acc": fs_acc}
    }
    fname = os.path.join(args.output_dir,
                         f"cross_{args.source}2{args.target}_{args.model_type}_{timestamp}.json")
    with open(fname, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport saved to {fname}")

    # ---- Summary ----
    print("\n" + "="*60)
    print(f"Cross‑transfer ({args.model_type.upper()}): {args.source.upper()} -> {args.target.upper()}")
    print(f"{'Metric':<10} {'Zero‑shot':<12} {'Few‑shot':<12}")
    print("-"*34)
    for name, zs, fs in [('AUC', zs_auc, fs_auc), ('AP', zs_ap, fs_ap), ('ACC', zs_acc, fs_acc)]:
        print(f"{name:<10} {zs:<12.4f} {fs:<12.4f}")
    print("="*60)

if __name__ == '__main__':
    main()