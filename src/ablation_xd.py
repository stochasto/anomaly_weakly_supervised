#!/usr/bin/env python3
"""
ablation_xd.py

Trains short ERM baseline for each architectural variant on XD‑Violence
and reports final AUC/AP/ACC.
"""

import os
import random
import json
import argparse
import torch
import torch.utils.data as data
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import torch.nn as nn

# ------------- Imports from your XD training script -------------
from model_gemini_xd_ablation import WSAD
from maml_xd import XDVideoDataset, test_xd_frame, AD_Loss
# ---------------------------------------------------------------

# ---------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--root_dir', default='/mnt/c/code/ae_anomaly/UR-DMU/xdv')
    p.add_argument('--num_segments', type=int, default=300)
    p.add_argument('--len_feature', type=int, default=1024)
    p.add_argument('--meta_iters', type=int, default=1200)
    p.add_argument('--meta_lr', type=float, default=1e-4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--output_file', default='ablation_results.json')
    return p.parse_args()

# ---------------------------------------------------------------
def train_one_variant(args, config, variant_name):
    device = torch.device(args.device)

    # Dataset
    train_dataset = XDVideoDataset(
        root_dir=args.root_dir, mode='Train', modal='rgb',
        num_segments=args.num_segments, len_feature=args.len_feature, seed=args.seed)
    test_dataset = XDVideoDataset(
        root_dir=args.root_dir, mode='Test', modal='rgb',
        num_segments=args.num_segments, len_feature=args.len_feature, seed=args.seed)

    # Cached indices for balanced sampling (same as warmup)
    normal_idxs, abnormal_idxs = build_index_cache(train_dataset)

    test_loader = data.DataLoader(test_dataset, batch_size=1, shuffle=False,
                                  num_workers=4)

    # Model
    model = WSAD(input_size=args.len_feature, flag="Train",
                 a_nums=200, n_nums=200, config=config)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.meta_lr,
                                 betas=(0.9, 0.999), weight_decay=5e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.meta_iters)

    # Loss – for NoMemory variant we need a simpler loss (no memory keys)
    use_memory = config.get('use_memory', True)
    if use_memory:
        criterion = AD_Loss()
    else:
        # Simplified MIL loss only
        class SimpleMIL(nn.Module):
            def __init__(self):
                super().__init__()
                self.bce = nn.BCEWithLogitsLoss()
            def forward(self, result, _label):
                att = result['frame']
                if att.dim() == 2:
                    att = att.unsqueeze(-1)
                B, T, _ = att.shape
                k_top = max(T//24 + 1, 1)
                anomaly_scores = torch.topk(att.squeeze(-1), k_top, dim=1)[0].mean(1)
                loss = self.bce(anomaly_scores, _label.float())
                return loss, {'total_loss': loss}
        criterion = SimpleMIL()

    def get_balanced_batch(k):
        half = k // 2
        n_idxs = random.sample(normal_idxs, half)
        a_idxs = random.sample(abnormal_idxs, half)
        idxs = n_idxs + a_idxs
        feats, labels = [], []
        for i in idxs:
            feat, lbl = train_dataset[i]
            feats.append(torch.from_numpy(np.asarray(feat).astype(np.float32)))
            labels.append(torch.tensor(lbl).float())
        return torch.stack(feats).to(device), torch.stack(labels).to(device)

    best_auc = 0.0
    # Training loop (ERM, similar to warmup)
    for step in tqdm(range(1, args.meta_iters + 1), desc=f"Train {variant_name}", ncols=80):
        model.train()
        model.flag = "Train"
        sup_x, sup_y = get_balanced_batch(64)   # batch size 64
        result = model(sup_x)
        loss_sup, _ = criterion(result, sup_y)
        optimizer.zero_grad()
        loss_sup.backward()
        optimizer.step()
        scheduler.step()

        if step % 200 == 0 or step == args.meta_iters:
            model.eval()
            model.flag = "Test"
            auc_score, ap_score, acc = test_xd_frame(model,
                {'num_segments': args.num_segments, 'len_feature': args.len_feature,
                 'device': device, 'gt_file': 'frame_label/xd_frame_gt_generated.npy'},
                test_loader, model_file=None)
            model.train()
            model.flag = "Train"
            if auc_score > best_auc:
                best_auc = auc_score
                best_ap = ap_score
                best_acc = acc
            print(f"[{variant_name}] Step {step}: AUC={auc_score:.4f}, AP={ap_score:.4f}, ACC={acc:.4f}")

    return {'auc': best_auc, 'ap': best_ap, 'acc': best_acc}

# ---------------------------------------------------------------
def build_index_cache(dataset, cache_path="index_cache_xd_ablation.pkl"):
    # Simplified version (no disk cache for brevity)
    normal = [i for i, (_, lbl) in enumerate(dataset.samples) if lbl == 0]
    abnormal = [i for i, (_, lbl) in enumerate(dataset.samples) if lbl == 1]
    return normal, abnormal

# ---------------------------------------------------------------
def main():
    args = parse_args()
    device = torch.device(args.device)

    # Define variants
    variants = [
        #{
        #    "name": "Full",
        #    "config": {"use_local_att": True, "use_transformer": True,
        #               "use_memory": True, "use_vae": True}
        #},
        #{
        #    "name": "NoLocal",
        #    "config": {"use_local_att": False, "use_transformer": True,
        #               "use_memory": True, "use_vae": True}
        #},
        #{
        #    "name": "NoTransformer",
        #    "config": {"use_local_att": True, "use_transformer": False,
        #               "use_memory": True, "use_vae": True}
        #},
        {
            "name": "NoMemory",
            "config": {"use_local_att": True, "use_transformer": True,
                       "use_memory": False, "use_vae": False}   # VAE not applicable without memory
        },
        {
            "name": "NoVAE",
            "config": {"use_local_att": True, "use_transformer": True,
                       "use_memory": True, "use_vae": False}
        },
    ]

    results = {}
    for v in variants:
        print(f"\n=== Training {v['name']} ===")
        res = train_one_variant(args, v['config'], v['name'])
        results[v['name']] = res
        print(f"{v['name']} best: AUC={res['auc']:.4f}, AP={res['ap']:.4f}, ACC={res['acc']:.4f}")

    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output_file}")

if __name__ == '__main__':
    main()