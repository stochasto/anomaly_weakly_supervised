#!/usr/bin/env python3
"""
Unified Meta-Learning Training Script for UR-DMU (UCF-Crime)
- Combines main, options, config, and meta-train
- Evaluates AUC every 5 iterations
- Saves best checkpoint automatically
- Fully stochastic (no fixed seed)
"""

import os
import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc, precision_recall_curve

# ============================================================
# OPTIONS (replaces options.py)
# ============================================================
import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Meta-Learning UR-DMU Training")
    parser.add_argument("--output_path", type=str, default="outputs/")
    parser.add_argument("--root_dir", type=str, default="outputs/")
    parser.add_argument("--log_path", type=str, default="logs/")
    parser.add_argument("--modal", type=str, default="rgb", choices=["rgb", "flow", "both"])
    parser.add_argument("--model_path", type=str, default="models/")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--meta_lr", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--num_segments", type=int, default=32)
    parser.add_argument("--num_iters", type=int, default=1000)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


# ============================================================
# CONFIG (replaces config.py)
# ============================================================
class Config:
    def __init__(self, args):
        self.root_dir = args.root_dir
        self.modal = args.modal
        self.output_path = args.output_path
        self.model_path = args.model_path
        self.num_workers = args.num_workers
        self.batch_size = args.batch_size
        self.num_segments = args.num_segments
        self.num_iters = args.num_iters
        self.lr = args.lr
        self.meta_lr = args.meta_lr
        self.len_feature = 2048


# ============================================================
# LOSS (simplified from train.py)
# ============================================================
class AD_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()

    def forward(self, result, _label):
        _label = _label.float()
        att = result["frame"]
        b = _label.size(0) // 2
        t = att.size(1)
        anomaly = torch.topk(att, t // 16 + 1, dim=-1)[0].mean(-1)
        anomaly_loss = self.bce(anomaly, _label)
        return anomaly_loss, {"total_loss": anomaly_loss}


# ============================================================
# META-LEARNING TRAINING LOOP (replaces meta_train)
# ============================================================
from torch.nn.utils import stateless

def meta_train_step(model, optimizer, normal_loader, abnormal_loader, criterion, beta=0.1):
    """One MAML-style meta-training step."""
    model.train()
    model.flag = "Train"

    # Sample batches
    ninput, nlabel = next(normal_loader)
    ainput, alabel = next(abnormal_loader)
    support_x = torch.cat((ninput, ainput), 0).cuda()
    support_y = torch.cat((nlabel, alabel), 0).cuda()

    # Inner loop (fast adaptation)
    result = model(support_x)
    loss, _ = criterion(result, support_y)
    grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
    fast_weights = {name: p - 5e-5 * g for (name, p), g in zip(model.named_parameters(), grads)}

    # Validation batch for meta-update
    ninput, nlabel = next(normal_loader)
    ainput, alabel = next(abnormal_loader)
    query_x = torch.cat((ninput, ainput), 0).cuda()
    query_y = torch.cat((nlabel, alabel), 0).cuda()

    result_query = stateless.functional_call(model, fast_weights, (query_x,))
    meta_val_loss, _ = criterion(result_query, query_y)

    total_loss = loss + beta * meta_val_loss
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return total_loss.item()


# ============================================================
# DATALOADER CREATOR
# ============================================================
def create_dataloader(dataset, batch_size, shuffle, num_workers, drop_last=False):
    loader_kwargs = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=True,
    )
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2
        loader_kwargs["persistent_workers"] = True
    return data.DataLoader(dataset, **loader_kwargs)


# ============================================================
# MAIN TRAINING SCRIPT
# ============================================================
def main():
    from dataset_loader import UCF_crime
    from model import WSAD
    from ucf_test import test

    args = parse_args()
    config = Config(args)

    # No fixed seed: full randomness
    print("[INFO] Training is fully stochastic (no seed).")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = WSAD(config.len_feature, flag="Train", a_nums=60, n_nums=60).to(device)
    print(f"[INFO] Model initialized on {device}")

    # Dataloaders
    normal_train_loader = create_dataloader(
        UCF_crime(root_dir=config.root_dir, mode='Train', modal=config.modal,
                  num_segments=200, len_feature=config.len_feature, is_normal=True),
        batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, drop_last=True
    )

    abnormal_train_loader = create_dataloader(
        UCF_crime(root_dir=config.root_dir, mode='Train', modal=config.modal,
                  num_segments=200, len_feature=config.len_feature, is_normal=False),
        batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, drop_last=True
    )

    test_loader = create_dataloader(
        UCF_crime(root_dir=config.root_dir, mode='Test', modal=config.modal,
                  num_segments=config.num_segments, len_feature=config.len_feature),
        batch_size=1, shuffle=False, num_workers=config.num_workers
    )

    # Optimizer and loss
    criterion = AD_Loss()
    optimizer = torch.optim.Adam(net.parameters(), lr=config.meta_lr, betas=(0.9, 0.999), weight_decay=5e-5)

    os.makedirs(config.model_path, exist_ok=True)
    os.makedirs(config.output_path, exist_ok=True)

    model_file = os.path.join(config.model_path, "ucf_meta_best.pkl")
    test_info = {"step": [], "auc": [], "ap": [], "ac": []}
    best_auc = 0.0

    normal_iter = iter(normal_train_loader)
    abnormal_iter = iter(abnormal_train_loader)

    print("[INFO] Starting Meta-Learning Training...")

    for step in tqdm(range(1, config.num_iters + 1), total=config.num_iters, dynamic_ncols=True):
        try:
            loss_value = meta_train_step(net, optimizer, normal_iter, abnormal_iter, criterion)
        except StopIteration:
            normal_iter = iter(normal_train_loader)
            abnormal_iter = iter(abnormal_train_loader)
            loss_value = meta_train_step(net, optimizer, normal_iter, abnormal_iter, criterion)

        #if step % 50 == 0:
        print(f"[STEP {step}] Meta-Loss: {loss_value:.6f}")

        # Evaluate every 5 iterations
        if step % 5 == 0 and step > 1:
            auc_score, ap_score, acc = test(net, config, test_loader, test_info, step)
            if auc_score > best_auc:
                best_auc = auc_score
                best_ckpt = 'dm_meta_best_auc_' + str(best_auc) + '.pkl'
                best_ckpt = os.path.join(config.model_path, best_ckpt)
                torch.save(net.state_dict(), best_ckpt)
                print(f"[INFO] ✅ New best AUC: {best_auc:.4f} at step {step}")

    print(f"[INFO] Training complete. Best AUC: {best_auc:.4f}")
    torch.save(net.state_dict(), os.path.join(config.model_path, "ucf_meta_final.pkl"))


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    main()
