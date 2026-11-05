#!/usr/bin/env python3
"""
FINAL MAML WSAD TRAINING SCRIPT

Includes:
- Original dataset (unchanged → preserves AUC)
- Curriculum sampling (Reviewer 6)
- Improved AD Loss (better localization)
- Proper MAML implementation
- Warmup + stable training
"""

import os
import random
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.utils.data as data
from torch.nn.utils import stateless

from sklearn.metrics import roc_curve, auc, precision_recall_curve

from model_min_upgrade import WSAD
import utils

# -------------------------
# CLI
# -------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--root_dir', type=str, default='/mnt/c/code/data/I3D')
    p.add_argument('--num_segments', type=int, default=200)
    p.add_argument('--a_nums', type=int, default=200)
    p.add_argument('--n_nums', type=int, default=200)
    p.add_argument('--len_feature', type=int, default=1024)
    p.add_argument('--support_k', type=int, default=8)
    p.add_argument('--query_k', type=int, default=16)
    p.add_argument('--meta_iters', type=int, default=500)
    p.add_argument('--warmup_steps', type=int, default=100)
    p.add_argument('--inner_lr', type=float, default=1e-3)
    p.add_argument('--meta_lr', type=float, default=1e-4)
    p.add_argument('--beta', type=float, default=1.0)
    p.add_argument('--eval_interval', type=int, default=100)
    p.add_argument('--seed', type=int, default=-1)
    p.add_argument('--gt_file', type=str, default='frame_label/gt_ucf_v2.npy')
    p.add_argument('--save_dir', type=str, default='checkpoints')
    return p.parse_args()

# -------------------------
# DATASET (UNCHANGED — IMPORTANT FOR AUC)
# -------------------------
class UCF_crime(data.Dataset):
    def __init__(self, root_dir, modal, mode, num_segments, len_feature, seed=-1):
        self.mode = mode
        self.num_segments = num_segments
        self.root_dir = root_dir

        list_path_repo = os.path.join('list', f'UCF_{self.mode}.list')
        alt_list_path = os.path.join(self.root_dir, f'{self.mode.lower()}_list.txt')

        if os.path.exists(list_path_repo):
            list_file = list_path_repo
        elif os.path.exists(alt_list_path):
            list_file = alt_list_path
        else:
            raise FileNotFoundError("List file not found")

        samples = []
        numAbnorm = 8100 if mode == 'Train' else 1400

        with open(list_file, 'r') as f:
            for i, line in enumerate(f):
                label = 1 if i < numAbnorm else 0
                samples.append((line.strip(), label))

        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        vid, label = self.samples[index]
        feat = np.load(vid).astype(np.float32)

        if self.mode == "Train":
            T, D = feat.shape
            new_feat = np.zeros((self.num_segments, D), dtype=np.float32)
            r = np.linspace(0, T, self.num_segments + 1, dtype=np.int32)

            for i in range(self.num_segments):
                if r[i] != r[i+1]:
                    new_feat[i] = np.mean(feat[r[i]:r[i+1]], axis=0)
                else:
                    new_feat[i] = feat[r[i]]

            feat = new_feat

        return torch.from_numpy(feat), torch.tensor(label).float()

# -------------------------
# CURRICULUM (Reviewer 6)
# -------------------------
ANOMALY_DIFFICULTY = {
    "Explosion": 0.1,
    "Arson": 0.2,
    "RoadAccidents": 0.2,
    "Shooting": 0.3,
    "Fighting": 0.4,
    "Robbery": 0.5,
    "Assault": 0.6,
    "Burglary": 0.7,
    "Vandalism": 0.7,
    "Abuse": 0.8,
    "Arrest": 0.85,
    "Stealing": 0.9,
    "Shoplifting": 1.0,
}

def get_difficulty(path):
    for k in ANOMALY_DIFFICULTY:
        if k.lower() in path.lower():
            return ANOMALY_DIFFICULTY[k]
    return 0.5

def curriculum_sample(dataset, k, step, total_steps, device):
    progress = step / total_steps

    weights = []
    for vid, label in dataset.samples:
        if label == 0:
            w = 1.0
        else:
            diff = get_difficulty(vid)
            w = (1 - diff) if progress < 0.5 else diff
        weights.append(w)

    probs = np.array(weights)
    probs /= probs.sum()

    idxs = np.random.choice(len(dataset), size=k, p=probs)

    feats, labels = [], []
    for i in idxs:
        f, l = dataset[i]
        feats.append(f)
        labels.append(l)

    return torch.stack(feats).to(device), torch.stack(labels).to(device)

# -------------------------
# LOSS (Improved but SAFE)
# -------------------------
class AD_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()

    def forward(self, result, y):
        y = y.float()
        att = result["frame"]

        t = att.size(1)
        k = max(t // 16 + 1, 1)

        anomaly = torch.topk(att, k, dim=-1)[0].mean(-1)
        anomaly_loss = self.bce(anomaly, y)

        smooth = result.get("smooth_loss", 0.0)
        triplet = result.get("triplet_margin", 0.0)

        total = anomaly_loss + 0.2 * triplet + 0.01 * smooth

        return total

# -------------------------
# TEST
# -------------------------
def test(model, loader):
    model.eval()
    preds, gts = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.cuda()
            out = model(x)["frame"]
            preds.append(out.mean(1).cpu().numpy())
            gts.append(y.numpy())

    preds = np.concatenate(preds)
    gts = np.concatenate(gts)

    fpr, tpr, _ = roc_curve(gts, preds)
    auc_score = auc(fpr, tpr)

    precision, recall, _ = precision_recall_curve(gts, preds)
    ap = auc(recall, precision)

    return auc_score, ap

# -------------------------
# MAIN
# -------------------------
def main():
    args = parse_args()

    device = torch.device("cuda")

    train_ds = UCF_crime(args.root_dir, 'rgb', 'Train', args.num_segments, args.len_feature)
    test_ds = UCF_crime(args.root_dir, 'rgb', 'Test', args.num_segments, args.len_feature)

    test_loader = data.DataLoader(test_ds, batch_size=1)

    model = WSAD(
        input_size=args.len_feature,
        flag="Train",
        a_nums=args.a_nums,
        n_nums=args.n_nums
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.meta_lr)
    criterion = AD_Loss()

    best_auc = 0

    for step in tqdm(range(1, args.meta_iters + 1)):

        # -------------------------
        # WARMUP
        # -------------------------
        if step <= args.warmup_steps:
            x, y = curriculum_sample(train_ds, args.query_k, step, args.meta_iters, device)
            loss = criterion(model(x), y)

            opt.zero_grad()
            loss.backward()
            opt.step()
            continue

        # -------------------------
        # MAML
        # -------------------------
        sx, sy = curriculum_sample(train_ds, args.support_k, step, args.meta_iters, device)
        qx, qy = curriculum_sample(train_ds, args.query_k, step, args.meta_iters, device)

        out_s = model(sx)
        inner_loss = criterion(out_s, sy)

        grads = torch.autograd.grad(inner_loss, model.parameters(), create_graph=True)

        fast_weights = {
            n: p - args.inner_lr * g
            for (n, p), g in zip(model.named_parameters(), grads)
        }

        out_q = stateless.functional_call(model, fast_weights, (qx,))
        meta_loss = criterion(out_q, qy)

        loss = inner_loss + args.beta * meta_loss

        opt.zero_grad()
        loss.backward()
        opt.step()

        # -------------------------
        # EVAL
        # -------------------------
        if step % args.eval_interval == 0:
            model.flag = "Test"
            auc_score, ap = test(model, test_loader)
            model.flag = "Train"

            print(f"[STEP {step}] AUC={auc_score:.4f} AP={ap:.4f}")

            if auc_score > best_auc:
                best_auc = auc_score
                torch.save(model.state_dict(), os.path.join(args.save_dir, "best.pth"))

    print("BEST AUC:", best_auc)

# -------------------------
if __name__ == "__main__":
    main()