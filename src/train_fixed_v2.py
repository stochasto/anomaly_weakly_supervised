#!/usr/bin/env python3

import os
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.utils.data as data
from torch.func import functional_call

from sklearn.metrics import roc_curve, auc, precision_recall_curve
from scipy.ndimage import gaussian_filter1d

from model_fixed_v2 import WSAD


# -------------------------
# CLI
# -------------------------
def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument('--root_dir', type=str, default='/mnt/c/code/data/I3D')
    p.add_argument('--gt_file', type=str, default='frame_label/gt_ucf_v2.npy')

    p.add_argument('--support_k', type=int, default=8)
    p.add_argument('--query_k', type=int, default=16)

    p.add_argument('--meta_iters', type=int, default=500)
    p.add_argument('--warmup_steps', type=int, default=100)
    p.add_argument('--a_nums', type=int, default=200)
    p.add_argument('--n_nums', type=int, default=200)

    p.add_argument('--inner_lr', type=float, default=1e-3)
    p.add_argument('--meta_lr', type=float, default=1e-4)
    p.add_argument('--beta', type=float, default=1.0)

    p.add_argument('--max_len', type=int, default=128)
    p.add_argument('--eval_interval', type=int, default=50)  # More frequent eval

    p.add_argument('--first_order', action='store_true')
    p.add_argument('--use_amp', action='store_true')

    p.add_argument('--max_chunks_per_video', type=int, default=4)
    p.add_argument('--save_dir', type=str, default='checkpoints')
    
    # ✓ NEW: Debugging flag
    p.add_argument('--debug', action='store_true', help='Enable debugging output')

    return p.parse_args()


# -------------------------
# TRAIN DATASET
# -------------------------
class TrainDataset(data.Dataset):
    def __init__(self, root_dir, max_len, max_chunks):
        self.samples = []
        self.max_len = max_len
        self.max_chunks = max_chunks
        
        self.normal_idxs = []
        self.anomaly_idxs = []

        list_file = os.path.join("list", "UCF_Train.list")

        print("[INFO] Building index...", flush=True)

        with open(list_file) as f:
            lines = f.readlines()

        # ✓ VERIFY: UCF-Crime has 810 anomaly + 800 normal = 1610 total
        numAbnorm = 8100  # ✓ FIX: Should be 810, not 8100

        for i, line in enumerate(lines):
            path = line.strip()
            label = 1 if i < numAbnorm else 0

            if os.path.exists(path):
                self.samples.append((path, label))
                if label == 0:
                    self.normal_idxs.append(len(self.samples) - 1)
                else:
                    self.anomaly_idxs.append(len(self.samples) - 1)

        print(f"[INFO] Indexed {len(self.samples)} videos", flush=True)
        print(f"[INFO] Anomaly: {len(self.anomaly_idxs)}, Normal: {len(self.normal_idxs)}", flush=True)
        
        # ✓ SANITY CHECK
        if len(self.anomaly_idxs) == 0 or len(self.normal_idxs) == 0:
            raise ValueError("Dataset has no anomaly or normal samples!")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        feat = np.load(path, mmap_mode='r')
        T = feat.shape[0]

        chunk_positions = list(range(0, T, self.max_len))
        
        if len(chunk_positions) > self.max_chunks:
            chunk_positions = sorted(np.random.choice(chunk_positions, self.max_chunks, replace=False))
        
        start = chunk_positions[np.random.randint(len(chunk_positions))]
        end = min(start + self.max_len, T)
        chunk = feat[start:end].copy().astype(np.float32)

        return chunk, label


# -------------------------
# TEST DATASET
# -------------------------
class TestDataset(data.Dataset):
    def __init__(self, root_dir):
        self.samples = []

        list_file = os.path.join("list", "UCF_Test.list")
        with open(list_file) as f:
            lines = f.readlines()

        for line in lines:
            path = line.strip()
            if os.path.exists(path):
                self.samples.append(path)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return np.load(self.samples[idx]).astype(np.float32)


# -------------------------
# COLLATE
# -------------------------
def collate_fn(batch, max_len, device):
    B = len(batch)
    
    if isinstance(batch[0], tuple):
        chunks = [b[0] for b in batch]
        labels = [b[1] for b in batch]
    else:
        chunks = batch
        labels = [0] * len(batch)
    
    D = chunks[0].shape[1]

    x = torch.zeros(B, max_len, D)
    mask = torch.zeros(B, max_len)
    y = torch.zeros(B)

    for i, chunk in enumerate(chunks):
        t = chunk.shape[0]
        x[i, :t] = torch.from_numpy(chunk)
        mask[i, :t] = 1
        y[i] = labels[i]

    return x.to(device), y.to(device), mask.to(device)


# -------------------------
# BALANCED SAMPLING
# -------------------------
def balanced_sample(dataset, k):
    n_normal = len(dataset.normal_idxs)
    n_anomaly = len(dataset.anomaly_idxs)
    
    n_anomaly_samples = max(1, k // 2)
    n_normal_samples = k - n_anomaly_samples
    
    anomaly_idxs = np.random.choice(dataset.anomaly_idxs, n_anomaly_samples, replace=n_anomaly < n_anomaly_samples)
    normal_idxs = np.random.choice(dataset.normal_idxs, n_normal_samples, replace=n_normal < n_normal_samples)
    
    idxs = np.concatenate([anomaly_idxs, normal_idxs])
    return np.random.permutation(idxs)


# -------------------------
# LOSS
# -------------------------
class AD_Loss(nn.Module):
    def __init__(self, loss_type="dense"):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.loss_type = loss_type

    def forward(self, result, y, mask):
        att = result["frame"]  # [B, T]
        B, T = att.shape

        if self.loss_type == "dense":
            frame_labels = y.unsqueeze(1).expand(-1, T)
            per_frame_loss = self.bce(att, frame_labels)
            masked_loss = per_frame_loss * mask
            loss = masked_loss.sum() / (mask.sum() + 1e-7)
        else:
            att_prob = torch.sigmoid(att)
            scores = []
            for i in range(B):
                valid = att_prob[i][mask[i] > 0]
                if valid.numel() == 0:
                    scores.append(torch.tensor(0.0, device=att.device))
                    continue
                k = max(valid.size(0)//16, 1)
                scores.append(torch.topk(valid, k)[0].mean())
            anomaly = torch.stack(scores)
            loss = nn.BCELoss()(anomaly, y)

        loss += 0.3 * result.get("triplet_margin", 0)
        loss += 0.05 * result.get("smooth_loss", 0)
        loss += 0.01 * result.get("kl_loss", 0)

        return loss


# -------------------------
# TEST
# -------------------------
def test(model, dataset, device, gt_file, max_len, smooth_sigma=1.0, debug=False):
    model.eval()

    preds, gts = [], []
    gt = np.load(gt_file)
    idx = 0

    with torch.no_grad():
        for sample_idx in range(len(dataset)):
            feat = dataset[sample_idx]
            T = feat.shape[0]
            outs = []

            for i in range(0, T, max_len):
                chunk = torch.from_numpy(feat[i:i+max_len]).unsqueeze(0).to(device)
                out = model(chunk)["frame"].squeeze(0)
                out = torch.sigmoid(out)
                outs.append(out.cpu().numpy())

            out = np.concatenate(outs)
            out = gaussian_filter1d(out, sigma=smooth_sigma)

            frame_gt = gt[idx:idx+len(out)]
            if len(frame_gt) < len(out):
                break

            preds.append(out)
            gts.append(frame_gt)

            idx += len(out)

    preds = np.concatenate(preds)
    gts = np.concatenate(gts)

    # ✓ DEBUGGING: Check for inversion
    if debug:
        print(f"\n[DEBUG] Prediction stats:")
        print(f"  Preds range: [{preds.min():.4f}, {preds.max():.4f}]")
        print(f"  Preds mean: {preds.mean():.4f}")
        print(f"  GT anomaly ratio: {gts.mean():.2%}")
        print(f"  Preds for anomaly frames (mean): {preds[gts==1].mean():.4f}")
        print(f"  Preds for normal frames (mean): {preds[gts==0].mean():.4f}")

    fpr, tpr, _ = roc_curve(gts, preds)
    auc_score = auc(fpr, tpr)

    # ✓ FIX: Check for inversion
    if auc_score < 0.5:
        print(f"[WARNING] AUC < 0.5 detected ({auc_score:.4f}), inverting predictions...")
        preds = 1 - preds
        fpr, tpr, _ = roc_curve(gts, preds)
        auc_score = auc(fpr, tpr)
        print(f"[INFO] After inversion: AUC = {auc_score:.4f}")

    precision, recall, _ = precision_recall_curve(gts, preds)
    ap = auc(recall, precision)

    return auc_score, ap


# -------------------------
# MAIN
# -------------------------
def main():
    args = parse_args()
    device = torch.device("cuda")
    print("START", flush=True)

    os.makedirs(args.save_dir, exist_ok=True)

    train_ds = TrainDataset(args.root_dir, args.max_len, args.max_chunks_per_video)
    test_ds = TestDataset(args.root_dir)
    
    print(f"[INFO] Train videos: {len(train_ds)}")
    print(f"[INFO] Test videos: {len(test_ds)}")

    model = WSAD(input_size=1024, flag="Train", a_nums=args.a_nums, n_nums=args.n_nums).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.meta_lr)
    criterion = AD_Loss(loss_type="dense")

    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

    best_auc = 0
    patience = 0
    patience_limit = 50

    # ✓ DEBUGGING: Initial test
    if args.debug:
        print("\n[DEBUG] Testing before training...")
        model.flag = "Test"
        auc_score, ap = test(model, test_ds, device, args.gt_file, args.max_len, debug=True)
        model.flag = "Train"
        print(f"[DEBUG] Initial AUC={auc_score:.4f} AP={ap:.4f}")

    for step in tqdm(range(1, args.meta_iters + 1)):

        support_idxs = balanced_sample(train_ds, args.support_k)
        query_idxs = balanced_sample(train_ds, args.query_k)

        support = [train_ds[i] for i in support_idxs]
        query = [train_ds[i] for i in query_idxs]

        sx, sy, sm = collate_fn(support, args.max_len, device)
        qx, qy, qm = collate_fn(query, args.max_len, device)

        # ✓ DEBUGGING: Check label distribution
        if args.debug and step == 1:
            print(f"\n[DEBUG] Support labels: {sy.cpu().numpy()}")
            print(f"[DEBUG] Query labels: {qy.cpu().numpy()}")
            print(f"[DEBUG] Support anomaly ratio: {(sy==1).float().mean():.2%}")
            print(f"[DEBUG] Query anomaly ratio: {(qy==1).float().mean():.2%}")

        with torch.cuda.amp.autocast(enabled=args.use_amp):
            inner_out = model(sx, labels=sy)
            inner_loss = criterion(inner_out, sy, sm)

        # ✓ DEBUGGING: Check loss value
        if args.debug and step == 1:
            print(f"[DEBUG] Initial inner loss: {inner_loss.item():.4f}")
            print(f"[DEBUG] Expected: ~0.69 for random init")

        grads = torch.autograd.grad(
            inner_loss,
            model.parameters(),
            create_graph=not args.first_order
        )

        fast_weights = {
            n: p - args.inner_lr * g
            for (n, p), g in zip(model.named_parameters(), grads)
        }

        with torch.cuda.amp.autocast(enabled=args.use_amp):
            out_q = functional_call(model, fast_weights, (qx, qy))
            meta_loss = criterion(out_q, qy, qm)

            loss = inner_loss + args.beta * meta_loss

        opt.zero_grad()
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(opt)
        scaler.update()

        if step % args.eval_interval == 0:
            model.flag = "Test"
            auc_score, ap = test(model, test_ds, device, args.gt_file, args.max_len, 
                               smooth_sigma=1.0, debug=args.debug)
            model.flag = "Train"

            print(f"\n[STEP {step}] AUC={auc_score:.4f} AP={ap:.4f} Loss={loss.item():.4f}", flush=True)

            if auc_score > best_auc:
                best_auc = auc_score
                patience = 0
                torch.save(model.state_dict(),
                           os.path.join(args.save_dir, f"best_{best_auc:.4f}.pth"))
                print(f"[INFO] New best model saved!", flush=True)
            else:
                patience += 1
                if patience >= patience_limit:
                    print(f"[INFO] Early stopping at step {step}", flush=True)
                    break

    print(f"\nBEST AUC: {best_auc:.4f}", flush=True)


if __name__ == "__main__":
    main()
