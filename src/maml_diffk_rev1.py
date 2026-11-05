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

from model_no_segment import WSAD


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
    p.add_argument('--eval_interval', type=int, default=100)

    p.add_argument('--first_order', action='store_true')
    p.add_argument('--use_amp', action='store_true')

    p.add_argument('--max_chunks_per_video', type=int, default=4)

    p.add_argument('--save_dir', type=str, default='checkpoints')

    return p.parse_args()


# -------------------------
# TRAIN DATASET (chunked)
# -------------------------
class TrainDataset(data.Dataset):
    def __init__(self, root_dir, max_len, max_chunks):
        self.samples = []
        self.max_len = max_len
        self.max_chunks = max_chunks

        list_file = os.path.join("list", "UCF_Train.list")

        print("[INFO] Building index...", flush=True)

        with open(list_file) as f:
            lines = f.readlines()

        numAbnorm = 8100

        for i, line in enumerate(lines):
            path = line.strip()
            label = 1 if i < numAbnorm else 0

            if os.path.exists(path):
                self.samples.append((path, label))

        print(f"[INFO] Indexed {len(self.samples)} videos", flush=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        feat = np.load(path, mmap_mode='r')  # 🔥 CRITICAL: no full load
        T = feat.shape[0]

        # ---- chunk sampling ----
        chunk_positions = list(range(0, T, self.max_len))
        
        # limit chunks per video
        if len(chunk_positions) > self.max_chunks:
            chunk_positions = sorted(np.random.choice(chunk_positions, self.max_chunks, replace=False))
        
        # pick ONE chunk randomly (important for batching)
        start = chunk_positions[np.random.randint(len(chunk_positions))]
        end = min(start + self.max_len, T)
        chunk = feat[start:end].astype(np.float32)

        return chunk, label

# -------------------------
# TEST DATASET (video-wise)
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
    
    # Handle both TrainDataset (returns (chunk, label)) and raw data
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
# LOSS
# -------------------------
class AD_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()

    def forward(self, result, y, mask):
        att = result["frame"]

        scores = []
        for i in range(att.size(0)):
            valid = att[i][mask[i] > 0]
            if valid.numel() == 0:
                scores.append(torch.tensor(0.0, device=att.device))
                continue

            k = max(valid.size(0)//16, 1)
            scores.append(torch.topk(valid, k)[0].mean())

        anomaly = torch.stack(scores)

        loss = self.bce(anomaly, y)

        loss += 0.2 * result.get("triplet_margin", 0)
        loss += 0.01 * result.get("smooth_loss", 0)

        return loss


# -------------------------
# TEST
# -------------------------
def test(model, dataset, device, gt_file, max_len):
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
                out = model(chunk)["frame"].squeeze(0).cpu().numpy()
                outs.append(out)

            out = np.concatenate(outs)

            frame_gt = gt[idx:idx+len(out)]
            if len(frame_gt) < len(out):
                break

            preds.append(out)
            gts.append(frame_gt)

            idx += len(out)

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
    print("START", flush=True)

    os.makedirs(args.save_dir, exist_ok=True)

    train_ds = TrainDataset(args.root_dir, args.max_len, args.max_chunks_per_video)
    test_ds = TestDataset(args.root_dir)
    
    print(f"[INFO] Train chunks: {len(train_ds)}")
    print(f"[INFO] Test videos: {len(test_ds)}")

    model = WSAD(input_size=1024, flag="Train", a_nums=args.a_nums, n_nums=args.n_nums).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.meta_lr)
    criterion = AD_Loss()

    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

    best_auc = 0

    for step in tqdm(range(1, args.meta_iters + 1)):

        idxs = np.random.choice(len(train_ds), args.support_k + args.query_k)

        support = [train_ds[i] for i in idxs[:args.support_k]]
        query = [train_ds[i] for i in idxs[args.support_k:]]

        sx, sy, sm = collate_fn(support, args.max_len, device)
        qx, qy, qm = collate_fn(query, args.max_len, device)

        with torch.cuda.amp.autocast(enabled=args.use_amp):

            inner_out = model(sx)
            inner_loss = criterion(inner_out, sy, sm)

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
            out_q = functional_call(model, fast_weights, (qx,))
            meta_loss = criterion(out_q, qy, qm)

            loss = inner_loss + args.beta * meta_loss

        opt.zero_grad()
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(opt)
        scaler.update()

        if step % args.eval_interval == 0:
            model.flag = "Test"
            auc_score, ap = test(model, test_ds, device, args.gt_file, args.max_len)
            model.flag = "Train"

            print(f"[STEP {step}] AUC={auc_score:.4f} AP={ap:.4f}")

            if auc_score > best_auc:
                best_auc = auc_score
                torch.save(model.state_dict(),
                           os.path.join(args.save_dir, f"best_{best_auc:.4f}.pth"))

    print("BEST AUC:", best_auc)


if __name__ == "__main__":
    main()