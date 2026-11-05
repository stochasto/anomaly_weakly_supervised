#!/usr/bin/env python3

import os
import random
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.nn.utils import stateless
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import warnings
warnings.filterwarnings("ignore")

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

    p.add_argument('--meta_iters', type=int, default=4000)
    p.add_argument('--warmup_steps', type=int, default=100)

    p.add_argument('--inner_lr', type=float, default=1e-3)
    p.add_argument('--meta_lr', type=float, default=1e-4)

    # 🔧 FIX: reduced beta (prevents query domination)
    p.add_argument('--beta', type=float, default=0.3)

    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--save_dir', type=str, default='checkpoints')
    p.add_argument('--gt_file', type=str, default='frame_label/gt_ucf_single_crop.npy')

    # 🔧 FIX: less frequent eval (reduce noise)
    p.add_argument('--eval_interval', type=int, default=200)

    p.add_argument('--seed', type=int, default=-1)
    return p.parse_args()

# -------------------------
# DATASET (UNCHANGED)
# -------------------------
class UCF_crime(data.Dataset):
    def __init__(self, root_dir, modal, mode, num_segments, len_feature, seed=-1):
        if seed >= 0 and hasattr(utils, "set_seed"):
            utils.set_seed(seed)

        self.mode = mode
        self.num_segments = num_segments
        self.len_feature = len_feature
        self.root_dir = root_dir

        list_path_repo = os.path.join('list', f'UCF_{self.mode}.list')
        with open(list_path_repo, 'r') as f:
            lines = [l.strip() for l in f if "__" not in os.path.basename(l.strip())]

        self.samples = [(p, 0 if "Normal" in p else 1) for p in lines]

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
                    new_feat[i] = feat[r[i]:r[i+1]].mean(axis=0)
                else:
                    new_feat[i] = feat[r[i]]
            feat = new_feat

        return feat, label

# -------------------------
# SAMPLER
# -------------------------
def get_random_batch_from_dataset(dataset, k, device):
    idxs = random.sample(range(len(dataset)), k)
    feats, labels = [], []
    for i in idxs:
        f, l = dataset[i]
        feats.append(torch.from_numpy(f))
        labels.append(torch.tensor(l).float())
    return torch.stack(feats).to(device), torch.stack(labels).to(device)

# -------------------------
# TEST (UNCHANGED)
# -------------------------
def test(net, config, test_loader, step):
    with torch.no_grad():
        net.eval()
        net.flag = "Test"

        frame_gt = np.load(config['gt_file'])
        frame_predict = None

        for data_, label_ in test_loader:
            data_ = data_.cuda()
            pred = net(data_)["frame"].squeeze(0).cpu().numpy()

            if frame_predict is None:
                frame_predict = pred
            else:
                frame_predict = np.concatenate([frame_predict, pred])

        fpr, tpr, _ = roc_curve(frame_gt, frame_predict)
        return auc(fpr, tpr), 0, 0

# -------------------------
# LOSS (kept intact, only safe clamp added)
# -------------------------
class AD_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()

    def forward(self, result, label, step):
        att = result['frame']

        # 🔧 FIX: clamp prevents saturation collapse
        att = torch.clamp(att, 0.01, 0.99)

        b = max(label.size(0)//2, 1)
        t = max(att.size(1), 1)
        topk = max(t//64, 1)

        anomaly = torch.topk(att, topk, dim=-1)[0].mean(-1)
        anomaly_loss = self.bce(anomaly, label)

        triplet = result.get("triplet_margin", 0)
        kl = result.get("kl_loss", 0)
        dist = result.get("distance", 0)
        smooth = result.get("smooth_loss", 0)

        sparsity = att.mean()

        total = (
            anomaly_loss
            + 0.3 * triplet
            + 0.005 * kl
            + 0.0001 * dist
            + 0.05 * smooth
            + 0.1 * sparsity
        )

        return total, {"total_loss": total}

# -------------------------
# MAIN
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

    print("[INFO] Loading datasets...")
    train_dataset = UCF_crime(root_dir=args.root_dir, modal='rgb', mode='Train',
                             num_segments=args.num_segments, len_feature=args.len_feature, seed=args.seed)

    test_dataset = UCF_crime(root_dir=args.root_dir, modal='rgb', mode='Test',
                            num_segments=args.num_segments, len_feature=args.len_feature, seed=args.seed)

    print("[INFO] Initializing model...")
    model = WSAD(input_size=args.len_feature, flag="Train",
                 a_nums=args.a_nums, n_nums=args.n_nums).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.meta_lr, weight_decay=5e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.meta_iters)

    criterion = AD_Loss()

    best_auc = 0.0

    print("[INFO] Training started")

    for step in tqdm(range(1, args.meta_iters + 1), ncols=100):

        model.train()

        # ============================================================
        # ✅ WARMUP (UNCHANGED LOGIC, just clean)
        # ============================================================
        if step <= args.warmup_steps:
            x, y = get_random_batch_from_dataset(train_dataset,
                                                 max(args.support_k, args.query_k),
                                                 device)

            result = model(x)
            loss, _ = criterion(result, y, step)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if step % 20 == 0:
                print(f"[WARMUP {step}] loss={loss.item():.4f}")

            continue

        # ============================================================
        # ✅ SUPPORT SET
        # ============================================================
        support_x, support_y = get_random_batch_from_dataset(
            train_dataset,
            args.support_k,
            device
        )

        result_support = model(support_x)
        inner_loss, _ = criterion(result_support, support_y, step)

        # ============================================================
        # ✅ FIX 1: FILTER ONLY TRAINABLE PARAMS
        # ============================================================
        named_params = [(n, p) for (n, p) in model.named_parameters() if p.requires_grad]
        params = [p for (_, p) in named_params]

        # ============================================================
        # ✅ FIX 2: SAFE GRAD COMPUTATION
        # ============================================================
        grads = torch.autograd.grad(
            inner_loss,
            params,
            create_graph=True,
            allow_unused=True   # CRITICAL FIX
        )

        # ============================================================
        # ✅ FIX 3: SAFE FAST WEIGHTS (HANDLE None GRADS)
        # ============================================================
        fast_weights = {}

        for (name, param), grad in zip(named_params, grads):

            if grad is None:
                fast_weights[name] = param
            else:
                # ✅ FIX 4: REDUCED INNER STEP (STABILITY)
                fast_weights[name] = param - 0.5 * args.inner_lr * grad

        # ============================================================
        # ✅ QUERY SET
        # ============================================================
        query_x, query_y = get_random_batch_from_dataset(
            train_dataset,
            args.query_k,
            device
        )

        # ============================================================
        # ✅ FUNCTIONAL FORWARD (MAML)
        # ============================================================
        result_query = stateless.functional_call(model, fast_weights, (query_x,))
        meta_val_loss, _ = criterion(result_query, query_y, step)

        # ============================================================
        # ✅ FIX 5: STABILIZED META LOSS (PREVENT COLLAPSE)
        # ============================================================
        total_loss = 0.3 * inner_loss + args.beta * meta_val_loss

        optimizer.zero_grad()
        total_loss.backward()

        # ============================================================
        # ✅ FIX 6: GRAD CLIPPING
        # ============================================================
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

        optimizer.step()
        scheduler.step()

        # ============================================================
        # ✅ FIX 7: MEMORY STABILIZATION (CRITICAL FOR UR-DMU)
        # ============================================================
        for name, param in model.named_parameters():
            if "memory" in name:
                param.data.clamp_(-1, 1)

        # ============================================================
        # LOGGING
        # ============================================================
        if step % 10 == 0:
            print(f"[STEP {step}] total={total_loss.item():.4f} | inner={inner_loss.item():.4f} | meta={meta_val_loss.item():.4f}")

        # ============================================================
        # EVALUATION (UNCHANGED)
        # ============================================================
        if step % args.eval_interval == 0:

            model.flag = "Test"

            test_loader = data.DataLoader(
                test_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=args.num_workers
            )

            auc_score, ap_score, acc = test(
                model, config, test_loader, step
            )

            model.flag = "Train"

            print(f"[EVAL {step}] AUC={auc_score:.4f} AP={ap_score:.4f} ACC={acc:.4f}")

            # ========================================================
            # SAVE BEST
            # ========================================================
            if auc_score > best_auc:
                best_auc = auc_score

                save_path = os.path.join(
                    args.save_dir,
                    f"best_model_auc_{best_auc:.4f}.pth"
                )

                torch.save(model.state_dict(), save_path)
                print(f"[INFO] Saved best model → {save_path}")

    print(f"[INFO] Training complete. Best AUC: {best_auc:.4f}")

if __name__ == "__main__":
    main()