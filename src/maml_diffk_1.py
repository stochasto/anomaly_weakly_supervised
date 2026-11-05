#!/usr/bin/env python3
"""
meta_train_wsad_runner.py

Meta-learning (MAML-style) training + evaluation for WSAD on UCF-Crime features.

Expect:
- model.py defines `WSAD`
- loss.py defines `AD_Loss`
- list/UCF_Train.list and list/UCF_Test.list exist and contain .npy paths (one per line)
- Ground-truth concatenated file path passed via --gt_file
"""

import os
import glob
import random
import argparse
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import stateless
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Import model and loss from separate files (do NOT duplicate model here)
# ---------------------------------------------------------------------
from model import WSAD          # your model file; must be available on PYTHONPATH

# ---------------------------------------------------------------------
# Use your provided UCF_crime dataset class exactly (copied from your message)
# ---------------------------------------------------------------------
import utils  # required by dataset class for set_seed
import os, glob, numpy as np
import torch.utils.data as data

class UCF_crime(data.Dataset): 
    def __init__(self, root_dir, modal, mode, num_segments, len_feature, seed=-1, is_normal=None):
        if seed >= 0:
            utils.set_seed(seed)
        self.root_dir = root_dir
        self.modal = modal
        self.mode = mode
        self.num_segments = num_segments
        self.len_feature = len_feature

        list_path = os.path.join('list', f'UCF_{self.mode}.list')
        if not os.path.exists(list_path):
            raise FileNotFoundError(f"Missing list file: {list_path}")
        
        # Load video paths
        with open(list_path, 'r') as f:
            lines = [x.strip().split()[0] for x in f if x.strip()]
        self.vid_list = lines

        # Optional subset (like reference code)
        if self.mode == "Train":
            if is_normal is True:
                self.vid_list = self.vid_list[8100:]
            elif is_normal is False:
                self.vid_list = self.vid_list[:8100]
            elif is_normal is None:
                print("Please specify is_normal=[True/False] for training split")

    def __len__(self):
        return len(self.vid_list)

    def __getitem__(self, index):
        if self.mode == "Test":
            data, label, name = self.get_data(index)
            return data, label, name
        else:
            data, label = self.get_data(index)
            return data, label

    def get_data(self, index):
        vid_path = self.vid_list[index]
        base = vid_path.replace('.npy', '').replace('__0', '')
        name = os.path.basename(base)

        # Load and average all crops if multiple exist
        crop_files = sorted(glob.glob(f"{base}*.npy"))
        if not crop_files:
            raise FileNotFoundError(f"No .npy files found for base: {base}")

        crops = [np.load(cf).astype(np.float32) for cf in crop_files]
        maxT = max(c.shape[0] for c in crops)
        feat_dim = crops[0].shape[1]

        # Pad shorter crops and average
        padded = []
        for c in crops:
            if c.shape[0] < maxT:
                pad = np.zeros((maxT - c.shape[0], feat_dim), dtype=np.float32)
                c = np.concatenate([c, pad], axis=0)
            padded.append(c)
        video_feature = np.mean(np.stack(padded, axis=0), axis=0)

        # Label assignment
        label = 0 if "Normal" in name else 1

        # Segment sampling during training
        if self.mode == "Train":
            T, D = video_feature.shape
            new_feat = np.zeros((self.num_segments, D), dtype=np.float32)
            r = np.linspace(0, T, self.num_segments + 1, dtype=np.int32)
            for i in range(self.num_segments):
                if r[i] != r[i + 1]:
                    new_feat[i, :] = np.mean(video_feature[r[i]:r[i + 1], :], axis=0)
                else:
                    new_feat[i, :] = video_feature[r[i], :]
            video_feature = new_feat

        if self.mode == "Test":
            return video_feature, label, name
        else:
            return video_feature, label

# ---------------------------------------------------------------------
# Helper: random sampling by video (k videos)
# ---------------------------------------------------------------------
def get_random_batch_from_dataset(dataset, k, device, do_norm=True):
    """
    Sample k videos from dataset (dataset.__getitem__ -> (T,D,label)).
    Returns x: (k, T, D) tensor, y: (k,) tensor
    """
    idxs = random.sample(range(len(dataset)), k)
    feats, labels = [], []
    for i in idxs:
        item = dataset[i]
        # dataset in Train mode returns (feat,label)
        if len(item) == 3:
            feat, label, _ = item
        else:
            feat, label = item
        arr = np.asarray(feat).astype(np.float32)
        if do_norm:
            mu = arr.mean(axis=0, keepdims=True)
            s = arr.std(axis=0, keepdims=True) + 1e-6
            arr = (arr - mu) / s
        feats.append(torch.from_numpy(arr))
        labels.append(torch.tensor(label).float())
    x = torch.stack(feats).to(device)   # (k, T, D)
    y = torch.stack(labels).to(device)  # (k,)
    return x, y

class AD_Loss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bce = nn.BCELoss()

    def forward(self, result, _label):
        loss = {}
        _label = _label.float()

        triplet = result.get("triplet_margin", torch.tensor(0.0, device=_label.device))
        att = result['frame']
        A_att = result.get("A_att", torch.zeros_like(att))
        N_att = result.get("N_att", torch.zeros_like(att))
        A_Natt = result.get("A_Natt", torch.zeros_like(att))
        N_Aatt = result.get("N_Aatt", torch.zeros_like(att))
        kl_loss = result.get("kl_loss", torch.tensor(0.0, device=_label.device))
        distance = result.get("distance", torch.tensor(0.0, device=_label.device))

        b = max(_label.size(0) // 2, 1)
        t = max(att.size(1), 1)

        anomaly_topk = max(t // 16 + 1, 1)
        anomaly = torch.topk(att, anomaly_topk, dim=-1)[0].mean(-1)
        anomaly_loss = self.bce(anomaly, _label)

        panomaly = torch.topk(1 - N_Aatt, anomaly_topk, dim=-1)[0].mean(-1)
        panomaly = panomaly[:b]
        panomaly_loss = self.bce(panomaly, torch.ones_like(panomaly))

        A_att_topk = torch.topk(A_att, anomaly_topk, dim=-1)[0].mean(-1)
        A_att_topk = A_att_topk[:b]
        A_loss = self.bce(A_att_topk, torch.ones_like(A_att_topk))

        N_loss = self.bce(N_att, torch.ones_like(N_att))
        A_Nloss = self.bce(A_Natt, torch.zeros_like(A_Natt))

        # stronger triplet/kl scaling to encourage sharper localization
        total_loss = (
            anomaly_loss
            + 0.1 * (A_loss + panomaly_loss + N_loss + A_Nloss)
            + 0.2 * triplet
            + 0.005 * kl_loss
            + 0.0001 * distance
        )

        loss["total_loss"] = total_loss
        loss["att_loss"] = anomaly_loss
        loss["N_Aatt"] = panomaly_loss
        loss["A_loss"] = A_loss
        loss["N_loss"] = N_loss
        loss["A_Nloss"] = A_Nloss
        loss["triplet"] = triplet
        loss["kl_loss"] = kl_loss

        return total_loss, loss

# ---------------------------------------------------------------------
# Evaluation: align_and_evaluate (keeps your logic)
# ---------------------------------------------------------------------
def align_and_evaluate(net, config, full_video_list, step, model_file=None, save_dir='checkpoints', do_norm=False, fixed_expand=None):
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, precision_recall_curve, auc
    import torch, os, glob, numpy as np

    device = config['device']
    net.eval()
    net.flag = "Test"
    if model_file is not None:
        net.load_state_dict(torch.load(model_file, map_location=device))

    frame_gt = np.load(config['gt_file'])
    frame_preds = np.zeros_like(frame_gt, dtype=np.float32)
    ptr = 0  # cumulative pointer into frame_gt

    per_video_stats = []
    cls_label, cls_pred = [], []

    best_norm_preds, worst_norm_preds, best_abn_preds, worst_abn_preds = [], [], [], []
    best_norm_gt, worst_norm_gt, best_abn_gt, worst_abn_gt = [], [], [], []
    best_norm_acc, worst_norm_acc = -1, 2
    best_abn_acc, worst_abn_acc = -1, 2

    with torch.no_grad():
        for base, label in full_video_list:
            # ----------------------
            # Load crops for the video
            # ----------------------
            crop_paths = []
            base_file = f"{base}.npy"
            if os.path.exists(base_file):
                crop_paths.append(base_file)
            for i in range(1, 10):
                p = f"{base}__{i}.npy"
                if os.path.exists(p):
                    crop_paths.append(p)
            if len(crop_paths) == 0:
                glob_paths = sorted(glob.glob(base + '*.npy'))
                if glob_paths:
                    crop_paths = glob_paths
            if len(crop_paths) == 0:
                raise FileNotFoundError(f"No crop files for {base}")

            # ----------------------
            # Load and stack features
            # ----------------------
            crops = [np.load(p).astype(np.float32) for p in crop_paths]
            maxT = max(c.shape[0] for c in crops)
            feat_dim = crops[0].shape[1]
            stacked = []
            for c in crops:
                if c.shape[0] < maxT:
                    pad = np.zeros((maxT - c.shape[0], feat_dim), dtype=np.float32)
                    c = np.concatenate([c, pad], axis=0)
                stacked.append(c)
            stacked = np.stack(stacked, axis=0)  # [num_crops, maxT, feat]
            avg_feat = stacked.mean(axis=0)      # [maxT, feat]

            # ----------------------
            # Check GT for normal videos
            # ----------------------
            if label == 0:
                gt_segment = frame_gt[ptr:ptr + maxT]
                if not np.all(gt_segment == 0):
                    raise ValueError(f"Normal video '{os.path.basename(base)}' has non-zero labels in GT!")

            # ----------------------
            # Temporal downsample to num_segments
            # ----------------------
            T = avg_feat.shape[0]
            num_segments = config['num_segments']
            if T != num_segments:
                r = np.linspace(0, T, num_segments + 1, dtype=np.int32)
                new_feat = np.zeros((num_segments, feat_dim), dtype=np.float32)
                for i in range(num_segments):
                    if r[i] != r[i+1]:
                        new_feat[i, :] = np.mean(avg_feat[r[i]:r[i+1], :], axis=0)
                    else:
                        new_feat[i, :] = avg_feat[r[i], :]
                seg_feat = new_feat
            else:
                seg_feat = avg_feat

            # ----------------------
            # Normalization if requested
            # ----------------------
            if do_norm:
                mu = seg_feat.mean(axis=0, keepdims=True)
                s = seg_feat.std(axis=0, keepdims=True) + 1e-6
                seg_feat = (seg_feat - mu) / s

            # ----------------------
            # Model forward
            # ----------------------
            x = torch.from_numpy(seg_feat).unsqueeze(0).to(device)
            res = net(x)
            preds = res["frame"]
            if preds.dim() == 3:
                preds = preds.mean(dim=2)
            preds = preds.squeeze(0).detach().cpu().numpy()  # [num_segments]

            # ----------------------
            # Frame-level expansion
            # ----------------------
            if fixed_expand is not None:
                expand_factor = fixed_expand
            else:
                expand_factor = int(np.ceil(maxT / max(preds.shape[0], 1)))
            frame_level_preds = np.repeat(preds, expand_factor)[:maxT]

            # ----------------------
            # Fill global predictions
            # ----------------------
            remaining = len(frame_gt) - ptr
            if remaining <= 0:
                break
            place = frame_level_preds[:remaining]
            frame_preds[ptr:ptr+len(place)] = place
            gt_clip = frame_gt[ptr:ptr+len(place)]
            ptr += len(place)

            # ----------------------
            # Per-video metrics
            # ----------------------
            vid_pred_mean = frame_level_preds.mean()
            vid_pred_label = 1 if frame_level_preds.max() > 0.5 else 0
            vid_acc = (frame_level_preds.round() == gt_clip).sum() / len(gt_clip)

            per_video_stats.append((os.path.basename(base), int(label), vid_pred_mean, vid_acc, len(gt_clip), len(place)))
            cls_label.append(int(label))
            cls_pred.append(vid_pred_label)

            # ----------------------
            # Update best/worst video lists
            # ----------------------
            if label == 0:  # normal
                if vid_acc > best_norm_acc:
                    best_norm_acc = vid_acc
                    best_norm_preds, best_norm_gt = frame_level_preds, gt_clip
                if vid_acc < worst_norm_acc:
                    worst_norm_acc = vid_acc
                    worst_norm_preds, worst_norm_gt = frame_level_preds, gt_clip
            else:  # abnormal
                if vid_acc > best_abn_acc:
                    best_abn_acc = vid_acc
                    best_abn_preds, best_abn_gt = frame_level_preds, gt_clip
                if vid_acc < worst_abn_acc:
                    worst_abn_acc = vid_acc
                    worst_abn_preds, worst_abn_gt = frame_level_preds, gt_clip

    # ----------------------
    # Global metrics
    # ----------------------
    fpr, tpr, _ = roc_curve(frame_gt[:ptr], frame_preds[:ptr])
    auc_score = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(frame_gt[:ptr], frame_preds[:ptr])
    ap_score = auc(recall, precision)
    accuracy = np.mean(np.array(cls_label) == np.array(cls_pred)) if len(cls_pred) > 0 else 0.0

    # ----------------------
    # Plots for diagnostics
    # ----------------------
    os.makedirs(os.path.join(save_dir, 'plots'), exist_ok=True)
    for tag, preds, gt in zip(
        ["best_norm", "worst_norm", "best_abn", "worst_abn"],
        [best_norm_preds, worst_norm_preds, best_abn_preds, worst_abn_preds],
        [best_norm_gt, worst_norm_gt, best_abn_gt, worst_abn_gt]
    ):
        if len(preds) == 0:
            continue
        plt.figure(figsize=(10,4))
        plt.plot(preds, label='pred', color='orange')
        plt.plot(gt, label='gt', alpha=0.6, color='blue')
        plt.legend()
        plt.title(f"{tag.upper()} @ step {step}")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'plots', f'{tag}_step{step}.png'))
        plt.close()

    print(f"[EVAL @ step {step}] AUC={auc_score:.4f} | AP={ap_score:.4f} | ACC={accuracy:.4f}")
    return auc_score, ap_score, accuracy

# ---------------------------------------------------------------------
# Main meta-learning training loop
# ---------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--root_dir', type=str, required=True, help='root where features live (used to resolve relative paths)')
    p.add_argument('--gt_file', type=str, required=True, help='path to concatenated frame GT .npy')
    p.add_argument('--num_segments', type=int, default=32)
    p.add_argument('--len_feature', type=int, default=1024)
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--save_dir', type=str, default='checkpoints')
    p.add_argument('--meta_iters', type=int, default=200)
    p.add_argument('--support_k', type=int, default=4)
    p.add_argument('--query_k', type=int, default=4)
    p.add_argument('--inner_lr', type=float, default=1e-3)
    p.add_argument('--meta_lr', type=float, default=1e-4)
    p.add_argument('--beta', type=float, default=1.0)
    p.add_argument('--warmup_steps', type=int, default=1)
    p.add_argument('--eval_interval', type=int, default=1)
    p.add_argument('--a_nums', type=int, default=60)
    p.add_argument('--n_nums', type=int, default=60)
    p.add_argument('--debug', action='store_true')
    return p.parse_args()

def main():
    args = parse_args()
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

    print("[INFO] Loading Train list via UCF_crime (to split normal/abnormal)...")
    full_train_dataset = UCF_crime(root_dir=args.root_dir, modal='rgb', mode='Train',
                                   num_segments=args.num_segments, len_feature=args.len_feature, seed=args.seed)

    # build sample lists (base path, label)
    all_vids = full_train_dataset.vid_list
    normal_samples = []
    abnormal_samples = []
    for p in all_vids:
        base = p.replace('.npy', '').replace('__0', '')
        name = os.path.basename(base)
        label = 0 if "Normal" in name else 1
        if label == 0:
            normal_samples.append((base, label))
        else:
            abnormal_samples.append((base, label))

    # Create small dataset-like wrappers (VERY small, just __len__ and __getitem__ returning (T,D),label)
    class SimpleSamplesDataset(data.Dataset):
        def __init__(self, samples, num_segments):
            self.samples = samples
            self.num_segments = num_segments
        def __len__(self):
            return len(self.samples)
        def __getitem__(self, idx):
            base, label = self.samples[idx]
            # reuse the same loading/averaging logic as UCF_crime.get_data (Train mode)
            crop_files = sorted(glob.glob(f"{base}*.npy"))
            if not crop_files:
                raise FileNotFoundError(f"No .npy for {base}")
            crops = [np.load(cf).astype(np.float32) for cf in crop_files]
            maxT = max(c.shape[0] for c in crops)
            feat_dim = crops[0].shape[1]
            padded = []
            for c in crops:
                if c.shape[0] < maxT:
                    pad = np.zeros((maxT - c.shape[0], feat_dim), dtype=np.float32)
                    c = np.concatenate([c, pad], axis=0)
                padded.append(c)
            avg = np.mean(np.stack(padded, axis=0), axis=0)
            # downsample to num_segments
            T = avg.shape[0]
            D = avg.shape[1]
            new_feat = np.zeros((self.num_segments, D), dtype=np.float32)
            r = np.linspace(0, T, self.num_segments + 1, dtype=np.int32)
            for i in range(self.num_segments):
                if r[i] != r[i+1]:
                    new_feat[i, :] = np.mean(avg[r[i]:r[i+1], :], axis=0)
                else:
                    new_feat[i, :] = avg[r[i], :]
            return new_feat, label

    normal_train_ds = SimpleSamplesDataset(normal_samples, args.num_segments)
    abnormal_train_ds = SimpleSamplesDataset(abnormal_samples, args.num_segments)

    # Prepare test dataset list from list/UCF_Test.list (create the full_video_list used by align_and_evaluate)
    test_list_file = os.path.join("list", "UCF_Test.list")
    if not os.path.exists(test_list_file):
        raise FileNotFoundError(f"{test_list_file} not found.")
    with open(test_list_file, 'r') as f:
        test_lines = [line.strip().split()[0] for line in f if line.strip()]
    test_samples = []
    for base in test_lines:
        # Normalize relative path if necessary
        if not os.path.isabs(base):
            candidate = os.path.join(args.root_dir, base)
            if os.path.exists(candidate):
                base = candidate
        base_name_only = os.path.basename(base)
        label = 0 if "Normal" in base_name_only else 1
        test_samples.append((base.replace('.npy','').replace('__0',''), label))

    test_dataset_for_eval = test_samples

    print("[INFO] Initializing WSAD model...")
    model = WSAD(input_size=args.len_feature, flag="Train", a_nums=args.a_nums, n_nums=args.n_nums)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.meta_lr, betas=(0.9, 0.999), weight_decay=5e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.meta_iters))
    criterion = AD_Loss()

    test_info = {'step': [], 'auc': [], 'ap': [], 'ac': []}
    best_auc = 0.0
    best_model_path = os.path.join(args.save_dir, 'best_meta_wsad.pth')

    print("[INFO] Training settings:")
    print(f" device={device} support_k={args.support_k} query_k={args.query_k} inner_lr={args.inner_lr} meta_lr={args.meta_lr} beta={args.beta} warmup_steps={args.warmup_steps}")

    for step in tqdm(range(1, args.meta_iters + 1), total=args.meta_iters, ncols=100):
        model.train()

        # Warmup supervised pretraining
        if step <= args.warmup_steps:
            k_sup = max(args.query_k, args.support_k)
            qx_n_x, qx_n_y = get_random_batch_from_dataset(normal_train_ds, k_sup, device, do_norm=False)
            qx_a_x, qx_a_y = get_random_batch_from_dataset(abnormal_train_ds, k_sup, device, do_norm=False)
            sup_x = torch.cat([qx_n_x, qx_a_x], dim=0)
            sup_y = torch.cat([qx_n_y, qx_a_y], dim=0)

            result = model(sup_x)
            loss_sup, loss_dict_sup = criterion(result, sup_y)

            optimizer.zero_grad()
            loss_sup.backward()
            optimizer.step()
            scheduler.step()

            if step % max(1, args.warmup_steps//5) == 0 or args.debug:
                print(f"[WARMUP {step}/{args.warmup_steps}] sup_loss={float(loss_sup.item()):.6f}")
            continue

        # ----------------------
        # Meta-training inner loop (support)
        # ----------------------
        sx_n_x, sx_n_y = get_random_batch_from_dataset(normal_train_ds, args.support_k, device, do_norm=False)
        sx_a_x, sx_a_y = get_random_batch_from_dataset(abnormal_train_ds, args.support_k, device, do_norm=False)
        support_x = torch.cat([sx_n_x, sx_a_x], dim=0)
        support_y = torch.cat([sx_n_y, sx_a_y], dim=0)

        # inner forward & loss
        result_support = model(support_x)
        inner_loss, inner_loss_dict = criterion(result_support, support_y)

        # gradients
        grads = torch.autograd.grad(inner_loss, model.parameters(), create_graph=True, allow_unused=False)
        alpha = args.inner_lr
        fast_weights = {name: param - alpha * g for (name, param), g in zip(model.named_parameters(), grads)}

        # ----------------------
        # Query
        # ----------------------
        qx_n_x, qx_n_y = get_random_batch_from_dataset(normal_train_ds, args.query_k, device, do_norm = False)
        qx_a_x, qx_a_y = get_random_batch_from_dataset(abnormal_train_ds, args.query_k, device, do_norm= False)
        query_x = torch.cat([qx_n_x, qx_a_x], dim=0)
        query_y = torch.cat([qx_n_y, qx_a_y], dim=0)

        result_query = stateless.functional_call(model, fast_weights, (query_x,))
        meta_val_loss, query_loss_dict = criterion(result_query, query_y)

        total_loss = inner_loss + args.beta * meta_val_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        def scalarize(v):
            try:
                if isinstance(v, (float, int)):
                    return float(v)
                if isinstance(v, torch.Tensor):
                    if v.numel() == 1:
                        return float(v.item())
                    else:
                        return float(v.mean().item())
                return float(np.array(v).mean())
            except Exception:
                return 0.0

        inner_items = ", ".join([f"{k}={scalarize(v):.6f}" for k, v in inner_loss_dict.items()])
        query_items = ", ".join([f"{k}={scalarize(v):.6f}" for k, v in query_loss_dict.items()])

        if step % 10 == 0 or args.debug:
            print(f"[STEP {step}] total_meta_loss={scalarize(total_loss):.6f} | inner_loss={scalarize(inner_loss):.6f} | meta_val_loss={scalarize(meta_val_loss):.6f}")
            print("  inner:", inner_items)
            print("  query:", query_items)

        # Evaluate & checkpoint
        if step % args.eval_interval == 0:
            model.flag = "Test"
            print(f"[EVAL] step {step} -> running align_and_evaluate() ...")
            auc_score, ap_score, acc = align_and_evaluate(model, config, test_dataset_for_eval, step, model_file=None, save_dir=args.save_dir)
            model.flag = "Train"
            test_info["step"].append(step)
            test_info["auc"].append(auc_score)
            test_info["ap"].append(ap_score)
            test_info["ac"].append(acc)
            print(f"[EVAL @ {step}] AUC={auc_score:.4f} | AP={ap_score:.4f} | ACC={acc:.4f}")
            if auc_score > best_auc:
                best_auc = auc_score
                torch.save(model.state_dict(), best_model_path)
                print(f"[INFO] New best AUC: {best_auc:.4f} saved to {best_model_path}")

    print("[INFO] Meta-training completed. Best AUC: {:.4f}".format(best_auc))


if __name__ == "__main__":
    main()
