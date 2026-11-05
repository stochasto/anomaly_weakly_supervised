#!/usr/bin/env python3
"""
MAML-style meta-training using WSAD + AD_Loss (keeps meta-learning intact).
- Dataset loader expects per-video crops saved as:
    base_name.npy        (may exist, acts as crop 0)
    base_name__1.npy ... base_name__9.npy
  (It handles missing base / __0 gracefully.)
- Uses AD_Loss only (inner + meta val), prints full loss breakdown every step,
  computes AUC/AP/ACC every eval_interval steps (default 5).
"""

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

# -------------------------
# CLI / Config
# -------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--root_dir', type=str, default='/mnt/c/code/data/I3D',
                   help="base data directory containing Train/RGB and Test/RGB")
    p.add_argument('--num_segments', type=int, default=200)
    p.add_argument('--len_feature', type=int, default=1024)   # 1024 for repo I3D
    p.add_argument('--batch_k', type=int, default=8, help="support/query size per normal/abnormal in meta step")
    p.add_argument('--meta_iters', type=int, default=2000, help="number of meta iterations (increase for better performance)")
    p.add_argument('--inner_lr', type=float, default=5e-5)
    p.add_argument('--meta_lr', type=float, default=5e-4)
    p.add_argument('--beta', type=float, default=0.1, help="weight for meta-val loss")
    p.add_argument('--num_workers', type=int, default=2)
    p.add_argument('--save_dir', type=str, default='checkpoints')
    p.add_argument('--gt_file', type=str, default='frame_label/gt-ucf.npy')
    p.add_argument('--eval_interval', type=int, default=100)
    p.add_argument('--debug', action='store_true')
    args = p.parse_args()
    return args

# -------------------------
# repo model & utils (must be available)
# -------------------------
from model import WSAD
import utils

# -------------------------
# Dataset: group crop files per video
# -------------------------
class UCF_crime(data.Dataset):
    """
    Builds a list of unique videos from list files and loads available crop files
    for each video (handles base .npy as crop 0 if present; otherwise picks __1..__9).
    Returns averaged crops -> [T, feat_dim]. For Train mode it downsamples to num_segments.
    """
    def __init__(self, root_dir, modal, mode, num_segments, len_feature, seed=-1):
        if seed >= 0 and hasattr(utils, "set_seed"):
            utils.set_seed(seed)
        self.mode = mode
        self.modal = modal
        self.num_segments = num_segments
        self.len_feature = len_feature
        self.root_dir = root_dir

        # Try standard repo list location first, fallback to generic file
        list_path_repo = os.path.join('list', f'UCF_{self.mode}.list')
        alt_list_path = os.path.join(self.root_dir, f'{self.mode.lower()}_list.txt')

        if os.path.exists(list_path_repo):
            list_file = list_path_repo
        elif os.path.exists(alt_list_path):
            list_file = alt_list_path
        else:
            raise FileNotFoundError(f"Cannot find list file for mode={self.mode}. "
                                    f"Looked for {list_path_repo} and {alt_list_path}.")

        # read lines, each line may point to e.g. /.../Abuse001_x264.npy or ...__3.npy
        with open(list_file, 'r') as f:
            raw = [line.strip().split()[0] for line in f if line.strip()]

        # normalize to absolute paths if relative
        # If entries are relative, try root_dir as base
        normalized = []
        for pth in raw:
            if os.path.isabs(pth):
                normalized.append(pth)
            else:
                # try root_dir + pth
                candidate = os.path.join(self.root_dir, pth)
                if os.path.exists(candidate):
                    normalized.append(candidate)
                elif os.path.exists(pth):
                    normalized.append(pth)
                else:
                    # keep original, may still be valid later
                    normalized.append(pth)

        # group by base_name (strip __\d and .npy)
        grouped = {}
        for pth in normalized:
            bn = pth
            if bn.endswith('.npy'):
                bn = bn[:-4]
            # strip trailing __\d if present
            if '__' in bn and bn.split('__')[-1].isdigit():
                base = '__'.join(bn.split('__')[:-1])
            else:
                base = bn
            # store original path entry
            grouped.setdefault(base, []).append(pth)

        # Build samples as list of (base_path, label)
        samples = []
        for base, files in sorted(grouped.items()):
            # decide label by base name containing 'Normal' (following repo)
            base_name_only = os.path.basename(base)
            label = 0 if "Normal" in base_name_only else 1
            # store base (directory + base) so we can find base.npy and base__i.npy
            samples.append((base, label))
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        if self.mode == "Test":
            data, label, name = self.get_data(index)
            return data, label, name
        else:
            data, label = self.get_data(index)
            return data, label

    def _load_and_average_crops(self, base):
        """
        Given base path (without extension), load available crops:
        base.npy (if exists) => crop 0
        base__1.npy ... base__9.npy (if exist)
        Stack them and average across crops -> [T, feat_dim]
        """
        crop_paths = []
        base_file = f"{base}.npy"
        if os.path.exists(base_file):
            crop_paths.append(base_file)
        # __1..__9
        for i in range(1, 10):
            p = f"{base}__{i}.npy"
            if os.path.exists(p):
                crop_paths.append(p)

        # if none found, try to find any file matching base* (fallback)
        if len(crop_paths) == 0:
            glob_paths = sorted([p for p in glob.glob(base + '*.npy')])
            if glob_paths:
                crop_paths = glob_paths

        if len(crop_paths) == 0:
            raise FileNotFoundError(f"No crop files found for base: {base}")

        crops = [np.load(p).astype(np.float32) for p in crop_paths]  # list of [T, feat]
        # verify shapes compatible
        shapes = [c.shape for c in crops]
        # convert to shape [num_crops, T, feat_dim]
        # if some crops have different T, pad shorter to max_T with zeros before stacking
        maxT = max([c.shape[0] for c in crops])
        feat_dim = crops[0].shape[1]
        stacked = []
        for c in crops:
            if c.shape[0] < maxT:
                pad = np.zeros((maxT - c.shape[0], feat_dim), dtype=np.float32)
                c = np.concatenate([c, pad], axis=0)
            stacked.append(c)
        stacked = np.stack(stacked, axis=0)  # [num_crops, maxT, feat_dim]
        # average across crop axis -> [maxT, feat_dim]
        avg = stacked.mean(axis=0)
        return avg

    def get_data(self, index):
        base, label = self.samples[index]
        name = os.path.basename(base)
        video_feature = self._load_and_average_crops(base)  # [T, feat_dim]

        # training: temporal downsample to num_segments
        if self.mode == "Train":
            T = video_feature.shape[0]
            D = video_feature.shape[1]
            new_feat = np.zeros((self.num_segments, D), dtype=np.float32)
            r = np.linspace(0, T, self.num_segments + 1, dtype=np.int32)
            for i in range(self.num_segments):
                if r[i] != r[i+1]:
                    new_feat[i, :] = np.mean(video_feature[r[i]:r[i+1], :], axis=0)
                else:
                    new_feat[i, :] = video_feature[r[i], :]
            video_feature = new_feat

        if self.mode == "Test":
            return video_feature, label, name
        else:
            return video_feature, label

# -------------------------
# Helper: random sampling by video (k videos)
# -------------------------
def get_random_batch_from_dataset(dataset, k):
    """
    Sample k videos from dataset (dataset.__getitem__ -> (T,D,label))
    Returns x: (k, T, D) tensor, y: (k,) tensor
    """
    idxs = random.sample(range(len(dataset)), k)
    feats, labels = [], []
    for i in idxs:
        feat, label = dataset[i]
        feats.append(torch.from_numpy(np.asarray(feat)).float())
        labels.append(torch.tensor(label).float())
    x = torch.stack(feats)   # (k, T, D)
    y = torch.stack(labels)  # (k,)
    return x, y

# -------------------------
# Efficient test() with frame alignment + zero padding
# -------------------------
import glob
def align_and_evaluate(net, config, full_video_list, step, model_file=None):
    """
    full_video_list: list of (base_path, label) as created by UCF_crime.samples
    Computes frame-level predictions for entire test set, aligning segments -> frames with zero-pad.
    """
    device = config['device']
    net.eval()
    net.flag = "Test"
    if model_file is not None:
        net.load_state_dict(torch.load(model_file, map_location=device))

    # Load global GT
    frame_gt = np.load(config['gt_file'])
    frame_preds = np.zeros_like(frame_gt, dtype=np.float32)
    ptr = 0

    cls_label = []
    cls_pred = []

    with torch.no_grad():
        #for base, label in tqdm(full_video_list, desc="Testing", ncols=120):
        for base, label in full_video_list:
            # load averaged crops then convert to num_segments (if needed) similar to train
            # _load_and_average_crops exists in UCF_crime, but here we do same simple logic:
            # find crop files for base, stack, average
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

            # if needed, temporal downsample to num_segments (to match training scale)
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
                seg_feat = avg_feat  # [num_segments, feat_dim]

            # Forward: create tensor [1, num_segments, feat_dim]
            x = torch.from_numpy(seg_feat).unsqueeze(0).to(device)
            res = net(x)
            # res["frame"] expected shape [1, num_segments] or [1, num_segments, crops]
            preds = res["frame"]
            if preds.dim() == 3:
                preds = preds.mean(dim=2)  # mean over crop axis if present
            preds = preds.squeeze(0).detach().cpu().numpy()  # [num_segments]

            # Align segment preds to frame-level by repeating each segment prediction
            # Here we expand preds to the original maxT frames (before downsample), then
            # place them into global frame_preds with zero padding where needed.
            # Use expansion factor = ceil(maxT / num_segments)
            expand_factor = int(np.ceil(maxT / preds.shape[0]))
            frame_level_preds = np.repeat(preds, expand_factor)[:maxT]

            # Place predictions into global frame_preds (zero-padded if not enough space)
            remaining = len(frame_gt) - ptr
            if remaining <= 0:
                break
            place = frame_level_preds[:remaining]
            frame_preds[ptr:ptr+len(place)] = place
            ptr += len(place)

            # video-level prediction
            cls_label.append(int(label))
            cls_pred.append(1 if preds.max() > 0.5 else 0)

            # stop early if we've filled all GT frames
            if ptr >= len(frame_gt):
                break

    # compute metrics
    fpr, tpr, _ = roc_curve(frame_gt, frame_preds)
    auc_score = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(frame_gt, frame_preds)
    ap_score = auc(recall, precision)
    accuracy = np.mean(np.array(cls_label) == np.array(cls_pred)) if len(cls_pred) > 0 else 0.0

    return auc_score, ap_score, accuracy

# -------------------------
# AD_Loss
# -------------------------
import torch
import torch.nn as nn

class AD_Loss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bce = nn.BCELoss()
      
    def forward(self, result, _label):
        loss = {}
        _label = _label.float()  # ensure float

        # Extract all outputs from model
        triplet = result["triplet_margin"]
        att = result['frame']
        A_att = result["A_att"]
        N_att = result["N_att"]
        A_Natt = result["A_Natt"]
        N_Aatt = result["N_Aatt"]
        kl_loss = result["kl_loss"]
        distance = result["distance"]

        # Number of abnormal samples
        b = max(_label.size(0) // 2, 1)  # safe fallback
        t = att.size(1)  # temporal length

        # ---- Anomaly attention loss ----
        anomaly_topk = max(t // 16 + 1, 1)  # ensure at least 1
        anomaly = torch.topk(att, anomaly_topk, dim=-1)[0].mean(-1)  # [batch]
        anomaly_loss = self.bce(anomaly, _label)

        # ---- Pseudo-anomaly loss (N_Aatt) ----
        panomaly = torch.topk(1 - N_Aatt, anomaly_topk, dim=-1)[0].mean(-1)
        # Ensure shape matches batch of abnormal samples
        if panomaly.numel() != b:
            panomaly = panomaly[:b]  # trim if needed
        panomaly_loss = self.bce(panomaly, torch.ones_like(panomaly))

        # ---- Abnormal attention loss ----
        A_att_topk = torch.topk(A_att, anomaly_topk, dim=-1)[0].mean(-1)
        if A_att_topk.numel() != b:
            A_att_topk = A_att_topk[:b]
        A_loss = self.bce(A_att_topk, torch.ones_like(A_att_topk))

        # ---- Normal attention loss ----
        N_loss = self.bce(N_att, torch.ones_like(N_att))
        A_Nloss = self.bce(A_Natt, torch.zeros_like(A_Natt))

        # ---- Total loss ----
        total_loss = (
            anomaly_loss
            + 0.1 * (A_loss + panomaly_loss + N_loss + A_Nloss)
            + 0.1 * triplet
            + 0.001 * kl_loss
            + 0.0001 * distance
        )

        # ---- Return dict ----
        loss["total_loss"] = total_loss
        loss["att_loss"] = anomaly_loss
        loss["N_Aatt"] = panomaly_loss
        loss["A_loss"] = A_loss
        loss["N_loss"] = N_loss
        loss["A_Nloss"] = A_Nloss
        loss["triplet"] = triplet
        loss["kl_loss"] = kl_loss

        return total_loss, loss


# -------------------------
# MAIN: meta-training (MAML kept intact)
# -------------------------
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = {
        'num_segments': args.num_segments,
        'len_feature': args.len_feature,
        'device': device,
        'gt_file': args.gt_file
    }
    os.makedirs(args.save_dir, exist_ok=True)

    print("[INFO] Loading datasets...")
    train_list_fallback = os.path.join(args.root_dir, 'Train', 'RGB')
    test_list_fallback = os.path.join(args.root_dir, 'Test', 'RGB')

    # instantiate dataset (expects list/UCF_Train.list or fallback)
    full_train_dataset = UCF_crime(root_dir=train_list_fallback, modal='rgb', mode='Train',
                                   num_segments=args.num_segments, len_feature=args.len_feature, seed=-1)
    all_samples = full_train_dataset.samples   # list of (base,label)
    normal_samples = [s for s in all_samples if s[1] == 0]
    abnormal_samples = [s for s in all_samples if s[1] == 1]

    # small DatasetFromSamples used for sampling during meta-train
    class DatasetFromSamples(data.Dataset):
        def __init__(self, samples, mode, num_segments, len_feature):
            self.samples = samples
            self.mode = mode
            self.num_segments = num_segments
            self.len_feature = len_feature
        def __len__(self):
            return len(self.samples)
        def __getitem__(self, index):
            base, label = self.samples[index]
            # load and average crops similar to UCF_crime._load_and_average_crops
            crop_paths = []
            base_file = f"{base}.npy"
            if os.path.exists(base_file):
                crop_paths.append(base_file)
            for i in range(1, 10):
                p = f"{base}__{i}.npy"
                if os.path.exists(p):
                    crop_paths.append(p)
            if len(crop_paths) == 0:
                import glob
                glob_paths = sorted(glob.glob(base + '*.npy'))
                if glob_paths:
                    crop_paths = glob_paths
            if len(crop_paths) == 0:
                raise FileNotFoundError(f"No crop files found for {base}")

            crops = [np.load(p).astype(np.float32) for p in crop_paths]
            maxT = max([c.shape[0] for c in crops])
            feat_dim = crops[0].shape[1]
            stacked = []
            for c in crops:
                if c.shape[0] < maxT:
                    pad = np.zeros((maxT - c.shape[0], feat_dim), dtype=np.float32)
                    c = np.concatenate([c, pad], axis=0)
                stacked.append(c)
            stacked = np.stack(stacked, axis=0)  # [num_crops, maxT, feat_dim]
            avg = stacked.mean(axis=0)           # [maxT, feat_dim]

            # temporal downsample to num_segments for training sampling
            if self.mode == "Train":
                T = avg.shape[0]
                D = avg.shape[1]
                new_feat = np.zeros((self.num_segments, D), dtype=np.float32)
                r = np.linspace(0, T, self.num_segments + 1, dtype=np.int32)
                for i in range(self.num_segments):
                    if r[i] != r[i+1]:
                        new_feat[i, :] = np.mean(avg[r[i]:r[i+1], :], axis=0)
                    else:
                        new_feat[i, :] = avg[r[i], :]
                feat = new_feat
            else:
                feat = avg

            return feat, label

    normal_train_ds = DatasetFromSamples(normal_samples, mode='Train', num_segments=args.num_segments, len_feature=args.len_feature)
    abnormal_train_ds = DatasetFromSamples(abnormal_samples, mode='Train', num_segments=args.num_segments, len_feature=args.len_feature)

    # Test samplings use the list of all samples (base,label)
    test_dataset_for_eval = all_samples
    # instantiate a DataLoader for other use (not used in test above)
    # Not needed for our align_and_evaluate function; kept for compatibility
    test_ds_for_loader = DatasetFromSamples(all_samples, mode='Test', num_segments=args.num_segments, len_feature=args.len_feature)
    test_loader = data.DataLoader(test_ds_for_loader, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # Model and optimizer
    print("[INFO] Initializing WSAD model...")
    model = WSAD(input_size=args.len_feature, flag="Train", a_nums=60, n_nums=60)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.meta_lr, betas=(0.9, 0.999), weight_decay=5e-5)

    # criterion replaced by AD_Loss only
    criterion = AD_Loss()

    test_info = {'step': [], 'auc': [], 'ap': [], 'ac': []}
    best_auc = 0.0
    best_model_path = os.path.join(args.save_dir, 'best_meta_wsad_1k_')

    print("[INFO] Starting meta-training with AD_Loss (meta intact)...")
    for step in tqdm(range(1, args.meta_iters + 1), total=args.meta_iters, ncols=100):
        model.train()
        k = args.batch_k

        # Sample support (k normal, k abnormal)
        sx_n_x, sx_n_y = get_random_batch_from_dataset(normal_train_ds, k)
        sx_a_x, sx_a_y = get_random_batch_from_dataset(abnormal_train_ds, k)
        support_x = torch.cat([sx_n_x, sx_a_x], dim=0).to(device, non_blocking=True)
        support_y = torch.cat([sx_n_y, sx_a_y], dim=0).to(device, non_blocking=True)

        # --- Inner loop (support)
        result_support = model(support_x)
        inner_loss, inner_loss_dict = criterion(result_support, support_y)

        # compute grads for inner update
        grads = torch.autograd.grad(inner_loss, model.parameters(), create_graph=True, allow_unused=False)
        alpha = args.inner_lr
        fast_weights = {name: param - alpha * g for (name, param), g in zip(model.named_parameters(), grads)}

        # --- Query step (new random draw)
        qx_n_x, qx_n_y = get_random_batch_from_dataset(normal_train_ds, k)
        qx_a_x, qx_a_y = get_random_batch_from_dataset(abnormal_train_ds, k)
        query_x = torch.cat([qx_n_x, qx_a_x], dim=0).to(device, non_blocking=True)
        query_y = torch.cat([qx_n_y, qx_a_y], dim=0).to(device, non_blocking=True)

        result_query = stateless.functional_call(model, fast_weights, (query_x,))
        meta_val_loss, query_loss_dict = criterion(result_query, query_y)

        # Total meta-loss (unchanged logic)
        total_loss = inner_loss + args.beta * meta_val_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Print losses: inner, meta_val, total and breakdowns (make scalars robustly)
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
                return None

        # Build display string showing main items
        inner_items = ", ".join([f"{k}={scalarize(v):.6f}" for k, v in inner_loss_dict.items()])
        query_items = ", ".join([f"{k}={scalarize(v):.6f}" for k, v in query_loss_dict.items()])

        print(f"[STEP {step}] total_meta_loss={scalarize(total_loss):.6f} | inner_loss={scalarize(inner_loss):.6f} | meta_val_loss={scalarize(meta_val_loss):.6f}")
        if args.debug:
            print("  inner loss breakdown:", inner_items)
            print("  query loss breakdown:", query_items)
        else:
            # still print key scalars succinctly
            print("  inner:", inner_items)
            print("  query:", query_items)

        # Evaluate every eval_interval steps
        if step % args.eval_interval == 0:
            model.flag = "Test"
            print(f'Testing step {step}')
            auc_score, ap_score, acc = align_and_evaluate(model, config, test_dataset_for_eval, step, model_file=None)
            model.flag = "Train"
            test_info["step"].append(step)
            test_info["auc"].append(auc_score)
            test_info["ap"].append(ap_score)
            test_info["ac"].append(acc)
            print(f"[EVAL @ step {step}] AUC={auc_score:.4f} | AP={ap_score:.4f} | ACC={acc:.4f}")
            if auc_score > best_auc:
                best_auc = auc_score
                best_model_path = best_model_path + str(best_auc) + '.pth'  
                torch.save(model.state_dict(), best_model_path)
                print(f"[INFO] New best AUC: {best_auc:.4f} (saved to {best_model_path})")

    print("[INFO] Meta-training completed. Best AUC: {:.4f}".format(best_auc))


if __name__ == '__main__':
    main()
