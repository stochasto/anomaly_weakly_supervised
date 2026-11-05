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
import pickle
from pathlib import Path
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
    p.add_argument('--dataset', type=str, default='ucf', choices=['ucf', 'shanghai'])
    p.add_argument('--train_list', type=str, default='list/UCF_Train.list')
    p.add_argument('--test_list', type=str, default='list/UCF_Test.list')
    p.add_argument('--train_abnormal_count', type=int, default=None)
    p.add_argument('--test_abnormal_count', type=int, default=None)
    p.add_argument('--num_segments', type=int, default=200)
    p.add_argument('--a_nums', type=int, default=300)
    p.add_argument('--n_nums', type=int, default=300)
    p.add_argument('--len_feature', type=int, default=1024)
    p.add_argument('--support_k', type=int, default=8)
    p.add_argument('--query_k', type=int, default=16)
    p.add_argument('--meta_iters', type=int, default=1000)
    p.add_argument('--warmup_steps', type=int, default=100)
    p.add_argument('--inner_lr', type=float, default=1e-3)
    p.add_argument('--meta_lr', type=float, default=1e-4)
    p.add_argument('--beta', type=float, default=1.0)
    p.add_argument('--eval_interval', type=int, default=100)
    p.add_argument('--seed', type=int, default=-1)
    p.add_argument('--gt_file', type=str, default='frame_label/gt_ucf_v2.npy')
    p.add_argument('--video_label_pickle', type=str, default=None)
    p.add_argument('--frame_label_pickle', type=str, default=None)
    p.add_argument('--use_frame_gt', action='store_true', help='If set, evaluate using frame-level ground truth when available')
    p.add_argument('--save_dir', type=str, default='checkpoints')
    return p.parse_args()

# -------------------------
# DATASET
# -------------------------
class VideoFeatureDataset(data.Dataset):
    def __init__(self, root_dir, list_path, mode, num_segments, len_feature,
                 abnormal_count=None, video_label_map=None):
        self.mode = mode
        self.num_segments = num_segments
        self.len_feature = len_feature
        self.root_dir = root_dir
        self.abnormal_count = abnormal_count
        self.video_label_map = video_label_map

        if not os.path.exists(list_path):
            raise FileNotFoundError(f"List file not found: {list_path}")

        self.samples = self._load_list(list_path)

    def _load_list(self, list_path):
        samples = []
        with open(list_path, 'r') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                vid_path = parts[0]
                if not os.path.isabs(vid_path) and self.root_dir:
                    vid_path = os.path.join(self.root_dir, vid_path)
                label = None
                if len(parts) >= 2:
                    try:
                        label = int(parts[1])
                    except ValueError:
                        label = int(float(parts[1]))
                samples.append((vid_path, label))

        if any(label is None for _, label in samples):
            if self.video_label_map is not None:
                labeled = []
                for path, _ in samples:
                    video_id = Path(path).parent.name
                    if video_id not in self.video_label_map:
                        raise ValueError(
                            f"video_label.pickle does not contain a label for video {video_id}"
                        )
                    video_label = self.video_label_map[video_id]
                    if isinstance(video_label, (list, tuple, np.ndarray)):
                        if len(video_label) == 0:
                            raise ValueError(f"Empty label container for video {video_id}")
                        video_label = video_label[0]
                    labeled.append((path, int(float(video_label))))
                samples = labeled
            elif self.abnormal_count is None:
                raise ValueError(
                    "List file has no explicit labels and abnormal_count was not provided. "
                    "Use a two-column list file or pass --train_abnormal_count / --test_abnormal_count."
                )
            else:
                labeled = []
                for i, (path, _) in enumerate(samples):
                    labeled.append((path, 1 if i < self.abnormal_count else 0))
                samples = labeled

        return samples

    def __len__(self):
        return len(self.samples)

    def _prepare_feat(self, feat):
        if feat.ndim == 3:
            if feat.shape[1] == 10:
                feat = feat.mean(axis=1)
            elif feat.shape[0] == 10:
                feat = feat.mean(axis=0)
            else:
                feat = feat.reshape(feat.shape[0], -1)
        elif feat.ndim > 2:
            feat = feat.reshape(feat.shape[0], -1)

        return feat.astype(np.float32)

    def _temporal_pool(self, feat):
        T, D = feat.shape
        new_feat = np.zeros((self.num_segments, D), dtype=np.float32)
        r = np.linspace(0, T, self.num_segments + 1, dtype=np.int32)
        for i in range(self.num_segments):
            if r[i] != r[i + 1]:
                new_feat[i] = np.mean(feat[r[i]:r[i + 1]], axis=0)
            else:
                new_feat[i] = feat[r[i]]
        return new_feat

    def __getitem__(self, index):
        vid, label = self.samples[index]
        feat = np.load(vid)
        feat = self._prepare_feat(feat)

        if self.mode == "Train":
            feat = self._temporal_pool(feat)

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
def test(model, loader, frame_label_map=None):
    model.eval()
    preds, gts = [], []
    frame_preds, frame_gts = [], []

    with torch.no_grad():
        for idx, (x, y) in enumerate(loader):
            x = x.cuda()
            out = model(x)["frame"]
            preds.append(out.mean(1).cpu().numpy())
            gts.append(y.numpy())

            if frame_label_map is not None:
                vid_path, _ = loader.dataset.samples[idx]
                video_id = Path(vid_path).parent.name
                if video_id not in frame_label_map:
                    raise ValueError(f"Frame label missing for video {video_id}")
                labels = np.asarray(frame_label_map[video_id], dtype=np.float32)
                pred_frames = out.squeeze(0).cpu().numpy()
                if pred_frames.shape[0] != labels.shape[0]:
                    min_len = min(pred_frames.shape[0], labels.shape[0])
                    pred_frames = pred_frames[:min_len]
                    labels = labels[:min_len]
                frame_preds.append(pred_frames)
                frame_gts.append(labels)

    preds = np.concatenate(preds)
    gts = np.concatenate(gts)

    fpr, tpr, _ = roc_curve(gts, preds)
    auc_score = auc(fpr, tpr)

    precision, recall, _ = precision_recall_curve(gts, preds)
    ap = auc(recall, precision)

    if frame_label_map is not None and len(frame_preds) > 0:
        frame_preds = np.concatenate(frame_preds)
        frame_gts = np.concatenate(frame_gts)
        fpr_f, tpr_f, _ = roc_curve(frame_gts, frame_preds)
        frame_auc = auc(fpr_f, tpr_f)
        precision_f, recall_f, _ = precision_recall_curve(frame_gts, frame_preds)
        frame_ap = auc(recall_f, precision_f)
        return auc_score, ap, frame_auc, frame_ap

    return auc_score, ap, None, None

# -------------------------
# MAIN
# -------------------------
def main():
    args = parse_args()

    if args.dataset == 'shanghai':
        if args.train_list == 'list/UCF_Train.list':
            args.train_list = 'list/shanghai-i3d-train-10crop-labeled.list'
        if args.test_list == 'list/UCF_Test.list':
            args.test_list = 'list/shanghai-i3d-test-10crop-labeled.list'
        if args.len_feature == 1024:
            args.len_feature = 2048
        args.train_abnormal_count = None
        args.test_abnormal_count = None

        root_path = Path(args.root_dir).resolve()
        shanghai_gt = root_path.parents[2] / 'GT'
        if args.video_label_pickle is None:
            args.video_label_pickle = str(shanghai_gt / 'video_label.pickle')
        if args.frame_label_pickle is None:
            args.frame_label_pickle = str(shanghai_gt / 'frame_label.pickle')

        if not os.path.exists(args.video_label_pickle):
            raise FileNotFoundError(f"video_label.pickle not found: {args.video_label_pickle}")

    if args.dataset == 'ucf':
        if args.train_abnormal_count is None:
            args.train_abnormal_count = 8100
        if args.test_abnormal_count is None:
            args.test_abnormal_count = 1400

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    video_label_map = None
    frame_label_map = None
    if args.dataset == 'shanghai':
        with open(args.video_label_pickle, 'rb') as f:
            video_label_map = pickle.load(f)
        if args.use_frame_gt and args.frame_label_pickle is not None:
            if not os.path.exists(args.frame_label_pickle):
                raise FileNotFoundError(f"frame_label.pickle not found: {args.frame_label_pickle}")
            with open(args.frame_label_pickle, 'rb') as f:
                frame_label_map = pickle.load(f)

    train_ds = VideoFeatureDataset(
        root_dir=args.root_dir,
        list_path=args.train_list,
        mode='Train',
        num_segments=args.num_segments,
        len_feature=args.len_feature,
        abnormal_count=args.train_abnormal_count,
        video_label_map=video_label_map,
    )

    test_ds = VideoFeatureDataset(
        root_dir=args.root_dir,
        list_path=args.test_list,
        mode='Test',
        num_segments=args.num_segments,
        len_feature=args.len_feature,
        abnormal_count=args.test_abnormal_count,
        video_label_map=video_label_map,
    )

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
        if args.eval_interval > 0 and step % args.eval_interval == 0:
            model.flag = "Test"
            auc_score, ap, frame_auc, frame_ap = test(model, test_loader, frame_label_map if args.use_frame_gt else None)
            model.flag = "Train"

            message = f"[STEP {step}] AUC={auc_score:.4f} AP={ap:.4f}"
            if frame_auc is not None:
                message += f" | frameAUC={frame_auc:.4f} frameAP={frame_ap:.4f}"
            print(message)

            if auc_score > best_auc:
                best_auc = auc_score
                best_model_path = os.path.join(args.save_dir, 'best_meta_wsad')
                best_model_path += '_sk_' + str(args.support_k) + '_qk_' + str(args.query_k) + '_ilr_' + str(args.inner_lr) + '_mlr_' + str(args.meta_lr) + '_beta_' + str(args.beta) + '_segments_' + str(args.num_segments)
                best_model_path += '_anums_' + str(args.a_nums) + '_nnums_' + str(args.n_nums) + '_auc_' + str(best_auc) + '.pth'
                torch.save(model.state_dict(), best_model_path)

    print("BEST AUC:", best_auc)

# -------------------------
if __name__ == "__main__":
    main()