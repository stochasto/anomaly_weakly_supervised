#!/usr/bin/env python3
"""
maml_dm_improved.py

MAML-style meta-training using WSAD + AD_Loss, improved:
- different k for support and query
- supervised warmup
- inner/outer LR tuning
- per-video normalization
- LR scheduler
- evaluation + plotting
- robust frame alignment
"""

import os
import glob
import random
import argparse
from tqdm import tqdm
import numpy as np
import matplotlib
import json
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.nn.utils import stateless
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
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
    p.add_argument('--a_nums', type=int, default=200)
    p.add_argument('--n_nums', type=int, default=200)
    p.add_argument('--len_feature', type=int, default=1024,   # change to 1024 if using repo features
                   help="feature dimension (2048 for your I3D)")
    p.add_argument('--support_k', type=int, default=8, help="support size per class (normal/abnormal)")
    p.add_argument('--query_k', type=int, default=16, help="query size per class (normal/abnormal)")
    p.add_argument('--meta_iters', type=int, default=4000, help="number of meta iterations")
    p.add_argument('--warmup_steps', type=int, default=100, help="supervised warmup iterations before meta updates")
    p.add_argument('--inner_lr', type=float, default=1e-3)
    p.add_argument('--meta_lr', type=float, default=1e-4)
    p.add_argument('--beta', type=float, default=1.0, help="weight for meta-val loss")
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--save_dir', type=str, default='checkpoints')
    p.add_argument('--gt_file', type=str, default='frame_label/gt_ucf_single_crop.npy')
    p.add_argument('--eval_interval', type=int, default=100)
    p.add_argument('--debug', action='store_true')
    p.add_argument('--no_norm', action='store_true', help="disable per-video normalization")
    p.add_argument('--seed', type=int, default=-1, help="set >=0 for deterministic runs, -1 for fully random")
    args = p.parse_args()
    return args

# -------------------------
# repo model & utils (must be present in repo)
# -------------------------
#from model import WSAD
# minimal upgrade bundle
from model_min_upgrade import WSAD
#from model_ablation import WSAD
#from model_full import WSAD
#from model_dpsk import WSAD


import utils

# -------------------------
# Dataset: group crop files per video
# (adapted from your code)
# -------------------------
def stable_norm(x):
    return (x - x.mean()) / (x.std() + 1e-6)


def save_log(log_dict, file_path):
    with open(file_path, "a") as f:
        f.write(json.dumps(log_dict) + "\n")

class UCF_crime(data.Dataset):
    def __init__(self, root_dir, modal, mode, num_segments, len_feature, seed=-1):
        if seed >= 0 and hasattr(utils, "set_seed"):
            utils.set_seed(seed)

        self.mode = mode
        self.modal = modal
        self.num_segments = num_segments
        self.len_feature = len_feature
        self.root_dir = root_dir

        list_path_repo = os.path.join('list', f'UCF_{self.mode}.list')
        alt_list_path = os.path.join(self.root_dir, f'{self.mode.lower()}_list.txt')

        if os.path.exists(list_path_repo):
            list_file = list_path_repo
        elif os.path.exists(alt_list_path):
            list_file = alt_list_path
        else:
            raise FileNotFoundError(
                f"Cannot find list file for mode={self.mode}. "
                f"Looked for {list_path_repo} and {alt_list_path}."
            )

        # -------------------------
        # STEP 1: filter only single-crop entries
        # -------------------------
        filtered_lines = []
        with open(list_file, 'r') as f:
            for line in f:
                path = line.strip()
                base = os.path.basename(path)

                # skip multi-crop files
                if "__" in base:
                    continue

                filtered_lines.append(path)

        # -------------------------
        # STEP 2: assign labels from path
        # -------------------------
        samples = []
        for path in filtered_lines:
            if "Normal" in path or "normal" in path:
                label = 0
            else:
                label = 1

            samples.append((path, label))

        self.samples = samples

        # Debug print (VERY useful)
        num_abn = sum([l for _, l in self.samples])
        print(f"[DATASET] Mode={self.mode} | Total={len(self.samples)} | Abnormal={num_abn} | Normal={len(self.samples)-num_abn}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.get_data(index)

    def get_data(self, index):
        vid, label = self.samples[index]

        video_feature = np.load(vid).astype(np.float32)  # [T, D]

        if self.mode == "Train":
            T, D = video_feature.shape

            new_feat = np.zeros((self.num_segments, D), dtype=np.float32)
            r = np.linspace(0, T, self.num_segments + 1, dtype=np.int32)

            for i in range(self.num_segments):
                if r[i] != r[i + 1]:
                    new_feat[i] = video_feature[r[i]:r[i + 1]].mean(axis=0)
                else:
                    new_feat[i] = video_feature[r[i]]

            video_feature = new_feat

        return video_feature, label

# -------------------------
# Helper: random sampling by video (k videos)
# -------------------------
def build_index_cache(dataset):
    normal_idxs = []
    abnormal_idxs = []

    for i in range(len(dataset)):
        label = dataset[i][1]   # one-time cost
        if label == 0:
            normal_idxs.append(i)
        else:
            abnormal_idxs.append(i)

    return normal_idxs, abnormal_idxs

def get_random_batch_from_dataset_fast(dataset, normal_idxs, abnormal_idxs, k, device):
    half_k = k // 2

    idxs = random.sample(normal_idxs, half_k) + \
           random.sample(abnormal_idxs, k - half_k)
    random.shuffle(idxs)

    feats, labels = [], []

    for i in idxs:
        feat, label = dataset[i]

        arr = np.asarray(feat).astype(np.float32)

        feats.append(torch.from_numpy(arr))
        labels.append(torch.tensor(label).float())

    x = torch.stack(feats).to(device)
    y = torch.stack(labels).to(device)

    return x, y

def get_random_batch_from_dataset_0(dataset, k, device, do_norm=False):
    """
    Sample k videos from dataset (dataset.__getitem__ -> (T,D,label)).
    Returns x: (k, T, D) tensor, y: (k,) tensor
    """
    idxs = random.sample(range(len(dataset)), k)
    feats, labels = [], []
    for i in idxs:
        feat, label = dataset[i]
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

def get_random_batch_from_dataset(dataset, k, device):
    normal_idxs = [i for i in range(len(dataset)) if dataset[i][1] == 0]
    abnormal_idxs = [i for i in range(len(dataset)) if dataset[i][1] == 1]

    half_k = k // 2

    idxs = random.sample(normal_idxs, half_k) + random.sample(abnormal_idxs, k - half_k)
    random.shuffle(idxs)

    feats, labels = [], []
    for i in idxs:
        f, l = dataset[i]
        feats.append(torch.from_numpy(f))
        labels.append(torch.tensor(l).float())

    return torch.stack(feats).to(device), torch.stack(labels).to(device)



## test ##
def test(net, config, test_loader, step, model_file=None, save_dir='checkpoints', do_norm=False):
    with torch.no_grad():
        net.eval()
        net.flag = "Test"

        if model_file is not None:
            net.load_state_dict(torch.load(model_file))

        load_iter = iter(test_loader)
        frame_gt = np.load(config['gt_file'])

        frame_predict = None
        cls_label = []
        cls_pre = []

        for i in range(len(test_loader.dataset)):
            data_, label_ = next(load_iter)

            data_ = data_.cuda()
            label_ = label_.cuda()

            res = net(data_)
            pred = res["frame"].squeeze(0).cpu().numpy()  # [T]

            # -------------------------
            # Video-level classification
            # -------------------------
            cls_label.append(int(label_))
            cls_pre.append(1 if pred.max() > 0.5 else 0)

            # -------------------------
            # Frame-level prediction
            # -------------------------
            fpre_ = pred   # already segment-level aligned

            if frame_predict is None:
                frame_predict = fpre_
            else:
                frame_predict = np.concatenate([frame_predict, fpre_])

        print(f'len gt {len(frame_gt)} len predict {len(frame_predict)}')

        # -------------------------
        # Metrics
        # -------------------------
        fpr, tpr, _ = roc_curve(frame_gt, frame_predict)
        auc_score = auc(fpr, tpr)

        precision, recall, _ = precision_recall_curve(frame_gt, frame_predict)
        ap_score = auc(recall, precision)

        accuracy = np.mean(np.array(cls_label) == np.array(cls_pre))

    return auc_score, ap_score, accuracy

## Ablation aware ADLoss

## Diagnostic
#import os
#import numpy as np
#import torch
#import matplotlib.pyplot as plt

def diagnostic_test(net, config, test_loader, save_dir="diagnostics", max_videos=5):
    os.makedirs(save_dir, exist_ok=True)

    net.eval()
    net.flag = "Test"

    frame_gt = np.load(config['gt_file'])

    all_preds = []
    video_ptr = 0  # pointer into GT

    with torch.no_grad():
        for i, (data_, label_) in enumerate(test_loader):

            data_ = data_.cuda()
            label_ = label_.cuda()

            res = net(data_)
            pred = res["frame"].squeeze(0).cpu().numpy()  # [T]

            # -------------------------
            # Stats
            # -------------------------
            print(f"\n[VIDEO {i}]")
            print(f"Label: {int(label_.item())}")
            print(f"Pred stats -> min: {pred.min():.4f}, max: {pred.max():.4f}, mean: {pred.mean():.4f}")

            # -------------------------
            # Top-k analysis
            # -------------------------
            t = len(pred)
            topk = max(t // 32, 1)
            topk_idx = np.argsort(pred)[-topk:]

            print(f"Top-{topk} indices:", sorted(topk_idx.tolist()))
            print(f"Top-{topk} values:", pred[topk_idx])

            # -------------------------
            # Expand to frame-level (IMPORTANT)
            # -------------------------
            pred_expanded = np.repeat(pred, 16)

            # Align with GT
            gt_segment = frame_gt[video_ptr: video_ptr + len(pred_expanded)]
            pred_expanded = pred_expanded[:len(gt_segment)]

            video_ptr += len(gt_segment)

            # -------------------------
            # Save raw arrays
            # -------------------------
            np.save(os.path.join(save_dir, f"video_{i}_pred.npy"), pred)
            np.save(os.path.join(save_dir, f"video_{i}_pred_expanded.npy"), pred_expanded)
            np.save(os.path.join(save_dir, f"video_{i}_gt.npy"), gt_segment)

            # -------------------------
            # Plot
            # -------------------------
            if i < max_videos:
                plt.figure(figsize=(12, 4))
                plt.plot(gt_segment, label="GT", linewidth=2)
                plt.plot(pred_expanded, label="Prediction", alpha=0.7)
                plt.title(f"Video {i} | Label={int(label_.item())}")
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f"video_{i}_plot.png"))
                plt.close()

            all_preds.append(pred_expanded)

    print("\n[INFO] Diagnostics complete. Saved to:", save_dir)



# -------------------------
# AD_Loss (slightly adjusted weights for stronger localization)
# -------------------------
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
        smooth_loss = result.get("smooth_loss", torch.tensor(0.0, device=_label.device))

        b = max(_label.size(0) // 2, 1)
        t = max(att.size(1), 1)

        # SAME AS BASELINE (important)
        anomaly_topk = max(t // 16 + 1, 1)

        # -------------------------
        # Main anomaly supervision
        # -------------------------
        anomaly = torch.topk(att, anomaly_topk, dim=-1)[0].mean(-1)
        anomaly_loss = self.bce(anomaly, _label)

        # -------------------------
        # Dual attention losses (unchanged)
        # -------------------------
        panomaly = torch.topk(1 - N_Aatt, anomaly_topk, dim=-1)[0].mean(-1)
        panomaly = panomaly[:b]
        panomaly_loss = self.bce(panomaly, torch.ones_like(panomaly))

        A_att_topk = torch.topk(A_att, anomaly_topk, dim=-1)[0].mean(-1)
        A_att_topk = A_att_topk[:b]
        A_loss = self.bce(A_att_topk, torch.ones_like(A_att_topk))

        N_loss = self.bce(N_att, torch.ones_like(N_att))
        A_Nloss = self.bce(A_Natt, torch.zeros_like(A_Natt))

        # -------------------------
        # ✅ NEW: margin separation (LIGHT, PER-SAMPLE)
        # -------------------------
        att_mean = att.mean(dim=1)                      # (B,)
        topk_mean = torch.topk(att, anomaly_topk, dim=-1)[0].mean(-1)  # (B,)

        margin = 0.15
        separation_loss = torch.relu(margin - (topk_mean - att_mean)).mean()

        # -------------------------
        # ✅ NEW: VERY WEAK sparsity (global)
        # -------------------------
        sparsity_loss = att.mean()

        # -------------------------
        # Weights (carefully chosen)
        # -------------------------
        mult_dual = 0.2
        mult_triplet = 0.2
        mult_kl = 0.005
        mult_dist = 0.0001
        mult_smooth = 0.01

        mult_sep = 0.1        # small!
        mult_sparse = 0.02    # VERY small

        # -------------------------
        # Final loss
        # -------------------------
        total_loss = (
            anomaly_loss
            + mult_dual * (A_loss + panomaly_loss + N_loss + A_Nloss)
            + mult_triplet * triplet
            + mult_kl * kl_loss
            + mult_dist * distance
            + mult_smooth * smooth_loss
            + mult_sep * separation_loss
            + mult_sparse * sparsity_loss
        )

        # logging
        loss["total_loss"] = total_loss
        loss["att_loss"] = anomaly_loss
        loss["sep_loss"] = separation_loss
        loss["sparsity"] = sparsity_loss

        return total_loss, loss



class AD_Loss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bce = nn.BCELoss()

    def forward(self, result, _label, step):
        loss = {}
        _label = _label.float()

        triplet = result.get("triplet_margin", torch.tensor(0.0, device=_label.device))
        
        # orig att
        #att = result['frame']
        
        # new att
        att_raw = result['frame']
        att = 0.7 * att_raw.detach() + 0.3 * att_raw
        # optional: clamp to avoid extreme logits (VERY helpful for stability)
        att = torch.clamp(att, 1e-4, 1 - 1e-4)
        
        if ((step + 1) % 5 == 0):
            print("Att raw mean:", att_raw.mean().item())
            print("Att stabilized mean:", att.mean().item())
            print("Att std:", att.std().item())
        
        A_att = result.get("A_att", torch.zeros_like(att))
        N_att = result.get("N_att", torch.zeros_like(att))
        A_Natt = result.get("A_Natt", torch.zeros_like(att))
        N_Aatt = result.get("N_Aatt", torch.zeros_like(att))
        kl_loss = result.get("kl_loss", torch.tensor(0.0, device=_label.device))
        distance = result.get("distance", torch.tensor(0.0, device=_label.device))

         # ✅ NEW: smooth loss
        smooth_loss = result.get("smooth_loss", torch.tensor(0.0, device=_label.device))

        b = max(_label.size(0) // 2, 1)
        t = max(att.size(1), 1)

        #orig
        anomaly_topk = max(t // 16 + 1, 1)
        #anomaly_topk = max(t // 32 + 1, 1)
        #anomaly_topk = max(t // 64, 1)

        anomaly = torch.topk(att, anomaly_topk, dim=-1)[0].mean(-1)
        anomaly_loss = self.bce(anomaly, _label)
        # ===== CURRICULUM WEIGHT FOR ATTENTION LOSS =====
        #att_weight = min(0.1, 0.01 + 0.001 * step)
        
        ## margin based
        # Compute means
        anomaly_mean = att.mean()
        topk_mean = torch.topk(att, anomaly_topk, dim=-1)[0].mean()
        # Margin separation (CRITICAL)
        margin = 0.2
        separation_loss = torch.relu(margin - (topk_mean - anomaly_mean))

        panomaly = torch.topk(1 - N_Aatt, anomaly_topk, dim=-1)[0].mean(-1)
        panomaly = panomaly[:b]
        panomaly_loss = self.bce(panomaly, torch.ones_like(panomaly))

        A_att_topk = torch.topk(A_att, anomaly_topk, dim=-1)[0].mean(-1)
        A_att_topk = A_att_topk[:b]
        A_loss = self.bce(A_att_topk, torch.ones_like(A_att_topk))

        N_loss = self.bce(N_att, torch.ones_like(N_att))
        A_Nloss = self.bce(A_Natt, torch.zeros_like(A_Natt))
        
        # sep loss
        normal_mask = (att < att.mean(dim=1, keepdim=True)).float()
        abn_mask = (att >= att.mean(dim=1, keepdim=True)).float()
        sep_loss = (att * normal_mask).mean() - (att * abn_mask).mean()
        
        # sparsit loss
        sparsity_loss = att.mean()
        
        if ((step + 1) % 5 == 0):
            print("Anomaly mean:", att.mean().item())
            print("Top-k mean:", torch.topk(att, anomaly_topk).values.mean().item())

        # stronger triplet/kl scaling to encourage sharper localization
        mult_dual = 0.15
        mult_triplet = 0.3
        mult_kl = 0.005
        mult_dist = 0.0001
        mult_smooth = 0.05
        mult_sep = 0.2
        mult_sparsity = 0.1
        mult_margin_sep = 0.5
        att_weight = 0.05
        total_loss = (
            att_weight*anomaly_loss
            + mult_dual * (A_loss + panomaly_loss + N_loss + A_Nloss)
            + mult_triplet * triplet
            + mult_kl * kl_loss
            + mult_dist * distance
            + mult_smooth * smooth_loss
            + mult_sep * sep_loss
            + mult_sparsity * sparsity_loss
            + mult_margin_sep * separation_loss
        )

        # gave 83.6% with num_seg 300, anum = nnum = 300
        #mult_dual = 0.2
        #mult_triplet = 0.2
        #mult_kl = 0.01
        #mult_dist = 0.0001
        #mult_smooth = 0.01
        #total_loss = (
        #    anomaly_loss
        #    + mult_dual * (A_loss + panomaly_loss + N_loss + A_Nloss)
        #    + mult_triplet * triplet
        #    + mult_kl * kl_loss
        #    + mult_dist * distance
        #    + mult_smooth * smooth_loss
        #)

        # orig loss
        #total_loss = (
        #    anomaly_loss
        #    + 0.1 * (A_loss + panomaly_loss + N_loss + A_Nloss)
        #    + 0.2 * triplet
        #    + 0.005 * kl_loss
        #    + 0.0001 * distance
        #)

        loss["total_loss"] = total_loss
        loss["att_loss"] = anomaly_loss
        loss["N_Aatt"] = panomaly_loss
        loss["A_loss"] = A_loss
        loss["N_loss"] = N_loss
        loss["A_Nloss"] = A_Nloss
        loss["triplet"] = triplet
        loss["kl_loss"] = kl_loss
        loss["smooth_loss"] = smooth_loss 

        return total_loss, loss

# -------------------------
# MAIN: improved meta-training
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
    os.makedirs(os.path.join(args.save_dir, 'plots'), exist_ok=True)

    print("[INFO] Loading dataset metadata...")
    train_list_fallback = os.path.join(args.root_dir, 'Train', 'RGB')
    #full_train_dataset = UCF_crime(root_dir=train_list_fallback, modal='rgb', mode='Train',
    #                               num_segments=args.num_segments, len_feature=args.len_feature, seed=args.seed)
    train_dataset = UCF_crime(root_dir=train_list_fallback, modal='rgb', mode='Train',
                                   num_segments=args.num_segments, len_feature=args.len_feature, seed=args.seed)
    test_dataset = UCF_crime(root_dir=train_list_fallback, modal='rgb', mode='Test',
                                   num_segments=args.num_segments, len_feature=args.len_feature, seed=args.seed)
    #all_samples = full_train_dataset.samples
    normal_idxs, abnormal_idxs = build_index_cache(train_dataset)
   

    print("[INFO] Initializing model...")
    #model = WSAD(input_size=args.len_feature, flag="Train", a_nums=100, n_nums=100)
    # working
    model = WSAD(input_size=args.len_feature, flag="Train", a_nums=args.a_nums, n_nums=args.n_nums)
    # ABLATION
    #config = {
    #    "use_dual_memory": False,
    #    "use_transformer": True,
    #    "use_local_att": True,
    #    "use_vae": True}
    #model = WSAD(input_size=args.len_feature, flag="Train", a_nums=args.a_nums, n_nums=args.n_nums, config=config)

    # test model, larger nums
    #model = WSAD(input_size=args.len_feature, flag="Train", a_nums=200, n_nums=200)

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.meta_lr, betas=(0.9, 0.999), weight_decay=5e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.meta_iters))
    criterion = AD_Loss()

    test_info = {'step': [], 'auc': [], 'ap': [], 'ac': []}
    best_auc = 0.0
    #best_model_path = os.path.join(args.save_dir, 'best_meta_wsad')
    #best_model_path += '_sk_' + str(args.support_k) + '_qk_' + str(args.query_k) + '_ilr_' + str(args.inner_lr) + '_mlr_' + str(args.meta_lr)

    print("[INFO] Training settings:")
    print(f" device={device} support_k={args.support_k} query_k={args.query_k} inner_lr={args.inner_lr} meta_lr={args.meta_lr} beta={args.beta} warmup_steps={args.warmup_steps}")

    for step in tqdm(range(1, args.meta_iters + 1), total=args.meta_iters, ncols=100):
        model.train()

        # Warmup supervised pretraining (simple AD_Loss) to stabilize representation
        if step <= args.warmup_steps:
            k_sup = max(args.query_k, args.support_k)
            #qx_n_x, qx_n_y = get_random_batch_from_dataset(normal_train_ds, k_sup, device, do_norm=False)
            #qx_a_x, qx_a_y = get_random_batch_from_dataset(abnormal_train_ds, k_sup, device, do_norm=False)
            #sup_x = torch.cat([qx_n_x, qx_a_x], dim=0)
            #sup_y = torch.cat([qx_n_y, qx_a_y], dim=0)
            #sup_x, sup_y = get_random_batch_from_dataset(train_dataset, k_sup, device)
            sup_x, sup_y = get_random_batch_from_dataset_fast(train_dataset, normal_idxs, abnormal_idxs, k_sup, device)
            result = model(sup_x)
            loss_sup, loss_dict_sup = criterion(result, sup_y, step)
            optimizer.zero_grad()
            loss_sup.backward()
            optimizer.step()
            scheduler.step()
            if step % max(1, args.warmup_steps//5) == 0 or args.debug:
                print(f"[WARMUP {step}/{args.warmup_steps}] sup_loss={float(loss_sup.item()):.6f}")
            continue

        # Sample support and query with different k's
        #sx_n_x, sx_n_y = get_random_batch_from_dataset(normal_train_ds, args.support_k, device, do_norm=False)
        #sx_a_x, sx_a_y = get_random_batch_from_dataset(abnormal_train_ds, args.support_k, device, do_norm=False)
        #support_x = torch.cat([sx_n_x, sx_a_x], dim=0)
        #support_y = torch.cat([sx_n_y, sx_a_y], dim=0)
        
        #support_x, support_y = get_random_batch_from_dataset(train_dataset, args.support_k, device, do_norm=False)
        support_x, support_y = get_random_batch_from_dataset_fast(train_dataset, normal_idxs, abnormal_idxs, args.support_k, device)

        # inner loop
        result_support = model(support_x)
        inner_loss, inner_loss_dict = criterion(result_support, support_y, step)

        grads = torch.autograd.grad(inner_loss, model.parameters(), create_graph=True, allow_unused=False)
        #grads = torch.autograd.grad(inner_loss, model.parameters(), create_graph=True, allow_unused=False)

        alpha = args.inner_lr
        fast_weights = {name: param - alpha * 0.5 * g for (name, param), g in zip(model.named_parameters(), grads)}

        # Query (different k)
        #qx_n_x, qx_n_y = get_random_batch_from_dataset(normal_train_ds, args.query_k, device, do_norm = False)
        #qx_a_x, qx_a_y = get_random_batch_from_dataset(abnormal_train_ds, args.query_k, device, do_norm= False)
        #query_x = torch.cat([qx_n_x, qx_a_x], dim=0)
        #query_y = torch.cat([qx_n_y, qx_a_y], dim=0)

        #query_x, query_y = get_random_batch_from_dataset(train_dataset, args.query_k, device, do_norm=False)
        query_x, query_y = get_random_batch_from_dataset_fast(train_dataset, normal_idxs, abnormal_idxs, args.query_k, device)

        result_query = stateless.functional_call(model, fast_weights, (query_x,))
        meta_val_loss, query_loss_dict = criterion(result_query, query_y, step)
        

        total_loss = inner_loss + args.beta * meta_val_loss
        #total_loss = inner_loss + args.beta * meta_val_loss_smooth
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
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
            test_loader = data.DataLoader(UCF_crime(root_dir = args.root_dir, mode = 'Test', modal = 'rgb', num_segments = args.num_segments, len_feature = args.len_feature),
                                          batch_size = 1, shuffle = False, num_workers = args.num_workers)
            #auc_score, ap_score, acc = align_and_evaluate(model, config, test_dataset, step, model_file=None, save_dir=args.save_dir)
            auc_score, ap_score, acc = test(model, config, test_loader, step, model_file=None, save_dir=args.save_dir)
            #diagnostic_test(model, config, test_loader)

            model.flag = "Train"
            test_info["step"].append(step)
            test_info["auc"].append(auc_score)
            test_info["ap"].append(ap_score)
            test_info["ac"].append(acc)
            print(f"[EVAL @ {step}] AUC={auc_score:.4f} | AP={ap_score:.4f} | ACC={acc:.4f}")
            if auc_score > best_auc:
                best_auc = auc_score
                best_model_path = os.path.join(args.save_dir, 'best_meta_wsad')
                best_model_path += '_sk_' + str(args.support_k) + '_qk_' + str(args.query_k) + '_ilr_' + str(args.inner_lr) + '_mlr_' + str(args.meta_lr) + '_beta_' + str(args.beta) + '_segments_' + str(args.num_segments)
                best_model_path += '_anums_' + str(args.a_nums) + '_nnums_' + str(args.n_nums) + '_auc_' + str(best_auc) + '.pth'
                torch.save(model.state_dict(), best_model_path)
                print(f"[INFO] New best AUC: {best_auc:.4f} saved to {best_model_path}")

    print("[INFO] Meta-training completed. Best AUC: {:.4f}".format(best_auc))


if __name__ == '__main__':
    main()
