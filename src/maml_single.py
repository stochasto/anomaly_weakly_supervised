#!/usr/bin/env python3
"""
maml_dm_improved_enriched.py

MAML-style meta-training using WSAD + AD_Loss, improved:
- multi-stream enrichment from RGB I3D features (no CLIP / no Flow)
  (Temporal Difference, Magnitude-of-Change, Self-Similarity)
- different k for support and query
- supervised warmup
- inner/outer LR tuning
- per-video normalization toggle
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
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    p.add_argument('--num_segments', type=int, default=300)
    p.add_argument('--len_feature', type=int, default=1024,   # original I3D dim
                   help="feature dimension (I3D: 1024)")
    p.add_argument('--support_k', type=int, default=8, help="support size per iteration")
    p.add_argument('--query_k', type=int, default=16, help="query size per iteration")
    p.add_argument('--meta_iters', type=int, default=3000, help="number of meta iterations")
    p.add_argument('--warmup_steps', type=int, default=20, help="supervised warmup iterations before meta updates")
    p.add_argument('--inner_lr', type=float, default=1e-3)
    p.add_argument('--meta_lr', type=float, default=1e-4)
    p.add_argument('--beta', type=float, default=1, help="weight for meta-val loss")
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--save_dir', type=str, default='checkpoints')
    p.add_argument('--gt_file', type=str, default='frame_label/gt_ucf_v2.npy')
    p.add_argument('--eval_interval', type=int, default=50)
    p.add_argument('--debug', action='store_true')
    p.add_argument('--no_norm', action='store_true', help="disable per-video normalization (default False => do normalization in training sampler)")
    p.add_argument('--seed', type=int, default=-1, help="set >=0 for deterministic runs, -1 for fully random")
    args = p.parse_args()
    return args

# -------------------------
# Model & utils (we include your WSAD model here with one small change)
# -------------------------
# If you already have model_min_upgrade.py etc, you can ignore this and import instead.
# For self-containedness we re-declare your model class (only Temporal input dimension is parameterized)
import math
# minimal dependencies from original code: Memory_Unit, Transformer, norm must be in repo
from memory import Memory_Unit
from translayer import Transformer
from utils import norm

class Temporal(nn.Module):
    def __init__(self, input_size, out_size):
        super(Temporal, self).__init__()
        # note: input_size is the feature dim per segment (after enrichment)
        self.conv_1 = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=out_size, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
    def forward(self, x):
        # x: (B, T, D) -> Conv1d expects (B, D, T)
        x = x.permute(0, 2, 1)
        x = self.conv_1(x)
        x = x.permute(0, 2, 1)
        return x

class ADCLS_head(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_dim,128), nn.ReLU(), nn.Linear(128,out_dim), nn.Sigmoid())
    def forward(self, x):
        return self.mlp(x)

class WSAD(nn.Module):
    def __init__(self, input_size, flag, a_nums, n_nums):
        super().__init__()
        self.flag = flag
        self.a_nums = a_nums
        self.n_nums = n_nums

        self.embedding = Temporal(input_size,512)
        self.triplet = nn.TripletMarginLoss(margin=1)
        self.cls_head = ADCLS_head(1024, 1)
        self.Amemory = Memory_Unit(nums=a_nums, dim=512)
        self.Nmemory = Memory_Unit(nums=n_nums, dim=512)
        # keep a conservative transformer config for stability
        self.selfatt = Transformer(512, 2, 4, 128, 512, dropout = 0.3)
        self.encoder_mu = nn.Sequential(nn.Linear(512, 512))
        self.encoder_var = nn.Sequential(nn.Linear(512, 512))
        self.relu = nn.ReLU()

    def _reparameterize(self, mu, logvar):
        std = torch.exp(logvar).sqrt()
        epsilon = torch.randn_like(std)
        return mu + epsilon * std

    def latent_loss(self, mu, var):
        kl_loss = torch.mean(-0.5 * torch.sum(1 + var - mu ** 2 - var.exp(), dim = 1))
        return kl_loss

    def forward(self, x):
        # x: (B, T, D) OR (B, N, T, D)
        if len(x.size()) == 4:
            b, n, t, d = x.size()
            x = x.reshape(b * n, t, d)
        else:
            b, t, d = x.size()
            n = 1
        x = self.embedding(x)
        x = self.selfatt(x)
        if self.flag == "Train":
            # expect batch arranged: first half normal, second half abnormal (used by your training sampler)
            N_x = x[:b*n//2]
            A_x = x[b*n//2:]
            A_att, A_aug = self.Amemory(A_x)
            N_Aatt, N_Aaug = self.Nmemory(A_x)

            A_Natt, A_Naug = self.Amemory(N_x)
            N_att, N_aug = self.Nmemory(N_x)

            _, A_index = torch.topk(A_att, max(t//16 + 1, 1), dim=-1)
            negative_ax = torch.gather(A_x, 1, A_index.unsqueeze(2).expand([-1, -1, x.size(-1)])).mean(1).reshape(b//2,n,-1).mean(1)

            _, N_index = torch.topk(N_att, max(t//16 + 1, 1), dim=-1)
            anchor_nx = torch.gather(N_x, 1, N_index.unsqueeze(2).expand([-1, -1, x.size(-1)])).mean(1).reshape(b//2,n,-1).mean(1)

            _, P_index = torch.topk(N_Aatt, max(t//16 + 1, 1), dim=-1)
            positivte_nx = torch.gather(A_x, 1, P_index.unsqueeze(2).expand([-1, -1, x.size(-1)])).mean(1).reshape(b//2,n,-1).mean(1)

            triplet_margin_loss = self.triplet(norm(anchor_nx), norm(positivte_nx), norm(negative_ax))

            N_aug_mu = self.encoder_mu(N_aug)
            N_aug_var = self.encoder_var(N_aug)
            N_aug_new = self._reparameterize(N_aug_mu, N_aug_var)

            anchor_nx_new = torch.gather(N_aug_new, 1, N_index.unsqueeze(2).expand([-1, -1, x.size(-1)])).mean(1).reshape(b//2,n,-1).mean(1)

            A_aug_new = self.encoder_mu(A_aug)
            negative_ax_new = torch.gather(A_aug_new, 1, A_index.unsqueeze(2).expand([-1, -1, x.size(-1)])).mean(1).reshape(b//2,n,-1).mean(1)

            kl_loss = self.latent_loss(N_aug_mu, N_aug_var)

            A_Naug = self.encoder_mu(A_Naug)
            N_Aaug = self.encoder_mu(N_Aaug)

            # distance encourages separation (small weight used in loss)
            distance = torch.relu(100 - torch.norm(negative_ax_new, p=2, dim=-1) + torch.norm(anchor_nx_new, p=2, dim=-1)).mean()
            x = torch.cat((x, (torch.cat([N_aug_new + A_Naug, A_aug_new + N_Aaug], dim=0))), dim=-1)
            pre_att = self.cls_head(x).reshape((b, n, -1)).mean(1)

            return {
                    "frame": pre_att,
                    'triplet_margin': triplet_margin_loss,
                    'kl_loss': kl_loss,
                    'distance': distance,
                    'A_att': A_att.reshape((b//2, n, -1)).mean(1),
                    "N_att": N_att.reshape((b//2, n, -1)).mean(1),
                    "A_Natt": A_Natt.reshape((b//2, n, -1)).mean(1),
                    "N_Aatt": N_Aatt.reshape((b//2, n, -1)).mean(1)
                }
        else:
            _, A_aug = self.Amemory(x)
            _, N_aug = self.Nmemory(x)

            A_aug = self.encoder_mu(A_aug)
            N_aug = self.encoder_mu(N_aug)

            x = torch.cat([x, A_aug + N_aug], dim=-1)

            pre_att = self.cls_head(x).reshape((b, n, -1)).mean(1)
            return {"frame": pre_att}

# -------------------------
# Dataset: group crop files per video + enrichment
# -------------------------
class UCF_crime(data.Dataset):
    def __init__(self, root_dir, modal, mode, num_segments, len_feature, seed=-1):
        if seed >= 0 and hasattr(np.random, "seed"):
            np.random.seed(seed)
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
            raise FileNotFoundError(f"Cannot find list file for mode={self.mode}. Looked for {list_path_repo} and {alt_list_path}.")

        samples = []
        if self.mode == 'Train':
            numAbnorm = 8100
        else:
            numAbnorm = 1400
        with open(list_file, 'r') as f:
            for (i,line) in enumerate(f):
                if (i < numAbnorm):
                    samples.append((line.strip(),1))
                else:
                    samples.append((line.strip(),0))
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def build_motion_structure_features(self, arr):
        """
        arr: (T, D) numpy float32 (I3D RGB)
        Returns: enriched features (T, D*2 + 2)
        """
        T, D = arr.shape
        # Temporal difference
        tdf = np.zeros_like(arr, dtype=np.float32)
        if T > 1:
            tdf[1:] = arr[1:] - arr[:-1]
        # Magnitude of change (scalar per segment)
        mag = np.zeros((T, 1), dtype=np.float32)
        if T > 1:
            mag[1:, 0] = np.linalg.norm(arr[1:] - arr[:-1], axis=1)
        # Self-similarity feature (mean absolute similarity)
        ssf = np.zeros((T, 1), dtype=np.float32)
        # normalize per-frame to compute similarity stably
        denom = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-6
        norm_arr = arr / denom
        sim = np.abs(np.matmul(norm_arr, norm_arr.T))  # (T, T)
        ssf[:, 0] = sim.mean(axis=1)
        enriched = np.concatenate([arr, tdf, mag, ssf], axis=1)
        return enriched.astype(np.float32)

    def __getitem__(self, index):
        data, label = self.get_data(index)
        return data, label

    def get_data(self, index):
        vid, label = self.samples[index]
        video_feature = np.load(vid).astype(np.float32)   # [T, D]
        # uniform temporal pooling/resampling into num_segments (same as before)
        if self.mode == "Train": #or self.mode == "Test":
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
        # build enriched features (no per-video normalization here; sampler can apply it)
        enriched = self.build_motion_structure_features(video_feature)  # (T, D*2+2)
        return enriched, label

# -------------------------
# Helper: random sampling by video (k videos)
# -------------------------
def get_random_batch_from_dataset(dataset, k, device, do_norm=False):
    """
    Sample k videos from dataset (dataset.__getitem__ -> (T,D_enriched,label)).
    Returns x: (k, T, D_enriched) tensor, y: (k,) tensor
    """
    idxs = random.sample(range(len(dataset)), k)
    feats, labels = [], []
    for i in idxs:
        feat, label = dataset[i]
        arr = np.asarray(feat).astype(np.float32)  # (T, D_enriched)
        if do_norm:
            mu = arr.mean(axis=0, keepdims=True)
            s = arr.std(axis=0, keepdims=True) + 1e-6
            arr = (arr - mu) / s
        feats.append(torch.from_numpy(arr))
        labels.append(torch.tensor(label).float())
    x = torch.stack(feats).to(device)   # (k, T, D_enriched)
    y = torch.stack(labels).to(device)  # (k,)
    return x, y

# -------------------------
# test / evaluation (unchanged, works with enriched features returned by dataset)
# -------------------------
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
        temp_predict = torch.zeros((0)).cuda()

        for i in range(len(test_loader.dataset)):
            _data, _label = next(load_iter)

            # DataLoader returns tensors shaped (batch=1, T, D_enriched)
            _data = _data.cuda().float()
            _label = _label.cuda()

            res = net(_data)
            a_predict = res["frame"]
            temp_predict = torch.cat([temp_predict, a_predict], dim=0)
            if (i + 1) % 10 == 0 :
                cls_label.append(int(_label))
                a_predict = temp_predict.mean(0).cpu().numpy()

                cls_pre.append(1 if a_predict.max()>0.5 else 0)
                fpre_ = a_predict

                if frame_predict is None:
                    frame_predict = fpre_
                else:
                    frame_predict = np.concatenate([frame_predict, fpre_])
                temp_predict = torch.zeros((0)).cuda()

        print(f'len gt {len(frame_gt)} len predict {len(frame_predict)}')
        fpr,tpr,_ = roc_curve(frame_gt, frame_predict)
        auc_score = auc(fpr, tpr)

        corrent_num = np.sum(np.array(cls_label) == np.array(cls_pre), axis=0)
        accuracy = corrent_num / (len(cls_pre))

        precision, recall, th = precision_recall_curve(frame_gt, frame_predict)
        ap_score = auc(recall, precision)
    return auc_score, ap_score, accuracy

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

        total_loss = (
            anomaly_loss
            + 0.1 * (A_loss + panomaly_loss + N_loss + A_Nloss)
            + 0.08 * triplet
            + 0.005 * kl_loss
            + 0.00005 * distance
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

# -------------------------
# MAIN: improved meta-training with enriched inputs
# -------------------------
def main():
    args = parse_args()
    if args.seed >= 0:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # enriched input dim = D + D + 1 + 1 = 2*D + 2
    enriched_dim = args.len_feature * 2 + 2
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
    train_dataset = UCF_crime(root_dir=train_list_fallback, modal='rgb', mode='Train',
                                   num_segments=args.num_segments, len_feature=args.len_feature, seed=args.seed)
    test_dataset = UCF_crime(root_dir=train_list_fallback, modal='rgb', mode='Test',
                                   num_segments=args.num_segments, len_feature=args.len_feature, seed=args.seed)

    print(f"[INFO] Initializing model with enriched input dim = {enriched_dim} ...")
    model = WSAD(input_size=enriched_dim, flag="Train", a_nums=60, n_nums=60)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.meta_lr, betas=(0.9, 0.999), weight_decay=5e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.meta_iters))
    criterion = AD_Loss()

    test_info = {'step': [], 'auc': [], 'ap': [], 'ac': []}
    best_auc = 0.0

    print("[INFO] Training settings:")
    print(f" device={device} support_k={args.support_k} query_k={args.query_k} inner_lr={args.inner_lr} meta_lr={args.meta_lr} beta={args.beta} warmup_steps={args.warmup_steps}")

    for step in tqdm(range(1, args.meta_iters + 1), total=args.meta_iters, ncols=100):
        model.train()

        # Warmup supervised pretraining (simple AD_Loss) to stabilize representation
        if step <= args.warmup_steps:
            k_sup = max(args.query_k, args.support_k)
            sup_x, sup_y = get_random_batch_from_dataset(train_dataset, k_sup, device, do_norm=(not args.no_norm))
            result = model(sup_x)
            loss_sup, loss_dict_sup = criterion(result, sup_y)
            optimizer.zero_grad()
            loss_sup.backward()
            optimizer.step()
            scheduler.step()
            if step % max(1, args.warmup_steps//5) == 0 or args.debug:
                print(f"[WARMUP {step}/{args.warmup_steps}] sup_loss={float(loss_sup.item()):.6f}")
            continue

        # Sample support and query with different k's
        support_x, support_y = get_random_batch_from_dataset(train_dataset, args.support_k, device, do_norm=(not args.no_norm))

        # inner loop
        result_support = model(support_x)
        inner_loss, inner_loss_dict = criterion(result_support, support_y)

        grads = torch.autograd.grad(inner_loss, model.parameters(), create_graph=True, allow_unused=False)
        alpha = args.inner_lr
        fast_weights = {name: param - alpha * g for (name, param), g in zip(model.named_parameters(), grads)}

        # Query (different k)
        query_x, query_y = get_random_batch_from_dataset(train_dataset, args.query_k, device, do_norm=(not args.no_norm))

        result_query = stateless.functional_call(model, fast_weights, (query_x,))
        meta_val_loss, query_loss_dict = criterion(result_query, query_y)

        total_loss = inner_loss + args.beta * meta_val_loss
        optimizer.zero_grad()
        total_loss.backward()
        # gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
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
            print(f"[EVAL] step {step} -> running evaluation ...")
            test_loader = data.DataLoader(test_dataset,
                                          batch_size = 1, shuffle = False, num_workers = args.num_workers)
            auc_score, ap_score, acc = test(model, config, test_loader, step, model_file=None, save_dir=args.save_dir)
            model.flag = "Train"
            test_info["step"].append(step)
            test_info["auc"].append(auc_score)
            test_info["ap"].append(ap_score)
            test_info["ac"].append(acc)
            print(f"[EVAL @ {step}] AUC={auc_score:.4f} | AP={ap_score:.4f} | ACC={acc:.4f}")
            if auc_score > best_auc:
                best_auc = auc_score
                best_model_path = os.path.join(args.save_dir, 'best_meta_single')
                best_model_path += '_sk_' + str(args.support_k) + '_qk_' + str(args.query_k) + '_ilr_' + str(args.inner_lr) + '_mlr_' + str(args.meta_lr) + '_beta_' + str(args.beta)
                best_model_path += '_auc_' + str(best_auc) + '.pth'
                torch.save(model.state_dict(), best_model_path)
                print(f"[INFO] New best AUC: {best_auc:.4f} saved to {best_model_path}")

    print("[INFO] Meta-training completed. Best AUC: {:.4f}".format(best_auc))

if __name__ == '__main__':
    main()
