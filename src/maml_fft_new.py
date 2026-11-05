#!/usr/bin/env python3
"""
maml_dm_improved.py (with FFT-Conv1D + Wavelet spectral embeddings)

Minimal, safe integration:
- keeps MAML/Sampling/AD_Loss/WSAD logic unchanged
- adds SpectralEmbed module and applies it before model forward
- embedder parameters are trained together with WSAD
- choose --spec_embed in {none, fft, wavelet, both}
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
    p.add_argument('--len_feature', type=int, default=1024,
                   help="feature dimension (I3D: 1024)")
    p.add_argument('--support_k', type=int, default=8, help="support size per class")
    p.add_argument('--query_k', type=int, default=16, help="query size per class")
    p.add_argument('--meta_iters', type=int, default=4000, help="number of meta iterations")
    p.add_argument('--warmup_steps', type=int, default=1, help="supervised warmup iterations before meta updates")
    p.add_argument('--inner_lr', type=float, default=1e-3)
    p.add_argument('--meta_lr', type=float, default=1e-4)
    p.add_argument('--beta', type=float, default=1.0, help="weight for meta-val loss")
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--save_dir', type=str, default='checkpoints')
    p.add_argument('--gt_file', type=str, default='frame_label/gt_ucf_v2.npy')
    p.add_argument('--eval_interval', type=int, default=50)
    p.add_argument('--debug', action='store_true')
    p.add_argument('--no_norm', action='store_true', help="disable per-video normalization")
    p.add_argument('--seed', type=int, default=-1, help="set >=0 for deterministic runs, -1 for fully random")
    # spectral embedding choice
    p.add_argument('--spec_embed', type=str, default='fft', choices=['none', 'fft', 'wavelet', 'both'],
                   help="Spectral embedding: 'fft' (FFT-Conv1D), 'wavelet' (learnable), 'both', or 'none'")
    # hyperparams for embedder
    p.add_argument('--fft_hidden', type=int, default=256, help="hidden channels for fft conv")
    p.add_argument('--embed_alpha', type=float, default=0.5, help="initial scaling for spectral embed addback")
    args = p.parse_args()
    return args

# -------------------------
# repo model & utils (must be present in repo)
# -------------------------
from model_min_upgrade import WSAD
import utils

# -------------------------
# Spectral embedding module (FFT-Conv1D + Wavelet depthwise)
# -------------------------
class SpectralEmbed(nn.Module):
    """
    Spectral embedding with conditional computation.
    Modes:
      - 'fft'     : only FFT → Conv1D
      - 'wavelet' : only wavelet path
      - 'both'    : fft + wavelet
    """
    def __init__(self, feat_dim, fft_hidden=256, alpha_init=0.5,
                 device='cpu', mode='fft'):
        super().__init__()
        self.D = feat_dim
        self.mode = mode.lower()
        self.device = device
        self.relu = nn.ReLU()

        # ---------------- FFT PATH ----------------
        if self.mode in ('fft', 'both'):
            self.fft_conv1 = nn.Conv1d(self.D, fft_hidden, kernel_size=3, padding=1)
            self.fft_bn1   = nn.BatchNorm1d(fft_hidden)
            self.fft_conv2 = nn.Conv1d(fft_hidden, self.D, kernel_size=3, padding=1)

        # ---------------- WAVELET PATH ----------------
        if self.mode in ('wavelet', 'both'):
            self.wave_dw = nn.Conv1d(self.D, self.D, kernel_size=2, padding=1,
                                     groups=self.D, bias=False)
            self.wave_conv = nn.Conv1d(self.D, self.D, kernel_size=3, padding=1)

            # Haar initialization
            with torch.no_grad():
                k = torch.zeros_like(self.wave_dw.weight)
                k[:, 0, 0] = 0.5
                k[:, 0, 1] = 0.5
                self.wave_dw.weight.copy_(k)

        # ---------------- Combination ----------------
        if self.mode != 'none':
            self.comb_bn = nn.BatchNorm1d(self.D)
            self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))

    def forward(self, x):
        """
        x: (B, T, D)
        """
        if self.mode == 'none':
            return x

        B, T, D = x.shape
        out = 0

        # ---------------- FFT FORWARD ----------------
        if self.mode in ('fft', 'both'):
            F = torch.fft.rfft(x, dim=1)                # (B, F, D) complex
            F_mag = torch.abs(F).permute(0, 2, 1)       # (B, D, F)

            h = self.fft_conv1(F_mag)
            h = self.relu(self.fft_bn1(h))
            h = self.fft_conv2(h)                       # (B, D, F)

            h_up = nn.functional.interpolate(h, size=T, mode='linear',
                                             align_corners=False)
            h_up = h_up.permute(0, 2, 1)                # (B, T, D)
            out = out + h_up

        # ---------------- WAVELET FORWARD ----------------
        if self.mode in ('wavelet', 'both'):
            Xc = x.permute(0, 2, 1)                     # (B, D, T)
            w = self.relu(self.wave_dw(Xc))
            w = self.wave_conv(w)
            w = w.permute(0, 2, 1)                      # (B, T, D)
            out = out + w

        # ---------------- COMBINE ----------------
        out_bn = self.comb_bn(out.permute(0, 2, 1))
        out_bn = out_bn.permute(0, 2, 1)

        return x + self.alpha * out_bn

# -------------------------
# Dataset: group crop files per video (unchanged except adding feat_aug arg to constructor)
# -------------------------
class UCF_crime(data.Dataset):
    def __init__(self, root_dir, modal, mode, num_segments, len_feature, seed=-1, feat_aug='none'):
        if seed >= 0 and hasattr(utils, "set_seed"):
            utils.set_seed(seed)
        self.mode = mode
        self.modal = modal
        self.num_segments = num_segments
        self.len_feature = len_feature
        self.root_dir = root_dir
        self.feat_aug = feat_aug

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

    def __getitem__(self, index):
        data, label = self.get_data(index)
        return data, label
      

    def get_data(self, index):
        vid, label = self.samples[index]
        video_feature = np.load(vid).astype(np.float32)   # [T, feat_dim]

        # uniform pooling/resampling (same as before)
        T_orig = video_feature.shape[0]
        D = video_feature.shape[1]
        if self.mode == "Train":
            T = T_orig
            new_feat = np.zeros((self.num_segments, D), dtype=np.float32)
            r = np.linspace(0, T, self.num_segments + 1, dtype=np.int32)
            for i in range(self.num_segments):
                if r[i] != r[i+1]:
                    new_feat[i, :] = np.mean(video_feature[r[i]:r[i+1], :], axis=0)
                else:
                    new_feat[i, :] = video_feature[r[i], :]
            video_feature = new_feat

        # NOTE: spectral embedding is applied as a *learned module* in main (not here),
        # so dataset returns plain video_feature (T', D)
        return video_feature, label

# -------------------------
# Helper: random sampling by video (k videos)
# -------------------------
def get_random_batch_from_dataset(dataset, k, device, do_norm=False):
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

# -------------------------
# test() - modified to accept embedder and use it before model forward
# -------------------------
def test(net, config, test_loader, step, model_file=None, save_dir='checkpoints', do_norm=False, embedder=None, device=None):
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
        temp_predict = torch.zeros((0)).to(device)
        
        for i in range(len(test_loader.dataset)):
            _data, _label = next(load_iter)
            # _data: (1, T, D)
            _data = _data.to(device).float()
            _label = _label.to(device)

            # apply embedder if provided
            if embedder is not None:
                _data = embedder(_data)   # (1, T, D)

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
                temp_predict = torch.zeros((0)).to(device)
   
        print(f'len gt {len(frame_gt)} len predict {len(frame_predict)}')
        fpr,tpr,_ = roc_curve(frame_gt, frame_predict)
        auc_score = auc(fpr, tpr)
    
        corrent_num = np.sum(np.array(cls_label) == np.array(cls_pre), axis=0)
        accuracy = corrent_num / (len(cls_pre))
        
        precision, recall, th = precision_recall_curve(frame_gt, frame_predict,)
        ap_score = auc(recall, precision)
    return auc_score, ap_score, accuracy

# -------------------------
# AD_Loss (unchanged)
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

# -------------------------
# MAIN: improved meta-training (only small changes to use embedder)
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

    train_dataset = UCF_crime(root_dir=train_list_fallback, modal='rgb', mode='Train',
                                   num_segments=args.num_segments, len_feature=args.len_feature, seed=args.seed, feat_aug='none')
    test_dataset = UCF_crime(root_dir=train_list_fallback, modal='rgb', mode='Test',
                                   num_segments=args.num_segments, len_feature=args.len_feature, seed=args.seed, feat_aug='none')

    # Spectral embedder (learned module). Keep it separate and apply before WSAD.
    embedder = None

    if args.spec_embed != 'none':
        embedder = SpectralEmbed(feat_dim=args.len_feature, fft_hidden=args.fft_hidden,
                                 alpha_init=args.embed_alpha, device=device, mode=args.spec_embed).to(device)
        print(f'embedder is {args.spec_embed}')
    else:
        print("[INFO] No spectral embedding used (spec_embed=none)")

    print("[INFO] Initializing model...")
    model = WSAD(input_size=args.len_feature, flag="Train", a_nums=200, n_nums=200)
    model.to(device)

    # optimizer: include embedder parameters if present
    opt_params = list(model.parameters())
    if embedder is not None:
        opt_params += list(embedder.parameters())
    optimizer = torch.optim.Adam(opt_params, lr=args.meta_lr, betas=(0.9, 0.999), weight_decay=5e-5)
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
            sup_x, sup_y = get_random_batch_from_dataset(train_dataset, k_sup, device, do_norm = False)
            # apply embedder if present
            if embedder is not None:
                sup_x = embedder(sup_x)
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
        support_x, support_y = get_random_batch_from_dataset(train_dataset, args.support_k, device, do_norm=False)

        # apply embedder BEFORE inner loop forward (embedder is not adapted in inner loop)
        if embedder is not None:
            support_x = embedder(support_x)

        # inner loop
        result_support = model(support_x)
        inner_loss, inner_loss_dict = criterion(result_support, support_y)

        grads = torch.autograd.grad(inner_loss, model.parameters(), create_graph=True, allow_unused=False)
        alpha = args.inner_lr
        fast_weights = {name: param - alpha * g for (name, param), g in zip(model.named_parameters(), grads)}

        # Query (different k)
        query_x, query_y = get_random_batch_from_dataset(train_dataset, args.query_k, device, do_norm=False)
        if embedder is not None:
            query_x = embedder(query_x)

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
            print(f"[EVAL] step {step} -> running evaluation ...")
            test_loader = data.DataLoader(UCF_crime(root_dir = args.root_dir, mode = 'Test', modal = 'rgb', num_segments = args.num_segments, len_feature = args.len_feature, feat_aug='none'),
                                          batch_size = 1, shuffle = False, num_workers = args.num_workers)
            auc_score, ap_score, acc = test(model, config, test_loader, step, model_file=None, save_dir=args.save_dir, do_norm=False, embedder=embedder, device=device)
            model.flag = "Train"
            test_info["step"].append(step)
            test_info["auc"].append(auc_score)
            test_info["ap"].append(ap_score)
            test_info["ac"].append(acc)
            print(f"[EVAL @ {step}] AUC={auc_score:.4f} | AP={ap_score:.4f} | ACC={acc:.4f}")
            if auc_score > best_auc:
                best_auc = auc_score
                best_model_path = os.path.join(args.save_dir, 'best_meta_wsad')
                best_model_path += '_sk_' + str(args.support_k) + '_qk_' + str(args.query_k) + '_ilr_' + str(args.inner_lr) + '_mlr_' + str(args.meta_lr) + '_beta_' + str(args.beta)
                best_model_path += '_auc_' + str(best_auc) + '.pth'
                torch.save(model.state_dict(), best_model_path)
                print(f"[INFO] New best AUC: {best_auc:.4f} saved to {best_model_path}")

    print("[INFO] Meta-training completed. Best AUC: {:.4f}".format(best_auc))


if __name__ == '__main__':
    main()
