#!/usr/bin/env python3
"""
maml_ucf.py

MAML-style meta-training for UCF-Crime (mirrors XD training script, constant inner LR).
"""

import os
import random
import pickle
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
from torch.utils.tensorboard import SummaryWriter
warnings.filterwarnings("ignore")

# -------------------------
# CLI / Config
# -------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--root_dir', type=str, default='/mnt/c/code/data/I3D',
                   help="base data directory")
    p.add_argument('--num_segments', type=int, default=300)
    p.add_argument('--a_nums', type=int, default=200)
    p.add_argument('--n_nums', type=int, default=200)
    p.add_argument('--len_feature', type=int, default=1024)
    p.add_argument('--support_k', type=int, default=8)
    p.add_argument('--query_k', type=int, default=16)
    p.add_argument('--meta_iters', type=int, default=4000)
    p.add_argument('--warmup_steps', type=int, default=100)
    p.add_argument('--inner_lr', type=float, default=1e-3)
    p.add_argument('--meta_lr', type=float, default=1e-4)
    p.add_argument('--max_grad_norm', type=float, default=2.5)
    p.add_argument('--beta', type=float, default=1.0)
    p.add_argument('--num_workers', type=int, default=8)
    p.add_argument('--save_dir', type=str, default='checkpoints')
    p.add_argument('--gt_file', type=str, default='frame_label/gt_ucf_v2.npy')
    p.add_argument('--eval_interval', type=int, default=100)
    p.add_argument('--debug', action='store_true')
    p.add_argument('--seed', type=int, default=-1)
    p.add_argument('--log_dir', type=str, default='logs_ucf')
    p.add_argument('--resume', type=str, default='')
    args = p.parse_args()
    return args

# -------------------------
# Model
# -------------------------
from model_gemini_xd_v3 import WSAD
import utils

# -------------------------
# Dataset
# -------------------------
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
            raise FileNotFoundError(f"Cannot find list file for mode={self.mode}.")

        samples = []
        numAbnorm = 8100 if self.mode == 'Train' else 1400
        with open(list_file, 'r') as f:
            for i, line in enumerate(f):
                if i < numAbnorm:
                    samples.append((line.strip(), 1))
                else:
                    samples.append((line.strip(), 0))
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        data, label = self.get_data(index)
        return data, label

    def get_data(self, index):
        vid, label = self.samples[index]
        video_feature = np.load(vid).astype(np.float32)
        if self.mode == "Train":
            T, D = video_feature.shape
            new_feat = np.zeros((self.num_segments, D), dtype=np.float32)
            r = np.linspace(0, T, self.num_segments + 1, dtype=np.int32)
            for i in range(self.num_segments):
                if r[i] != r[i+1]:
                    new_feat[i, :] = np.mean(video_feature[r[i]:r[i+1], :], axis=0)
                else:
                    new_feat[i, :] = video_feature[r[i], :]
            video_feature = new_feat
            # temporal masking augmentation
            if random.random() < 0.25:
                mask_len = random.randint(int(self.num_segments * 0.05), int(self.num_segments * 0.15))
                mask_start = random.randint(0, self.num_segments - mask_len)
                video_feature[mask_start:mask_start + mask_len] *= random.uniform(0.2, 0.6)
        return video_feature, label

def build_index_cache(dataset, cache_path="index_cache_ucf.pkl", rebuild=False):
    if os.path.exists(cache_path) and not rebuild:
        print(f"[INFO] Loading index cache from {cache_path}")
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
        return cache["normal"], cache["abnormal"]

    print("[INFO] Building index cache (one-time cost)...")
    normal_idxs = []
    abnormal_idxs = []
    for i in range(len(dataset)):
        label = dataset.samples[i][1]
        if label == 0:
            normal_idxs.append(i)
        else:
            abnormal_idxs.append(i)
    cache = {"normal": normal_idxs, "abnormal": abnormal_idxs}
    with open(cache_path, "wb") as f:
        pickle.dump(cache, f)
    print(f"[INFO] Saved index cache to {cache_path}")
    return normal_idxs, abnormal_idxs

# -------------------------
# Batches
# -------------------------
def get_balanced_batch(dataset, normal_idxs, abnormal_idxs, k, device):
    half = k // 2
    n_idxs = random.sample(normal_idxs, half)
    a_idxs = random.sample(abnormal_idxs, half)
    idxs = n_idxs + a_idxs
    feats, labels = [], []
    for i in idxs:
        feat, label = dataset[i]
        feats.append(torch.from_numpy(np.asarray(feat).astype(np.float32)))
        labels.append(torch.tensor(label).float())
    x = torch.stack(feats).to(device)
    y = torch.stack(labels).to(device)
    return x, y

def compute_grad_norm(parameters):
    total_norm = 0.0
    for p in parameters:
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

# -------------------------
# Test function (original 10-crop, no repeat, gt_ucf_v2.npy)
# -------------------------
def test(net, config, test_loader, step, model_file=None, save_dir='checkpoints', do_norm=False):
    with torch.no_grad():
        net.eval()
        net.flag = "Test"
        if model_file is not None:
            net.load_state_dict(torch.load(model_file))

        load_iter = iter(test_loader)
        frame_gt = np.load("frame_label/gt_ucf_v2.npy")
        frame_predict = None
        
        cls_label = []
        cls_pre = []
        temp_predict = torch.zeros((0)).cuda()
        
        for i in range(len(test_loader.dataset)):
            _data, _label = next(load_iter)
            _data = _data.cuda()
            _label = _label.cuda()

            res = net(_data)
            a_predict = torch.sigmoid(res["frame"])
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
        
        precision, recall, th = precision_recall_curve(frame_gt, frame_predict,)
        ap_score = auc(recall, precision)
    return auc_score, ap_score, accuracy

# -------------------------
# AD_Loss with UCF weights
# -------------------------
class AD_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.topk_frac_override = None   # uses default 16

    def forward(self, result, _label):
        loss = {}
        _label = _label.float()
        device = _label.device

        att = result['frame']
        if att.dim() == 2:
            att = att.unsqueeze(-1)
        B, T, _ = att.shape
        b = B // 2

        A_att  = result["A_att"]
        N_att  = result["N_att"]
        N_Aatt = result["N_Aatt"]
        A_Natt = result["A_Natt_raw"]

        triplet = result.get("triplet_margin", torch.tensor(0.0, device=device))
        kl      = result.get("kl_loss", torch.tensor(0.0, device=device))
        ortho   = result.get("ortho_loss", torch.tensor(0.0, device=device))

        frac = self.topk_frac_override if self.topk_frac_override is not None else 16
        k_top = max(T // frac + 1, 1)

        anomaly_scores = torch.topk(att.squeeze(-1), k_top, dim=1)[0].mean(1)
        anomaly_loss = self.bce(anomaly_scores, _label)

        A_topk = torch.topk(A_att, k_top, dim=1)[0].mean(1)
        A_loss = self.bce(A_topk, torch.ones(b, device=device))

        panomaly_topk = torch.topk(N_Aatt, k_top, dim=1)[0].mean(1)
        panomaly_loss = self.bce(panomaly_topk, torch.ones(b, device=device))

        N_topk = torch.topk(N_att, k_top, dim=1)[0].mean(1)
        N_loss = self.bce(N_topk, torch.ones(b, device=device))

        A_N_topk = torch.topk(A_Natt, k_top, dim=1)[0].mean(1)
        A_Nloss = self.bce(A_N_topk, torch.zeros(b, device=device))

        sparsity_loss = torch.mean(torch.abs(A_Natt))
        k_sparse = max(T // 8, 1)
        stubborn_noise = torch.topk(A_Natt, k_sparse, dim=1)[0].mean()
        max_A_on_N = A_Natt.max(dim=1)[0]
        max_N_on_N = torch.topk(N_att, k_sparse, dim=1)[0].mean(1)
        contrastive_A_N = torch.relu(max_A_on_N - max_N_on_N + 0.7).mean()

        # UCF loss weights
        total_loss = (
            anomaly_loss
            + 2.0 * (A_loss + panomaly_loss + N_loss)
            + 1.5 * A_Nloss
            + 0.5 * triplet
            + 0.1 * kl
            + 2.25 * ortho
            + 1.0 * sparsity_loss
            + 1.0 * stubborn_noise
            + 1.5 * contrastive_A_N
        )

        loss["total_loss"]      = total_loss
        loss["att_loss"]        = anomaly_loss
        loss["A_loss"]          = A_loss
        loss["panomaly_loss"]   = panomaly_loss
        loss["N_loss"]          = N_loss
        loss["A_Nloss"]         = A_Nloss
        loss["triplet"]         = triplet
        loss["kl_loss"]         = kl
        loss["ortho_loss"]      = ortho
        loss["sparsity_loss"]   = sparsity_loss
        loss["stubborn_noise"]  = stubborn_noise
        loss["contrastive_A_N"] = contrastive_A_N
        return total_loss, loss

def evaluate_and_save(model, optimizer, test_loader, config, step, best_auc, save_dir, args):
    model.eval()
    model.flag = "Test"
    auc_score, ap_score, acc = test(model, config, test_loader, step)
    model.train()
    model.flag = "Train"
    print(f"[EVAL @ {step}] AUC={auc_score:.4f} | AP={ap_score:.4f} | ACC={acc:.4f}")
    if auc_score > best_auc:
        best_auc = auc_score
        best_model_path = os.path.join(save_dir, 'best_ucf')
        best_model_path += f'_sk_{args.support_k}_qk_{args.query_k}_ilr_{args.inner_lr}_mlr_{args.meta_lr}_beta_{args.beta}_segments_{args.num_segments}'
        best_model_path += f'_anums_{args.a_nums}_nnums_{args.n_nums}_ap_{ap_score:.4f}_auc_{best_auc:.4f}.pth'
        torch.save({"model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step}, best_model_path)
        print(f"[INFO] New best AUC: {best_auc:.4f} saved to {best_model_path}")
    return auc_score, ap_score, acc, best_auc

# -------------------------
# Main
# -------------------------
def main():
    args = parse_args()
    if args.seed >= 0:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = {'num_segments': args.num_segments, 'len_feature': args.len_feature, 'device': device, 'gt_file': args.gt_file}
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, 'plots'), exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(args.log_dir, "tb_logs"))

    print("[INFO] Loading dataset metadata...")
    train_list_fallback = os.path.join(args.root_dir, 'Train', 'RGB')
    train_dataset = UCF_crime(root_dir=train_list_fallback, modal='rgb', mode='Train',
                               num_segments=args.num_segments, len_feature=args.len_feature, seed=args.seed)
    test_dataset = UCF_crime(root_dir=train_list_fallback, modal='rgb', mode='Test',
                              num_segments=args.num_segments, len_feature=args.len_feature, seed=args.seed)
    normal_idxs, abnormal_idxs = build_index_cache(train_dataset)
    test_loader = data.DataLoader(UCF_crime(root_dir=args.root_dir, mode='Test', modal='rgb',
                                            num_segments=args.num_segments, len_feature=args.len_feature),
                                  batch_size=1, shuffle=False, num_workers=args.num_workers)

    print("[INFO] Initializing model...")
    model = WSAD(input_size=args.len_feature, flag="Train", a_nums=args.a_nums, n_nums=args.n_nums)
    model.to(device)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        start_step = checkpoint.get("step", 0)
        best_auc = 0.0
        memory_params = list(model.Amemory.parameters()) + list(model.Nmemory.parameters())
        base_params = [p for n, p in model.named_parameters() if 'memory' not in n]
        optimizer = torch.optim.Adam([{'params': base_params, 'lr': 1e-5},
                                      {'params': memory_params, 'lr': 1e-7}], weight_decay=5e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.meta_iters)
    else:
        start_step = 0
        optimizer = torch.optim.Adam(model.parameters(), lr=args.meta_lr, betas=(0.9, 0.999), weight_decay=5e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.meta_iters))

    criterion = AD_Loss()
    test_info = {'step': [], 'auc': [], 'ap': [], 'ac': []}
    best_auc = 0.0

    print("[INFO] Training settings:")
    print(f" device={device} support_k={args.support_k} query_k={args.query_k} inner_lr={args.inner_lr} meta_lr={args.meta_lr} beta={args.beta} warmup_steps={args.warmup_steps}")

    for step in tqdm(range(1, args.meta_iters + 1), total=args.meta_iters, ncols=100):
        model.train()
        model.flag = "Train"

        if step <= args.warmup_steps:
            k_sup = 64
            sup_x, sup_y = get_balanced_batch(train_dataset, normal_idxs, abnormal_idxs, k_sup, device)
            result = model(sup_x)
            loss_sup, _ = criterion(result, sup_y)
            optimizer.zero_grad()
            loss_sup.backward()
            optimizer.step()
            scheduler.step()
            if step % args.eval_interval == 0:
                auc_score, ap_score, acc, best_auc = evaluate_and_save(model, optimizer, test_loader, config, step, best_auc, args.save_dir, args)
            if step % max(1, args.warmup_steps//5) == 0 or args.debug:
                print(f"[WARMUP {step}/{args.warmup_steps}] sup_loss={float(loss_sup.item()):.6f}")
            continue

        num_tasks = 4
        task_losses = []
        for _ in range(num_tasks):
            support_x, support_y = get_balanced_batch(train_dataset, normal_idxs, abnormal_idxs, args.support_k, device)
            query_x, query_y = get_balanced_batch(train_dataset, normal_idxs, abnormal_idxs, args.query_k, device)

            result_support = model(support_x)
            inner_loss, inner_loss_dict = criterion(result_support, support_y)
            if step % 5 == 0:
                for k, v in inner_loss_dict.items():
                    writer.add_scalar(f"support/{k}", v.item(), step)
            grads = torch.autograd.grad(inner_loss, model.parameters(), create_graph=True, allow_unused=False)

            fast_weights = {}
            for (name, param), g in zip(model.named_parameters(), grads):
                if 'encoder' in name or 'Amemory' in name or 'Nmemory' in name:
                    fast_weights[name] = param
                else:
                    fast_weights[name] = param - args.inner_lr * g

            res_query = stateless.functional_call(model, fast_weights, (query_x,))
            meta_val_loss, query_loss_dict = criterion(res_query, query_y)
            if step % 5 == 0:
                for k, v in query_loss_dict.items():
                    writer.add_scalar(f"query/{k}", v.item(), step)

            task_loss = inner_loss + args.beta * meta_val_loss
            task_losses.append(task_loss)

        loss_tensor = torch.stack(task_losses)
        if len(task_losses) >= 4:
            median = loss_tensor.median()
            mask = loss_tensor < 2.0 * median
            if mask.sum() >= 2:
                loss_tensor = loss_tensor[mask]
        meta_loss = loss_tensor.mean()

        if torch.isnan(meta_loss):
            optimizer.zero_grad()
            print(f"[WARN] NaN meta loss at step {step}, skipping")
            continue

        optimizer.zero_grad()
        meta_loss.backward()
        if step % 5 == 0:
            writer.add_scalar("grad_norm/before", compute_grad_norm(model.parameters()), step)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
        if step % 5 == 0:
            writer.add_scalar("grad_norm/after", compute_grad_norm(model.parameters()), step)
        optimizer.step()
        scheduler.step()

        if step % args.eval_interval == 0:
            auc_score, ap_score, acc, best_auc = evaluate_and_save(model, optimizer, test_loader, config, step, best_auc, args.save_dir, args)

    print("[INFO] Meta-training completed. Best AUC: {:.4f}".format(best_auc))

if __name__ == '__main__':
    main()