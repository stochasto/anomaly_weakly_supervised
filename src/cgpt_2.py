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

from model_min_upgrade import WSAD


# =========================================================
# CLI
# =========================================================
def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument('--root_dir', type=str, default='/mnt/c/code/data/I3D')
    p.add_argument('--num_segments', type=int, default=200)
    p.add_argument('--len_feature', type=int, default=1024)

    p.add_argument('--a_nums', type=int, default=200)
    p.add_argument('--n_nums', type=int, default=200)

    p.add_argument('--support_k', type=int, default=8)
    p.add_argument('--query_k', type=int, default=16)

    p.add_argument('--meta_iters', type=int, default=4000)
    p.add_argument('--warmup_steps', type=int, default=100)

    p.add_argument('--inner_lr', type=float, default=1e-3)
    p.add_argument('--meta_lr', type=float, default=1e-4)
    p.add_argument('--beta', type=float, default=1.0)

    p.add_argument('--eval_interval', type=int, default=100)
    p.add_argument('--gt_file', type=str, default='frame_label/gt_ucf_v2.npy')

    p.add_argument('--save_dir', type=str, default='checkpoints')
    p.add_argument('--seed', type=int, default=-1)

    # Improvements
    p.add_argument('--mil_temp_start', type=float, default=5.0)
    p.add_argument('--mil_temp_end', type=float, default=12.0)
    p.add_argument('--ema_alpha', type=float, default=0.7)
    p.add_argument('--use_norm', action='store_true')

    return p.parse_args()


# =========================================================
# DATASET
# =========================================================
class UCF_crime(data.Dataset):
    def __init__(self, root_dir, mode, num_segments):
        self.mode = mode
        self.num_segments = num_segments

        list_file = f'list/UCF_{mode}.list'

        samples = []
        numAbnorm = 8100 if mode == 'Train' else 1400

        with open(list_file) as f:
            for i, line in enumerate(f):
                label = 1 if i < numAbnorm else 0
                samples.append((line.strip(), label))

        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        feat = np.load(path).astype(np.float32)

        if self.mode == "Train":
            T, D = feat.shape
            new_feat = np.zeros((self.num_segments, D), dtype=np.float32)
            r = np.linspace(0, T, self.num_segments + 1, dtype=int)

            for i in range(self.num_segments):
                if r[i] != r[i+1]:
                    new_feat[i] = feat[r[i]:r[i+1]].mean(0)
                else:
                    new_feat[i] = feat[r[i]]

            feat = new_feat

        return feat, label


# =========================================================
# SAMPLING (unchanged distribution)
# =========================================================
def get_batch(dataset, k, device, use_norm=False):
    idxs = random.sample(range(len(dataset)), k)

    feats, labels = [], []

    for i in idxs:
        f, y = dataset[i]

        if use_norm:
            mu = f.mean(0, keepdims=True)
            std = f.std(0, keepdims=True) + 1e-6
            f = (f - mu) / std

        feats.append(torch.from_numpy(f))
        labels.append(torch.tensor(y).float())

    return torch.stack(feats).to(device), torch.stack(labels).to(device)


# =========================================================
# EMA SMOOTHING
# =========================================================
def temporal_ema(x, alpha=0.7):
    out = x.clone()
    for t in range(1, x.size(1)):
        out[:, t] = alpha * out[:, t-1] + (1 - alpha) * x[:, t]
    return out


# =========================================================
# LOSS (SOFT MIL + EMA + FULL TERMS)
# =========================================================
class AD_Loss(nn.Module):
    def __init__(self, temp=10.0, ema_alpha=0.7):
        super().__init__()
        self.bce = nn.BCELoss()
        self.temp = temp
        self.ema_alpha = ema_alpha

    def soft_topk(self, att):
        weights = torch.softmax(att * self.temp, dim=-1)
        return (att * weights).sum(-1)

    def forward(self, result, label):
        loss = {}
        label = label.float()

        att = result["frame"]
        att = temporal_ema(att, self.ema_alpha)

        anomaly = self.soft_topk(att)
        anomaly_loss = self.bce(anomaly, label)

        A_att = result.get("A_att", att)
        N_att = result.get("N_att", att)
        A_Natt = result.get("A_Natt", att)
        N_Aatt = result.get("N_Aatt", att)

        b = max(label.size(0)//2, 1)

        panomaly = self.soft_topk(1 - N_Aatt)[:b]
        panomaly_loss = self.bce(panomaly, torch.ones_like(panomaly))

        A_loss = self.bce(self.soft_topk(A_att)[:b], torch.ones(b, device=label.device))
        N_loss = self.bce(N_att, torch.ones_like(N_att))
        A_Nloss = self.bce(A_Natt, torch.zeros_like(A_Natt))

        triplet = result.get("triplet_margin", torch.tensor(0.0, device=label.device))
        kl = result.get("kl_loss", torch.tensor(0.0, device=label.device))
        smooth = result.get("smooth_loss", torch.tensor(0.0, device=label.device))
        dist = result.get("distance", torch.tensor(0.0, device=label.device))

        total = (
            anomaly_loss
            + 0.2*(A_loss + panomaly_loss + N_loss + A_Nloss)
            + 0.2*triplet
            + 0.005*kl
            + 0.0001*dist
            + 0.01*smooth
        )

        loss["total_loss"] = total
        loss["att_loss"] = anomaly_loss
        loss["N_Aatt"] = panomaly_loss
        loss["A_loss"] = A_loss
        loss["N_loss"] = N_loss
        loss["A_Nloss"] = A_Nloss
        loss["triplet"] = triplet
        loss["kl_loss"] = kl
        loss["smooth_loss"] = smooth

        return total, loss


# =========================================================
# TEST FUNCTION (EXACT — UNCHANGED)
# =========================================================
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
            a_predict = res["frame"]
            temp_predict = torch.cat([temp_predict, a_predict], dim=0)

            if (i + 1) % 10 == 0:
                cls_label.append(int(_label))
                a_predict = temp_predict.mean(0).cpu().numpy()
                
                cls_pre.append(1 if a_predict.max()>0.5 else 0)          
                fpre_ = a_predict

                if frame_predict is None:         
                    frame_predict = fpre_
                else:
                    frame_predict = np.concatenate([frame_predict, fpre_])  

                temp_predict = torch.zeros((0)).cuda()
   
        fpr,tpr,_ = roc_curve(frame_gt, frame_predict)
        auc_score = auc(fpr, tpr)
    
        corrent_num = np.sum(np.array(cls_label) == np.array(cls_pre), axis=0)
        accuracy = corrent_num / (len(cls_pre))
        
        precision, recall, th = precision_recall_curve(frame_gt, frame_predict)
        ap_score = auc(recall, precision)

    return auc_score, ap_score, accuracy


# =========================================================
# TRAIN
# =========================================================
def main():
    args = parse_args()

    if args.seed >= 0:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    device = torch.device("cuda")

    train_ds = UCF_crime(args.root_dir, "Train", args.num_segments)
    test_ds = UCF_crime(args.root_dir, "Test", args.num_segments)

    test_loader = data.DataLoader(test_ds, batch_size=1, shuffle=False)

    model = WSAD(args.len_feature, "Train", args.a_nums, args.n_nums).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.meta_lr, weight_decay=5e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.meta_iters)

    loss_fn = AD_Loss(temp=args.mil_temp_start, ema_alpha=args.ema_alpha)

    best_auc = 0

    for step in tqdm(range(1, args.meta_iters+1)):

        # -------- curriculum MIL temp ----------
        progress = step / args.meta_iters
        curr_temp = args.mil_temp_start + (args.mil_temp_end - args.mil_temp_start) * progress
        loss_fn.temp = curr_temp

        model.train()

        # -------- WARMUP --------
        if step <= args.warmup_steps:
            x, y = get_batch(train_ds, args.query_k, device, args.use_norm)
            res = model(x)
            loss, _ = loss_fn(res, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            continue

        # -------- META --------
        sx, sy = get_batch(train_ds, args.support_k, device, args.use_norm)
        qx, qy = get_batch(train_ds, args.query_k, device, args.use_norm)

        res_s = model(sx)
        inner_loss, _ = loss_fn(res_s, sy)

        grads = torch.autograd.grad(inner_loss, model.parameters(), create_graph=True)

        fast_weights = {
            n: p - args.inner_lr * g
            for (n,p), g in zip(model.named_parameters(), grads)
        }

        res_q = stateless.functional_call(model, fast_weights, (qx,))
        meta_loss, _ = loss_fn(res_q, qy)

        total_loss = inner_loss + args.beta * meta_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        # -------- EVAL --------
        if step % args.eval_interval == 0:
            model.flag = "Test"
            auc_score, ap, acc = test(model, None, test_loader, step)
            model.flag = "Train"

            print(f"[STEP {step}] AUC={auc_score:.4f} | temp={curr_temp:.2f}")

            if auc_score > best_auc:
                best_auc = auc_score
                os.makedirs(args.save_dir, exist_ok=True)
                torch.save(model.state_dict(),
                           os.path.join(args.save_dir, f"best_{best_auc:.4f}.pth"))


if __name__ == "__main__":
    main()