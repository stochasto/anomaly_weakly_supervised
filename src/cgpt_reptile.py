#!/usr/bin/env python3

import os
import random
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
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

    p.add_argument('--support_k', type=int, default=16)
    p.add_argument('--inner_steps', type=int, default=5)

    p.add_argument('--inner_lr', type=float, default=1e-3)
    p.add_argument('--meta_lr', type=float, default=1e-5)

    p.add_argument('--meta_iters', type=int, default=4000)
    p.add_argument('--warmup_steps', type=int, default=200)

    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--save_dir', type=str, default='checkpoints')
    p.add_argument('--gt_file', type=str, default='frame_label/gt_ucf_v2.npy')
    p.add_argument('--eval_interval', type=int, default=100)
    p.add_argument('--seed', type=int, default=-1)
    return p.parse_args()

# -------------------------
# Dataset (UNCHANGED)
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
            raise FileNotFoundError

        samples = []
        numAbnorm = 8100 if self.mode == 'Train' else 1400
        with open(list_file, 'r') as f:
            for i, line in enumerate(f):
                samples.append((line.strip(), 1 if i < numAbnorm else 0))
        self.samples = samples

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
                    new_feat[i] = np.mean(feat[r[i]:r[i+1]], axis=0)
                else:
                    new_feat[i] = feat[r[i]]
            feat = new_feat
        return feat, label

# -------------------------
# Sampling
# -------------------------
def get_batch(dataset, k, device):
    idxs = random.sample(range(len(dataset)), k)
    x, y = [], []
    for i in idxs:
        f, l = dataset[i]
        x.append(torch.from_numpy(f))
        y.append(torch.tensor(l).float())
    return torch.stack(x).to(device), torch.stack(y).to(device)

# -------------------------
# TEST (UNCHANGED)
# -------------------------
def test(net, config, test_loader, step):
    with torch.no_grad():
        net.eval()
        net.flag = "Test"
        frame_gt = np.load("frame_label/gt_ucf_v2.npy")
        frame_predict = None

        temp_predict = torch.zeros((0)).cuda()
        cls_label, cls_pre = [], []

        for i, (_data, _label) in enumerate(test_loader):
            _data = _data.cuda()
            res = net(_data)
            a_predict = res["frame"]
            temp_predict = torch.cat([temp_predict, a_predict], dim=0)

            if (i+1) % 10 == 0:
                cls_label.append(int(_label))
                a_predict = temp_predict.mean(0).cpu().numpy()
                cls_pre.append(1 if a_predict.max()>0.5 else 0)

                if frame_predict is None:
                    frame_predict = a_predict
                else:
                    frame_predict = np.concatenate([frame_predict, a_predict])
                temp_predict = torch.zeros((0)).cuda()

        fpr,tpr,_ = roc_curve(frame_gt, frame_predict)
        auc_score = auc(fpr, tpr)
        acc = np.mean(np.array(cls_label)==np.array(cls_pre))
        precision, recall, _ = precision_recall_curve(frame_gt, frame_predict)
        ap = auc(recall, precision)

    return auc_score, ap, acc

# -------------------------
# LOSS (UNCHANGED)
# -------------------------
class AD_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()

    def forward(self, result, _label):
        loss = {}
        _label = _label.float()

        att = result['frame']

        A_att = result.get("A_att", torch.zeros_like(att))
        N_att = result.get("N_att", torch.zeros_like(att))
        A_Natt = result.get("A_Natt", torch.zeros_like(att))
        N_Aatt = result.get("N_Aatt", torch.zeros_like(att))
        kl_loss = result.get("kl_loss", torch.tensor(0.0, device=_label.device))
        distance = result.get("distance", torch.tensor(0.0, device=_label.device))
        smooth_loss = result.get("smooth_loss", torch.tensor(0.0, device=_label.device))

        b = max(_label.size(0)//2, 1)
        t = max(att.size(1), 1)

        anomaly_topk = max(t // 64, 1)

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

        triplet = result.get("triplet_margin", torch.tensor(0.0))
        sep_loss = 0
        sparsity_loss = att.mean()

        total_loss = (
            anomaly_loss
            + 0.15 * (A_loss + panomaly_loss + N_loss + A_Nloss)
            + 0.3 * triplet
            + 0.005 * kl_loss
            + 0.0001 * distance
            + 0.05 * smooth_loss
            + 0.2 * sep_loss
            + 0.1 * sparsity_loss
        )

        loss["total_loss"] = total_loss
        return total_loss, loss

# -------------------------
# REPTILE PARAM FILTER
# -------------------------
def is_reptile_param(name):
    return any(k in name for k in ["embedding", "localatt", "cls_head"])

# -------------------------
# MAIN
# -------------------------
def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = UCF_crime(args.root_dir, 'rgb', 'Train',
                             args.num_segments, args.len_feature)
    test_dataset = UCF_crime(args.root_dir, 'rgb', 'Test',
                            args.num_segments, args.len_feature)

    model = WSAD(input_size=args.len_feature, flag="Train",
                 a_nums=args.a_nums, n_nums=args.n_nums).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = AD_Loss()

    best_auc = 0.0

    for step in tqdm(range(1, args.meta_iters+1)):

        model.train()

        # ---------------- WARMUP ----------------
        if step < args.warmup_steps:
            x, y = get_batch(train_dataset, args.support_k, device)
            out = model(x)
            loss, _ = criterion(out, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            continue

        # ---------------- REPTILE ----------------
        import copy
        fast_model = copy.deepcopy(model)

        # Freeze memory + VAE
        for name, p in fast_model.named_parameters():
            if not is_reptile_param(name):
                p.requires_grad = False

        inner_opt = torch.optim.SGD(
            [p for p in fast_model.parameters() if p.requires_grad],
            lr=args.inner_lr
        )

        # SAME TASK (important!)
        x, y = get_batch(train_dataset, args.support_k, device)

        for _ in range(args.inner_steps):
            out = fast_model(x)
            loss, _ = criterion(out, y)

            inner_opt.zero_grad()
            loss.backward()
            inner_opt.step()

        # -------- SAFE REPTILE UPDATE --------
        with torch.no_grad():
            for (name, p), (_, fp) in zip(model.named_parameters(),
                                         fast_model.named_parameters()):
                if is_reptile_param(name):
                    p.data += args.meta_lr * (fp.data - p.data)

        # ---------------- EVAL ----------------
        if step % args.eval_interval == 0:
            model.flag = "Test"
            test_loader = data.DataLoader(test_dataset,
                                         batch_size=1,
                                         shuffle=False,
                                         num_workers=4)

            auc_score, ap, acc = test(model, None, test_loader, step)
            model.flag = "Train"

            print(f"[EVAL] AUC={auc_score:.4f} AP={ap:.4f} ACC={acc:.4f}")

            if auc_score > best_auc:
                best_auc = auc_score
                torch.save(model.state_dict(),
                           f"{args.save_dir}/best_auc_{best_auc:.4f}.pth")

    print("Best AUC:", best_auc)

if __name__ == "__main__":
    main()