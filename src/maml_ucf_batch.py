#!/usr/bin/env python3
"""
maml_ucf.py

MAML-style meta-training for UCF-Crime (mirrors XD training, constant inner LR,
increased batch/num_tasks for better GPU utilisation).
"""

import os, random, pickle, argparse
from tqdm import tqdm
import numpy as np
import torch, torch.nn as nn, torch.utils.data as data
from torch.nn.utils import stateless
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.filterwarnings("ignore")

# ---------- Args ----------
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
    p.add_argument('--max_grad_norm', type=float, default=2.5)
    p.add_argument('--beta', type=float, default=1.0)
    p.add_argument('--num_workers', type=int, default=8)          # ← increased
    p.add_argument('--save_dir', type=str, default='checkpoints')
    p.add_argument('--gt_file', type=str, default='frame_label/gt_ucf_v2.npy')
    p.add_argument('--eval_interval', type=int, default=100)
    p.add_argument('--debug', action='store_true')
    p.add_argument('--seed', type=int, default=-1)
    p.add_argument('--log_dir', type=str, default='logs_ucf')
    p.add_argument('--resume', type=str, default='')
    return p.parse_args()

# ---------- Model ----------
from model_gemini_xd_v3 import WSAD
import utils

# ---------- Dataset ----------
class UCF_crime(data.Dataset):
    def __init__(self, root_dir, modal, mode, num_segments, len_feature, seed=-1):
        if seed >= 0 and hasattr(utils, "set_seed"):
            utils.set_seed(seed)
        self.mode = mode
        self.num_segments = num_segments
        self.len_feature = len_feature
        list_path_repo = os.path.join('list', f'UCF_{self.mode}.list')
        alt_list_path = os.path.join(root_dir, f'{self.mode.lower()}_list.txt')
        if os.path.exists(list_path_repo):
            list_file = list_path_repo
        elif os.path.exists(alt_list_path):
            list_file = alt_list_path
        else:
            raise FileNotFoundError(f"List file not found for mode={self.mode}")
        samples = []
        numAbnorm = 8100 if self.mode == 'Train' else 1400
        with open(list_file, 'r') as f:
            for i, line in enumerate(f):
                label = 1 if i < numAbnorm else 0
                samples.append((line.strip(), label))
        self.samples = samples

    def __len__(self): return len(self.samples)

    def __getitem__(self, index):
        vid, label = self.samples[index]
        feat = np.load(vid).astype(np.float32)   # [T, D]
        if self.mode == "Train":
            T, D = feat.shape
            new_feat = np.zeros((self.num_segments, D), dtype=np.float32)
            r = np.linspace(0, T, self.num_segments + 1, dtype=np.int32)
            for i in range(self.num_segments):
                if r[i] != r[i+1]:
                    new_feat[i] = feat[r[i]:r[i+1]].mean(0)
                else:
                    new_feat[i] = feat[r[i]]
            feat = new_feat
            # temporal masking augmentation
            if random.random() < 0.25:
                mask_len = random.randint(int(self.num_segments*0.05),
                                          int(self.num_segments*0.15))
                mask_start = random.randint(0, self.num_segments - mask_len)
                feat[mask_start:mask_start+mask_len] *= random.uniform(0.2, 0.6)
        return feat, label

# ---------- Index cache ----------
def build_index_cache(dataset, cache_path="index_cache_ucf.pkl", rebuild=False):
    if os.path.exists(cache_path) and not rebuild:
        with open(cache_path, "rb") as f: cache = pickle.load(f)
        return cache["normal"], cache["abnormal"]
    normal_idxs, abnormal_idxs = [], []
    for i in range(len(dataset)):
        _, label = dataset.samples[i]
        (normal_idxs if label == 0 else abnormal_idxs).append(i)
    cache = {"normal": normal_idxs, "abnormal": abnormal_idxs}
    with open(cache_path, "wb") as f: pickle.dump(cache, f)
    return normal_idxs, abnormal_idxs

# ---------- Balanced batch ----------
def get_balanced_batch(dataset, normal_idxs, abnormal_idxs, k, device):
    half = k // 2
    idxs = random.sample(normal_idxs, half) + random.sample(abnormal_idxs, half)
    feats, labels = [], []
    for i in idxs:
        feat, label = dataset[i]
        feats.append(torch.from_numpy(feat.astype(np.float32)))
        labels.append(torch.tensor(label).float())
    return torch.stack(feats).to(device), torch.stack(labels).to(device)

def compute_grad_norm(params):
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += p.grad.data.norm(2).item() ** 2
    return total ** 0.5

# ---------- Test (original, no repeat) ----------
def test(net, config, test_loader, step, model_file=None, save_dir='checkpoints', do_norm=False):
    with torch.no_grad():
        net.eval()
        net.flag = "Test"
        if model_file is not None:
            net.load_state_dict(torch.load(model_file))
        load_iter = iter(test_loader)
        frame_gt = np.load("frame_label/gt_ucf_v2.npy")
        frame_predict = None
        cls_label, cls_pre = [], []
        temp_predict = torch.zeros((0)).cuda()
        for i in range(len(test_loader.dataset)):
            _data, _label = next(load_iter)
            _data, _label = _data.cuda(), _label.cuda()
            res = net(_data)
            a_predict = torch.sigmoid(res["frame"])
            temp_predict = torch.cat([temp_predict, a_predict], dim=0)
            if (i + 1) % 10 == 0:
                cls_label.append(int(_label))
                avg_pred = temp_predict.mean(0).cpu().numpy()
                cls_pre.append(1 if avg_pred.max() > 0.5 else 0)
                if frame_predict is None:
                    frame_predict = avg_pred
                else:
                    frame_predict = np.concatenate([frame_predict, avg_pred])
                temp_predict = torch.zeros((0)).cuda()
        fpr, tpr, _ = roc_curve(frame_gt, frame_predict)
        auc_score = auc(fpr, tpr)
        acc = np.sum(np.array(cls_label) == np.array(cls_pre)) / len(cls_pre)
        prec, rec, _ = precision_recall_curve(frame_gt, frame_predict)
        ap_score = auc(rec, prec)
    return auc_score, ap_score, acc

# ---------- AD_Loss (UCF weights) ----------
class AD_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.topk_frac_override = None

    def forward(self, result, _label):
        loss = {}
        _label = _label.float()
        device = _label.device
        att = result['frame']
        if att.dim() == 2: att = att.unsqueeze(-1)
        B, T, _ = att.shape
        b = B // 2

        A_att  = result["A_att"]
        N_att  = result["N_att"]
        N_Aatt = result["N_Aatt"]
        A_Natt = result["A_Natt_raw"]

        triplet = result.get("triplet_margin", torch.tensor(0.0, device=device))
        kl      = result.get("kl_loss", torch.tensor(0.0, device=device))
        ortho   = result.get("ortho_loss", torch.tensor(0.0, device=device))

        frac = self.topk_frac_override or 16
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

        total_loss = (
            anomaly_loss
            + 1.0 * (A_loss + panomaly_loss + N_loss)
            + 1.5 * A_Nloss
            + 0.5 * triplet
            + 0.1 * kl
            + 0.25 * ortho
            + 1.0 * sparsity_loss
            + 1.0 * stubborn_noise
            + 1.5 * contrastive_A_N
        )

        loss.update({k: v for k, v in locals().items() if isinstance(v, torch.Tensor)})
        loss["total_loss"] = total_loss
        return total_loss, loss

# ---------- Eval & save ----------
def evaluate_and_save(model, optimizer, test_loader, config, step, best_auc, save_dir, args):
    model.eval(); model.flag = "Test"
    auc_score, ap_score, acc = test(model, config, test_loader, step)
    model.train(); model.flag = "Train"
    print(f"[EVAL @ {step}] AUC={auc_score:.4f} | AP={ap_score:.4f} | ACC={acc:.4f}")
    if auc_score > best_auc:
        best_auc = auc_score
        best_path = os.path.join(save_dir,
            f'best_ucf_sk_{args.support_k}_qk_{args.query_k}_ilr_{args.inner_lr}_mlr_{args.meta_lr}'
            f'_beta_{args.beta}_segments_{args.num_segments}_anums_{args.a_nums}_nnums_{args.n_nums}'
            f'_ap_{ap_score:.4f}_auc_{best_auc:.4f}.pth')
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step}, best_path)
        print(f"[INFO] New best AUC: {best_auc:.4f} saved to {best_path}")
    return auc_score, ap_score, acc, best_auc

# ---------- Main ----------
def main():
    args = parse_args()
    if args.seed >= 0:
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = {'num_segments': args.num_segments, 'len_feature': args.len_feature, 'device': device, 'gt_file': args.gt_file}
    os.makedirs(args.save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(args.log_dir, "tb_logs"))

    train_dataset = UCF_crime(root_dir=os.path.join(args.root_dir, 'Train', 'RGB'), modal='rgb', mode='Train',
                               num_segments=args.num_segments, len_feature=args.len_feature, seed=args.seed)
    test_loader = data.DataLoader(UCF_crime(root_dir=args.root_dir, mode='Test', modal='rgb',
                                            num_segments=args.num_segments, len_feature=args.len_feature),
                                  batch_size=1, shuffle=False, num_workers=args.num_workers)

    normal_idxs, abnormal_idxs = build_index_cache(train_dataset)

    model = WSAD(input_size=args.len_feature, flag="Train", a_nums=args.a_nums, n_nums=args.n_nums).to(device)

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        start_step = ckpt.get("step", 0)
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
    best_auc = 0.0

    for step in tqdm(range(1, args.meta_iters + 1), total=args.meta_iters, ncols=100):
        model.train()
        model.flag = "Train"

        # ---------- warmup (now with k_sup=256) ----------
        if step <= args.warmup_steps:
            sup_x, sup_y = get_balanced_batch(train_dataset, normal_idxs, abnormal_idxs,
                                              256, device)          # ← batch size 256
            result = model(sup_x)
            loss_sup, _ = criterion(result, sup_y)
            optimizer.zero_grad()
            loss_sup.backward()
            optimizer.step()
            scheduler.step()
            if step % args.eval_interval == 0:
                best_auc = evaluate_and_save(model, optimizer, test_loader, config, step, best_auc, args.save_dir, args)[-1]
            continue

        # ---------- meta-training (8 tasks) ----------
        num_tasks = 8
        task_losses = []
        for _ in range(num_tasks):
            support_x, support_y = get_balanced_batch(train_dataset, normal_idxs, abnormal_idxs, args.support_k, device)
            query_x, query_y = get_balanced_batch(train_dataset, normal_idxs, abnormal_idxs, args.query_k, device)

            result_support = model(support_x)
            inner_loss, _ = criterion(result_support, support_y)

            grads = torch.autograd.grad(inner_loss, model.parameters(), create_graph=True, allow_unused=False)

            fast_weights = {}
            for (name, param), g in zip(model.named_parameters(), grads):
                if 'encoder' in name or 'Amemory' in name or 'Nmemory' in name:
                    fast_weights[name] = param
                else:
                    fast_weights[name] = param - args.inner_lr * g

            res_query = stateless.functional_call(model, fast_weights, (query_x,))
            meta_val_loss, _ = criterion(res_query, query_y)

            task_loss = inner_loss + args.beta * meta_val_loss
            task_losses.append(task_loss)

        # outlier rejection
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
        optimizer.step()
        scheduler.step()

        # logging (skip for speed)
        if step % 5 == 0:
            writer.add_scalar("grad_norm/before", compute_grad_norm(model.parameters()), step)

        if step % args.eval_interval == 0:
            best_auc = evaluate_and_save(model, optimizer, test_loader, config, step, best_auc, args.save_dir, args)[-1]

    print(f"[INFO] Meta-training completed. Best AUC: {best_auc:.4f}")

if __name__ == '__main__':
    main()