#!/usr/bin/env python3
"""
maml_curriculum.py

MAML-style meta-training with curriculum learning, soft memory attention,
staged loss weights, and improved mixed-hardness task sampling.
"""

import os
import glob
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

# ---------- Global variables for difficulty lists ----------
hard_normal_scores = {}
hard_normal_list = []

hard_anomaly_scores = {}
easy_anomaly_list = []
hard_anomaly_list = []

# ---------- CLI / Config ----------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--root_dir', type=str, default='/mnt/c/code/data/I3D')
    p.add_argument('--num_segments', type=int, default=300)
    p.add_argument('--a_nums', type=int, default=200)
    p.add_argument('--n_nums', type=int, default=200)
    p.add_argument('--len_feature', type=int, default=1024)
    p.add_argument('--support_k', type=int, default=8)
    p.add_argument('--query_k', type=int, default=16)
    p.add_argument('--meta_iters', type=int, default=4000)
    p.add_argument('--warmup_steps', type=int, default=50)
    p.add_argument('--inner_lr', type=float, default=1e-3)
    p.add_argument('--meta_lr', type=float, default=1e-4)
    p.add_argument('--beta', type=float, default=1.0)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--save_dir', type=str, default='checkpoints')
    p.add_argument('--gt_file', type=str, default='frame_label/gt_ucf_v2.npy')
    p.add_argument('--eval_interval', type=int, default=25)
    p.add_argument('--debug', action='store_true')
    p.add_argument('--no_norm', action='store_true')
    p.add_argument('--seed', type=int, default=-1)
    p.add_argument('--log_dir', type=str, default='logs_curriculum')
    args = p.parse_args()
    return args

# ---------- Model import ----------
from model_min_upgrade_dsk1 import WSAD   # your model (with corrected triplet and softmax memory)

import utils

# ---------- Dataset ----------
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
            raise FileNotFoundError(f"List file missing for {self.mode}")
        samples = []
        numAbnorm = 8100 if self.mode == 'Train' else 1400
        with open(list_file, 'r') as f:
            for (i, line) in enumerate(f):
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
            # temporal masking augmentation
            if random.random() < 0.25:
                mask_len = random.randint(int(self.num_segments * 0.05), int(self.num_segments * 0.15))
                mask_start = random.randint(0, self.num_segments - mask_len)
                video_feature[mask_start:mask_start + mask_len] *= random.uniform(0.2, 0.6)
        return video_feature, label

# ---------- Balanced batch with curriculum ----------
def get_balanced_batch(dataset, normal_idxs, abnormal_idxs, k, device,
                       hard_normal_list=None, hard_prob=0.0,
                       easy_anomaly_list=None, hard_anomaly_list=None,
                       abnormal_hard_prob=0.0):
    half = k // 2
    # Normal sampling with hard mining
    if hard_normal_list and random.random() < hard_prob:
        n_hard = max(1, half // 2)
        hard_n_idxs = random.sample(hard_normal_list, min(n_hard, len(hard_normal_list)))
        regular_n_idxs = random.sample(normal_idxs, half - len(hard_n_idxs))
        n_idxs = hard_n_idxs + regular_n_idxs
    else:
        n_idxs = random.sample(normal_idxs, half)

    # Abnormal sampling with difficulty
    if easy_anomaly_list and hard_anomaly_list and random.random() < abnormal_hard_prob:
        a_hard = max(1, half // 2)
        hard_a_idxs = random.sample(hard_anomaly_list, min(a_hard, len(hard_anomaly_list)))
        easy_a_idxs = random.sample(easy_anomaly_list, half - len(hard_a_idxs))
        a_idxs = hard_a_idxs + easy_a_idxs
    else:
        a_idxs = random.sample(abnormal_idxs, half)

    idxs = n_idxs + a_idxs   # fixed order: normals first
    feats, labels = [], []
    for i in idxs:
        feat, label = dataset[i]
        feats.append(torch.from_numpy(np.asarray(feat).astype(np.float32)))
        labels.append(torch.tensor(label).float())
    x = torch.stack(feats).to(device)
    y = torch.stack(labels).to(device)
    return x, y

def build_index_cache(dataset, cache_path="index_cache.pkl", rebuild=False):
    if os.path.exists(cache_path) and not rebuild:
        print(f"[INFO] Loading index cache from {cache_path}")
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
        return cache["normal"], cache["abnormal"]
    print("[INFO] Building index cache...")
    normal_idxs, abnormal_idxs = [], []
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

def update_anomaly_difficulty(model, train_dataset, abnormal_idxs, device,
                              hard_anomaly_scores, num_sample=200):
    sample_a_idxs = random.sample(abnormal_idxs, min(num_sample, len(abnormal_idxs)))
    for idx in sample_a_idxs:
        feat, _ = train_dataset[idx]
        feat_t = torch.from_numpy(np.asarray(feat).astype(np.float32)).unsqueeze(0).to(device)
        with torch.no_grad():
            score = torch.sigmoid(model(feat_t)['frame']).max().item()
        hard_anomaly_scores[idx] = score

    # fill missing with neutral score 0.5
    for idx in abnormal_idxs:
        if idx not in hard_anomaly_scores:
            hard_anomaly_scores[idx] = 0.5

    sorted_items = sorted(hard_anomaly_scores.items(), key=lambda x: x[1], reverse=True)
    split = len(sorted_items)//2
    easy_list = [idx for idx, _ in sorted_items[:split]]
    hard_list = [idx for idx, _ in sorted_items[-split:]]
    return easy_list, hard_list

# ---------- Curriculum phase parameters (now includes loss weights) ----------
def get_curriculum_params(step):
    phase1_end = 500
    phase2_end = 1500
    if step <= phase1_end:
        # Phase 1: gentle loss weights, no hard mining, no TAML
        return 0.0, 0.0, 0.0, False, 1e-3, 0.05, 0.05, 0.005
    elif step <= phase2_end:
        progress = (step - phase1_end) / (phase2_end - phase1_end)
        hard_prob_support = 0.0 + progress * 0.5
        hard_prob_query = 0.0 + progress * 0.2
        abnormal_hard_prob = 0.0 + progress * 0.2
        mult_dual = 0.05 + progress * 0.25   # 0.05 → 0.3
        mult_triplet = 0.05 + progress * 0.15 # 0.05 → 0.2
        mult_kl = 0.005
        return hard_prob_support, hard_prob_query, abnormal_hard_prob, True, 1e-3, mult_dual, mult_triplet, mult_kl
    else:
        # Phase 3: aggressive
        return 0.5, 0.2, 0.5, True, 1e-3, 0.3, 0.2, 0.005

# ---------- Losses (with dynamic weights) ----------
class AD_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        # These will be updated from outside
        self.mult_dual = 0.2
        self.mult_triplet = 0.2
        self.mult_kl = 0.005

    def forward(self, result, _label):
        _label = _label.float()
        triplet = result.get("triplet_margin", torch.tensor(0.0, device=_label.device))
        att = result['frame']
        A_att = result.get("A_att", torch.zeros_like(att))
        N_att = result.get("N_att", torch.zeros_like(att))
        A_Natt = result.get("A_Natt", torch.zeros_like(att))
        N_Aatt = result.get("N_Aatt", torch.zeros_like(att))
        kl_loss = result.get("kl_loss", torch.tensor(0.0, device=_label.device))
        smooth_loss = result.get("smooth_loss", torch.tensor(0.0, device=_label.device))

        b = max(_label.size(0) // 2, 1)
        t = max(att.size(1), 1)
        anomaly_topk = max(t // 16 + 1, 1)

        anomaly = torch.topk(att, anomaly_topk, dim=-1)[0].mean(-1)
        anomaly_loss = self.bce(anomaly, _label)

        panomaly = torch.topk(1 - N_Aatt, anomaly_topk, dim=-1)[0].mean(-1)[:b]
        panomaly_loss = self.bce(panomaly, torch.ones_like(panomaly))

        A_att_topk = torch.topk(A_att, anomaly_topk, dim=-1)[0].mean(-1)[:b]
        A_loss = self.bce(A_att_topk, torch.ones_like(A_att_topk))

        N_loss = self.bce(N_att, torch.ones_like(N_att))
        A_Nloss = self.bce(A_Natt, torch.zeros_like(A_Natt))

        total_loss = (anomaly_loss
                      + self.mult_dual * (A_loss + panomaly_loss + N_loss + A_Nloss)
                      + self.mult_triplet * triplet
                      + self.mult_kl * kl_loss)

        return total_loss, {"att_loss": anomaly_loss, "N_Aatt": panomaly_loss,
                            "A_loss": A_loss, "N_loss": N_loss,
                            "A_Nloss": A_Nloss, "triplet": triplet,
                            "kl_loss": kl_loss, "smooth_loss": smooth_loss}

class InnerLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, result, label):
        att = result['frame']
        k = max(att.size(1)//16+1, 1)
        anomaly = torch.topk(att, k, dim=-1)[0].mean(-1) * 1.5
        loss = self.bce(anomaly, label.float())
        triplet = result.get("triplet_margin", torch.tensor(0.0, device=att.device))
        return loss + 0.05 * triplet

# ---------- Test ----------
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
            _data = _data.cuda()
            _label = _label.cuda()
            res = net(_data)
            a_predict = torch.sigmoid(res["frame"])
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
        correct = np.sum(np.array(cls_label) == np.array(cls_pre))
        accuracy = correct / len(cls_pre)
        precision, recall, th = precision_recall_curve(frame_gt, frame_predict)
        ap_score = auc(recall, precision)
    return auc_score, ap_score, accuracy

# ---------- Main ----------
def main():
    args = parse_args()
    global hard_normal_scores, hard_normal_list
    global hard_anomaly_scores, easy_anomaly_list, hard_anomaly_list

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
    writer = SummaryWriter(log_dir=os.path.join(args.log_dir, "tb_logs"))

    print("[INFO] Loading dataset metadata...")
    train_list_fallback = os.path.join(args.root_dir, 'Train', 'RGB')
    train_dataset = UCF_crime(root_dir=train_list_fallback, modal='rgb', mode='Train',
                              num_segments=args.num_segments, len_feature=args.len_feature, seed=args.seed)
    test_dataset = UCF_crime(root_dir=train_list_fallback, modal='rgb', mode='Test',
                             num_segments=args.num_segments, len_feature=args.len_feature, seed=args.seed)
    normal_idxs, abnormal_idxs = build_index_cache(train_dataset)

    print("[INFO] Initializing model...")
    model = WSAD(input_size=args.len_feature, flag="Train", a_nums=args.a_nums, n_nums=args.n_nums)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.meta_lr, betas=(0.9, 0.999), weight_decay=5e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.meta_iters)

    criterion_inner = InnerLoss()
    criterion = AD_Loss()

    test_info = {'step': [], 'auc': [], 'ap': [], 'ac': []}
    best_auc = 0.0

    print("[INFO] Starting curriculum meta-training...")
    for step in tqdm(range(1, args.meta_iters + 1), total=args.meta_iters, ncols=100):
        model.train()
        model.flag = "Train"

        # ---------- Warmup ----------
        if step <= args.warmup_steps:
            # For warmup we still use a moderate loss setting (Phase 1 weights)
            # but we set them here to be safe.
            criterion.mult_dual = 0.05
            criterion.mult_triplet = 0.05
            criterion.mult_kl = 0.005
            k_sup = max(args.query_k, args.support_k)
            sup_x, sup_y = get_balanced_batch(train_dataset, normal_idxs, abnormal_idxs, k_sup, device)
            result = model(sup_x)
            loss_sup, _ = criterion(result, sup_y)
            optimizer.zero_grad()
            loss_sup.backward()
            optimizer.step()
            scheduler.step()
            if step % max(1, args.warmup_steps//5) == 0 or args.debug:
                print(f"[WARMUP {step}/{args.warmup_steps}] sup_loss={float(loss_sup.item()):.6f}")
            continue

        # ---------- Curriculum params ----------
        hard_prob_s, hard_prob_q, abnorm_hard_prob, use_taml, cur_inner_lr, mult_dual, mult_triplet, mult_kl = get_curriculum_params(step)

        # Update the criterion weights dynamically
        criterion.mult_dual = mult_dual
        criterion.mult_triplet = mult_triplet
        criterion.mult_kl = mult_kl

        num_tasks = 4
        task_losses = []

        for _ in range(num_tasks):
            # Support with curriculum
            support_x, support_y = get_balanced_batch(
                train_dataset, normal_idxs, abnormal_idxs, args.support_k, device,
                hard_normal_list=hard_normal_list, hard_prob=hard_prob_s,
                easy_anomaly_list=easy_anomaly_list, hard_anomaly_list=hard_anomaly_list,
                abnormal_hard_prob=abnorm_hard_prob)
            # Query with lighter mining
            query_x, query_y = get_balanced_batch(
                train_dataset, normal_idxs, abnormal_idxs, args.query_k, device,
                hard_normal_list=hard_normal_list, hard_prob=hard_prob_q,
                easy_anomaly_list=easy_anomaly_list, hard_anomaly_list=hard_anomaly_list,
                abnormal_hard_prob=abnorm_hard_prob)

            # Inner loop
            result_support = model(support_x)
            inner_loss = criterion_inner(result_support, support_y)

            grads = torch.autograd.grad(inner_loss, model.parameters(), create_graph=True)
            fast_weights = {n: p - cur_inner_lr * g for (n, p), g in zip(model.named_parameters(), grads)}

            # Query evaluation
            res_query = stateless.functional_call(model, fast_weights, (query_x,))
            meta_val_loss, query_loss_dict = criterion(res_query, query_y)

            task_loss = inner_loss + args.beta * meta_val_loss
            task_losses.append(task_loss)

            # Console logging (every 10 steps)
            if step % 10 == 0:
                print(f"[STEP {step}] inner_loss={inner_loss.item():.4f}  query_loss={meta_val_loss.item():.4f}")
                print(f"  att_loss={query_loss_dict['att_loss']:.4f}  A_Nloss={query_loss_dict['A_Nloss']:.4f}  triplet={query_loss_dict['triplet']:.4f}  kl={query_loss_dict['kl_loss']:.4f}")

            # TensorBoard logging
            if step % 5 == 0:
                for k, v in query_loss_dict.items():
                    writer.add_scalar(f"query/{k}", v.item(), step)
                writer.add_scalar("query/total_loss", meta_val_loss.item(), step)

        # ---------- TAML ----------
        losses_stack = torch.stack(task_losses)
        meta_loss = losses_stack.mean()
        if use_taml:
            var_pen = losses_stack.var(unbiased=False)
            weights = torch.softmax(losses_stack.detach(), dim=0)
            entropy = -torch.sum(weights * torch.log(weights + 1e-8))
            meta_loss += 0.005 * var_pen - 0.1 * entropy
            writer.add_scalar("taml/var_pen", var_pen.item(), step)
            writer.add_scalar("taml/entropy", entropy.item(), step)

        optimizer.zero_grad()
        meta_loss.backward()
        optimizer.step()
        scheduler.step()

        # ---------- Evaluation & difficulty update ----------
        if step % args.eval_interval == 0:
            model.eval()
            model.flag = "Test"
            test_loader = data.DataLoader(
                UCF_crime(root_dir=args.root_dir, mode='Test', modal='rgb',
                          num_segments=args.num_segments, len_feature=args.len_feature),
                batch_size=1, shuffle=False, num_workers=args.num_workers)
            auc_score, ap_score, acc = test(model, config, test_loader, step)

            test_info["step"].append(step)
            test_info["auc"].append(auc_score)
            test_info["ap"].append(ap_score)
            test_info["ac"].append(acc)

            print(f"[EVAL @ {step}] AUC={auc_score:.4f} | AP={ap_score:.4f} | ACC={acc:.4f}")

            # Hard normal mining
            sample_n = random.sample(normal_idxs, min(200, len(normal_idxs)))
            for idx in sample_n:
                feat, _ = train_dataset[idx]
                feat_t = torch.from_numpy(np.asarray(feat).astype(np.float32)).unsqueeze(0).to(device)
                with torch.no_grad():
                    score = torch.sigmoid(model(feat_t)['frame']).max().item()
                hard_normal_scores[idx] = score
            hard_normal_list[:] = sorted(hard_normal_scores, key=hard_normal_scores.get, reverse=True)[:50]

            # Anomaly difficulty mining
            easy_anomaly_list[:], hard_anomaly_list[:] = update_anomaly_difficulty(
                model, train_dataset, abnormal_idxs, device, hard_anomaly_scores, num_sample=200)

            # Save best model
            if auc_score > best_auc:
                best_auc = auc_score
                best_path = os.path.join(args.save_dir,
                    f'best_curriculum_auc_{best_auc:.4f}_step_{step}.pth')
                torch.save(model.state_dict(), best_path)
                print(f"[INFO] New best AUC: {best_auc:.4f} saved to {best_path}")

            model.train()
            model.flag = "Train"

    print(f"[INFO] Training completed. Best AUC: {best_auc:.4f}")

if __name__ == '__main__':
    main()