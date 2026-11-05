#!/usr/bin/env python3 
"""
Full MAML-style training + testing using your WSAD model and UCF_crime dataset.
- Uses your model.py (WSAD), memory.py, translayer.py, utils.py from the repo
- True random support/query sampling
- AUC/AP printed every 5 iterations, best checkpoint saved
- Test loop uses your robust frame expansion (handles 10-crop averaging)
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
    p.add_argument('--root_dir', type=str, default='/mnt/c/code/ae_anomaly/temp',
                   help="root dir / base path (not used directly; dataset file provides full paths in lists)")
    p.add_argument('--num_segments', type=int, default=200)
    p.add_argument('--len_feature', type=int, default=2048)
    p.add_argument('--batch_k', type=int, default=8, help="support/query size per normal/abnormal in meta step")
    p.add_argument('--meta_iters', type=int, default=10, help="number of meta iterations")
    p.add_argument('--inner_lr', type=float, default=5e-5)
    p.add_argument('--meta_lr', type=float, default=5e-5)
    p.add_argument('--beta', type=float, default=0.1, help="weight for meta-val loss")
    p.add_argument('--num_workers', type=int, default=2)
    p.add_argument('--save_dir', type=str, default='checkpointsml')
    p.add_argument('--gt_file', type=str, default='frame_label/gt-ucf.npy')
    p.add_argument('--eval_interval', type=int, default=5)
    p.add_argument('--debug', action='store_true')
    args = p.parse_args()
    return args

# -------------------------
# Use your repo model & utils
# -------------------------
from model import WSAD
import utils

# -------------------------
# UCF dataset
# -------------------------
class UCF_crime(data.Dataset):
    def __init__(self, root_dir, modal, mode, num_segments, len_feature, seed=-1):
        if seed >= 0 and hasattr(utils, "set_seed"):
            utils.set_seed(seed)
        self.mode = mode
        self.modal = modal
        self.num_segments = num_segments
        self.len_feature = len_feature
        self.samples = []

        split_path = os.path.join('list', f'UCF_{self.mode}.list')
        with open(split_path, 'r') as f:
            vid_list = [line.strip().split()[0] for line in f]

        for vid_info in vid_list:
            label = 0 if "Normal" in vid_info.split("/")[-1] else 1
            self.samples.append((vid_info, label))

    def __len__(self):
        return len(self.samples) * 10

    def __getitem__(self, index):
        video_idx = index // 10
        crop_idx = index % 10
        vid_info, label = self.samples[video_idx]
        name = vid_info.split("/")[-1].split("_x264")[0]
        video_feature = np.load(vid_info).astype(np.float32)  # [T,10,D]
        video_feature = video_feature[:, crop_idx, :]         # [T,D]

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

        if self.mode == "Test":
            return video_feature, label, f"{name}__{crop_idx}"
        else:
            return video_feature, label

# -------------------------
# Helper: random sampling
# -------------------------
def get_random_batch_from_dataset(dataset, k):
    idxs = random.sample(range(len(dataset)), k)
    feats, labels = [], []
    for i in idxs:
        item = dataset[i]
        feats.append(torch.from_numpy(np.asarray(item[0])).float())
        labels.append(torch.tensor(item[1]).float())
    x = torch.stack(feats)
    y = torch.stack(labels)
    return x, y

# -------------------------
# Test loop
# -------------------------
def test(net, config, test_loader, test_info, step, model_file=None):
    with torch.no_grad():
        net.eval()
        net.flag = "Test"
        if model_file is not None:
            print(f"[INFO] Loading checkpoint from: {model_file}")
            net.load_state_dict(torch.load(model_file))

        frame_gt = np.load("frame_label/gt-ucf.npy")
        frame_predict = None
        cls_label, cls_pre = [], []
        temp_predict = torch.zeros((0)).cuda()
        count = 0
        load_iter = iter(test_loader)

        for i in range(len(test_loader.dataset)):
            _data, _label, _name = next(load_iter)
            _data = _data.to(config['device'], non_blocking=True)
            _label = _label.to(config['device'], non_blocking=True)

            if _data.dim() == 2:
                _data = _data.unsqueeze(0)
            elif _data.dim() == 4:
                _data = _data.squeeze(1)

            res = net(_data)
            a_predict = res["frame"]
            temp_predict = torch.cat([temp_predict, a_predict], dim=0)

            if (i + 1) % 10 == 0:
                cls_label.append(int(_label))
                a_predict = temp_predict.mean(0).cpu().numpy()
                cls_pre.append(1 if a_predict.max() > 0.5 else 0)

                num_frames_remaining = len(frame_gt) - (0 if frame_predict is None else len(frame_predict))
                num_frames_clip = len(a_predict)
                repeat_factor = max(num_frames_remaining // num_frames_clip, 1)
                fpre_ = np.repeat(a_predict, repeat_factor)[:num_frames_remaining]

                if frame_predict is None:
                    frame_predict = fpre_
                else:
                    frame_predict = np.concatenate([frame_predict, fpre_])
                temp_predict = torch.zeros((0)).cuda()
                count += len(fpre_)

        fpr, tpr, _ = roc_curve(frame_gt, frame_predict)
        auc_score = auc(fpr, tpr)
        precision, recall, _ = precision_recall_curve(frame_gt, frame_predict)
        ap_score = auc(recall, precision)
        correct_num = np.sum(np.array(cls_label) == np.array(cls_pre))
        accuracy = correct_num / len(cls_pre) if len(cls_pre) > 0 else 0.0

        test_info["step"].append(step)
        test_info["auc"].append(auc_score)
        test_info["ap"].append(ap_score)
        test_info["ac"].append(accuracy)

        print(f"[STEP {step:04d}] AUC={auc_score:.4f} | AP={ap_score:.4f} | ACC={accuracy:.4f}")
        return auc_score, ap_score, accuracy

# -------------------------
# AD_Loss
# -------------------------
class AD_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()

    def forward(self, result, _label):
        loss = {}
        _label = _label.float()
        triplet = result["triplet_margin"]
        att = result["frame"]
        A_att = result["A_att"]
        N_att = result["N_att"]
        A_Natt = result["A_Natt"]
        N_Aatt = result["N_Aatt"]
        kl_loss = result["kl_loss"]
        distance = result["distance"]
        b = _label.size(0) // 2
        t = att.size(1)

        anomaly = torch.topk(att, t // 16 + 1, dim=-1)[0].mean(-1)
        anomaly_loss = self.bce(anomaly, _label)
        panomaly = torch.topk(1 - N_Aatt, t // 16 + 1, dim=-1)[0].mean(-1)
        panomaly_loss = self.bce(panomaly, torch.ones((b)).cuda())
        A_att = torch.topk(A_att, t // 16 + 1, dim=-1)[0].mean(-1)
        A_loss = self.bce(A_att, torch.ones((b)).cuda())
        N_loss = self.bce(N_att, torch.ones_like(N_att).cuda())
        A_Nloss = self.bce(A_Natt, torch.zeros_like(A_Natt).cuda())

        cost = (
            anomaly_loss
            + 0.1 * (A_loss + panomaly_loss + N_loss + A_Nloss)
            + 0.1 * triplet
            + 0.001 * kl_loss
            + 0.0001 * distance
        )

        loss["total_loss"] = cost
        loss["att_loss"] = anomaly_loss
        loss["N_Aatt"] = panomaly_loss
        loss["A_loss"] = A_loss
        loss["N_loss"] = N_loss
        loss["A_Nloss"] = A_Nloss
        loss["triplet"] = triplet
        loss["kl_loss"] = kl_loss
        return cost, loss

# -------------------------
# MAIN: meta-training
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
    full_train_dataset = UCF_crime(root_dir=args.root_dir, modal='rgb', mode='Train',
                                   num_segments=args.num_segments, len_feature=args.len_feature, seed=-1)
    all_samples = full_train_dataset.samples
    normal_samples = [s for s in all_samples if s[1] == 0]
    abnormal_samples = [s for s in all_samples if s[1] == 1]

    class DatasetFromSamples(data.Dataset):
        def __init__(self, samples, mode, num_segments, len_feature):
            self.samples = samples
            self.mode = mode
            self.num_segments = num_segments
            self.len_feature = len_feature
        def __len__(self):
            return len(self.samples) * 10
        def __getitem__(self, index):
            video_idx = index // 10
            crop_idx = index % 10
            vid_info, label = self.samples[video_idx]
            video_feature = np.load(vid_info).astype(np.float32)
            video_feature = video_feature[:, crop_idx, :]
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
            if self.mode == "Test":
                return video_feature, label, vid_info.split("/")[-1].split("_x264")[0] + f"__{crop_idx}"
            else:
                return video_feature, label

    normal_train_ds = DatasetFromSamples(normal_samples, mode='Train', num_segments=args.num_segments, len_feature=args.len_feature)
    abnormal_train_ds = DatasetFromSamples(abnormal_samples, mode='Train', num_segments=args.num_segments, len_feature=args.len_feature)
    test_ds = DatasetFromSamples(all_samples, mode='Test', num_segments=args.num_segments, len_feature=args.len_feature)
    test_loader = data.DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # Model and optimizer
    model = WSAD(input_size=args.len_feature, flag="Train", a_nums=60, n_nums=60)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.meta_lr, betas=(0.9, 0.999), weight_decay=5e-5)

    # REPLACE criterion ONLY
    criterion = AD_Loss()

    test_info = {'step': [], 'auc': [], 'ap': [], 'ac': []}
    best_auc = 0.0
    best_model_path = os.path.join(args.save_dir, 'best_meta_wsad.pth')

    print("[INFO] Starting meta-training with AD_Loss only...")
    for step in tqdm(range(1, args.meta_iters + 1), total=args.meta_iters, ncols=100):
        model.train()
        k = args.batch_k
        sx_n_x, sx_n_y = get_random_batch_from_dataset(normal_train_ds, k)
        sx_a_x, sx_a_y = get_random_batch_from_dataset(abnormal_train_ds, k)
        support_x = torch.cat([sx_n_x, sx_a_x], dim=0).to(device, non_blocking=True)
        support_y = torch.cat([sx_n_y, sx_a_y], dim=0).to(device, non_blocking=True)

        # --- Inner loop
        result_support = model(support_x)
        inner_loss, _ = criterion(result_support, support_y)
        grads = torch.autograd.grad(inner_loss, model.parameters(), create_graph=True, allow_unused=False)
        alpha = args.inner_lr
        fast_weights = {name: param - alpha * g for (name, param), g in zip(model.named_parameters(), grads)}

        # --- Query step
        qx_n_x, qx_n_y = get_random_batch_from_dataset(normal_train_ds, k)
        qx_a_x, qx_a_y = get_random_batch_from_dataset(abnormal_train_ds, k)
        query_x = torch.cat([qx_n_x, qx_a_x], dim=0).to(device, non_blocking=True)
        query_y = torch.cat([qx_n_y, qx_a_y], dim=0).to(device, non_blocking=True)

        result_query = stateless.functional_call(model, fast_weights, (query_x,))
        meta_val_loss, _ = criterion(result_query, query_y)

        # Total meta-loss
        total_loss = inner_loss + args.beta * meta_val_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if step % 1 == 0 or step == 1:
            print(f"[STEP {step}] meta_loss={total_loss.item():.6f}")

        if step % args.eval_interval == 0:
            model.flag = "Test"
            auc_score, ap_score, acc = test(model, config, test_loader, test_info, step, model_file=None)
            model.flag = "Train"
            if auc_score > best_auc:
                best_auc = auc_score
                torch.save(model.state_dict(), best_model_path)
                print(f"[INFO] New best AUC: {best_auc:.4f} (saved to {best_model_path})")

    print("[INFO] Meta-training completed. Best AUC: {:.4f}".format(best_auc))


# -------------------------
# Run
# -------------------------
if __name__ == '__main__':
    main()
