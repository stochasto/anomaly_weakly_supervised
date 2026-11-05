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
import matplotlib.pyplot as plt
import warnings
from torch.utils.tensorboard import SummaryWriter
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
    p.add_argument('--gt_file', type=str, default='frame_label/gt_ucf_v2.npy')
    p.add_argument('--eval_interval', type=int, default=100)
    p.add_argument('--debug', action='store_true')
    p.add_argument('--no_norm', action='store_true', help="disable per-video normalization")
    p.add_argument('--seed', type=int, default=-1, help="set >=0 for deterministic runs, -1 for fully random")
    p.add_argument('--log_dir', type=str, default='logs')
    args = p.parse_args()
    return args

# -------------------------
# repo model & utils (must be present in repo)
# -------------------------
#from model import WSAD
# minimal upgrade bundle
from model_min_upgrade_dsk1_ref import WSAD
#from model_min_upgrade import WSAD
#from model_ablation import WSAD
#from model_full import WSAD
#from model_dpsk import WSAD


import utils

# -------------------------
# Dataset: group crop files per video
# (adapted from your code)
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
        #name = os.path.basename(base)
        video_feature = np.load(vid).astype(np.float32)   # [T, feat_dim]

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
            # -------- temporal masking augmentation --------
            if random.random() < 0.25:
                mask_len = random.randint(
                    int(self.num_segments * 0.05),
                    int(self.num_segments * 0.15)
                )
                mask_start = random.randint(0, self.num_segments - mask_len)
                #video_feature[mask_start:mask_start + mask_len, :] *= 0.3
                video_feature[mask_start:mask_start + mask_len] *= random.uniform(0.2, 0.6)
        return video_feature, label

def build_index_cache(dataset, cache_path="index_cache.pkl", rebuild=False):
    """
    Builds or loads cached indices for normal/abnormal samples.
    """

    # -------------------------
    # LOAD CACHE
    # -------------------------
    if os.path.exists(cache_path) and not rebuild:
        print(f"[INFO] Loading index cache from {cache_path}")
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
        return cache["normal"], cache["abnormal"]

    # -------------------------
    # BUILD CACHE
    # -------------------------
    print("[INFO] Building index cache (one-time cost)...")

    normal_idxs = []
    abnormal_idxs = []

    for i in range(len(dataset)):
        label = dataset.samples[i][1]   # 🔴 FAST: avoids loading npy file
        if label == 0:
            normal_idxs.append(i)
        else:
            abnormal_idxs.append(i)

    # -------------------------
    # SAVE CACHE
    # -------------------------
    cache = {
        "normal": normal_idxs,
        "abnormal": abnormal_idxs
    }

    with open(cache_path, "wb") as f:
        pickle.dump(cache, f)

    print(f"[INFO] Saved index cache to {cache_path}")

    return normal_idxs, abnormal_idxs

# -------------------------
# Helper: random sampling by video (k videos)
# -------------------------
def get_balanced_batch(dataset, normal_idxs, abnormal_idxs, k, device):
    half = k // 2
    n_idxs = random.sample(normal_idxs, half)
    a_idxs = random.sample(abnormal_idxs, half)
    idxs = n_idxs + a_idxs   # normal first, then abnormal
    feats, labels = [], []
    for i in idxs:
        feat, label = dataset[i]
        feats.append(torch.from_numpy(np.asarray(feat).astype(np.float32)))
        labels.append(torch.tensor(label).float())
    x = torch.stack(feats).to(device)
    y = torch.stack(labels).to(device)
    return x, y
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

## global memory for test
def build_global_memory(model, dataset, normal_idxs, abnormal_idxs, device, n_support=100):
    # Reset both memory units
    model.Amemory.reset()
    model.Nmemory.reset()

    # Sample balanced support: normal first, then abnormal
    half = n_support // 2
    n_idxs = random.sample(normal_idxs, half)
    a_idxs = random.sample(abnormal_idxs, half)
    idxs = n_idxs + a_idxs

    feats = []
    for i in idxs:
        feat, _ = dataset[i]
        feats.append(torch.from_numpy(np.asarray(feat).astype(np.float32)))
    support_x = torch.stack(feats).to(device)

    # Extract features through the model's backbone (no gradients)
    with torch.no_grad():
        x = model.embedding(support_x)
        x = model.localatt(x)
        x = model.selfatt(x)

        # Split exactly as the model's forward does: first half normal, second half abnormal
        half_batch = x.size(0) // 2
        N_x = x[:half_batch]
        A_x = x[half_batch:]

        # Write features to the corresponding memory units
        model.Amemory.write(A_x.detach())
        model.Nmemory.write(N_x.detach())


## test ##
#def test(net, config, wind, test_loader, test_info, step, model_file = None):
def test(net, config, test_loader, step, model_file=None, save_dir='checkpoints', do_norm=False):
    with torch.no_grad():
        net.eval()
        net.flag = "Test"
        #net.flag = "Train"
        if model_file is not None:
            net.load_state_dict(torch.load(model_file))

        load_iter = iter(test_loader)
        frame_gt = np.load("frame_label/gt_ucf_v2.npy")
        frame_predict = None
        
        cls_label = []
        cls_pre = []
        temp_predict = torch.zeros((0)).cuda()
        best_norm_preds, worst_norm_preds, best_abn_preds, worst_abn_preds = [], [], [], []
        best_norm_gt, worst_norm_gt, best_abn_gt, worst_abn_gt = [], [], [], []
        best_norm_acc, worst_norm_acc = -1, 2
        best_abn_acc, worst_abn_acc = -1, 2
        
        for i in range(len(test_loader.dataset)):
            _data, _label = next(load_iter)
            
            _data = _data.cuda()
            _label = _label.cuda()

            
            res = net(_data)
               
            #a_predict = res["frame"]
            #sigmoid
            a_predict = torch.sigmoid(res["frame"])
            temp_predict = torch.cat([temp_predict, a_predict], dim=0)
            if (i + 1) % 10 == 0 :
                cls_label.append(int(_label))
                a_predict = temp_predict.mean(0).cpu().numpy()
                
                cls_pre.append(1 if a_predict.max()>0.5 else 0)          
                #fpre_ = np.repeat(a_predict, 16)
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
# AD_Loss (slightly adjusted weights for stronger localization)
# -------------------------
class AD_Loss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        #self.bce = nn.BCELoss()
        self.bce = nn.BCEWithLogitsLoss()

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

         # ✅ NEW: smooth loss
        smooth_loss = result.get("smooth_loss", torch.tensor(0.0, device=_label.device))

        b = max(_label.size(0) // 2, 1)
        t = max(att.size(1), 1)

        #orig
        anomaly_topk = max(t // 16 + 1, 1)
        #anomaly_topk = max(t // 24 + 1, 1)

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
        mult_dual = 0.2
        mult_triplet = 0.2
        mult_kl = 0.005
        #mult_dist = 0.0001
        mult_dist = 0.0
        mult_smooth = 0.0
        total_loss = (
            anomaly_loss
            + mult_dual * (A_loss + panomaly_loss + N_loss + A_Nloss)
            + mult_triplet * triplet
            + mult_kl * kl_loss
            + mult_dist * distance
            + mult_smooth * smooth_loss
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
    writer = SummaryWriter(log_dir=os.path.join(args.log_dir, "tb_logs"))

    print("[INFO] Loading dataset metadata...")
    train_list_fallback = os.path.join(args.root_dir, 'Train', 'RGB')
    #full_train_dataset = UCF_crime(root_dir=train_list_fallback, modal='rgb', mode='Train',
    #                               num_segments=args.num_segments, len_feature=args.len_feature, seed=args.seed)
    train_dataset = UCF_crime(root_dir=train_list_fallback, modal='rgb', mode='Train',
                                   num_segments=args.num_segments, len_feature=args.len_feature, seed=args.seed)
    test_dataset = UCF_crime(root_dir=train_list_fallback, modal='rgb', mode='Test',
                                   num_segments=args.num_segments, len_feature=args.len_feature, seed=args.seed)
    #all_samples = full_train_dataset.samples
    # FAST BALANCED INDEX CACHE
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
        model.flag = "Train"

        # Warmup supervised pretraining (simple AD_Loss) to stabilize representation
        if step <= args.warmup_steps:
            k_sup = max(args.query_k, args.support_k)
            
            
            #sup_x, sup_y = get_random_batch_from_dataset(train_dataset, k_sup, device, do_norm=False)
            sup_x, sup_y = get_balanced_batch(train_dataset, normal_idxs, abnormal_idxs, k_sup, device)
            
            result = model(sup_x)
            loss_sup, loss_dict_sup = criterion(result, sup_y)
            optimizer.zero_grad()
            loss_sup.backward()
            optimizer.step()
            scheduler.step()
            if step % max(1, args.warmup_steps//5) == 0 or args.debug:
                print(f"[WARMUP {step}/{args.warmup_steps}] sup_loss={float(loss_sup.item()):.6f}")
            continue
        num_tasks = 4
        task_losses = []
        for _ in range(num_tasks):
            support_x, support_y = get_balanced_batch(train_dataset, normal_idxs, abnormal_idxs, args.support_k, device)
            query_x, query_y = get_balanced_batch(train_dataset, normal_idxs, abnormal_idxs, args.query_k, device)

            # inner loop
            result_support = model(support_x)
            inner_loss, inner_loss_dict = criterion(result_support, support_y)
            # inner loss dump
            if step % 5 == 0:
                for k, v in inner_loss_dict.items():
                    writer.add_scalar(f"support/{k}", v.item(), step)
                    writer.add_scalar("support/total_loss", inner_loss.item(), step)
            if step % 10 == 0:
                print(f"[STEP {step}][TASK SUPPORT]")
                for k, v in inner_loss_dict.items():
                    print(f"  {k}: {v.item():.6f}")

            grads = torch.autograd.grad(inner_loss, model.parameters(), create_graph=True)
            fast_weights = {n: p - args.inner_lr * g for (n, p), g in zip(model.named_parameters(), grads)}

            # query
            res_query = stateless.functional_call(model, fast_weights, (query_x,))
            meta_val_loss, query_loss_dict = criterion(res_query, query_y)
            # query loss dump
            if step % 5 == 0:
                for k, v in query_loss_dict.items():
                    writer.add_scalar(f"query/{k}", v.item(), step)
                    writer.add_scalar("query/total_loss", meta_val_loss.item(), step)
            if step % 10 == 0:
                print(f"[STEP {step}][TASK QUERY]")
                for k, v in query_loss_dict.items():
                    print(f"  {k}: {v.item():.6f}")

            task_loss = inner_loss + args.beta * meta_val_loss
            task_losses.append(task_loss)

        meta_loss = torch.stack(task_losses).mean()
        optimizer.zero_grad()
        meta_loss.backward()
        optimizer.step()
        scheduler.step()

        # Evaluate & checkpoint
        if step % args.eval_interval == 0:
            
            #model.flag = "Test"
            
            # new flag for global memory
            model.eval()
            #model.flag = "Train"
            
            #build_global_memory(model,train_dataset,normal_idxs,abnormal_idxs,device,n_support=100)
            model.flag = "Test"
            #build_global_memory(model, train_dataset, normal_idxs, abnormal_idxs, device, n_support=100)
            
            print(f"[EVAL] step {step} -> running align_and_evaluate() ...")
            
            test_loader = data.DataLoader(UCF_crime(root_dir = args.root_dir, mode = 'Test', modal = 'rgb', num_segments = args.num_segments, len_feature = args.len_feature),
                                          batch_size = 1, shuffle = False, num_workers = args.num_workers)
            #auc_score, ap_score, acc = align_and_evaluate(model, config, test_dataset, step, model_file=None, save_dir=args.save_dir)
            
            auc_score, ap_score, acc = test(model, config, test_loader, step, model_file=None, save_dir=args.save_dir)
            
            model.train()
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
