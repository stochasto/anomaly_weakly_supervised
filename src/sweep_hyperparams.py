#!/usr/bin/env python3
"""
sweep_full.py

Run coarse + fine hyperparameter sweep for a given parameter.
Usage:
    python sweep_full.py --sweep_param kl_weight
"""

import os, random, json, argparse
import numpy as np
import torch
import torch.utils.data as data
from torch.nn.utils import stateless
from tqdm import tqdm

# ---------- Your UCF imports ----------
from maml_resume_gem import UCF_crime, test_ucf_urdmu_style
from model_gemini_xd_ablation import WSAD

# ---------- Configurable AD_Loss ----------
class SweepADLoss(torch.nn.Module):
    def __init__(self, w_triplet=0.5, w_kl=0.25, w_ortho=1.5,
                 w_sparsity=0.75, w_stubborn=1.0, w_contrast=1.0,
                 w_anloss=1.5):
        super().__init__()
        self.bce = torch.nn.BCEWithLogitsLoss()
        self.w_triplet = w_triplet
        self.w_kl = w_kl
        self.w_ortho = w_ortho
        self.w_sparsity = w_sparsity
        self.w_stubborn = w_stubborn
        self.w_contrast = w_contrast
        self.w_anloss = w_anloss

    def forward(self, result, _label):
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

        k_top = max(T // 16 + 1, 1)

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
            + self.w_anloss * A_Nloss
            + self.w_triplet * triplet
            + self.w_kl * kl
            + self.w_ortho * ortho
            + self.w_sparsity * sparsity_loss
            + self.w_stubborn * stubborn_noise
            + self.w_contrast * contrastive_A_N
        )
        return total_loss, {"total_loss": total_loss}

# ---------- Helpers ----------
def build_index_cache(dataset):
    normal = [i for i, (_, lbl) in enumerate(dataset.samples) if lbl == 0]
    abnormal = [i for i, (_, lbl) in enumerate(dataset.samples) if lbl == 1]
    return normal, abnormal

def get_balanced_batch(dataset, normal_idxs, abnormal_idxs, k, device):
    half = k // 2
    n_idxs = random.sample(normal_idxs, half)
    a_idxs = random.sample(abnormal_idxs, half)
    idxs = n_idxs + a_idxs
    feats, labels = [], []
    for i in idxs:
        feat, lbl = dataset[i]
        feats.append(torch.from_numpy(np.asarray(feat).astype(np.float32)))
        labels.append(torch.tensor(lbl).float())
    return torch.stack(feats).to(device), torch.stack(labels).to(device)

# ---------- MAML training for one config ----------
def run_one_config(args, config, loss_weights, inner_lr, meta_lr,
                   a_nums, n_nums, name):
    device = torch.device(args.device)
    train_dataset = UCF_crime(root_dir=args.root_dir, modal='rgb', mode='Train',
                              num_segments=args.num_segments, len_feature=args.len_feature,
                              seed=args.seed)
    test_dataset = UCF_crime(root_dir=args.root_dir, modal='rgb', mode='Test',
                             num_segments=args.num_segments, len_feature=args.len_feature,
                             seed=args.seed)
    normal_idxs, abnormal_idxs = build_index_cache(train_dataset)
    test_loader = data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    model = WSAD(input_size=args.len_feature, flag="Train",
                 a_nums=a_nums, n_nums=n_nums, config=config)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=meta_lr, betas=(0.9, 0.999), weight_decay=5e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.meta_iters)
    criterion = SweepADLoss(**loss_weights)

    best_auc = 0.0
    best_ap = 0.0
    best_acc = 0.0

    for step in range(1, args.meta_iters + 1):
        model.train()
        model.flag = "Train"
        if step <= args.warmup_steps:
            sup_x, sup_y = get_balanced_batch(train_dataset, normal_idxs, abnormal_idxs, 64, device)
            result = model(sup_x)
            loss_sup, _ = criterion(result, sup_y)
            optimizer.zero_grad()
            loss_sup.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.5)
            optimizer.step()
            scheduler.step()
        else:
            task_losses = []
            for _ in range(4):
                support_x, support_y = get_balanced_batch(train_dataset, normal_idxs, abnormal_idxs, 8, device)
                query_x, query_y = get_balanced_batch(train_dataset, normal_idxs, abnormal_idxs, 16, device)
                result_support = model(support_x)
                inner_loss, _ = criterion(result_support, support_y)
                grads = torch.autograd.grad(inner_loss, model.parameters(),
                                            create_graph=True, allow_unused=True)
                fast_weights = {}
                for (n, p), g in zip(model.named_parameters(), grads):
                    if g is None or 'encoder' in n or 'Amemory' in n or 'Nmemory' in n:
                        fast_weights[n] = p
                    else:
                        fast_weights[n] = p - inner_lr * g
                res_query = stateless.functional_call(model, fast_weights, (query_x,))
                meta_val_loss, _ = criterion(res_query, query_y)
                task_loss = inner_loss + meta_val_loss
                task_losses.append(task_loss)

            loss_tensor = torch.stack(task_losses)
            if len(task_losses) >= 4:
                median = loss_tensor.median()
                mask = loss_tensor < 2.0 * median
                if mask.sum() >= 2:
                    loss_tensor = loss_tensor[mask]
            meta_loss = loss_tensor.mean()
            if torch.isnan(meta_loss):
                continue
            optimizer.zero_grad()
            meta_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.5)
            optimizer.step()
            scheduler.step()

        if step % 50 == 0 or step == args.meta_iters:
            model.eval()
            model.flag = "Test"
            auc_score, ap_score, acc = test_ucf_urdmu_style(
                model,
                {'num_segments': args.num_segments, 'len_feature': args.len_feature,
                 'device': device, 'gt_file': 'frame_label/gt-ucf_urd.npy'},
                test_loader, model_file=None)
            if auc_score > best_auc:
                best_auc = auc_score
                best_ap = ap_score
                best_acc = acc
            print(f"  [{name}] Step {step}: AUC={auc_score:.4f} (best={best_auc:.4f})")

    return {'auc': best_auc, 'ap': best_ap, 'acc': best_acc}

# ---------- Parameter ranges ----------
BEST = {
    'kl_weight': 0.1,
    'ortho_weight': 0.25,
    'triplet_weight': 0.5,
    'sparsity_weight': 1.0,
    'stubborn_weight': 1.0,
    'contrastive_weight': 1.5,
    'A_Nloss_weight': 1.5,
    'inner_lr': 1e-3,
    'meta_lr': 1e-4,
    'memory_size': 200
}

COARSE = {
    'kl_weight': [0.01, 0.03, 0.1, 0.3, 1.0],
    'ortho_weight': [0.1, 0.3, 1.0, 3.0, 10.0],
    'triplet_weight': [0.1, 0.3, 0.5, 1.0],
    'sparsity_weight': [0.1, 0.3, 1.0, 3.0],
    'stubborn_weight': [0.1, 0.5, 1.0, 2.0],
    'contrastive_weight': [0.1, 0.5, 1.0, 2.0],
    'A_Nloss_weight': [0.5, 1.0, 2.0, 5.0],
    'inner_lr': [5e-4, 1e-3, 2e-3, 5e-3],
    'meta_lr': [5e-5, 1e-4, 2e-4, 5e-4],
    'memory_size': [50, 100, 200, 400]
}

FINE = {
    'kl_weight': [0.08, 0.1, 0.12, 0.15],
    'ortho_weight': [0.2, 0.25, 0.3, 0.35],
    'triplet_weight': [0.4, 0.5, 0.6],
    'sparsity_weight': [0.8, 1.0, 1.2],
    'stubborn_weight': [0.8, 1.0, 1.2],
    'contrastive_weight': [1.2, 1.5, 1.8],
    'A_Nloss_weight': [1.2, 1.5, 1.8],
    'inner_lr': [8e-4, 1e-3, 1.2e-3],
    'meta_lr': [8e-5, 1e-4, 1.2e-4],
    'memory_size': [150, 200, 250]
}

LOSS_WEIGHT_PARAMS = {
    'kl_weight': 'w_kl',
    'ortho_weight': 'w_ortho',
    'triplet_weight': 'w_triplet',
    'sparsity_weight': 'w_sparsity',
    'stubborn_weight': 'w_stubborn',
    'contrastive_weight': 'w_contrast',
    'A_Nloss_weight': 'w_anloss'
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sweep_param', required=True, choices=list(COARSE.keys()))
    parser.add_argument('--root_dir', default='/mnt/c/code/data/I3D')
    parser.add_argument('--num_segments', type=int, default=300)
    parser.add_argument('--len_feature', type=int, default=1024)
    parser.add_argument('--meta_iters', type=int, default=150)
    parser.add_argument('--warmup_steps', type=int, default=50)
    parser.add_argument('--inner_lr_default', type=float, default=1e-3)
    parser.add_argument('--meta_lr_default', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=-1)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output_file', default='sweep_results.json')
    args = parser.parse_args()

    results = []
    if os.path.exists(args.output_file):
        with open(args.output_file, 'r') as f:
            results = json.load(f)

    # Default loss weights (from BEST)
    default_loss_weights = {
        'w_triplet': BEST['triplet_weight'],
        'w_kl': BEST['kl_weight'],
        'w_ortho': BEST['ortho_weight'],
        'w_sparsity': BEST['sparsity_weight'],
        'w_stubborn': BEST['stubborn_weight'],
        'w_contrast': BEST['contrastive_weight'],
        'w_anloss': BEST['A_Nloss_weight']
    }

    inner_lr = args.inner_lr_default
    meta_lr = args.meta_lr_default
    base_config = {"use_local_att": True, "use_transformer": True,
                   "use_memory": True, "use_vae": True}

    # Run coarse
    for val in COARSE[args.sweep_param]:
        a_nums = n_nums = 200
        if args.sweep_param == 'memory_size':
            a_nums = n_nums = int(val)
        loss_weights = default_loss_weights.copy()
        if args.sweep_param in LOSS_WEIGHT_PARAMS:
            loss_weights[LOSS_WEIGHT_PARAMS[args.sweep_param]] = val
        elif args.sweep_param == 'inner_lr':
            inner_lr = val
        elif args.sweep_param == 'meta_lr':
            meta_lr = val

        name = f"coarse_{args.sweep_param}={val}"
        print(f"\n=== {name} ===")
        res = run_one_config(args, base_config, loss_weights, inner_lr, meta_lr,
                             a_nums, n_nums, name)
        entry = {'param': args.sweep_param, 'value': val, 'phase': 'coarse',
                 'auc': res['auc'], 'ap': res['ap'], 'acc': res['acc']}
        results.append(entry)
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)

        # Early skip if AUC drops below 0.7
        if res['auc'] < 0.7:
            print(f"  AUC below 0.7, skipping further coarse values? (only skip this value itself)")
            # We'll still continue to next values; you can add logic if desired

    # Now fine sweep
    for val in FINE[args.sweep_param]:
        a_nums = n_nums = 200
        if args.sweep_param == 'memory_size':
            a_nums = n_nums = int(val)
        loss_weights = default_loss_weights.copy()
        if args.sweep_param in LOSS_WEIGHT_PARAMS:
            loss_weights[LOSS_WEIGHT_PARAMS[args.sweep_param]] = val
        elif args.sweep_param == 'inner_lr':
            inner_lr = val
        elif args.sweep_param == 'meta_lr':
            meta_lr = val

        name = f"fine_{args.sweep_param}={val}"
        print(f"\n=== {name} ===")
        res = run_one_config(args, base_config, loss_weights, inner_lr, meta_lr,
                             a_nums, n_nums, name)
        entry = {'param': args.sweep_param, 'value': val, 'phase': 'fine',
                 'auc': res['auc'], 'ap': res['ap'], 'acc': res['acc']}
        results.append(entry)
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)

    print(f"\nSweep complete. Results in {args.output_file}")

if __name__ == '__main__':
    main()