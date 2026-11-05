#!/usr/bin/env python3
"""
Sensitivity sweep – one JSON per parameter, prints metrics for each value as it completes.
Usage:
    python sweep_sensitivity_xd_v2.py --sweep_param mem_weight  [--meta_iters 250]
"""

import os, sys, json, argparse, random
import numpy as np
import torch
import torch.utils.data as data
from torch.nn.utils import stateless
from tqdm import tqdm

# ---------- XD imports ----------
from maml_xd import XDVideoDataset, test_xd_frame, get_balanced_batch, build_index_cache
from model_gemini_xd_v3 import WSAD

# ---------- Configurable AD_Loss ----------
class SweepADLossXD(torch.nn.Module):
    def __init__(self, w_mem=1.0, w_anloss=4.0, w_triplet=0.5, w_kl=0.2,
                 w_ortho=4.0, w_sparsity=1.0, w_stubborn=2.0, w_contrast=2.0):
        super().__init__()
        self.bce = torch.nn.BCEWithLogitsLoss()
        self.w_mem = w_mem
        self.w_anloss = w_anloss
        self.w_triplet = w_triplet
        self.w_kl = w_kl
        self.w_ortho = w_ortho
        self.w_sparsity = w_sparsity
        self.w_stubborn = w_stubborn
        self.w_contrast = w_contrast
        self.topk_frac = 24

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

        k_top = max(T // self.topk_frac + 1, 1)

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
            + self.w_mem * (A_loss + panomaly_loss + N_loss)
            + self.w_anloss * A_Nloss
            + self.w_triplet * triplet
            + self.w_kl * kl
            + self.w_ortho * ortho
            + self.w_sparsity * sparsity_loss
            + self.w_stubborn * stubborn_noise
            + self.w_contrast * contrastive_A_N
        )
        return total_loss, {"total_loss": total_loss}

# ---------- Single training run ----------
def run_one_config(args, loss_weights, inner_lr, meta_lr, param_name, val):
    device = torch.device('cuda')
    train_dataset = XDVideoDataset(root_dir=args.root_dir, mode='Train', modal='rgb',
                                   num_segments=args.num_segments, len_feature=args.len_feature,
                                   seed=42)
    test_dataset = XDVideoDataset(root_dir=args.root_dir, mode='Test', modal='rgb',
                                  num_segments=args.num_segments, len_feature=args.len_feature,
                                  seed=42)
    normal_idxs, abnormal_idxs = build_index_cache(train_dataset,
                                                   cache_path="index_cache_xd_sweep.pkl")
    test_loader = data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    model = WSAD(input_size=args.len_feature, flag="Train", a_nums=200, n_nums=200)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=meta_lr, betas=(0.9, 0.999), weight_decay=5e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.meta_iters)
    criterion = SweepADLossXD(**loss_weights)

    best_auc = best_ap = best_acc = 0.0
    auc_history = []

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
                                            create_graph=True, allow_unused=False)
                fast_weights = {}
                for (n, p), g in zip(model.named_parameters(), grads):
                    if 'encoder' in n or 'Amemory' in n or 'Nmemory' in n:
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
            auc, ap, acc = test_xd_frame(model,
                                         {'num_segments': args.num_segments, 'len_feature': args.len_feature,
                                          'device': device, 'gt_file': 'frame_label/xd_frame_gt_generated.npy'},
                                         test_loader)
            auc_history.append((step, auc))
            if auc > best_auc:
                best_auc = auc
                best_ap = ap
                best_acc = acc
            # print(f"  [{param_name}={val}] Step {step}: AUC={auc:.4f} (best={best_auc:.4f})")

    # Save per‑run history (optional)
    hist_file = os.path.join(args.save_dir, f"history_{param_name}_{val}.json")
    with open(hist_file, 'w') as f:
        json.dump({'param': param_name, 'value': val, 'history': auc_history}, f, indent=2)

    return best_auc, best_ap, best_acc

# ---------- Sweep definitions ----------
SWEEP_RANGES = {
    'mem_weight': [0.5, 1.0, 1.5, 2.0, 3.0],
    'ortho_weight': [2.0, 3.0, 4.0, 5.0, 6.0],
    'AtoN_weight': [2.0, 3.0, 4.0, 5.0, 6.0]
}

DEFAULT_WEIGHTS = {
    'w_mem': 1.0,
    'w_anloss': 4.0,
    'w_triplet': 0.5,
    'w_kl': 0.2,
    'w_ortho': 4.0,
    'w_sparsity': 1.0,
    'w_stubborn': 2.0,
    'w_contrast': 2.0
}

# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sweep_param', required=True, choices=list(SWEEP_RANGES.keys()))
    parser.add_argument('--meta_iters', type=int, default=250)
    parser.add_argument('--warmup_steps', type=int, default=50)
    parser.add_argument('--inner_lr', type=float, default=5e-4)
    parser.add_argument('--meta_lr', type=float, default=1e-5)
    parser.add_argument('--root_dir', default='/mnt/c/code/ae_anomaly/UR-DMU/xdv')
    parser.add_argument('--num_segments', type=int, default=200)
    parser.add_argument('--len_feature', type=int, default=1024)
    parser.add_argument('--save_dir', default='sweep_xd_results')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    param = args.sweep_param
    values = SWEEP_RANGES[param]
    master_json = os.path.join(args.save_dir, f"sweep_{param}.json")

    # Load existing master dict if present
    if os.path.exists(master_json):
        with open(master_json, 'r') as f:
            master = json.load(f)
    else:
        master = {}

    for val in values:
        # Skip if already in master
        if str(val) in master:
            print(f"Value {val} already present in {master_json}, skipping.")
            continue

        weights = DEFAULT_WEIGHTS.copy()
        if param == 'mem_weight':
            weights['w_mem'] = val
        elif param == 'ortho_weight':
            weights['w_ortho'] = val
        elif param == 'AtoN_weight':
            weights['w_anloss'] = val

        print(f"\n=== Running {param}={val} ===")
        best_auc, best_ap, best_acc = run_one_config(args, weights, args.inner_lr, args.meta_lr, param, val)

        # Immediately print results for this value
        print(f">>> {param}={val}: AUC={best_auc:.4f}, AP={best_ap:.4f}, Acc={best_acc:.4f}")

        # Store in master dict
        master[str(val)] = {
            'auc': best_auc,
            'ap': best_ap,
            'acc': best_acc
        }

        # Write back after each run
        with open(master_json, 'w') as f:
            json.dump(master, f, indent=2)
        print(f"Updated {master_json}")

    # Generate final report from master JSON
    generate_report(param, master, args.save_dir)

def generate_report(param, master, save_dir):
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 11, 'font.family': 'serif', 'pdf.fonttype': 42, 'ps.fonttype': 42})

    # Prepare sorted list
    items = sorted(master.items(), key=lambda x: float(x[0]))
    vals = [float(v) for v, _ in items]
    aucs = [d['auc'] for _, d in items]
    aps = [d['ap'] for _, d in items]
    accs = [d['acc'] for _, d in items]

    # Summary table
    print(f"\n=== Final Summary for {param} ===")
    print("Value\tAUC\tAP\tAcc")
    for v, auc, ap, acc in zip(vals, aucs, aps, accs):
        print(f"{v}\t{auc:.4f}\t{ap:.4f}\t{acc:.4f}")

    # Save CSV
    with open(os.path.join(save_dir, f"summary_{param}.csv"), 'w') as f:
        f.write("Value,AUC,AP,Acc\n")
        for v, auc, ap, acc in zip(vals, aucs, aps, accs):
            f.write(f"{v},{auc:.4f},{ap:.4f},{acc:.4f}\n")

    # Plot AUC vs value
    plt.figure(figsize=(6,4))
    plt.plot(vals, aucs, 'o-', color='#2166ac')
    opt_key = {'mem_weight': 'w_mem', 'ortho_weight': 'w_ortho', 'AtoN_weight': 'w_anloss'}[param]
    opt_val = DEFAULT_WEIGHTS[opt_key]
    plt.axvline(x=opt_val, color='red', linestyle='--', label=f'Chosen ({opt_val})')
    plt.xlabel(param)
    plt.ylabel('Frame-level AUC')
    plt.title(f'Sensitivity to {param}')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'sensitivity_{param}.pdf'), bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    main()