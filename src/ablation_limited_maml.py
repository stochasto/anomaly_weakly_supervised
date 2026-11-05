#!/usr/bin/env python3
"""
Limited‑data ablation with MAML on UCF‑Crime.
Trains the full WSAD model using only a fraction of the training videos.
Usage:
    python ablation_limited_maml.py --frac 0.25 --save_dir sweeps/
"""

import os, random, json, argparse
import numpy as np
import torch
import torch.utils.data as data
from torch.nn.utils import stateless
from tqdm import tqdm

# ---------- UCF imports ----------
from maml_resume_gem import UCF_crime, test_ucf_urdmu_style, AD_Loss
from model_gemini_xd_ablation import WSAD

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--frac', type=float, required=True,
                   help='Fraction of training videos to keep (e.g., 0.25, 0.5, 0.75)')
    p.add_argument('--save_dir', default='./')
    p.add_argument('--root_dir', default='/mnt/c/code/data/I3D')
    p.add_argument('--num_segments', type=int, default=300)
    p.add_argument('--len_feature', type=int, default=1024)
    p.add_argument('--meta_iters', type=int, default=400,
                   help='Shorter meta-training for ablation')
    p.add_argument('--warmup_steps', type=int, default=50)
    p.add_argument('--support_k', type=int, default=8)
    p.add_argument('--query_k', type=int, default=16)
    p.add_argument('--inner_lr', type=float, default=1e-3)
    p.add_argument('--meta_lr', type=float, default=1e-4)
    p.add_argument('--beta', type=float, default=1.0)
    p.add_argument('--max_grad_norm', type=float, default=2.5)
    p.add_argument('--device', type=str, default='cuda')
    return p.parse_args()

def build_index_cache(dataset):
    normal = [i for i, (_, lbl) in enumerate(dataset.samples) if lbl == 0]
    abnormal = [i for i, (_, lbl) in enumerate(dataset.samples) if lbl == 1]
    return normal, abnormal

def get_balanced_batch(dataset, normal_idxs, abnormal_idxs, k, device):
    half = k // 2
    n_idxs = random.sample(normal_idxs, min(half, len(normal_idxs)))
    a_idxs = random.sample(abnormal_idxs, min(half, len(abnormal_idxs)))
    # If one class runs out, fill with the other (shouldn't happen with decent fraction)
    idxs = n_idxs + a_idxs
    feats, labels = [], []
    for i in idxs:
        feat, lbl = dataset[i]
        feats.append(torch.from_numpy(np.asarray(feat).astype(np.float32)))
        labels.append(torch.tensor(lbl).float())
    return torch.stack(feats).to(device), torch.stack(labels).to(device)

def compute_grad_norm(parameters):
    total_norm = 0.0
    for p in parameters:
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

def main():
    args = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.save_dir, exist_ok=True)

    # Load full training dataset to get list of videos
    full_train = UCF_crime(root_dir=args.root_dir, modal='rgb', mode='Train',
                           num_segments=args.num_segments, len_feature=args.len_feature,
                           seed=-1)

    # Separate indices by label
    normal_idx_full = [i for i, (_, lbl) in enumerate(full_train.samples) if lbl == 0]
    abnormal_idx_full = [i for i, (_, lbl) in enumerate(full_train.samples) if lbl == 1]

    # Randomly select fraction of each class
    random.shuffle(normal_idx_full)
    random.shuffle(abnormal_idx_full)
    keep_n = int(len(normal_idx_full) * args.frac)
    keep_a = int(len(abnormal_idx_full) * args.frac)
    selected_normal = normal_idx_full[:keep_n]
    selected_abnormal = abnormal_idx_full[:keep_a]

    # Build a new dataset with only the selected samples
    # We'll create a copy of the full dataset and replace its samples list.
    # The easiest way: instantiate a new dataset and then override .samples
    train_dataset = UCF_crime(root_dir=args.root_dir, modal='rgb', mode='Train',
                              num_segments=args.num_segments, len_feature=args.len_feature,
                              seed=-1)
    # Reorder the samples list to contain only the selected indices
    all_selected = selected_normal + selected_abnormal
    new_samples = [train_dataset.samples[i] for i in all_selected]
    train_dataset.samples = new_samples
    # Build index cache from this subset
    normal_idxs, abnormal_idxs = build_index_cache(train_dataset)

    # Test set (unchanged)
    test_dataset = UCF_crime(root_dir=args.root_dir, modal='rgb', mode='Test',
                             num_segments=args.num_segments, len_feature=args.len_feature,
                             seed=-1)
    test_loader = data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    # Model
    model = WSAD(input_size=args.len_feature, flag="Train", a_nums=200, n_nums=200,
                 config={"use_local_att": True, "use_transformer": True,
                         "use_memory": True, "use_vae": True})
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.meta_lr,
                                 betas=(0.9, 0.999), weight_decay=5e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.meta_iters)
    criterion = AD_Loss()

    best_auc = 0.0
    best_ap = 0.0
    best_acc = 0.0

    print(f"Training with {args.frac*100:.0f}% data ({len(normal_idxs)} normal, {len(abnormal_idxs)} abnormal)")

    for step in tqdm(range(1, args.meta_iters + 1), desc=f"Frac={args.frac}", ncols=80):
        model.train()
        model.flag = "Train"

        if step <= args.warmup_steps:
            k_sup = min(64, 2*(len(normal_idxs)//2))  # ensure we can sample enough
            if k_sup < 2:
                continue  # not enough data
            sup_x, sup_y = get_balanced_batch(train_dataset, normal_idxs, abnormal_idxs, k_sup, device)
            result = model(sup_x)
            loss_sup, _ = criterion(result, sup_y)
            optimizer.zero_grad()
            loss_sup.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
        else:
            num_tasks = 4 if len(normal_idxs) >= 4 and len(abnormal_idxs) >= 4 else 1
            task_losses = []
            for _ in range(num_tasks):
                # Adapt support/query size to available data
                sup_k = min(args.support_k, len(normal_idxs)//2*2)
                q_k = min(args.query_k, len(normal_idxs)//2*2)
                if sup_k < 2 or q_k < 2:
                    continue
                support_x, support_y = get_balanced_batch(train_dataset, normal_idxs, abnormal_idxs, sup_k, device)
                query_x, query_y = get_balanced_batch(train_dataset, normal_idxs, abnormal_idxs, q_k, device)

                result_support = model(support_x)
                inner_loss, _ = criterion(result_support, support_y)
                grads = torch.autograd.grad(inner_loss, model.parameters(),
                                            create_graph=True, allow_unused=True)
                fast_weights = {}
                for (n, p), g in zip(model.named_parameters(), grads):
                    if g is None or 'encoder' in n or 'Amemory' in n or 'Nmemory' in n:
                        fast_weights[n] = p
                    else:
                        fast_weights[n] = p - args.inner_lr * g
                res_query = stateless.functional_call(model, fast_weights, (query_x,))
                meta_val_loss, _ = criterion(res_query, query_y)
                task_loss = inner_loss + args.beta * meta_val_loss
                task_losses.append(task_loss)

            if len(task_losses) == 0:
                continue
            loss_tensor = torch.stack(task_losses)
            if len(task_losses) >= 2:
                median = loss_tensor.median()
                mask = loss_tensor < 2.0 * median
                if mask.sum() >= 1:
                    loss_tensor = loss_tensor[mask]
            meta_loss = loss_tensor.mean()
            if torch.isnan(meta_loss):
                continue
            optimizer.zero_grad()
            meta_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()

        # Evaluate every 100 steps and at the end
        if step % 100 == 0 or step == args.meta_iters:
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
            print(f"[Frac={args.frac}] Step {step}: AUC={auc_score:.4f} (best={best_auc:.4f})")

    # Save results
    result = {
        'frac': args.frac,
        'best_auc': best_auc,
        'best_ap': best_ap,
        'best_acc': best_acc,
        'num_normal': len(normal_idxs),
        'num_abnormal': len(abnormal_idxs)
    }
    output_file = os.path.join(args.save_dir, f"limited_data_{args.frac}.json")
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {output_file}")

if __name__ == '__main__':
    main()