#!/usr/bin/env python3
"""
urdmu_fewshot_xd2ucf.py

Cross‑dataset evaluation of UR‑DMU:  XD‑Violence → UCF‑Crime.

- Loads an XD‑trained UR‑DMU checkpoint.
- Builds a balanced support set of 32 UCF training videos (16 normal, 16 abnormal)
  and saves their paths to a text file.
- Evaluates in three modes:
    --mode zero_shot       : direct test on UCF (no adaptation)
    --mode fewshot         : fine‑tune on support set with memory + VAE frozen
    --mode fewshot_nofreeze : fine‑tune with all parameters updated
- Saves all metrics (AUC, AP, Accuracy) to a JSON file.
"""

import os
import json
import sys
import numpy as np
import torch
import torch.utils.data as data
from tqdm import tqdm

# ---------- original UR‑DMU imports ----------
from options import parse_args            # original argument parser
from config import Config                 # original Config class
from model import WSAD                    # original UR‑DMU model
from dataset_loader import UCF_crime      # UCF dataset class
from train import train, AD_Loss          # original training step
import utils
from ucf_test import test                 # original UCF test function


def fewshot_adapt_and_test(args, source_ckpt, target_root, support_size=32,
                           finetune_steps=50, lr=1e-4,
                           support_file="support_ucf_videos.txt",
                           modes=["zero_shot", "fewshot"]):
    """
    Cross‑dataset adaptation and testing for UR‑DMU (XD → UCF).
    Args:
        args: namespace from parse_args()
        source_ckpt: path to XD‑trained checkpoint
        target_root: root directory for UCF dataset
        support_size: total videos in support set (must be even)
        finetune_steps: number of adaptation steps
        lr: learning rate for fine‑tuning
        support_file: file where support video names are saved
        modes: list of evaluation modes to run
    """
    # ---------- Config for target dataset (UCF) ----------
    config = Config(args)
    config.len_feature = 1024
    config.root_dir = target_root          # UCF root
    if config.seed >= 0:
        utils.set_seed(config.seed)

    # ---------- Build model and load XD checkpoint ----------
    net = WSAD(config.len_feature, flag="Train", a_nums=60, n_nums=60)
    net = net.cuda()
    print(f"Loading XD checkpoint: {source_ckpt}")
    net.load_state_dict(torch.load(source_ckpt))

    # ---------- Support set from UCF training split ----------
    normal_dataset = UCF_crime(root_dir=config.root_dir, mode='Train',
                               modal=config.modal, num_segments=config.num_segments,
                               len_feature=config.len_feature, is_normal=True)
    abnormal_dataset = UCF_crime(root_dir=config.root_dir, mode='Train',
                                 modal=config.modal, num_segments=config.num_segments,
                                 len_feature=config.len_feature, is_normal=False)

    half = support_size // 2
    # Random selection with fixed seed for reproducibility
    rng = np.random.RandomState(config.seed)
    normal_indices = rng.choice(len(normal_dataset), half, replace=False)
    abnormal_indices = rng.choice(len(abnormal_dataset), half, replace=False)

    # ---- Save support video names to file ----
    normal_paths = [normal_dataset.vid_list[i][0] for i in normal_indices]
    abnormal_paths = [abnormal_dataset.vid_list[i][0] for i in abnormal_indices]
    def video_name(path):
        return os.path.basename(path).split("_x264")[0]  # compact name
    with open(support_file, "w") as f:
        f.write("Normal videos:\n")
        for p in normal_paths:
            f.write(f"{video_name(p)}\t{p}\n")
        f.write("Abnormal videos:\n")
        for p in abnormal_paths:
            f.write(f"{video_name(p)}\t{p}\n")
    print(f"Support set of {support_size} videos saved to {support_file}")

    # ---------- Helper: evaluate on UCF test set ----------
    def evaluate_model(model):
        test_loader = data.DataLoader(
            UCF_crime(root_dir=config.root_dir, mode='Test', modal=config.modal,
                      num_segments=config.num_segments, len_feature=config.len_feature),
            batch_size=1, shuffle=False, num_workers=config.num_workers)
        test_info = {"step": [], "auc": [], "ap": [], "ac": []}
        model.eval()
        model.flag = "Test"
        test(model, config, test_loader, test_info, step=0)
        return {
            "auc": float(test_info["auc"][-1]),
            "ap": float(test_info["ap"][-1]),
            "acc": float(test_info["ac"][-1])
        }

    # ---------- Freezing utilities ----------
    def freeze_memory(model):
        for name, param in model.named_parameters():
            if 'Amemory' in name or 'Nmemory' in name or 'encoder_mu' in name or 'encoder_var' in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

    def unfreeze_all(model):
        for param in model.parameters():
            param.requires_grad = True

    # ---------- Iterator for original train() ----------
    class SingleIterator:
        def __init__(self, data, label):
            self.data = data
            self.label = label
        def __iter__(self):
            return self
        def __next__(self):
            return self.data, self.label

    results = {}

    # ---------- Zero‑shot (no adaptation) ----------
    if "zero_shot" in modes:
        print("\n=== Zero‑Shot UCF Evaluation ===")
        model_zero = WSAD(config.len_feature, flag="Test", a_nums=60, n_nums=60).cuda()
        model_zero.load_state_dict(net.state_dict())
        results["zero_shot"] = evaluate_model(model_zero)
        del model_zero

    # ---------- Few‑shot (frozen memory/VAE) ----------
    if "fewshot" in modes:
        print(f"\n=== Few‑Shot Adaptation (frozen memory, {finetune_steps} steps) ===")
        model_f = WSAD(config.len_feature, flag="Train", a_nums=60, n_nums=60).cuda()
        model_f.load_state_dict(net.state_dict())
        freeze_memory(model_f)
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model_f.parameters()),
            lr=lr, betas=(0.9, 0.999), weight_decay=0.00005)
        criterion = AD_Loss()
        model_f.train()
        for step in tqdm(range(1, finetune_steps + 1)):
            n_idx = np.random.choice(half)
            a_idx = np.random.choice(half)
            n_feat, n_lbl = normal_dataset[normal_indices[n_idx]]
            a_feat, a_lbl = abnormal_dataset[abnormal_indices[a_idx]]
            n_feat = torch.from_numpy(n_feat).unsqueeze(0).cuda().float()
            a_feat = torch.from_numpy(a_feat).unsqueeze(0).cuda().float()
            n_lbl  = torch.tensor([n_lbl]).cuda().float()
            a_lbl  = torch.tensor([a_lbl]).cuda().float()
            train(model_f,
                  SingleIterator(n_feat, n_lbl),
                  SingleIterator(a_feat, a_lbl),
                  optimizer, criterion, step)
        results["fewshot"] = evaluate_model(model_f)
        del model_f

    # ---------- Few‑shot (no freeze) ----------
    if "fewshot_nofreeze" in modes:
        print(f"\n=== Few‑Shot Adaptation (full fine‑tuning, {finetune_steps} steps) ===")
        model_nf = WSAD(config.len_feature, flag="Train", a_nums=60, n_nums=60).cuda()
        model_nf.load_state_dict(net.state_dict())
        unfreeze_all(model_nf)
        optimizer = torch.optim.Adam(model_nf.parameters(),
                                     lr=lr, betas=(0.9, 0.999), weight_decay=0.00005)
        criterion = AD_Loss()
        model_nf.train()
        for step in tqdm(range(1, finetune_steps + 1)):
            n_idx = np.random.choice(half)
            a_idx = np.random.choice(half)
            n_feat, n_lbl = normal_dataset[normal_indices[n_idx]]
            a_feat, a_lbl = abnormal_dataset[abnormal_indices[a_idx]]
            n_feat = torch.from_numpy(n_feat).unsqueeze(0).cuda().float()
            a_feat = torch.from_numpy(a_feat).unsqueeze(0).cuda().float()
            n_lbl  = torch.tensor([n_lbl]).cuda().float()
            a_lbl  = torch.tensor([a_lbl]).cuda().float()
            train(model_nf,
                  SingleIterator(n_feat, n_lbl),
                  SingleIterator(a_feat, a_lbl),
                  optimizer, criterion, step)
        results["fewshot_nofreeze"] = evaluate_model(model_nf)
        del model_nf

    return results


if __name__ == "__main__":
    # ---------- Step 1: Get all original UR-DMU arguments ----------
    args = parse_args()

    # ---------- Step 2: Override ONLY what's needed BEFORE any Config usage ----------
    # UCF root directory (required by Config and datasets)
    args.root_dir = "/mnt/c/code/data/I3D"

    # ---------- Step 3: Add custom parameters for this script ----------
    import argparse
    custom_parser = argparse.ArgumentParser()
    custom_parser.add_argument("--source_ckpt", required=True,
                               help="Path to XD‑trained UR‑DMU checkpoint")
    custom_parser.add_argument("--mode", nargs="+", default=["zero_shot", "fewshot"],
                               choices=["zero_shot", "fewshot", "fewshot_nofreeze"])
    custom_parser.add_argument("--support_size", type=int, default=32)
    custom_parser.add_argument("--finetune_steps", type=int, default=50)
    custom_parser.add_argument("--finetune_lr", type=float, default=1e-4)
    custom_parser.add_argument("--output_json", default="xd2ucf_urdmu_results.json")
    custom_parser.add_argument("--support_file", default="support_ucf_xd2ucf.txt")
    # Parse known custom args, ignore everything else (i.e., the original UR-DMU args)
    custom_args, _ = custom_parser.parse_known_args()

    # Attach custom args to the main namespace (no collisions)
    for key, value in vars(custom_args).items():
        setattr(args, key, value)

    # ---------- Step 4: Run the experiment ----------
    results = fewshot_adapt_and_test(
        args,
        source_ckpt=args.source_ckpt,
        target_root=args.root_dir,               # UCF root
        support_size=args.support_size,
        finetune_steps=args.finetune_steps,
        lr=args.finetune_lr,
        support_file=args.support_file,
        modes=args.mode
    )

    # ---------- Step 5: Save results ----------
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {args.output_json}")
    print(json.dumps(results, indent=2))