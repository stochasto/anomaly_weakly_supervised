#!/usr/bin/env python3
"""Quick evaluation of a saved checkpoint on UCF-Crime."""

import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc, precision_recall_curve

# Adjust the import to the model file you are currently using
from model_gemini_xd_v3 import WSAD

# Re-use the dataset and evaluation function from your training script.
# If you have them in a separate module, import them; otherwise copy here.
# For simplicity, we assume you have the UCF_crime class and the test_ucf_urdmu_style
# function in the same directory (you can copy them into this script if needed).
from maml_resume_gem import UCF_crime, test_ucf_urdmu_style, test
# (Replace "your_ucf_script" with the actual .py file name without extension)

CKPT_PATH = "checkpoints/THEBESTSOFAR_ucf_sk_8_qk_16_ilr_0.001_mlr_0.0001_beta_1.0_segments_300_anums_200_nnums_200_ap_0.3212_auc_0.8610.pth"

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = {"num_segments": 300, "len_feature": 1024, "device": device}

    # Build test loader (identical to your training script)
    test_dataset = UCF_crime(
        root_dir="/mnt/c/code/data/I3D",
        mode="Test",
        modal="rgb",
        num_segments=300,
        len_feature=1024,
        seed=-1
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    # Load model
    model = WSAD(input_size=1024, flag="Test", a_nums=200, n_nums=200)
    model.to(device)

    checkpoint = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model"])
    print(f"Loaded checkpoint from step {checkpoint.get('step', 'unknown')}")

    # Evaluate
    #auc_score, ap_score, acc = test_ucf_urdmu_style(model, config, test_loader)
    auc_score, ap_score, acc = test(model, config, test_loader, 1)
    print(f"AUC: {auc_score:.4f}")
    print(f"AP:  {ap_score:.4f}")
    print(f"Acc: {acc:.4f}")

if __name__ == "__main__":
    main()