#!/usr/bin/env python3
"""
profile_model.py

Profile your WSAD model with thop for parameters, FLOPs,
disk size, and GPU inference time.
"""

import os
import time
import torch
import numpy as np
from thop import profile, clever_format

# ---------------------------
# 1. Import your model (exactly as in training)
# ---------------------------
from model_gemini_xd_v3 import WSAD

# ---------------------------
# 2. Configuration – match your training script
# ---------------------------
INPUT_DIM = 1024          # len_feature
A_NUMS   = 200            # a_nums
N_NUMS   = 200            # n_nums
T        = 200            # num_segments (typical snippet count)
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH    = 1              # single video for inference timing

print(f"Using device: {DEVICE}")

# ---------------------------
# 3. Instantiate the model
# ---------------------------
# flag="Test" disables any training‑specific behaviour (dropout etc.)
model = WSAD(input_size=INPUT_DIM, flag="Test",
             a_nums=A_NUMS, n_nums=N_NUMS).to(DEVICE)
model.eval()

# ---------------------------
# 4. Dummy input matching your data
# ---------------------------
dummy = torch.randn(BATCH, T, INPUT_DIM).to(DEVICE)

# ---------------------------
# 5. Count parameters and FLOPs with thop
# ---------------------------
print("\n[INFO] Running thop profile...")
flops, params = profile(model, inputs=(dummy,), verbose=False)
flops_str, params_str = clever_format([flops, params], "%.2f")
print("-" * 50)
print(f"Total trainable parameters : {params_str}")
print(f"Total FLOPs (forward pass) : {flops_str}")
print("-" * 50)

# Optional raw numbers for your paper
print(f"Raw parameters: {params}")
print(f"Raw FLOPs    : {flops}")

# ---------------------------
# 6. Model disk size (state_dict only)
# ---------------------------
temp_file = "temp_model_weights.pth"
torch.save(model.state_dict(), temp_file)
size_mb = os.path.getsize(temp_file) / (1024 * 1024)
os.remove(temp_file)
print(f"\nState dict size on disk: {size_mb:.2f} MB")

# ---------------------------
# 7. Inference time measurement
# ---------------------------
if DEVICE.type == "cuda":
    # Warmup
    with torch.no_grad():
        for _ in range(20):
            _ = model(dummy)
        torch.cuda.synchronize()

    # Timed runs
    n_runs = 100
    start = time.time()
    with torch.no_grad():
        for _ in range(n_runs):
            _ = model(dummy)
        torch.cuda.synchronize()
    end = time.time()

    avg_ms = 1000 * (end - start) / n_runs
    print(f"Average inference time per video: {avg_ms:.2f} ms")
    print(f"Throughput (videos/second)      : {1000/avg_ms:.1f}")
else:
    print("GPU not available – inference timing skipped.")