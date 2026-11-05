#!/usr/bin/env python3
"""
Profile key modules of the WSAD model:
  - Temporal Conv1D (embedding)
  - Local Attention
  - Dual‑branch Transformer
  - Dual Memory readout (abnormal + normal banks)
  - Variational Encoder (mu + var)
  - Classifier head
"""

import torch
import time
import numpy as np
from model_gemini_xd_v3 import WSAD, Temporal, LocalAttention, Transformer, Memory_Unit, ADCLS_head
from thop import profile, clever_format

# -------------------- Setup --------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
T, D = 200, 1024           # typical input: 200 snippets, 1024-dim I3D features
B = 1                      # single video for inference timing

# Create dummy input
x = torch.randn(B, T, D).to(device)

# Instantiate full model (only for submodule extraction)
model = WSAD(input_size=D, flag='Test', a_nums=200, n_nums=200).to(device)
model.eval()

# Extract submodules
temporal      = model.embedding        # Temporal (Conv1D)
local_att     = model.localatt         # LocalAttention
transformer   = model.selfatt          # Transformer
memory_A      = model.Amemory          # abnormal memory bank
memory_N      = model.Nmemory          # normal memory bank
encoder_mu    = model.encoder_mu       # VAE mu
encoder_var   = model.encoder_var      # VAE var
classifier    = model.cls_head         # ADCLS_head (1024 -> 1)

# -------------------- Helper functions --------------------
def measure_macs_params(module, input_tensor):
    """Return MACs (M) and Params (M) using thop."""
    macs, params = profile(module, inputs=(input_tensor,), verbose=False)
    return macs / 1e6, params / 1e6   # convert to millions

def measure_time(module, input_tensor, repeats=100):
    """Return average forward time in ms using CUDA events."""
    # Warm-up
    for _ in range(10):
        _ = module(input_tensor)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        _ = module(input_tensor)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / repeats

# -------------------- Run profiling --------------------
results = []

# 1. Temporal Conv1D
print("Profiling Temporal Conv1D...")
macs, params = measure_macs_params(temporal, x)
t = measure_time(temporal, x)
results.append(("Temporal Conv1D", params, macs, t))

# 2. Local Attention (input from temporal)
x1 = temporal(x)
print("Profiling Local Attention...")
macs, params = measure_macs_params(local_att, x1)
t = measure_time(local_att, x1)
results.append(("Local Attention", params, macs, t))

# 3. Dual‑branch Transformer (input after local att)
x2 = local_att(x1)
print("Profiling Transformer...")
macs, params = measure_macs_params(transformer, x2)
t = measure_time(transformer, x2)
results.append(("Dual‑Branch Transformer", params, macs, t))

x3 = transformer(x2)

# 4. Dual Memory readout (we need both banks, take average of two readings)
print("Profiling Dual Memory (abnormal bank)...")
macs_A, params_A = measure_macs_params(memory_A, x3)
t_A = measure_time(memory_A, x3)
print("Profiling Dual Memory (normal bank)...")
macs_N, params_N = measure_macs_params(memory_N, x3)
t_N = measure_time(memory_N, x3)
# Memory readout includes both banks, so sum their parameters/MACs/time
results.append(("Dual Memory (readout)", params_A + params_N, macs_A + macs_N, t_A + t_N))

# For memory readout we need the actual outputs (just to get dimensions for VAE)
# In the full forward, A_aug and N_aug are then passed to VAE.
# We'll use dummy augmented features of the same shape.
A_att, A_aug = memory_A(x3)   # both (B, T, D)
N_att, N_aug = memory_N(x3)

# 5. Variational Encoder (mu + var)
print("Profiling Variational Encoder...")
# The encoder_mu and encoder_var are simple Linear layers, we profile them separately and sum.
macs_mu, params_mu = measure_macs_params(encoder_mu, A_aug)
t_mu = measure_time(encoder_mu, A_aug)
macs_var, params_var = measure_macs_params(encoder_var, A_aug)
t_var = measure_time(encoder_var, A_aug)
results.append(("Variational Encoder (mu+var)", params_mu + params_var, macs_mu + macs_var, t_mu + t_var))

# 6. Classifier
# In inference, the fused features are [x, gated augmentations], dim = 1024.
# Let's build a dummy fused feature:
fused = torch.randn(B, T, 1024).to(device)
print("Profiling Classifier...")
macs, params = measure_macs_params(classifier, fused)
t = measure_time(classifier, fused)
results.append(("Classifier", params, macs, t))

# -------------------- Print table --------------------
print("\n" + "="*80)
print(f"{'Module':<30} {'Params (M)':>10} {'MACs (M)':>10} {'Time (ms)':>10}")
print("-"*80)
total_params = total_macs = total_time = 0.0
for name, p, m, t in results:
    print(f"{name:<30} {p:>10.2f} {m:>10.0f} {t:>10.2f}")
    total_params += p
    total_macs   += m
    total_time   += t
print("-"*80)
print(f"{'Total':<30} {total_params:>10.2f} {total_macs:>10.0f} {total_time:>10.2f}")
print("="*80)