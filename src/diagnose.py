#!/usr/bin/env python3
"""
Diagnostic script to identify issues causing 40% AUC
"""

import torch
import numpy as np

print("="*60)
print("DIAGNOSTIC CHECKS")
print("="*60)

# Check 1: Label distribution
print("\n[CHECK 1] Label Distribution")
print("Expected: ~50% anomaly in support/query sets")
labels_test = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0])
anomaly_ratio = (labels_test == 1).float().mean()
print(f"Anomaly ratio: {anomaly_ratio:.2%}")

# Check 2: Model output range
print("\n[CHECK 2] Model Output Range")
print("After sigmoid, should be [0, 1]")
logits = torch.randn(4, 128) * 2  # simulate model output
probs = torch.sigmoid(logits)
print(f"Logits range: [{logits.min():.2f}, {logits.max():.2f}]")
print(f"Probs range: [{probs.min():.4f}, {probs.max():.4f}]")

# Check 3: Loss computation
print("\n[CHECK 3] Loss Computation")
pred_logits = torch.randn(4, 128)
labels = torch.tensor([1., 0., 1., 0.]).unsqueeze(1).expand(-1, 128)
mask = torch.ones(4, 128)

bce_logits = torch.nn.BCEWithLogitsLoss(reduction='none')
loss = (bce_logits(pred_logits, labels) * mask).sum() / mask.sum()
print(f"Loss value: {loss.item():.4f}")
print(f"Expected: ~0.69 (random init)")

# Check 4: Prediction inversion test
print("\n[CHECK 4] Prediction Inversion")
print("If AUC < 50%, predictions might be inverted")
gts = np.array([0, 0, 0, 1, 1, 1])
preds_correct = np.array([0.2, 0.3, 0.4, 0.6, 0.7, 0.8])
preds_inverted = 1 - preds_correct

from sklearn.metrics import roc_auc_score
auc_correct = roc_auc_score(gts, preds_correct)
auc_inverted = roc_auc_score(gts, preds_inverted)

print(f"Correct predictions AUC: {auc_correct:.2%}")
print(f"Inverted predictions AUC: {auc_inverted:.2%}")
print("⚠️  If your AUC ≈ 40%, predictions are INVERTED")

# Check 5: Label encoding
print("\n[CHECK 5] Label Encoding")
print("Verify: Anomaly=1, Normal=0")
print("If swapped, model learns opposite")

# Check 6: Memory read logic
print("\n[CHECK 6] Memory Read Logic")
print("Potential issue: Reading from wrong memory bank")
print("- Anomaly samples should read Anomaly memory strongly")
print("- Normal samples should read Normal memory strongly")

print("\n" + "="*60)
print("LIKELY ROOT CAUSES (40% AUC):")
print("="*60)
print("""
1. 🔴 INVERTED PREDICTIONS: Model outputs are flipped
   - Fix: Check memory read logic in model.forward()
   - Or: Invert test predictions: preds = 1 - preds

2. 🔴 LABEL SWAP: Anomaly=0, Normal=1 in dataset
   - Fix: Check UCF_Train.list ordering

3. 🔴 WRONG MEMORY BANK: A_x reads N_memory, N_x reads A_memory
   - Fix: Swap memory reads in model

4. 🔴 FEATURE CONCATENATION: Wrong order in combined features
   - Fix: Check order in torch.cat([x, aug_features], dim=-1)

5. 🔴 GROUND TRUTH MISMATCH: GT file labels don't match data
   - Fix: Verify gt_ucf_v2.npy matches test set order
""")

print("\nRun this first, then check your training logs for:")
print("- Initial loss (should be ~0.69)")
print("- Are anomaly/normal counts balanced?")
print("- Does loss decrease over time?")
