#!/usr/bin/env python3
"""t‑SNE visualisation of dual‑memory prototypes: MAML vs. No‑MAML."""

import os
import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Use the same model class as your training
from model_gemini_xd_v3 import WSAD

# ---- Paths to your checkpoints (adjust these) ----
MAML_CKPT   = "checkpoints/THEBESTSOFAR_ucf_sk_8_qk_16_ilr_0.001_mlr_0.0001_beta_1.0_segments_300_anums_200_nnums_200_ap_0.3212_auc_0.8610.pth"     # best MAML UCF
NOMAML_CKPT = "results/no_maml_ucf_sk_8_qk_16_ilr_0.001_mlr_0.0001_beta_1.0_segments_300_anums_200_nnums_200_ap_0.2854_auc_0.8496.pth"            # No‑MAML full model best

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def extract_memory_prototypes(ckpt_path):
    """Load model, restore checkpoint, return (amem, nmem) each (200, 512)."""
    model = WSAD(input_size=1024, flag="Test", a_nums=200, n_nums=200)
    model.to(DEVICE)
    state = torch.load(ckpt_path, map_location=DEVICE)
    # Checkpoints saved by your script have a 'model' key
    if 'model' in state:
        model.load_state_dict(state['model'])
    else:
        model.load_state_dict(state)
    model.eval()
    # Memory blocks are nn.Parameters, .data gives the tensor
    amem = model.Amemory.memory_block.data.cpu().numpy()   # (200, 512)
    nmem = model.Nmemory.memory_block.data.cpu().numpy()   # (200, 512)
    return amem, nmem

def plot_tsne(maml_am, maml_nm, nomaml_am, nomaml_nm, save_path="results/tsne_memory_ucf.pdf"):
    """Combine all prototypes, run t‑SNE, and plot."""
    # Stack with labels
    data = np.vstack([maml_am, maml_nm, nomaml_am, nomaml_nm])
    labels_bank = (['Abnormal']*200 + ['Normal']*200) * 2
    labels_method = (['MAML']*400 + ['No‑MAML']*400)

    # t‑SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, learning_rate='auto', init='pca')
    proj = tsne.fit_transform(data)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, method in zip(axes, ['MAML', 'No‑MAML']):
        mask = np.array(labels_method) == method
        for bank, color in [('Abnormal', 'red'), ('Normal', 'blue')]:
            idx = mask & (np.array(labels_bank) == bank)
            ax.scatter(proj[idx, 0], proj[idx, 1], c=color, label=bank, alpha=0.6, s=25)
        ax.set_title(f"{method} – memory prototypes")
        ax.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved t‑SNE plot to {save_path}")

if __name__ == "__main__":
    print("Loading MAML checkpoint...")
    mam_am, mam_nm = extract_memory_prototypes(MAML_CKPT)
    print("Loading No‑MAML checkpoint...")
    nomam_am, nomam_nm = extract_memory_prototypes(NOMAML_CKPT)
    plot_tsne(mam_am, mam_nm, nomam_am, nomam_nm)