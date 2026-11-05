#!/usr/bin/env python3
"""t‑SNE visualisation + quantitative analysis of dual‑memory separation."""

import os
import torch
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
import pandas as pd

from model_gemini_xd_v3 import WSAD

# ---- Paths to your checkpoints (adjust) ----
#MAML_CKPT   = "checkpoints/THE_XD_sk_8_qk_16_ilr_0.0005_mlr_1e-05_beta_1.0_segments_300_anums_200_nnums_200_ap_0.8040_auc_0.9360.pth"     # best MAML UCF
#NOMAML_CKPT = "results/no_maml_xd_sk_8_qk_16_ilr_0.0005_mlr_1e-05_beta_1.0_segments_300_anums_200_nnums_200_ap_0.7210_auc_0.8959.pth"            # No‑MAML full model best

MAML_CKPT   = "checkpoints/THEBESTSOFAR_ucf_sk_8_qk_16_ilr_0.001_mlr_0.0001_beta_1.0_segments_300_anums_200_nnums_200_ap_0.3212_auc_0.8610.pth"     # best MAML UCF
NOMAML_CKPT = "results/no_maml_ucf_sk_8_qk_16_ilr_0.001_mlr_0.0001_beta_1.0_segments_300_anums_200_nnums_200_ap_0.2832_auc_0.8217.pth"            # No‑MAML full model best

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def extract_memory_prototypes(ckpt_path):
    model = WSAD(input_size=1024, flag="Test", a_nums=200, n_nums=200)
    model.to(DEVICE)
    state = torch.load(ckpt_path, map_location=DEVICE)
    if 'model' in state:
        model.load_state_dict(state['model'])
    else:
        model.load_state_dict(state)
    model.eval()
    amem = model.Amemory.memory_block.data.cpu().numpy()
    nmem = model.Nmemory.memory_block.data.cpu().numpy()
    return amem, nmem

def compute_separation_metrics(amem, nmem):
    """Compute clustering metrics treating abnormal and normal as two clusters."""
    X = np.vstack([amem, nmem])
    labels = np.array([0]*200 + [1]*200)   # 0: abnormal, 1: normal
    sil = silhouette_score(X, labels)
    db = davies_bouldin_score(X, labels)
    return sil, db

def plot_and_save(maml_am, maml_nm, nomaml_am, nomaml_nm, save_path="results/tsne_memory_ucf.pdf"):
    # Combine all for t‑SNE
    data = np.vstack([maml_am, maml_nm, nomaml_am, nomaml_nm])
    labels_bank = (['Abnormal']*200 + ['Normal']*200) * 2
    labels_method = (['MAML']*400 + ['No‑MAML']*400)

    tsne = TSNE(n_components=2, random_state=42, perplexity=30, learning_rate='auto', init='pca')
    proj = tsne.fit_transform(data)
    
    # Save tabular data
    df = pd.DataFrame({
        'method': labels_method,
        'bank': labels_bank,
        'tsne_x': proj[:, 0],
        'tsne_y': proj[:, 1]
    })
    csv_path = save_path.replace('.pdf', '_data.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved t‑SNE data table to {csv_path}")

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

    # Compute metrics for each method separately
    sil_m, db_m = compute_separation_metrics(maml_am, maml_nm)
    sil_nm, db_nm = compute_separation_metrics(nomaml_am, nomaml_nm)

    # Print table
    print("\nQuantitative separation metrics (higher Silhouette = better, lower DB = better):")
    print("| Method      | Silhouette Score | Davies‑Bouldin Index |")
    print("|-------------|------------------|----------------------|")
    print(f"| MAML        | {sil_m:.4f}          | {db_m:.4f}             |")
    print(f"| No‑MAML     | {sil_nm:.4f}          | {db_nm:.4f}             |")
    
    # Save as text file for later inclusion
    with open("results/tsne_metrics_ucf.txt", "w") as f:
        f.write(f"MAML Silhouette: {sil_m:.4f}, DB Index: {db_m:.4f}\n")
        f.write(f"No‑MAML Silhouette: {sil_nm:.4f}, DB Index: {db_nm:.4f}\n")

if __name__ == "__main__":
    print("Loading MAML checkpoint...")
    mam_am, mam_nm = extract_memory_prototypes(MAML_CKPT)
    print("Loading No‑MAML checkpoint...")
    nomam_am, nomam_nm = extract_memory_prototypes(NOMAML_CKPT)
    plot_and_save(mam_am, mam_nm, nomam_am, nomam_nm)