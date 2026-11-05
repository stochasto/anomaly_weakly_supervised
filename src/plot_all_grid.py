#!/usr/bin/env python3
"""
2×6 panel overlay plots for all loss components – support & query.
Full model (blue) vs. ortho‑ablated (red).
Generates two PDFs: support_overlay.pdf and query_overlay.pdf
"""

import os, glob
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# ================== Configuration ==================
LOG_DIR_FULL = "/mnt/c/code/ae_anomaly/UR-DMU/logs_xd_rev2_all_losses"
LOG_DIR_ABL  = "/mnt/c/code/ae_anomaly/UR-DMU/logs_xd_rev2_ortho_zero"
SAVE_DIR = "."

# ================== Styling (larger, readable) ==================
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'font.family': 'serif',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'lines.linewidth': 1.2,
})

LOSS_NAMES = [
    "total_loss", "att_loss", "A_loss", "panomaly_loss",
    "N_loss", "A_Nloss", "triplet", "kl_loss",
    "ortho_loss", "sparsity_loss", "stubborn_noise", "contrastive_A_N"
]

def load_series(logdir, tag):
    event_files = glob.glob(os.path.join(logdir, "tb_logs", "events.out.tfevents.*"))
    if not event_files:
        event_files = glob.glob(os.path.join(logdir, "events.out.tfevents.*"))
    if not event_files:
        raise FileNotFoundError(f"No event files in {logdir}")
    event_file = max(event_files, key=os.path.getmtime)
    ea = EventAccumulator(event_file)
    ea.Reload()
    if tag not in ea.Tags()["scalars"]:
        return np.array([]), np.array([])
    events = ea.Scalars(tag)
    steps = np.array([e.step for e in events])
    values = np.array([e.value for e in events])
    return steps, values

def make_grid(data_full, data_abl, mode, out_pdf):
    """2x6 grid overlaying full vs ablated for a given mode (support/query)."""
    fig, axes = plt.subplots(2, 6, figsize=(20, 10), sharex=True)
    axes = axes.flatten()

    # Global max step (min of both runs across all tags)
    max_steps = []
    for name in LOSS_NAMES:
        tag = f"{mode}/{name}"
        s_full, _ = data_full.get(tag, (np.array([]), np.array([])))
        s_abl, _ = data_abl.get(tag, (np.array([]), np.array([])))
        if len(s_full) > 0: max_steps.append(s_full[-1])
        if len(s_abl) > 0: max_steps.append(s_abl[-1])
    global_max = min(max_steps) if max_steps else 0

    for i, name in enumerate(LOSS_NAMES):
        ax = axes[i]
        tag = f"{mode}/{name}"
        s_full, v_full = data_full.get(tag, (np.array([]), np.array([])))
        s_abl, v_abl   = data_abl.get(tag, (np.array([]), np.array([])))

        # Truncate to global_max
        if len(s_full) > 0:
            mask = s_full <= global_max
            s_full, v_full = s_full[mask], v_full[mask]
        if len(s_abl) > 0:
            mask = s_abl <= global_max
            s_abl, v_abl = s_abl[mask], v_abl[mask]

        if len(s_full) > 0:
            ax.plot(s_full, v_full, color='#2166ac', linewidth=1.2, label='Full')
        if len(s_abl) > 0:
            ax.plot(s_abl, v_abl, color='#d6604d', linewidth=1.2, label='Ablated')
        ax.set_title(name, fontsize=13)
        ax.grid(alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=10)
        if i >= 6:   # bottom row
            ax.set_xlabel("Meta‑iteration", fontsize=12)

    # Single legend at the top
    handles = [
        plt.Line2D([0], [0], color='#2166ac', lw=1.2, label='Full'),
        plt.Line2D([0], [0], color='#d6604d', lw=1.2, label='Without $\\mathcal{L}_{\\mathrm{ortho}}$')
    ]
    fig.legend(handles=handles, loc='upper center', ncol=2, fontsize=12, bbox_to_anchor=(0.5, 1.02))

    fig.suptitle(f"{mode.capitalize()} losses – Full vs. ortho‑ablated", fontsize=14, y=1.10)
    plt.tight_layout(pad=1.2, rect=[0, 0, 1, 0.95])
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_pdf}")

# Load all data once
print("Loading full model logs...")
data_full = {}
for name in LOSS_NAMES:
    for mode in ["support", "query"]:
        tag = f"{mode}/{name}"
        s, v = load_series(LOG_DIR_FULL, tag)
        if len(s) > 0:
            data_full[tag] = (s, v)

print("Loading ablated model logs...")
data_abl = {}
for name in LOSS_NAMES:
    for mode in ["support", "query"]:
        tag = f"{mode}/{name}"
        s, v = load_series(LOG_DIR_ABL, tag)
        if len(s) > 0:
            data_abl[tag] = (s, v)

# Generate both grids
make_grid(data_full, data_abl, "support", os.path.join(SAVE_DIR, "support_overlay.pdf"))
make_grid(data_full, data_abl, "query",   os.path.join(SAVE_DIR, "query_overlay.pdf"))