#!/usr/bin/env python3
"""Four 2x6 panel PDFs: support & query loss components for full model and ablation.

Each PDF shows all 12 loss components in a 2x6 grid.
Both runs are truncated to the minimum global step.
"""

import glob, os
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# ================== Configuration ==================
LOG_DIR_FULL   = "/mnt/c/code/ae_anomaly/UR-DMU/logs_xd_rev2_all_losses"      # full model run
LOG_DIR_ABL = "/mnt/c/code/ae_anomaly/UR-DMU/logs_xd_rev2_ortho_zero"  # orthogonality ablated run

# ================== Styling ==================
plt.rcParams.update({
    'font.size': 6,
    'axes.titlesize': 7,
    'axes.labelsize': 6,
    'font.family': 'serif',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'lines.linewidth': 0.6,
})

def extract_scalars(logdir):
    event_files = glob.glob(os.path.join(logdir, "tb_logs", "events.out.tfevents.*"))
    if not event_files:
        event_files = glob.glob(os.path.join(logdir, "events.out.tfevents.*"))
    if not event_files:
        raise FileNotFoundError(f"No event files in {logdir}")
    event_file = max(event_files, key=os.path.getmtime)
    print(f"Reading {event_file}")
    ea = EventAccumulator(event_file)
    ea.Reload()
    data = {}
    for tag in ea.Tags()["scalars"]:
        events = ea.Scalars(tag)
        data[tag] = (np.array([e.step for e in events]),
                     np.array([e.value for e in events]))
    return data

data_full = extract_scalars(LOG_DIR_FULL)
data_abl  = extract_scalars(LOG_DIR_ABL)

# Global max step (minimum of both runs)
def get_max_step(data):
    max_s = 0
    for (steps, _) in data.values():
        if len(steps) > 0 and steps[-1] > max_s:
            max_s = steps[-1]
    return max_s

max_step = min(get_max_step(data_full), get_max_step(data_abl))
print(f"Truncating to step {max_step}")

def truncate(steps, vals, limit):
    mask = steps <= limit
    return steps[mask], vals[mask]

loss_names = [
    "total_loss", "att_loss", "A_loss", "panomaly_loss",
    "N_loss", "A_Nloss", "triplet", "kl_loss",
    "ortho_loss", "sparsity_loss", "stubborn_noise", "contrastive_A_N"
]

def plot_grid(data_dict, mode, title, out_pdf):
    """2x6 grid of loss components for a given mode (support/query)."""
    fig, axes = plt.subplots(2, 6, figsize=(16, 5), sharex=True)
    axes = axes.flatten()
    for i, name in enumerate(loss_names):
        ax = axes[i]
        tag = f"{mode}/{name}"
        if tag in data_dict:
            steps, vals = data_dict[tag]
            steps, vals = truncate(steps, vals, max_step)
            ax.plot(steps, vals, color='#2166ac', linewidth=0.6)
        else:
            ax.text(0.5, 0.5, 'N/A', transform=ax.transAxes, ha='center', va='center')
        ax.set_title(name, fontsize=7)
        ax.grid(alpha=0.3)
        # Show xlabel only on bottom row
        if i >= 6:
            ax.set_xlabel("Meta‑iteration", fontsize=6)
    # Global title
    fig.suptitle(title, fontsize=9, y=1.02)
    plt.tight_layout(pad=0.5, rect=[0, 0, 1, 0.98])
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_pdf}")

# Generate four PDFs
plot_grid(data_full, "support", "Full model – support loss components", "support_full.pdf")
plot_grid(data_full, "query",   "Full model – query loss components",   "query_full.pdf")
plot_grid(data_abl,  "support", "Ablated model – support loss components", "support_abl.pdf")
plot_grid(data_abl,  "query",   "Ablated model – query loss components",   "query_abl.pdf")