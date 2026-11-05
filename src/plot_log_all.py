#!/usr/bin/env python3
"""Compare all loss components between two MAML training runs with step truncation.

Both curves are drawn only up to the minimum of the two run lengths.
"""

import glob, os
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# ================== Configuration ==================
LOG_DIR_A = "logs_xd_full"      # Reference run (e.g., full model)
LOG_DIR_B = "logs_xd_no_ortho"  # Ablated run (e.g., ortho removed)
OUTPUT_PREFIX = "loss_comp"     # prefix for output files

# ================== Publication styling ==================
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'legend.fontsize': 9,
    'font.family': 'serif',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'lines.linewidth': 1.1,
})

def extract_scalars(logdir):
    """Return dict {tag: (steps, values)} for all scalar events."""
    event_files = glob.glob(os.path.join(logdir, "tb_logs", "events.out.tfevents.*"))
    if not event_files:
        event_files = glob.glob(os.path.join(logdir, "events.out.tfevents.*"))
    if not event_files:
        raise FileNotFoundError(f"No event files found in {logdir}")
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

def truncate_to_common_steps(steps_a, vals_a, steps_b, vals_b):
    """Truncate both series to the minimum of their last steps."""
    if len(steps_a) == 0 or len(steps_b) == 0:
        return steps_a, vals_a, steps_b, vals_b
    max_step = min(steps_a[-1], steps_b[-1])
    mask_a = steps_a <= max_step
    mask_b = steps_b <= max_step
    return steps_a[mask_a], vals_a[mask_a], steps_b[mask_b], vals_b[mask_b]

# Load both logs
data_a = extract_scalars(LOG_DIR_A)
data_b = extract_scalars(LOG_DIR_B)

# ================== Loss components to plot ==================
loss_names = [
    "total_loss", "att_loss", "A_loss", "panomaly_loss",
    "N_loss", "A_Nloss", "triplet", "kl_loss",
    "ortho_loss", "sparsity_loss", "stubborn_noise", "contrastive_A_N"
]

# ================== 1. Individual component plots ==================
for name in loss_names:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, mode in zip(axes, ["support", "query"]):
        tag = f"{mode}/{name}"
        # Retrieve data for both logs if present
        series_a = None
        series_b = None
        if tag in data_a:
            series_a = data_a[tag]
        if tag in data_b:
            series_b = data_b[tag]
        # If only one log has the data, plot it alone (no truncation needed)
        if series_a is not None and series_b is not None:
            steps_a, vals_a, steps_b, vals_b = truncate_to_common_steps(
                series_a[0], series_a[1], series_b[0], series_b[1])
            ax.plot(steps_a, vals_a, color="#2166ac", linestyle="-",
                    label="Full model", alpha=0.8)
            ax.plot(steps_b, vals_b, color="#b2182b", linestyle="--",
                    label="Ablated", alpha=0.8)
        elif series_a is not None:
            ax.plot(series_a[0], series_a[1], color="#2166ac", linestyle="-",
                    label="Full model", alpha=0.8)
        elif series_b is not None:
            ax.plot(series_b[0], series_b[1], color="#b2182b", linestyle="--",
                    label="Ablated", alpha=0.8)
        ax.set_title(f"{mode} – {name}")
        ax.set_xlabel("Meta‑iteration")
        ax.set_ylabel(name)
        ax.legend(frameon=True, fancybox=True)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    fname = f"{OUTPUT_PREFIX}_{name}_comparison.pdf"
    plt.savefig(fname, bbox_inches='tight')
    plt.close()
    print(f"Saved {fname}")

# ================== 2. Total loss comparison (support + query) ==================
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, mode in zip(axes, ["support", "query"]):
    tag = f"{mode}/total_loss"
    series_a = data_a.get(tag, None)
    series_b = data_b.get(tag, None)
    if series_a is not None and series_b is not None:
        steps_a, vals_a, steps_b, vals_b = truncate_to_common_steps(
            series_a[0], series_a[1], series_b[0], series_b[1])
        ax.plot(steps_a, vals_a, color="#2166ac", linestyle="-",
                label="Full model")
        ax.plot(steps_b, vals_b, color="#b2182b", linestyle="--",
                label="Ablated")
    elif series_a is not None:
        ax.plot(series_a[0], series_a[1], color="#2166ac", linestyle="-",
                label="Full model")
    elif series_b is not None:
        ax.plot(series_b[0], series_b[1], color="#b2182b", linestyle="--",
                label="Ablated")
    ax.set_title(f"{mode} – total loss")
    ax.set_xlabel("Meta‑iteration")
    ax.set_ylabel("Total loss")
    ax.legend(frameon=True, fancybox=True)
    ax.grid(alpha=0.3)
plt.tight_layout()
fname_total = f"{OUTPUT_PREFIX}_total_loss_comparison.pdf"
plt.savefig(fname_total, bbox_inches='tight')
plt.close()
print(f"Saved {fname_total}")