#!/usr/bin/env python3
"""Comparative loss curves: full model vs. ablated ortho on XD-Violence."""

import glob, os
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def extract_scalars(logdir):
    """Return dict {tag: (steps, values)} for all scalar events."""
    event_files = glob.glob(os.path.join(logdir, "tb_logs", "events.out.tfevents.*"))
    if not event_files:
        event_files = glob.glob(os.path.join(logdir, "events.out.tfevents.*"))
    event_file = max(event_files, key=os.path.getmtime)
    ea = EventAccumulator(event_file)
    ea.Reload()
    data = {}
    for tag in ea.Tags()["scalars"]:
        events = ea.Scalars(tag)
        data[tag] = (np.array([e.step for e in events]),
                     np.array([e.value for e in events]))
    return data

# ---- Paths to the two log directories (adjust) ----
LOG_FULL   = "/mnt/c/code/ae_anomaly/UR-DMU/logs_xd_rev2_all_losses"      # full model run
LOG_ABLATE = "/mnt/c/code/ae_anomaly/UR-DMU/logs_xd_rev2_ortho_zero"  # orthogonality ablated run

data_full   = extract_scalars(LOG_FULL)
data_ablate = extract_scalars(LOG_ABLATE)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Left: ortho_loss vs step
ax = axes[0]
for label, data, color in [("Full model", data_full, "blue"),
                           ("No ortho", data_ablate, "red")]:
    if "support/ortho_loss" in data:
        steps, vals = data["support/ortho_loss"]
        ax.plot(steps, vals, color=color, label=label, alpha=0.8)
ax.set_xlabel("Meta‑iteration")
ax.set_ylabel("Orthogonality loss")
ax.set_title("Orthogonality loss evolution")
ax.legend()
ax.grid(alpha=0.3)

# Right: AUC vs step (if you logged AUC during training; else use step‑wise checkpoints)
# For this example we assume AUC was logged as "eval/auc"
ax = axes[1]
for label, data, color in [("Full model", data_full, "blue"),
                           ("No ortho", data_ablate, "red")]:
    if "eval/auc" in data:
        steps, aucs = data["eval/auc"]
        ax.plot(steps, aucs, color=color, label=label, alpha=0.8)
ax.set_xlabel("Meta‑iteration")
ax.set_ylabel("Frame‑level AUC")
ax.set_title("Validation AUC")
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("loss_comparison_xd.pdf")
print("Saved loss_comparison_xd.pdf")