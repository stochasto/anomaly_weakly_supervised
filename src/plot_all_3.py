    #!/usr/bin/env python3
"""Plot support and query losses for full vs. ortho‑ablated runs (two panels)."""

import os, glob
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

LOG_DIR_FULL = "/mnt/c/code/ae_anomaly/UR-DMU/logs_xd_rev2_all_losses"
LOG_DIR_ABL  = "/mnt/c/code/ae_anomaly/UR-DMU/logs_xd_rev2_ortho_zero"

plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'lines.linewidth': 1.0,
})

def load_series(logdir, tag):
    event_files = glob.glob(os.path.join(logdir, "tb_logs", "events.out.tfevents.*"))
    if not event_files:
        event_files = glob.glob(os.path.join(logdir, "events.out.tfevents.*"))
    event_file = max(event_files, key=os.path.getmtime)
    ea = EventAccumulator(event_file)
    ea.Reload()
    events = ea.Scalars(tag)
    steps = np.array([e.step for e in events])
    values = np.array([e.value for e in events])
    return steps, values

# Load data
s_full_s, v_full_s = load_series(LOG_DIR_FULL, "support/total_loss")
s_full_q, v_full_q = load_series(LOG_DIR_FULL, "query/total_loss")
s_abl_s, v_abl_s = load_series(LOG_DIR_ABL, "support/total_loss")
s_abl_q, v_abl_q = load_series(LOG_DIR_ABL, "query/total_loss")

# Align support and query steps for each run (they might differ)
def align_pair(steps1, vals1, steps2, vals2):
    common = np.intersect1d(steps1, steps2)
    mask1 = np.isin(steps1, common)
    mask2 = np.isin(steps2, common)
    return steps1[mask1], vals1[mask1], vals2[mask2]

# We want the same x-axis for both curves in each panel, so align full support and query, and separately ablated
steps_full, v_full_s, v_full_q = align_pair(s_full_s, v_full_s, s_full_q, v_full_q)
steps_abl,  v_abl_s,  v_abl_q  = align_pair(s_abl_s, v_abl_s, s_abl_q, v_abl_q)

# Truncate to same length
min_len = min(len(steps_full), len(steps_abl))
steps = steps_full[:min_len]
v_full_s = v_full_s[:min_len]
v_full_q = v_full_q[:min_len]
v_abl_s = v_abl_s[:min_len]
v_abl_q = v_abl_q[:min_len]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(steps, v_full_s, label='Full model', color='#2166ac')
ax1.plot(steps, v_abl_s, label='Without $\\mathcal{L}_{\\mathrm{ortho}}$', color='#d6604d')
ax1.set_xlabel('Meta‑iteration')
ax1.set_ylabel('Support loss')
ax1.legend()
ax1.grid(alpha=0.3)

ax2.plot(steps, v_full_q, label='Full model (AUC 93.60%)', color='#2166ac')
ax2.plot(steps, v_abl_q, label='Without $\\mathcal{L}_{\\mathrm{ortho}}$ (AUC 92.21%)', color='#d6604d')
ax2.set_xlabel('Meta‑iteration')
ax2.set_ylabel('Query loss')
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('fig_loss_dynamics_two_panel.pdf', bbox_inches='tight')
print("Saved fig_loss_dynamics_two_panel.pdf")