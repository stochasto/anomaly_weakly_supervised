import os
import random
import numpy as np
import torch
import torch.utils.data as data
import plotly.graph_objects as go
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc, precision_recall_curve

# ---------- adjust these imports to your actual files ----------
from model_gemini_xd_v3 import WSAD
from maml_xd import XDVideoDataset               # your XD dataset class

# ========================== USER SETTINGS ==========================
SEED = 42
ROOT_DIR = "/mnt/c/code/ae_anomaly/UR-DMU/xdv"   # XD feature base directory
GT_FILE = "frame_label/xd_frame_gt_generated.npy" # your generated frame‑level GT
MODEL_FILE = "checkpoints/THE_XD_sk_8_qk_16_ilr_0.0005_mlr_1e-05_beta_1.0_segments_300_anums_200_nnums_200_ap_0.8040_auc_0.9360.pth"  # best XD checkpoint
SAVE_DIR = "test_results_interactive_urdmu_style_xd"
NUM_SEGMENTS = 300       # must match the checkpoint (300)
LEN_FEATURE = 1024
NUM_WORKERS = 4
A_NUMS = 200
N_NUMS = 200
NUM_CROPS = 5            # XD uses 5 crops per video
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ===================================================================

@torch.no_grad()
def test_and_interactive_plot_xd():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    os.makedirs(SAVE_DIR, exist_ok=True)

    # ---------- Dataset ----------
    test_dataset = XDVideoDataset(
        root_dir=ROOT_DIR,
        mode='Test',
        modal='rgb',          # 'rgb' or whatever your XDVideoDataset expects
        num_segments=NUM_SEGMENTS,
        len_feature=LEN_FEATURE,
        seed=SEED
    )
    test_loader = data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    # ---------- Model ----------
    model = WSAD(
        input_size=LEN_FEATURE,
        flag="Test",
        a_nums=A_NUMS,
        n_nums=N_NUMS
    ).to(DEVICE)

    checkpoint = torch.load(MODEL_FILE, map_location=DEVICE)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    # ---------- Ground truth ----------
    frame_gt = np.load(GT_FILE)
    print('len gt', len(frame_gt))

    # ---------- Inference (XD style: 5 crops -> average -> 16x repeat) ----------
    load_iter = iter(test_loader)
    frame_predict = None
    cls_label = []
    cls_pre = []
    temp_predict = torch.zeros((0)).cuda()
    video_boundaries = []       # (start_frame, end_frame, label, video_index)
    count = 0
    video_idx = 0

    for i in range(len(test_loader.dataset)):
        _data, _label = next(load_iter)
        _data = _data.cuda()
        _label = _label.cuda()

        res = model(_data)
        a_predict = res["frame"]              # (1, T) logits
        temp_predict = torch.cat([temp_predict, a_predict], dim=0)

        if (i + 1) % NUM_CROPS == 0:
            cls_label.append(int(_label))
            # Average over the 5 crops
            avg_predict = temp_predict.mean(0).cpu().numpy()   # (T,)
            T_snip = len(avg_predict)
            # Expand to frame‑level: repeat each snippet score 16 times
            fpre_ = np.repeat(avg_predict, 16)
            pl = len(fpre_)

            # Record video boundary (frame‑level)
            video_boundaries.append((count, count + pl - 1, int(_label), video_idx))

            # Video‑level prediction (threshold 0.5)
            cls_pre.append(1 if fpre_.max() > 0.5 else 0)

            if frame_predict is None:
                frame_predict = fpre_
            else:
                frame_predict = np.concatenate([frame_predict, fpre_])

            count += pl
            temp_predict = torch.zeros((0)).cuda()
            video_idx += 1

    # Ensure length match
    if len(frame_gt) != len(frame_predict):
        min_len = min(len(frame_gt), len(frame_predict))
        frame_gt = frame_gt[:min_len]
        frame_predict = frame_predict[:min_len]

    # ---------- Metrics ----------
    fpr, tpr, _ = roc_curve(frame_gt, frame_predict)
    auc_score = auc(fpr, tpr)
    correct_num = np.sum(np.array(cls_label) == np.array(cls_pre))
    accuracy = correct_num / len(cls_pre)
    precision, recall, _ = precision_recall_curve(frame_gt, frame_predict)
    ap_score = auc(recall, precision)

    print(f"AUC: {auc_score:.4f}  AP: {ap_score:.4f}  Acc: {accuracy:.4f}")

    # ---------- Save scores and per‑video metadata ----------
    # 1. Full frame‑level predictions
    np.save(os.path.join(SAVE_DIR, "frame_scores.npy"), frame_predict)

    # 2. Per‑video range text files
    range_dir = os.path.join(SAVE_DIR, "video_ranges")
    os.makedirs(range_dir, exist_ok=True)
    for start, end, lbl, vid in video_boundaries:
        file_path = os.path.join(range_dir, f"video_{vid:04d}.txt")
        with open(file_path, "w") as f:
            f.write(f"video_index: {vid}\n")
            f.write(f"label: {lbl}\n")
            f.write(f"start_frame: {start}\n")
            f.write(f"end_frame: {end}\n")
            f.write(f"num_frames: {end - start + 1}\n")

    # ---------- Interactive Plot ----------
    frames = np.arange(len(frame_predict))
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=frames,
        y=frame_predict,
        mode='lines',
        name='Predicted Score',
        line=dict(color='orange', width=2),
        hovertemplate='Frame %{x}<br>Score: %{y:.4f}<extra></extra>'
    ))

    fig.add_trace(go.Scatter(
        x=frames,
        y=frame_gt,
        mode='lines',
        name='Ground Truth',
        line=dict(color='blue', dash='dot', width=1.5),
        hovertemplate='Frame %{x}<br>GT: %{y}<extra></extra>'
    ))

    # Video background rectangles
    for start, end, lbl, vid in video_boundaries:
        color = 'rgba(255,0,0,0.08)' if lbl == 1 else 'rgba(0,255,0,0.08)'
        fig.add_vrect(
            x0=start, x1=end,
            fillcolor=color, line_width=0,
            annotation_text=f"V{vid}", annotation_position="top left",
            layer="below"
        )

    fig.update_layout(
        title="Interactive Anomaly Score vs Ground Truth (XD-Violence) – 5‑crop + 16× repeat<br>"
              "Green = Normal, Red = Abnormal. Zoom/Pan with mouse.",
        xaxis_title="Frame Index",
        yaxis_title="Anomaly Score",
        hovermode='x unified',
        template='plotly_white',
        width=1400,
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )

    html_path = os.path.join(SAVE_DIR, "interactive_plot_xd.html")
    fig.write_html(html_path)
    print(f"Interactive plot saved to {html_path}")

    # Save per‑video boundaries array (optional)
    np.save(os.path.join(SAVE_DIR, "video_boundaries.npy"), np.array(video_boundaries, dtype=object))

if __name__ == "__main__":
    test_and_interactive_plot_xd()