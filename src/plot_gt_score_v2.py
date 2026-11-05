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
from maml_resume_gem import UCF_crime

# ========================== USER SETTINGS ==========================
SEED = 42
ROOT_DIR = "/mnt/c/code/data/I3D"
GT_FILE = "frame_label/gt-ucf_urd.npy"      # UR‑DMU frame‑level GT
MODEL_FILE = "checkpoints/THEBESTSOFAR_ucf_sk_8_qk_16_ilr_0.001_mlr_0.0001_beta_1.0_segments_300_anums_200_nnums_200_ap_0.3212_auc_0.8610.pth"
SAVE_DIR = "test_results_interactive_urdmu_style_v2"
NUM_SEGMENTS = 300
LEN_FEATURE = 1024
NUM_WORKERS = 4
A_NUMS = 200
N_NUMS = 200
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ===================================================================

@torch.no_grad()
def test_and_interactive_plot_urdmu_style():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    os.makedirs(SAVE_DIR, exist_ok=True)

    # ---------- Dataset ----------
    test_dataset = UCF_crime(
        root_dir=ROOT_DIR,
        modal="rgb",
        mode="Test",
        num_segments=NUM_SEGMENTS,
        len_feature=LEN_FEATURE
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

    # ---------- Inference (UR‑DMU style) ----------
    load_iter = iter(test_loader)
    frame_predict = None
    cls_label = []
    cls_pre = []
    temp_predict = torch.zeros((0)).cuda()
    video_boundaries = []       # (start_frame, end_frame, label, video_index)
    count = 0
    num_crops = 10
    video_idx = 0

    for i in range(len(test_loader.dataset)):
        _data, _label = next(load_iter)
        _data = _data.cuda()
        _label = _label.cuda()

        res = model(_data)
        a_predict = res["frame"]              # (1, T)
        temp_predict = torch.cat([temp_predict, a_predict], dim=0)

        if (i + 1) % num_crops == 0:
            cls_label.append(int(_label))
            avg_predict = temp_predict.mean(0).cpu().numpy()   # (T,)
            T_snip = len(avg_predict)
            fpre_ = np.repeat(avg_predict, 16)
            pl = len(fpre_)

            # Record video boundary (frame‑level)
            video_boundaries.append((count, count + pl - 1, int(_label), video_idx))

            # Video‑level prediction
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

    # ---------- Save scores and per‑video metadata (NEW) ----------
    # 1. Full frame‑level predictions (same as what the plotting script expects)
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

    for start, end, lbl, vid in video_boundaries:
        color = 'rgba(255,0,0,0.08)' if lbl == 1 else 'rgba(0,255,0,0.08)'
        fig.add_vrect(
            x0=start, x1=end,
            fillcolor=color, line_width=0,
            annotation_text=f"V{vid}", annotation_position="top left",
            layer="below"
        )

    fig.update_layout(
        title="Interactive Anomaly Score vs Ground Truth (UCF-Crime) – UR-DMU evaluation<br>"
              "Green = Normal, Red = Abnormal. Zoom/Pan with mouse.",
        xaxis_title="Frame Index",
        yaxis_title="Anomaly Score",
        hovermode='x unified',
        template='plotly_white',
        width=1400,
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )

    html_path = os.path.join(SAVE_DIR, "interactive_plot_urdmu_style.html")
    fig.write_html(html_path)
    print(f"Interactive plot saved to {html_path}")

    # Save per‑video boundaries array (optional)
    np.save(os.path.join(SAVE_DIR, "video_boundaries.npy"), np.array(video_boundaries, dtype=object))

if __name__ == "__main__":
    test_and_interactive_plot_urdmu_style()