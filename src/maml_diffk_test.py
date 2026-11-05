import os
import random
import numpy as np
import torch
import torch.utils.data as data
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.metrics import roc_curve, auc, precision_recall_curve

from model_min_upgrade import WSAD
#from model import WSAD

from maml_diffk import UCF_crime


# ============================================================
# ====================== USER ARGS ============================
# ============================================================

# Reproducibility
SEED = 42

# Paths
ROOT_DIR = "/mnt/c/code/data/I3D"
GT_FILE = "frame_label/gt_ucf_v2.npy"
MODEL_FILE = "checkpoints/best_meta_wsad_sk_8_qk_16_ilr_0.001_mlr_0.0001_beta_1.0_auc_0.8332629847488588.pth"
SAVE_DIR = "test_results_temp_with_labels"

# Dataset parameters
NUM_SEGMENTS = 400
LEN_FEATURE = 1024
NUM_WORKERS = 4

# Model parameters
A_NUMS = 200
N_NUMS = 200

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# ============================================================


@torch.no_grad()
def test_and_plot():
    # --------------------------------------------------
    # Seed
    # --------------------------------------------------
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    os.makedirs(SAVE_DIR, exist_ok=True)
    plot_dir = os.path.join(SAVE_DIR, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    range_dir = os.path.join(SAVE_DIR, "video_ranges")
    os.makedirs(range_dir, exist_ok=True)

    # --------------------------------------------------
    # Dataset
    # --------------------------------------------------
    print("[INFO] Loading test dataset...")
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
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    # --------------------------------------------------
    # Model
    # --------------------------------------------------
    print("[INFO] Loading model...")
    model = WSAD(
        input_size=LEN_FEATURE,
        flag="Test",
        a_nums=A_NUMS,
        n_nums=N_NUMS
    ).to(DEVICE)

    model.load_state_dict(torch.load(MODEL_FILE, map_location=DEVICE))
    model.eval()

    # --------------------------------------------------
    # Ground truth
    # --------------------------------------------------
    frame_gt = np.load(GT_FILE)

    # --------------------------------------------------
    # Containers
    # --------------------------------------------------
    all_frame_scores = []
    video_scores = []
    video_labels = []
    video_ranges = []

    temp_scores = []
    global_frame_ptr = 0
    video_idx = 0

    # --------------------------------------------------
    # Inference loop
    # --------------------------------------------------
    print("[INFO] Running inference...")
    for idx, (x, label) in enumerate(tqdm(test_loader, ncols=100)):
        x = x.to(DEVICE)

        out = model(x)
        scores = out["frame"].squeeze(0).cpu().numpy()  # [T]

        temp_scores.append(scores)

        # UCF-Crime protocol: 10 clips = 1 video
        if (idx + 1) % 10 == 0:
            video_score = np.mean(temp_scores, axis=0)
            T = len(video_score)
            
            start_frame = global_frame_ptr
            end_frame = global_frame_ptr + T - 1

            all_frame_scores.append(video_score)

            video_scores.append(video_score.max())
            video_labels.append(int(label.item()))
            video_ranges.append((start_frame, end_frame))

            # --------------------------------------------------
            # Save per-video range info
            # --------------------------------------------------
            range_file = os.path.join(range_dir, f"video_{video_idx:04d}.txt")
            with open(range_file, "w") as f:
                f.write(f"video_index: {video_idx}\n")
                f.write(f"label: {int(label.item())}\n")
                f.write(f"start_frame: {start_frame}\n")
                f.write(f"end_frame: {end_frame}\n")
                f.write(f"num_frames: {T}\n")
                f.write(f"max_score: {video_score.max():.6f}\n")

            global_frame_ptr += T
            temp_scores = []
            video_idx += 1

    # --------------------------------------------------
    # Frame-level aggregation
    # --------------------------------------------------
    frame_scores = np.concatenate(all_frame_scores, axis=0)
    frame_gt = frame_gt[: len(frame_scores)]

    # --------------------------------------------------
    # Metrics
    # --------------------------------------------------
    fpr, tpr, _ = roc_curve(frame_gt, frame_scores)
    auc_score = auc(fpr, tpr)

    precision, recall, _ = precision_recall_curve(frame_gt, frame_scores)
    ap_score = auc(recall, precision)

    video_preds = (np.array(video_scores) > 0.5).astype(int)
    accuracy = (video_preds == np.array(video_labels)).mean()

    print("\n========== RESULTS ==========")
    print(f"Frame-level AUC : {auc_score:.4f}")
    print(f"Frame-level AP  : {ap_score:.4f}")
    print(f"Video Accuracy  : {accuracy:.4f}")
    print("============================\n")

    # --------------------------------------------------
    # Save raw outputs
    # --------------------------------------------------
    np.save(os.path.join(SAVE_DIR, "frame_scores.npy"), frame_scores)
    np.save(os.path.join(SAVE_DIR, "frame_gt.npy"), frame_gt)
    np.save(os.path.join(SAVE_DIR, "video_scores.npy"), np.array(video_scores))
    np.save(os.path.join(SAVE_DIR, "video_labels.npy"), np.array(video_labels))
    np.save(os.path.join(SAVE_DIR, "video_ranges.npy"), np.array(video_ranges))


    # --------------------------------------------------
    # Plot: frame score vs ground truth
    # --------------------------------------------------
    plt.figure(figsize=(16, 4))
    plt.plot(frame_scores, label="Predicted Anomaly Score", linewidth=2)
    plt.plot(frame_gt, label="Ground Truth", linestyle="--", alpha=0.8)
    plt.xlabel("Frame Index")
    plt.ylabel("Anomaly Score")
    plt.title("Frame-level Anomaly Score vs Ground Truth (UCF-Crime)")
    plt.legend()
    plt.tight_layout()

    plot_path = os.path.join(plot_dir, "frame_scores_vs_gt_all.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"[INFO] Frame-level plot saved to {plot_path}")
    print(f"[INFO] Saved per-video ranges to {range_dir}")


if __name__ == "__main__":
    test_and_plot()
