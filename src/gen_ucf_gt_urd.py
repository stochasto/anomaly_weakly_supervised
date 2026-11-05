#!/usr/bin/env python3
"""
Generate clip-level ground truth for UCF-Crime (single-crop setup)

- Uses only first crop (ignores __1, __2, ...)
- Converts frame-level annotations → clip-level (16 frames = 1 clip)
- Handles up to 2 anomaly segments per video
- Drops leftover frames (matches feature extraction behavior)

Output:
    frame_label/gt_ucf_single_crop.npy
"""

import os
import numpy as np

# -------------------------
# CONFIG
# -------------------------
ANNOTATION_FILE = "Temporal_Anomaly_Annotation.txt"
TEST_LIST_FILE = "list/UCF_Test.list"
OUTPUT_GT_FILE = "frame_label/gt_ucf_single_crop.npy"

FRAMES_PER_CLIP = 16


# -------------------------
# Parse annotation file
# -------------------------
def parse_annotation_file(annotation_file):
    """
    Returns:
        dict: video_name -> list of (start_frame, end_frame)
    """
    gt_dict = {}

    with open(annotation_file, "r") as f:
        for line in f:
            parts = line.strip().split()

            if len(parts) < 4:
                continue  # skip malformed lines

            vid_name = parts[0].replace(".mp4", "")
            nums = list(map(int, parts[2:]))

            segments = []

            # First anomaly
            if nums[0] != -1:
                segments.append((nums[0], nums[1]))

            # Second anomaly (if exists)
            if len(nums) >= 4 and nums[2] != -1:
                segments.append((nums[2], nums[3]))

            gt_dict[vid_name] = segments

    return gt_dict


# -------------------------
# Build GT
# -------------------------
def build_gt():
    gt_dict = parse_annotation_file(ANNOTATION_FILE)

    gt_all = []
    video_count = 0

    with open(TEST_LIST_FILE, "r") as f:
        for line in f:
            path = line.strip()

            # ✅ Skip extra crops
            if "__" in path:
                continue

            if not os.path.exists(path):
                print(f"[WARN] Missing file: {path}")
                continue

            # Extract video name
            base = os.path.basename(path).replace(".npy", "")

            if base not in gt_dict:
                print(f"[WARN] {base} not in annotation file. Skipping.")
                continue

            segments = gt_dict[base]

            # Load feature to determine clip count
            feat = np.load(path)
            if feat.ndim != 2:
                raise ValueError(f"[ERROR] Unexpected feature shape for {base}: {feat.shape}")

            num_clips = feat.shape[0]

            # Initialize GT
            gt = np.zeros(num_clips, dtype=np.float32)

            # Fill anomaly regions
            for (start_f, end_f) in segments:
                if start_f < 0 or end_f < 0:
                    continue

                start_clip = start_f // FRAMES_PER_CLIP
                end_clip = (end_f - 1) // FRAMES_PER_CLIP  # ✅ better boundary handling

                start_clip = max(0, start_clip)
                end_clip = min(num_clips - 1, end_clip)

                if start_clip <= end_clip:
                    gt[start_clip:end_clip + 1] = 1

            gt_all.append(gt)
            video_count += 1

    if len(gt_all) == 0:
        raise RuntimeError("[ERROR] No valid videos processed. Check paths and annotation file.")

    gt_all = np.concatenate(gt_all)

    print(f"[INFO] Processed videos: {video_count}")
    print(f"[INFO] Total GT length (clips): {len(gt_all)}")

    os.makedirs(os.path.dirname(OUTPUT_GT_FILE), exist_ok=True)
    np.save(OUTPUT_GT_FILE, gt_all)

    print(f"[INFO] Saved GT to: {OUTPUT_GT_FILE}")


# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    build_gt()