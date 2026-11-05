#!/usr/bin/env python3
"""
Generate ground truth (GT) for UCF-Crime I3D 10-crop test features.
---------------------------------------------------------------
- Uses annotation file (e.g., Temporal_Anomaly_Annotation.txt)
- Uses list file (e.g., test_rgb_list.txt) for correct ordering
- Matches I3D crops: 
    Explosion010_x264.npy, Explosion010_x264__1.npy, ..., Explosion010_x264__9.npy
  (note that the first crop has no "__0" suffix)
- Accounts for 16 RGB frames per I3D feature
- Each GT vector corresponds to T clips per crop file
- Normal videos → all zeros
- Abnormal → 1 in anomaly intervals (converted to I3D clip indices)
"""

import os
import numpy as np
from tqdm import tqdm


def parse_annotation_file(annotation_file):
    """Parse UCF-Crime annotation file into a dict."""
    annotations = {}
    with open(annotation_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue

            video_name = parts[0]  # e.g., "Explosion010_x264.mp4"
            label = parts[1]       # e.g., "Explosion" or "Normal"
            frame_values = [int(x) for x in parts[2:] if x != "-1"]

            if label.lower() == "normal":
                annotations[video_name] = []
            else:
                # Each pair (start, end) defines an anomaly
                anomalies = []
                for i in range(0, len(frame_values), 2):
                    if i + 1 < len(frame_values):
                        anomalies.append((frame_values[i], frame_values[i + 1]))
                annotations[video_name] = anomalies
    return annotations


def normalize_video_name(path):
    """Convert a feature path to corresponding mp4 name in annotations."""
    base = os.path.basename(path)
    if "__" in base:
        base = base.split("__")[0] + ".mp4"
    else:
        base = base.replace(".npy", ".mp4")
    return base


def generate_clip_gt(video_name, num_clips, annotations, frames_per_clip=16):
    """
    Generate clip-level ground truth (0/1) for a given video.
    """
    gt = np.zeros(num_clips, dtype=np.int32)
    anomaly_intervals = annotations.get(video_name, [])

    for (start_f, end_f) in anomaly_intervals:
        start_clip = start_f // frames_per_clip
        end_clip = end_f // frames_per_clip
        gt[start_clip:end_clip + 1] = 1
    return gt


def generate_gt_from_list(list_file, annotation_file, save_path="gt-ucf.npy"):
    """
    Generate GT aligned with list file order and save as .npy
    """
    annotations = parse_annotation_file(annotation_file)

    with open(list_file, "r") as f:
        paths = [line.strip() for line in f if line.strip()]

    all_gt = []
    prev_video = None
    video_clips = []
    video_name = None

    for (i,path) in enumerate(paths):
        if ((i%10) == 0):
            video_name = normalize_video_name(path)
            # Load one crop to get clip length (T)
            if not os.path.exists(path):
                print(f"[WARN] Missing file: {path}")
                continue
            try:
                arr = np.load(path, mmap_mode="r")
            except Exception as e:
                print(f"[ERROR] Cannot load {path}: {e}")
                continue
            
            num_clips = arr.shape[0]
            arr = None  # free memory
            gt = generate_clip_gt(video_name, num_clips, annotations)
            all_gt.append(gt)
            
            if not all_gt:
                raise ValueError("No GT arrays were generated! Check file paths or annotations.")
            
    full_gt = np.concatenate(all_gt, axis=0)
    np.save(save_path, full_gt)
    print(f"\n✅ Saved GT: {save_path} | shape={full_gt.shape}")
            
    # Sanity check for normal videos
    '''for video, intervals in annotations.items():
        if not intervals:  # normal video
            mask = [p for p in paths if normalize_video_name(p) == video]
            if mask:
                # pick one for verification
                gt_sample = generate_clip_gt(video, np.load(mask[0]).shape[0], annotations)
                if np.any(gt_sample != 0):
                    raise ValueError(f"Normal video '{video}' has non-zero GT!")
                print("✅ All normal videos confirmed 0-labeled.")'''


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate UCF-Crime GT matching I3D 10-crop features")
    parser.add_argument("--list_file", type=str, required=True, help="Path to test list file")
    parser.add_argument("--annotation_file", type=str, required=True, help="Path to Temporal_Anomaly_Annotation.txt")
    parser.add_argument("--save_path", type=str, default="gt_ucf_v2.npy", help="Output npy file")
    args = parser.parse_args()

    generate_gt_from_list(args.list_file, args.annotation_file, args.save_path)
