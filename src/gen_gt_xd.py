import os
import re
import numpy as np
import math
import argparse

def parse_annotations(ann_file):
    """
    Returns dict: {video_id: [(start_frame, end_frame), ...]}
    video_id is the first column without any extension.
    """
    ann = {}
    with open(ann_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            raw_name = parts[0]
            # Remove .mp4 extension if present
            vid_id = raw_name.replace('.mp4', '')
            # Parse frame windows (pairs of ints)
            windows = []
            for i in range(1, len(parts) - 1, 2):
                try:
                    s = int(parts[i])
                    e = int(parts[i+1])
                    windows.append((s, e))
                except (ValueError, IndexError):
                    break
            if windows:
                ann[vid_id] = windows
    return ann

def match_annotation_to_feature(vid_id, feature_names):
    """
    Try to find the feature .npy filename that corresponds to the annotation video_id.
    Handles common variations: with/without extension, 'v=' prefix, etc.
    """
    # Exact match
    for fn in feature_names:
        base = fn.replace('.npy', '')
        if base == vid_id:
            return fn

    # Remove 'v=' prefix in annotation
    clean_id = vid_id.replace('v=', '')
    for fn in feature_names:
        base = fn.replace('.npy', '')
        if base == clean_id:
            return fn
        # Also try matching the end of the filename
        if base.endswith(clean_id) or clean_id.endswith(base):
            return fn

    # Fuzzy match: check if the core ID appears in the filename
    for fn in feature_names:
        base = fn.replace('.npy', '')
        # Remove common prefixes/suffixes
        if clean_id in base or base in clean_id:
            return fn

    return None

def resample_mask(windows, T_snippets, target_len):
    """
    windows: list of (start_frame, end_frame) in original video frames
    T_snippets: number of I3D snippets (16 frames each)
    target_len: desired number of segments (e.g., 200)
    Returns: binary mask of shape (target_len,)
    """
    mask_snippets = np.zeros(T_snippets, dtype=np.int32)
    for start_frame, end_frame in windows:
        s_chunk = start_frame // 16
        e_chunk = int(math.ceil(end_frame / 16))
        s_chunk = max(0, min(s_chunk, T_snippets - 1))
        e_chunk = max(0, min(e_chunk, T_snippets))
        if s_chunk < e_chunk:
            mask_snippets[s_chunk:e_chunk] = 1

    # Resample to target_len via uniform averaging
    mask_target = np.zeros(target_len, dtype=np.int32)
    r = np.linspace(0, T_snippets, target_len + 1, dtype=np.int32)
    for i in range(target_len):
        if r[i] < r[i+1]:
            segment_votes = mask_snippets[r[i]:r[i+1]]
            mask_target[i] = 1 if segment_votes.mean() >= 0.5 else 0
        else:
            mask_target[i] = mask_snippets[r[i]]
    return mask_target

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature_dir', type=str,
                        default='xdv/i3d-features/RGBTest',
                        help='Path to test I3D features')
    parser.add_argument('--test_list', type=str,
                        default='list/XD_Test.list',
                        help='Path to test list file')
    parser.add_argument('--ann_file', type=str,
                        default='xdv/annotations.txt',
                        help='Path to annotations.txt')
    parser.add_argument('--num_segments', type=int, default=200,
                        help='Target number of segments per video')
    parser.add_argument('--output', type=str,
                        default='frame_label/gt_xd.npy',
                        help='Output path for ground truth')
    args = parser.parse_args()

    # Load test list
    with open(args.test_list, 'r') as f:
        test_files = [line.strip() for line in f if line.strip()]
    print(f"Test videos in list: {len(test_files)}")

    # Load annotations
    annotations = parse_annotations(args.ann_file)
    print(f"Annotated videos: {len(annotations)}")

    # Match and build masks
    all_masks = []
    matched = 0
    unmatched = []
    for fn in test_files:
        # The filename from list is like 'video_name.npy'
        base = fn.replace('.npy', '')
        # Try to find matching annotation
        vid_id = base
        # Try different matching strategies
        windows = None
        if vid_id in annotations:
            windows = annotations[vid_id]
        else:
            # Try removing common suffixes
            for ann_id in annotations:
                if ann_id in vid_id or vid_id in ann_id:
                    windows = annotations[ann_id]
                    vid_id = ann_id
                    break

        if windows is None:
            unmatched.append(fn)
            windows = []

        # Load feature to get T_snippets
        feat_path = os.path.join(args.feature_dir, fn)
        if not os.path.exists(feat_path):
            print(f"WARNING: Feature not found: {feat_path}")
            # Assume T_snippets from annotations max
            T_snippets = args.num_segments  # fallback
        else:
            feat = np.load(feat_path)
            # Handle both (T, D) and (10, T, D)
            if feat.ndim == 3:
                feat = feat.mean(axis=0)
            T_snippets = feat.shape[0]

        mask = resample_mask(windows, T_snippets, args.num_segments)
        all_masks.append(mask)
        if windows:
            matched += 1

    gt_xd = np.concatenate(all_masks)
    print(f"Matched: {matched}/{len(test_files)} videos")
    if unmatched:
        print(f"Unmatched ({len(unmatched)}):")
        for u in unmatched[:5]:
            print(f"  {u}")
    print(f"Total GT length: {len(gt_xd)} (should be {len(test_files)} × {args.num_segments})")
    print(f"Positive ratio: {gt_xd.mean():.3f}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.save(args.output, gt_xd)
    print(f"Saved to {args.output}")

if __name__ == '__main__':
    main()