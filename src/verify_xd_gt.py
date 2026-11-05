import os
import re
import numpy as np
from collections import defaultdict

# ========== CONFIGURATION ==========
root_dir = '/mnt/c/code/ae_anomaly/UR-DMU'
annotation_file = os.path.join(root_dir, 'list', 'XD_Annotation.txt')
feature_dir = os.path.join(root_dir, 'xdv', 'i3d-features', 'RGBTest')
test_list_file = os.path.join(root_dir, 'list', 'XD_Test.list')   # still the crop‑based list
existing_gt_npy = os.path.join(root_dir, 'frame_label', 'xd_gt.npy')
num_segments = 300
# ===================================

def parse_annotations(ann_path):
    """Returns dict: video_id -> list of (start_frame, end_frame)"""
    annots = defaultdict(list)
    with open(ann_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            vid = parts[0]
            nums = [int(x) for x in parts[1:]]
            if len(nums) % 2 != 0:
                print(f"WARNING: odd number of frames in line: {line}")
                continue
            for i in range(0, len(nums), 2):
                annots[vid].append((nums[i], nums[i+1]))
    return annots

def snippet_mask_from_windows(windows, num_snippets, snippets_per_clip=16):
    mask = np.zeros(num_snippets, dtype=bool)
    max_frame = num_snippets * snippets_per_clip - 1
    for start, end in windows:
        if start > max_frame:
            print(f"  [WARN] Window ({start},{end}) completely beyond last frame {max_frame}")
            continue
        if end > max_frame:
            print(f"  [WARN] Window ({start},{end}) truncated")
            end = max_frame
        start_clip = start // snippets_per_clip
        end_clip = end // snippets_per_clip
        end_clip = min(end_clip, num_snippets - 1)
        mask[start_clip : end_clip+1] = True
    return mask

def resample_to_num_segments(mask, target_length):
    T = len(mask)
    if T == 0:
        return np.zeros(target_length, dtype=bool)
    indices = np.linspace(0, T-1, target_length+1).astype(int)
    resampled = np.zeros(target_length, dtype=bool)
    for i in range(target_length):
        resampled[i] = np.any(mask[indices[i]:indices[i+1]+1])
    return resampled

def extract_base_name(filename):
    """Remove __X.npy suffix to get the base video identifier."""
    # filename could be e.g. 'v=...__0.npy' or 'Bad.Boys...__0.npy'
    # The pattern is always '__X.npy' at the end.
    m = re.match(r'^(.*)__\d+\.npy$', filename)
    if m:
        return m.group(1)
    # Fallback: strip .npy if no crop suffix
    if filename.endswith('.npy'):
        return filename[:-4]
    return filename

def main():
    # 1. Parse annotations
    print("Parsing annotations...")
    annot_dict = parse_annotations(annotation_file)
    print(f"Found annotations for {len(annot_dict)} unique videos.\n")

    # 2. Read the crop‑based test list, extract unique base names in order
    with open(test_list_file, 'r') as f:
        crop_files = [line.strip() for line in f if line.strip()]
    print(f"Test list contains {len(crop_files)} crop files.")

    # Keep the original order of unique base names
    base_names = []
    seen = set()
    for fname in crop_files:
        filename_only = os.path.basename(fname)  
        base = extract_base_name(filename_only)
        if base not in seen:
            seen.add(base)
            base_names.append(base)
    print(f"Unique base videos: {len(base_names)}\n")

    # 3. Load existing GT if provided
    compare_gt = os.path.exists(existing_gt_npy)
    if compare_gt:
        existing_gt = np.load(existing_gt_npy)   # shape (num_unique, num_segments) or flattened? Usually 2D.
        print(f"Loaded existing GT from {existing_gt_npy}, shape {existing_gt.shape}")
        # The original xd_gt.npy is sometimes 1D (flattened) or 2D.
        # We'll assume it's 2D (rows = videos, cols = segments) because the inference script
        # slices it per video using a running frame counter. But the verification script expects 2D.
        # If it's 1D, we can reshape later.
        if existing_gt.ndim == 1:
            # Flattened ground truth – but we can't easily compare row‑wise.
            # We'll assume it's 2D for this script; if not, you might need to reshape.
            print("WARNING: existing GT is 1D; comparison will be skipped.")
            compare_gt = False
    else:
        print("No existing xd_gt.npy found, will not compare.")

    generated_300_masks = []
    all_warnings = 0

    print("Verifying annotation-feature alignment...")
    for idx, base in enumerate(base_names):
        # Match base name with annotation (handling 'v=' prefix)
        match = None
        if base in annot_dict:
            match = base
        elif base.startswith('v=') and base[2:] in annot_dict:
            match = base[2:]
        elif ('v=' + base) in annot_dict:
            match = 'v=' + base

        if match is None:
            print(f"  [MISS] No annotation for '{base}'")
            all_warnings += 1
            generated_300_masks.append(np.zeros(num_segments, dtype=bool))
            continue

        windows = annot_dict[match]
        # Load crop 0 to get snippet count T (all crops have same T)
        crop0_path = os.path.join(feature_dir, base + '__0.npy')
        if not os.path.exists(crop0_path):
            print(f"  [ERROR] Feature file not found: {crop0_path}")
            all_warnings += 1
            generated_300_masks.append(np.zeros(num_segments, dtype=bool))
            continue
        features = np.load(crop0_path)
        T = features.shape[0]

        mask = snippet_mask_from_windows(windows, T)
        violent = np.sum(mask)
        print(f"  [{base}] T={T:4d}, windows={windows}, violent snippets={violent}/{T} ({100*violent/T:.1f}%)")

        mask300 = resample_to_num_segments(mask, num_segments)
        generated_300_masks.append(mask300)

    # 4. Compare with existing GT
    if compare_gt and len(generated_300_masks) == existing_gt.shape[0]:
        print("\nComparing generated GT with existing xd_gt.npy...")
        mismatches = 0
        for i, gen_mask in enumerate(generated_300_masks):
            exist_mask = existing_gt[i]
            if not np.array_equal(gen_mask, exist_mask):
                diff = np.sum(gen_mask != exist_mask)
                mismatches += 1
                if mismatches <= 5:
                    print(f"  [MISMATCH] {base_names[i]}: {diff} segments differ (out of {num_segments})")
        if mismatches == 0:
            print("  All videos match perfectly. Existing xd_gt.npy is consistent.")
        else:
            print(f"  Total mismatched videos: {mismatches}/{len(generated_300_masks)}")
    elif compare_gt:
        print(f"\nCannot compare: generated masks count {len(generated_300_masks)} != GT rows {existing_gt.shape[0]}")

    # 5. Save recomputed GT for reference
    if len(generated_300_masks) == len(base_names):
        stacked = np.stack(generated_300_masks)
        np.save(os.path.join(root_dir, 'frame_label', 'gt_xd_verified.npy'), stacked)
        print("\nSaved verified GT to frame_label/gt_xd_verified.npy")

    print("Done.")

if __name__ == '__main__':
    main()