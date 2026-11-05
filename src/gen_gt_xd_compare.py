import os
import re
import argparse
import numpy as np
from collections import defaultdict

# ---------- configuration defaults ----------
ROOT = '/mnt/c/code/ae_anomaly/UR-DMU'
ANNOTATION_FILE = os.path.join(ROOT, 'list', 'XD_Annotation.txt')
TEST_LIST_CROPS = os.path.join(ROOT, 'list', 'XD_Test.list')     # per‑crop list
FEATURE_DIR = os.path.join(ROOT, 'xdv', 'i3d-features', 'RGBTest')
OUTPUT_GT = os.path.join(ROOT, 'frame_label', 'xd_frame_gt_generated.npy')
# --------------------------------------------

def parse_annotations(ann_path):
    """Returns dict: video_id -> list of (start_frame, end_frame)."""
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

def extract_base_name(filename):
    """Remove __X.npy suffix to get the base video identifier."""
    # filename e.g. 'v=...__0.npy' or 'Bad.Boys...__0.npy'
    m = re.match(r'^(.*)__\d+\.npy$', filename)
    if m:
        return m.group(1)
    if filename.endswith('.npy'):
        return filename[:-4]
    return filename

def snippet_mask_from_windows(windows, num_snippets, snippets_per_clip=16):
    """Convert frame windows to a boolean snippet mask of length num_snippets."""
    mask = np.zeros(num_snippets, dtype=bool)
    max_frame = num_snippets * snippets_per_clip - 1
    for start, end in windows:
        if start > max_frame:
            print(f"  [WARN] Window ({start},{end}) completely beyond last frame {max_frame}")
            continue
        if end > max_frame:
            print(f"  [WARN] Window ({start},{end}) truncated at frame {max_frame}")
            end = max_frame
        start_clip = start // snippets_per_clip
        end_clip = end // snippets_per_clip
        end_clip = min(end_clip, num_snippets - 1)
        mask[start_clip : end_clip+1] = True
    return mask

def generate_frame_gt(annotation_file, test_list_file, feature_dir):
    # 1. parse annotations
    print("Parsing annotations...")
    annot_dict = parse_annotations(annotation_file)
    print(f"Found annotations for {len(annot_dict)} unique videos.\n")

    # 2. read crop test list, extract unique base names in order
    with open(test_list_file, 'r') as f:
        crop_files = [line.strip() for line in f if line.strip()]
    base_names = []
    seen = set()
    for fname in crop_files:
        base = extract_base_name(os.path.basename(fname))
        if base not in seen:
            seen.add(base)
            base_names.append(base)
    print(f"Unique base videos: {len(base_names)}")

    # 3. build frame-level GT for each video, then concatenate
    frame_masks = []
    for base in base_names:
        # match annotation (handle 'v=' prefix)
        match = None
        if base in annot_dict:
            match = base
        elif base.startswith('v=') and base[2:] in annot_dict:
            match = base[2:]
        elif ('v=' + base) in annot_dict:
            match = 'v=' + base

        # load crop 0 to get snippet count T
        crop0_path = os.path.join(feature_dir, base + '__0.npy')
        if not os.path.exists(crop0_path):
            raise FileNotFoundError(f"Feature file not found: {crop0_path}")
        features = np.load(crop0_path)
        T = features.shape[0]

        if match:
            snippet_mask = snippet_mask_from_windows(annot_dict[match], T)
        else:
            snippet_mask = np.zeros(T, dtype=bool)   # normal video

        # repeat each snippet label 16 times to get frame-level labels
        frame_mask = np.repeat(snippet_mask, 16).astype(np.uint8)
        frame_masks.append(frame_mask)
        violent_frames = np.sum(frame_mask)
        print(f"  [{base}] T={T}, frames={len(frame_mask)}, violent={violent_frames}")

    concatenated = np.concatenate(frame_masks)
    print(f"\nTotal frame-level GT length: {len(concatenated)}")
    return concatenated, base_names

def compare_gt(generated, provided_path):
    provided = np.load(provided_path)
    print(f"Provided GT length: {len(provided)}")
    if len(generated) != len(provided):
        print(f"WARNING: Length mismatch! generated={len(generated)} provided={len(provided)}")
        min_len = min(len(generated), len(provided))
        # compare only up to min_len
        diff = np.sum(generated[:min_len] != provided[:min_len])
        print(f"Comparing first {min_len} frames: {diff} differences.")
        if diff > 0:
            idxs = np.where(generated[:min_len] != provided[:min_len])[0]
            print(f"First 10 mismatched indices: {idxs[:10]}")
    else:
        if np.array_equal(generated, provided):
            print("Ground truth arrays are identical!")
        else:
            diff = np.sum(generated != provided)
            print(f"Mismatch! {diff} frames differ out of {len(generated)}.")
            idxs = np.where(generated != provided)[0]
            print(f"First 10 mismatched indices: {idxs[:10]}")

def main():
    parser = argparse.ArgumentParser(description='Generate XD frame-level ground truth from annotations.')
    parser.add_argument('--compare', type=str, default=None,
                        help='Path to an existing GT .npy file to compare with (e.g., frame_label/xd_gt.npy)')
    args = parser.parse_args()

    gt, base_names = generate_frame_gt(ANNOTATION_FILE, TEST_LIST_CROPS, FEATURE_DIR)
    os.makedirs(os.path.dirname(OUTPUT_GT), exist_ok=True)
    np.save(OUTPUT_GT, gt)
    print(f"\nSaved generated GT to {OUTPUT_GT}")

    if args.compare:
        print(f"\nComparing with: {args.compare}")
        compare_gt(gt, args.compare)

if __name__ == '__main__':
    main()