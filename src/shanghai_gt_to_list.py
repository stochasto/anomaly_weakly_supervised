#!/usr/bin/env python3
import argparse
import pickle
from pathlib import Path
import numpy as np


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def extract_video_id(feature_path: Path):
    # Path is expected to be .../combine/<video_id>/feature.npy
    return feature_path.parent.name


def normalize_label(label):
    if isinstance(label, (list, tuple, np.ndarray)):
        if len(label) == 0:
            raise ValueError('Empty label container')
        label = label[0]
    return int(float(label))


def build_labeled_list(list_path: Path, video_label_map: dict, output_path: Path):
    lines = [line.strip() for line in list_path.read_text().splitlines() if line.strip()]
    labels = []
    missing = []
    with output_path.open('w') as out:
        for line in lines:
            feature_path = Path(line)
            vid = extract_video_id(feature_path)
            if vid not in video_label_map:
                missing.append(vid)
                continue
            label = normalize_label(video_label_map[vid])
            labels.append(label)
            out.write(f"{feature_path} {label}\n")
    if missing:
        raise ValueError(f'Video IDs missing from video_label.pickle: {missing[:10]}... total {len(missing)}')
    return len(lines), sum(labels), len(labels) - sum(labels)


def summarize_frame_labels(frame_label_map: dict, test_video_ids: list, feature_list: list):
    total_frame_labels = 0
    video_matches = []
    missing_labels = []
    total_feature_frames = 0
    mismatch_videos = []

    test_video_ids_set = set(test_video_ids)

    for feature_path in feature_list:
        path = Path(feature_path)
        video_id = extract_video_id(path)
        if video_id not in frame_label_map:
            missing_labels.append(video_id)
            continue
        labels = frame_label_map[video_id]
        total_frame_labels += len(labels)
        if not path.exists():
            raise FileNotFoundError(f'Feature file not found: {path}')
        feature = np.load(path)
        total_feature_frames += feature.shape[0]
        if feature.shape[0] != len(labels):
            mismatch_videos.append((video_id, feature.shape[0], len(labels)))
        video_matches.append(video_id)

    return {
        'total_feature_videos': len(feature_list),
        'total_feature_frames': total_feature_frames,
        'total_frame_label_videos': len(test_video_ids),
        'total_frame_label_frames': total_frame_labels,
        'missing_labels': missing_labels,
        'mismatch_videos': mismatch_videos,
        'matched_videos': len(video_matches),
    }


def main():
    parser = argparse.ArgumentParser(description='Generate ShanghaiTech labeled feature lists from GT pickle files.')
    parser.add_argument('--root_dir', type=Path, default=Path('/mnt/c/code/data/shanghai_tech/dataset/dataset'), help='ShanghaiTech dataset root directory')
    parser.add_argument('--train_list', type=Path, default=Path('/mnt/c/code/ae_anomaly/UR-DMU/list/shanghai-i3d-train-10crop.list'), help='Existing ShanghaiTech train list without labels')
    parser.add_argument('--test_list', type=Path, default=Path('/mnt/c/code/ae_anomaly/UR-DMU/list/shanghai-i3d-test-10crop.list'), help='Existing ShanghaiTech test list without labels')
    parser.add_argument('--video_label_pickle', type=Path, default=None, help='Path to video_label.pickle')
    parser.add_argument('--frame_label_pickle', type=Path, default=None, help='Path to frame_label.pickle')
    parser.add_argument('--output_prefix', type=str, default='shanghai-i3d', help='Prefix for generated list files')
    args = parser.parse_args()

    if args.video_label_pickle is None:
        args.video_label_pickle = args.root_dir / 'GT' / 'video_label.pickle'
    if args.frame_label_pickle is None:
        args.frame_label_pickle = args.root_dir / 'GT' / 'frame_label.pickle'

    print('Loading video_label from', args.video_label_pickle)
    video_label_map = load_pickle(args.video_label_pickle)
    print('Loading frame_label from', args.frame_label_pickle)
    frame_label_map = load_pickle(args.frame_label_pickle)

    output_dir = args.train_list.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    train_out = output_dir / f'{args.output_prefix}-train-10crop-labeled.list'
    test_out = output_dir / f'{args.output_prefix}-test-10crop-labeled.list'
    all_out = output_dir / f'{args.output_prefix}-all-10crop-labeled.list'

    print('Processing train list...')
    train_total, train_abn, train_norm = build_labeled_list(args.train_list, video_label_map, train_out)
    print(f'  train entries {train_total}, abnormal {train_abn}, normal {train_norm}')

    print('Processing test list...')
    test_total, test_abn, test_norm = build_labeled_list(args.test_list, video_label_map, test_out)
    print(f'  test entries {test_total}, abnormal {test_abn}, normal {test_norm}')

    print('Writing combined list...')
    with all_out.open('w') as out:
        for path in train_out.read_text().splitlines() + test_out.read_text().splitlines():
            out.write(path + '\n')
    print('  combined entries', train_total + test_total)

    # Analyze frame_label relation for test set
    test_paths = [line.split()[0] for line in test_out.read_text().splitlines() if line.strip()]
    test_video_ids = [extract_video_id(Path(p)) for p in test_paths]
    print('Analyzing frame labels for test videos...')
    summary = summarize_frame_labels(frame_label_map, test_video_ids, test_paths)
    print('Frame label summary:')
    print('  test video count', summary['total_feature_videos'])
    print('  total test feature frames', summary['total_feature_frames'])
    print('  total frame-label videos expected', summary['total_frame_label_videos'])
    print('  total frame-label frames expected', summary['total_frame_label_frames'])
    print('  matched videos', summary['matched_videos'])
    print('  missing labels', len(summary['missing_labels']))
    print('  mismatched frame length videos', len(summary['mismatch_videos']))
    if summary['mismatch_videos']:
        print('  mismatches (video_id, feature_frames, label_frames):')
        for item in summary['mismatch_videos'][:10]:
            print('   ', item)

    print('\nGenerated labeled list files:')
    print('  ', train_out)
    print('  ', test_out)
    print('  ', all_out)


if __name__ == '__main__':
    main()
