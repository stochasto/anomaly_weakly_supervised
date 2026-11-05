def align_and_evaluate(net, config, full_video_list, step, model_file=None, save_dir='checkpoints', do_norm=False, fixed_expand=None):
    import matplotlib.pyplot as plt
    device = config['device']
    net.eval()
    net.flag = "Test"
    if model_file is not None:
        net.load_state_dict(torch.load(model_file, map_location=device))

    frame_gt = np.load(config['gt_file'])
    frame_preds = np.zeros_like(frame_gt, dtype=np.float32)
    ptr = 0  # cumulative pointer into frame_gt

    per_video_stats = []
    cls_label, cls_pred = [], []

    best_norm_preds, worst_norm_preds, best_abn_preds, worst_abn_preds = [], [], [], []
    best_norm_gt, worst_norm_gt, best_abn_gt, worst_abn_gt = [], [], [], []
    best_norm_acc, worst_norm_acc = -1, 2
    best_abn_acc, worst_abn_acc = -1, 2

    with torch.no_grad():
        for feat, label in full_video_list:
            # ----------------------
            # Load crops for the video
            # ----------------------
            '''crop_paths = []
            base_file = f"{base}.npy"
            if os.path.exists(base_file):
                crop_paths.append(base_file)
            for i in range(1, 10):
                p = f"{base}__{i}.npy"
                if os.path.exists(p):
                    crop_paths.append(p)
            if len(crop_paths) == 0:
                glob_paths = sorted(glob.glob(base + '*.npy'))
                if glob_paths:
                    crop_paths = glob_paths
            if len(crop_paths) == 0:
                raise FileNotFoundError(f"No crop files for {base}")'''

            # ----------------------
            # Load and stack features
            # ----------------------
            crops = [np.load(p).astype(np.float32) for p in crop_paths]
            maxT = max(c.shape[0] for c in crops)
            feat_dim = crops[0].shape[1]
            stacked = []
            for c in crops:
                if c.shape[0] < maxT:
                    pad = np.zeros((maxT - c.shape[0], feat_dim), dtype=np.float32)
                    c = np.concatenate([c, pad], axis=0)
                stacked.append(c)
            stacked = np.stack(stacked, axis=0)  # [num_crops, maxT, feat]
            avg_feat = stacked.mean(axis=0)      # [maxT, feat]

            # ----------------------
            # Check GT for normal videos
            # ----------------------
            if label == 0:
                gt_segment = frame_gt[ptr:ptr + maxT]
                if not np.all(gt_segment == 0):
                    raise ValueError(f"Normal video '{os.path.basename(base)}' has non-zero labels in GT!")

            # ----------------------
            # Temporal downsample to num_segments
            # ----------------------
            T = avg_feat.shape[0]
            num_segments = config['num_segments']
            if T != num_segments:
                r = np.linspace(0, T, num_segments + 1, dtype=np.int32)
                new_feat = np.zeros((num_segments, feat_dim), dtype=np.float32)
                for i in range(num_segments):
                    if r[i] != r[i+1]:
                        new_feat[i, :] = np.mean(avg_feat[r[i]:r[i+1], :], axis=0)
                    else:
                        new_feat[i, :] = avg_feat[r[i], :]
                seg_feat = new_feat
            else:
                seg_feat = avg_feat

            # ----------------------
            # Normalization if requested
            # ----------------------
            if do_norm:
                mu = seg_feat.mean(axis=0, keepdims=True)
                s = seg_feat.std(axis=0, keepdims=True) + 1e-6
                seg_feat = (seg_feat - mu) / s

            # ----------------------
            # Model forward
            # ----------------------
            x = torch.from_numpy(seg_feat).unsqueeze(0).to(device)
            res = net(x)
            preds = res["frame"]
            if preds.dim() == 3:
                preds = preds.mean(dim=2)
            preds = preds.squeeze(0).detach().cpu().numpy()  # [num_segments]

            # ----------------------
            # Frame-level expansion
            # ----------------------
            if fixed_expand is not None:
                expand_factor = fixed_expand
            else:
                expand_factor = int(np.ceil(maxT / max(preds.shape[0], 1)))
            frame_level_preds = np.repeat(preds, expand_factor)[:maxT]

            # ----------------------
            # Fill global predictions
            # ----------------------
            remaining = len(frame_gt) - ptr
            if remaining <= 0:
                break
            place = frame_level_preds[:remaining]
            frame_preds[ptr:ptr+len(place)] = place
            gt_clip = frame_gt[ptr:ptr+len(place)]
            ptr += len(place)

            # ----------------------
            # Per-video metrics
            # ----------------------
            vid_pred_mean = frame_level_preds.mean()
            vid_pred_label = 1 if frame_level_preds.max() > 0.5 else 0
            vid_acc = (frame_level_preds.round() == gt_clip).sum() / len(gt_clip)

            per_video_stats.append((os.path.basename(base), int(label), vid_pred_mean, vid_acc, len(gt_clip), len(place)))
            cls_label.append(int(label))
            cls_pred.append(vid_pred_label)

            # ----------------------
            # Update best/worst video lists
            # ----------------------
            if label == 0:  # normal
                if vid_acc > best_norm_acc:
                    best_norm_acc = vid_acc
                    best_norm_preds, best_norm_gt = frame_level_preds, gt_clip
                if vid_acc < worst_norm_acc:
                    worst_norm_acc = vid_acc
                    worst_norm_preds, worst_norm_gt = frame_level_preds, gt_clip
            else:  # abnormal
                if vid_acc > best_abn_acc:
                    best_abn_acc = vid_acc
                    best_abn_preds, best_abn_gt = frame_level_preds, gt_clip
                if vid_acc < worst_abn_acc:
                    worst_abn_acc = vid_acc
                    worst_abn_preds, worst_abn_gt = frame_level_preds, gt_clip

    # ----------------------
    # Global metrics
    # ----------------------
    fpr, tpr, _ = roc_curve(frame_gt[:ptr], frame_preds[:ptr])
    auc_score = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(frame_gt[:ptr], frame_preds[:ptr])
    ap_score = auc(recall, precision)
    accuracy = np.mean(np.array(cls_label) == np.array(cls_pred)) if len(cls_pred) > 0 else 0.0

    # ----------------------
    # Plots for diagnostics
    # ----------------------
    os.makedirs(os.path.join(save_dir, 'plots'), exist_ok=True)
    for tag, preds, gt in zip(
        ["best_norm", "worst_norm", "best_abn", "worst_abn"],
        [best_norm_preds, worst_norm_preds, best_abn_preds, worst_abn_preds],
        [best_norm_gt, worst_norm_gt, best_abn_gt, worst_abn_gt]
    ):
        if len(preds) == 0:
            continue
        plt.figure(figsize=(10,4))
        plt.plot(preds, label='pred', color='orange')
        plt.plot(gt, label='gt', alpha=0.6, color='blue')
        plt.legend()
        plt.title(f"{tag.upper()} @ step {step}")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'plots', f'{tag}_step{step}.png'))
        plt.close()

    print(f"[EVAL @ step {step}] AUC={auc_score:.4f} | AP={ap_score:.4f} | ACC={accuracy:.4f}")
    return auc_score, ap_score, accuracy