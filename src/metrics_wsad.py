import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score
)


# --------------------------------------------------
# Basic metrics
# --------------------------------------------------

def compute_basic_metrics(y_true, y_score):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score)

    roc_auc = roc_auc_score(y_true, y_score)
    ap = average_precision_score(y_true, y_score)

    precision, recall, thresholds = precision_recall_curve(y_true, y_score)

    f1s = 2 * precision * recall / (precision + recall + 1e-8)
    best_idx = np.argmax(f1s)

    best_thresh = 0.5
    if best_idx < len(thresholds):
        best_thresh = thresholds[best_idx]

    y_pred = (y_score >= best_thresh).astype(int)

    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return {
        "roc_auc": roc_auc,
        "ap": ap,
        "precision": p,
        "recall": r,
        "f1": f1,
        "best_threshold": best_thresh
    }


# --------------------------------------------------
# Point-adjustment
# --------------------------------------------------

def point_adjust(y_pred, y_true):
    """
    If any frame in an anomaly segment is detected,
    mark whole GT segment as detected.
    """
    y_pred = y_pred.copy()
    y_true = y_true.copy()

    in_segment = False
    start = 0

    for i in range(len(y_true)):
        if y_true[i] == 1 and not in_segment:
            start = i
            in_segment = True

        if in_segment and (i == len(y_true) - 1 or y_true[i + 1] == 0):
            end = i

            if y_pred[start:end + 1].sum() > 0:
                y_pred[start:end + 1] = 1

            in_segment = False

    return y_pred


def point_adjust_metrics(y_true, y_score, threshold):
    y_pred = (y_score >= threshold).astype(int)
    y_pred = point_adjust(y_pred, y_true)

    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return {
        "pa_precision": p,
        "pa_recall": r,
        "pa_f1": f1
    }


# --------------------------------------------------
# Detection delay
# --------------------------------------------------

def compute_detection_delay(y_true, y_score, threshold):
    y_pred = (y_score >= threshold).astype(int)

    delays = []

    in_segment = False
    start = 0

    for i in range(len(y_true)):
        if y_true[i] == 1 and not in_segment:
            start = i
            in_segment = True

        if in_segment and (i == len(y_true) - 1 or y_true[i + 1] == 0):
            end = i

            detected = np.where(y_pred[start:end + 1] == 1)[0]

            if len(detected) > 0:
                delays.append(detected[0])

            in_segment = False

    if len(delays) == 0:
        return None

    return float(np.mean(delays))


# --------------------------------------------------
# Segment extraction
# --------------------------------------------------

def extract_segments(binary_seq):
    segments = []
    start = None

    for i, val in enumerate(binary_seq):
        if val == 1 and start is None:
            start = i

        if start is not None and (i == len(binary_seq)-1 or binary_seq[i+1] == 0):
            segments.append((start, i))
            start = None

    return segments


def iou(seg1, seg2):
    inter = max(0, min(seg1[1], seg2[1]) - max(seg1[0], seg2[0]) + 1)
    union = max(seg1[1], seg2[1]) - min(seg1[0], seg2[0]) + 1
    return inter / union


# --------------------------------------------------
# Segment mAP@IoU
# --------------------------------------------------

def segment_map(y_true, y_score, threshold=0.5, iou_thresh=0.5):
    y_pred = (y_score >= threshold).astype(int)

    pred_segments = extract_segments(y_pred)
    gt_segments = extract_segments(y_true)

    matched = set()
    tp = 0

    for p in pred_segments:
        found = False
        for j, g in enumerate(gt_segments):
            if j in matched:
                continue
            if iou(p, g) >= iou_thresh:
                tp += 1
                matched.add(j)
                found = True
                break

    fp = len(pred_segments) - tp
    fn = len(gt_segments) - tp

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)

    return {
        "precision": precision,
        "recall": recall
    }

# --------------------------------------------------
# Full evaluation wrapper
# --------------------------------------------------

def full_evaluate(y_true, y_score):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score)

    # 1. Basic metrics + threshold selection
    basic = compute_basic_metrics(y_true, y_score)
    threshold = basic["best_threshold"]

    # 2. Point-adjusted metrics
    pa = point_adjust_metrics(y_true, y_score, threshold)

    # 3. Detection delay
    delay = compute_detection_delay(y_true, y_score, threshold)

    # 4. Segment mAP-like metrics at different IoU thresholds
    seg_01 = segment_map(
        y_true,
        y_score,
        threshold=threshold,
        iou_thresh=0.1
    )

    seg_03 = segment_map(
        y_true,
        y_score,
        threshold=threshold,
        iou_thresh=0.3
    )

    seg_05 = segment_map(
        y_true,
        y_score,
        threshold=threshold,
        iou_thresh=0.5
    )

    results = {}
    results.update(basic)
    results.update(pa)

    results["delay"] = delay

    results["map@0.1_precision"] = seg_01["precision"]
    results["map@0.1_recall"] = seg_01["recall"]

    results["map@0.3_precision"] = seg_03["precision"]
    results["map@0.3_recall"] = seg_03["recall"]

    results["map@0.5_precision"] = seg_05["precision"]
    results["map@0.5_recall"] = seg_05["recall"]

    return results