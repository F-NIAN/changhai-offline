"""Segment-oriented post-processing and evaluation helpers."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np

from data_transfer import CLASSES
from run_optimization_experiments import evaluate_predictions

TARGET_CLASS_IDS = [idx for idx, name in enumerate(CLASSES) if name != "idle"]


@dataclass(frozen=True)
class PostprocessConfig:
    prob_smooth: int = 1
    min_segment: int = 1
    merge_gap: int = 0
    confidence_threshold: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "prob_smooth": self.prob_smooth,
            "min_segment": self.min_segment,
            "merge_gap": self.merge_gap,
            "confidence_threshold": self.confidence_threshold,
        }


def smooth_probabilities(probabilities: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(probabilities) == 0:
        return probabilities.astype(np.float32)
    radius = window // 2
    out = np.zeros_like(probabilities, dtype=np.float32)
    for idx in range(len(probabilities)):
        lo = max(0, idx - radius)
        hi = min(len(probabilities), idx + radius + 1)
        out[idx] = probabilities[lo:hi].mean(axis=0)
    return out


def labels_to_spans(labels: np.ndarray, include_idle: bool = False) -> list[tuple[int, int, int]]:
    spans: list[tuple[int, int, int]] = []
    if len(labels) == 0:
        return spans
    start = 0
    cur = int(labels[0])
    for idx in range(1, len(labels) + 1):
        nxt = int(labels[idx]) if idx < len(labels) else None
        if nxt != cur:
            if include_idle or cur != 0:
                spans.append((start, idx - 1, cur))
            start = idx
            cur = nxt if nxt is not None else 0
    return spans


def drop_short_segments(labels: np.ndarray, min_len: int) -> np.ndarray:
    if min_len <= 1 or len(labels) == 0:
        return labels.astype(np.int64)
    out = labels.astype(np.int64).copy()
    spans = labels_to_spans(out, include_idle=False)
    for start, end, class_id in spans:
        if end - start + 1 >= min_len:
            continue
        left = int(out[start - 1]) if start > 0 else 0
        right = int(out[end + 1]) if end + 1 < len(out) else left
        out[start : end + 1] = left if left == right else 0
    return out


def merge_short_gaps(labels: np.ndarray, max_gap: int) -> np.ndarray:
    if max_gap <= 0 or len(labels) == 0:
        return labels.astype(np.int64)
    out = labels.astype(np.int64).copy()
    changed = True
    while changed:
        changed = False
        spans = labels_to_spans(out, include_idle=True)
        for idx in range(1, len(spans) - 1):
            start, end, class_id = spans[idx]
            if class_id != 0 or end - start + 1 > max_gap:
                continue
            prev_start, prev_end, prev_class = spans[idx - 1]
            next_start, next_end, next_class = spans[idx + 1]
            if prev_class != 0 and prev_class == next_class:
                out[start : end + 1] = prev_class
                changed = True
                break
    return out


def majority_smooth(labels: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(labels) == 0:
        return labels.astype(np.int64)
    out = labels.astype(np.int64).copy()
    radius = window // 2
    for idx in range(len(labels)):
        values = labels[max(0, idx - radius) : min(len(labels), idx + radius + 1)]
        out[idx] = Counter(int(v) for v in values).most_common(1)[0][0]
    return out


def postprocess_probabilities(probabilities: np.ndarray, config: PostprocessConfig) -> np.ndarray:
    probs = smooth_probabilities(probabilities, config.prob_smooth)
    pred = probs.argmax(axis=1).astype(np.int64)
    if config.confidence_threshold > 0:
        low_conf = probs.max(axis=1) < config.confidence_threshold
        pred[low_conf] = 0
    pred = merge_short_gaps(pred, config.merge_gap)
    pred = drop_short_segments(pred, config.min_segment)
    pred = majority_smooth(pred, 3 if config.prob_smooth >= 5 else 1)
    pred = merge_short_gaps(pred, config.merge_gap)
    pred = drop_short_segments(pred, config.min_segment)
    return pred.astype(np.int64)


def span_iou(a: tuple[int, int, int], b: tuple[int, int, int]) -> float:
    a_start, a_end, _ = a
    b_start, b_end, _ = b
    inter = max(0, min(a_end, b_end) - max(a_start, b_start) + 1)
    union = max(a_end, b_end) - min(a_start, b_start) + 1
    return inter / union if union > 0 else 0.0


def match_spans(
    gt_spans: list[tuple[int, int, int]],
    pred_spans: list[tuple[int, int, int]],
    threshold: float,
) -> list[tuple[tuple[int, int, int], tuple[int, int, int], float]]:
    used: set[int] = set()
    matches = []
    for pred in pred_spans:
        best_idx = None
        best_iou = 0.0
        for idx, gt in enumerate(gt_spans):
            if idx in used or pred[2] != gt[2]:
                continue
            iou = span_iou(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        if best_idx is not None and best_iou >= threshold:
            used.add(best_idx)
            matches.append((gt_spans[best_idx], pred, best_iou))
    return matches


def segment_quality(records: list[tuple[dict[str, Any], np.ndarray]]) -> dict[str, Any]:
    per_class: dict[str, dict[str, Any]] = {}
    total_gt = 0
    total_pred = 0
    total_count_abs_error = 0
    boundary_errors = []

    for class_id in TARGET_CLASS_IDS:
        name = CLASSES[class_id]
        gt_count = 0
        pred_count = 0
        class_boundary_errors = []
        for item, pred in records:
            gt_spans = [span for span in labels_to_spans(item["labels"]) if span[2] == class_id]
            pred_spans = [span for span in labels_to_spans(pred) if span[2] == class_id]
            gt_count += len(gt_spans)
            pred_count += len(pred_spans)
            for gt, matched_pred, _ in match_spans(gt_spans, pred_spans, threshold=0.25):
                start_err = abs(matched_pred[0] - gt[0])
                end_err = abs(matched_pred[1] - gt[1])
                class_boundary_errors.append((start_err + end_err) / 2.0)
        count_abs_error = abs(pred_count - gt_count)
        total_gt += gt_count
        total_pred += pred_count
        total_count_abs_error += count_abs_error
        boundary_errors.extend(class_boundary_errors)
        per_class[name] = {
            "gt_segments": gt_count,
            "pred_segments": pred_count,
            "count_abs_error": count_abs_error,
            "boundary_error_mean_frames": float(np.mean(class_boundary_errors)) if class_boundary_errors else None,
            "boundary_error_median_frames": float(np.median(class_boundary_errors)) if class_boundary_errors else None,
        }

    return {
        "total_gt_segments": total_gt,
        "total_pred_segments": total_pred,
        "total_count_abs_error": total_count_abs_error,
        "boundary_error_mean_frames": float(np.mean(boundary_errors)) if boundary_errors else None,
        "boundary_error_median_frames": float(np.median(boundary_errors)) if boundary_errors else None,
        "per_class": per_class,
    }


def evaluate_segment_records(records: list[tuple[dict[str, Any], np.ndarray]]) -> dict[str, Any]:
    metrics = evaluate_predictions(records)
    metrics["segment_quality"] = segment_quality(records)
    return metrics


def objective_score(metrics: dict[str, Any]) -> float:
    frame = metrics.get("frame", {})
    segment = metrics.get("segment", {})
    quality = metrics.get("segment_quality", {})
    f50 = float(segment.get("target_macro_segment_f1@0.5", 0.0))
    f25 = float(segment.get("target_macro_segment_f1@0.25", 0.0))
    frame_f1 = float(frame.get("target_macro_frame_f1", 0.0))
    gt_segments = max(1, int(quality.get("total_gt_segments", 1)))
    count_error = float(quality.get("total_count_abs_error", gt_segments)) / gt_segments
    boundary = quality.get("boundary_error_mean_frames")
    boundary_penalty = min(float(boundary) / 20.0, 1.0) if boundary is not None else 1.0
    return 3.0 * f50 + 2.0 * f25 + 0.25 * frame_f1 - 0.35 * count_error - 0.15 * boundary_penalty
