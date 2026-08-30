#!/usr/bin/env python3
"""可视化最佳模型在 test split 上的预测时间线和真值时间线。

Usage:
    python scripts/visualize_results.py \
        --feature-dir output_actionmixed_best_models/feature_store_v2 \
        --model-dir output_actionmixed_best_models/models \
        --out-dir output_actionmixed_best_models/visualizations

脚本加载 FeatureStore 和 ``best_<model>_offline_segmenter.pt``。默认选择最长 test 样本；
``--all-tests`` 会把全部 test 序列画进一张图。该脚本只读 checkpoint，不修改模型或指标。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

import torch

# 允许从仓库根直接执行 ``python scripts/visualize_results.py``。
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_transfer import FeatureStore
from run_pipeline import OfflineSegmenter
from run_optimization_experiments import apply_feature_method


# ============================ 集中参数区 ============================
DEFAULT_FEATURE_DIR = Path("output_actionmixed_best_models/feature_store_v2")
DEFAULT_MODEL_DIR = Path("output_actionmixed_best_models/models")
DEFAULT_OUTPUT_DIR = Path("output_actionmixed_best_models/visualizations")
DEFAULT_MODELS = ["ms_tcn", "asformer", "bigru"]
DEFAULT_DEVICE = "cpu"

def labels_to_segments(labels: np.ndarray) -> List[Tuple[int, int, int]]:
    """Convert frame-wise labels to list of (start, end, label). Inclusive indices."""
    if labels.size == 0:
        return []
    segs: List[Tuple[int, int, int]] = []
    cur = int(labels[0])
    start = 0
    for i in range(1, labels.shape[0]):
        if int(labels[i]) != cur:
            segs.append((start, i - 1, cur))
            start = i
            cur = int(labels[i])
    segs.append((start, labels.shape[0] - 1, cur))
    return segs


def color_map(num: int) -> List[str]:
    cmap = plt.get_cmap("tab20")
    return [cmap(i % 20) for i in range(num)]


def plot_two_timelines(
    gt: np.ndarray,
    pred: np.ndarray,
    class_names: List[str],
    out_path: Path,
    title: str = "",
    fps: float = 1.0,
):
    labels = class_names
    nclass = len(labels)
    colors = color_map(nclass)

    gt_segs = labels_to_segments(gt)
    pred_segs = labels_to_segments(pred)

    fig, ax = plt.subplots(figsize=(16, 3))
    ax.set_title(title)

    height = 0.35
    y_top = 0.6
    y_bot = 0.2

    for s, e, l in gt_segs:
        ax.add_patch(
            plt.Rectangle((s / fps, y_top - height / 2), (e - s + 1) / fps, height, color=colors[l], ec="none")
        )
    for s, e, l in pred_segs:
        ax.add_patch(
            plt.Rectangle((s / fps, y_bot - height / 2), (e - s + 1) / fps, height, color=colors[l], ec="none", alpha=0.9)
        )

    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([y_bot, y_top])
    ax.set_yticklabels(["Prediction", "GroundTruth"])
    ax.set_xlabel("time (s)")

    # Legend
    legend_patches = [plt.Rectangle((0, 0), 1, 1, color=colors[i]) for i in range(nclass)]
    ax.legend(legend_patches, labels, bbox_to_anchor=(1.01, 1.0), loc="upper left")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_stack_for_model(
    items: List[dict[str, Any]],
    seg: OfflineSegmenter,
    class_names: List[str],
    out_path: Path,
    fps_fallback: float = 1.0,
):
    """Create a stacked figure: for each item, two small timelines (GT top, pred bottom)."""
    n = len(items)
    if n == 0:
        raise ValueError("no items to plot")

    colors = color_map(len(class_names))
    height_per = 0.9  # inches per sample
    fig_height = max(3, n * 0.9)
    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(16, fig_height), constrained_layout=True)
    if n == 1:
        axes = [axes]

    for ax, item in zip(axes, items):
        fps = float(item.get("fps", fps_fallback))
        labels = item["labels"].astype(np.int64)
        try:
            pred, _ = seg.predict(item)
        except Exception as e:
            pred = np.zeros_like(labels)

        gt_segs = labels_to_segments(labels)
        pred_segs = labels_to_segments(pred)

        y_top = 0.65
        y_bot = 0.35
        height = 0.25

        for s, e, l in gt_segs:
            ax.add_patch(plt.Rectangle((s / fps, y_top - height / 2), (e - s + 1) / fps, height, color=colors[l], ec="none"))
        for s, e, l in pred_segs:
            ax.add_patch(plt.Rectangle((s / fps, y_bot - height / 2), (e - s + 1) / fps, height, color=colors[l], ec="none", alpha=0.9))

        ax.set_ylim(0.0, 1.0)
        ax.set_yticks([y_bot, y_top])
        ax.set_yticklabels(["Pred", "GT"])
        ax.set_xlabel("time (s)")
        title = str(item.get("video_ref") or f"task{item.get('task_id')}")
        ax.set_title(title, fontsize=8)

    # common legend
    legend_patches = [plt.Rectangle((0, 0), 1, 1, color=colors[i]) for i in range(len(class_names))]
    axes[0].legend(legend_patches, class_names, bbox_to_anchor=(1.01, 1.0), loc="upper left")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR, help="基础 FeatureStore NPZ 目录")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR, help="best_*.pt 权重目录")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="PNG 输出目录")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--all-tests", action="store_true", help="Summarize all test split items into one figure per model.")
    args = parser.parse_args()

    store = FeatureStore(args.feature_dir)
    items = store.load_all()
    if not items:
        raise RuntimeError(f"no feature sequences found in {args.feature_dir}")

    # Prefer explicit test split; otherwise choose the longest sequence
    test_items = [it for it in items if str(it.get("split", "")).lower() == "test"]
    if not test_items:
        test_items = items

    # choose a representative sample from base items (longest)
    base_rep = max(test_items, key=lambda it: int(it.get("frames", 0)))
    rep_task = int(base_rep.get("task_id", -1))
    rep_video_ref = str(base_rep.get("video_ref", ""))

    for model_name in args.models:
        ckpt_path = args.model_dir / f"best_{model_name}_offline_segmenter.pt"
        if not ckpt_path.exists():
            print(f"checkpoint not found for {model_name}: {ckpt_path}, skipping")
            continue

        try:
            # newer PyTorch may require explicit weights_only flag to avoid unsafe unpickling
            ckpt = torch.load(ckpt_path, map_location=args.device, weights_only=False)
        except TypeError:
            # older PyTorch versions don't support weights_only parameter
            ckpt = torch.load(ckpt_path, map_location=args.device)
        except Exception:
            # fallback: try without forcing weights_only (may raise unsafe-unpickling warnings)
            ckpt = torch.load(ckpt_path, map_location=args.device)
        feature_dim = int(ckpt.get("feature_dim", 0))

        # determine feature transformation method used for this checkpoint
        feature_method = None
        if isinstance(ckpt, dict):
            best = ckpt.get("best_recipe") or {}
            if isinstance(best, dict):
                feature_method = best.get("feature_method")
            feature_method = feature_method or ckpt.get("feature_method")

        # apply feature_method (if any) to base items to get transformed items
        transformed = items
        if feature_method:
            try:
                transformed = apply_feature_method(items, feature_method)
            except Exception:
                transformed = items

        # pick representative item from transformed items by task_id or video_ref
        item = None
        for it in transformed:
            try:
                if int(it.get("task_id", -1)) == rep_task:
                    item = it
                    break
            except Exception:
                pass
        if item is None:
            for it in transformed:
                if str(it.get("video_ref", "")) == rep_video_ref:
                    item = it
                    break
        if item is None:
            item = max(transformed, key=lambda it: int(it.get("frames", 0)))

        features = item["features"]
        labels = item["labels"].astype(np.int64)
        fps = float(item.get("fps", 1.0))
        video_ref = item.get("video_ref", f"task{item.get('task_id')}")

        device = torch.device(args.device)
        seg = OfflineSegmenter(model_name, feature_dim, len(ckpt.get("class_names", [])) or 6, device)
        state = ckpt.get("state_dict") or ckpt.get("model_state_dict") or ckpt
        if isinstance(state, dict) and "state_dict" in ckpt:
            seg.model.load_state_dict(ckpt["state_dict"])
        elif isinstance(state, dict) and hasattr(seg.model, "load_state_dict"):
            try:
                seg.model.load_state_dict(state)
            except Exception:
                # try if ckpt contains nested
                seg.model.load_state_dict(ckpt)

        seg.mean = ckpt.get("normalizer_mean")
        seg.std = ckpt.get("normalizer_std")

        # fallback: if mean/std are numpy arrays saved as lists
        if seg.mean is not None:
            seg.mean = np.asarray(seg.mean)
        if seg.std is not None:
            seg.std = np.asarray(seg.std)

        pred, _ = seg.predict(item)

        class_names = list(ckpt.get("class_names") or ["idle", "long_brush_insert", "long_brush_withdraw", "short_brush_cleaning", "flush", "air_injection"])  # fallback

        if args.all_tests:
            # build list of test items from transformed items
            test_items = [it for it in transformed if str(it.get("split", "")).lower() == "test"]
            if not test_items:
                test_items = transformed
            out_path = args.out_dir / f"{model_name}_vs_gt_alltests.png"
            try:
                plot_stack_for_model(test_items, seg, class_names, out_path, fps_fallback=fps)
                print(f"wrote: {out_path}")
            except Exception as e:
                print(f"failed to plot stacked figure for {model_name}: {e}")
        else:
            out_path = args.out_dir / f"{model_name}_vs_gt_{video_ref}.png"
            title = f"{model_name} predictions vs GT — {video_ref}"
            plot_two_timelines(labels, pred, class_names, out_path, title=title, fps=fps)
            print(f"wrote: {out_path}")


if __name__ == "__main__":
    main()
