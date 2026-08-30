"""ActionMixed image-feature v3 experiment.

v3 optimizes action segments instead of frame accuracy:
- temporal model architectures remain shared from segmenter/*.py
- ROI manifest/cache remains reusable through image_train/common
- validation selects post-processing parameters by segment F1 and segment count
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_transfer import CLASSES, FeatureStore
from dataset import actionmixed_to_feature_store, split_by_declared_split
from image_train.common.roi_cache import (
    CRITICAL_SLOTS,
    load_cached_rgb_features,
    make_roi_manifest,
    make_roi_rgb_v2_cache,
    roi_rgb_v2_names,
)
from image_train.common.segment_postprocess import (
    PostprocessConfig,
    evaluate_segment_records,
    objective_score,
    postprocess_probabilities,
)
from run_optimization_experiments import SEED, add_business_priors, add_centered_window_stats, make_windows
from run_pipeline import OfflineSegmenter


V3_RECIPES: dict[str, dict[str, Any]] = {
    "ms_tcn": {
        "feature_method": "v2+segment_postprocess",
        "base_feature_method": "v2",
        "use_rgb": False,
        "train_mode": "full_sequence",
        "reason": "MS-TCN 在 v2 的高维 RGB+滑窗配置退化明显，v3 回到低维 v2 bbox 特征，把改进重点放在段级后处理。",
    },
    "asformer": {
        "feature_method": "business_priors+roi_rgb_v2+segment_postprocess",
        "base_feature_method": "business_priors",
        "use_rgb": True,
        "train_mode": "full_sequence",
        "reason": "ASFormer 保留 v2 中较有效的 business_priors+ROI RGB v2，使用段级后处理控制预测段数量和边界。",
    },
    "bigru": {
        "feature_method": "window_stats+business_priors+roi_rgb_v2+segment_postprocess",
        "base_feature_method": "window_stats+business_priors",
        "use_rgb": True,
        "train_mode": "full_sequence",
        "reason": "BiGRU 是 v1/v2 中最稳的主线；v3 为了快速验证段级后处理，先使用 full_sequence 训练，滑窗长训放到后续单独实验。",
    },
}


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def clone_item(item: dict[str, Any]) -> dict[str, Any]:
    out = dict(item)
    out["features"] = np.asarray(item["features"], dtype=np.float32).copy()
    out["labels"] = np.asarray(item["labels"], dtype=np.int64).copy()
    out["feature_names"] = list(item["feature_names"])
    return out


def clear_npz_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for file in path.glob("*.npz"):
        file.unlink()


def apply_base_feature_method(item: dict[str, Any], method: str) -> dict[str, Any]:
    if method == "v2":
        return clone_item(item)
    if method == "business_priors":
        return add_business_priors(item)
    if method == "window_stats+business_priors":
        return add_business_priors(add_centered_window_stats(item))
    raise ValueError(f"unknown base feature method: {method}")


def rgb_feature_names_v2() -> list[str]:
    names = roi_rgb_v2_names(CRITICAL_SLOTS)
    return names + [f"{name}_center_mean_w5" for name in names] + [f"{name}_delta" for name in names]


def add_rgb_to_item(item: dict[str, Any], rgb: np.ndarray, rgb_names: list[str]) -> dict[str, Any]:
    out = clone_item(item)
    out["features"] = np.concatenate([out["features"], rgb], axis=1).astype(np.float32)
    out["feature_names"] = list(out["feature_names"]) + list(rgb_names)
    out["feature_version"] = f"{out.get('feature_version', 'unknown')}+roi_rgb_v2_quality_smooth"
    out["sources"] = list(out.get("sources", ["bbox", "geometry", "motion"])) + ["roi_rgb_v2_quality_smooth"]
    return out


def make_model_items(
    base_items: list[dict[str, Any]],
    rgb_cache_index: dict[tuple[str, str], Path],
    model_name: str,
    rgb_dim: int,
) -> list[dict[str, Any]]:
    recipe = V3_RECIPES[model_name]
    items = []
    for item in base_items:
        base = apply_base_feature_method(item, str(recipe["base_feature_method"]))
        if recipe["use_rgb"]:
            rgb, rgb_names = load_cached_rgb_features(rgb_cache_index, item, rgb_dim)
            base = add_rgb_to_item(base, rgb, rgb_names or rgb_feature_names_v2())
        else:
            base["feature_version"] = f"{base.get('feature_version', 'unknown')}+segment_postprocess_v3"
        items.append(base)
    return items


def raw_probabilities(segmenter: OfflineSegmenter, item: dict[str, Any]) -> np.ndarray:
    if segmenter.mean is None or segmenter.std is None:
        raise RuntimeError("segmenter must be fitted before raw prediction")
    segmenter.model.eval()
    with torch.no_grad():
        x = torch.tensor(
            ((item["features"] - segmenter.mean) / segmenter.std)[None, :, :],
            dtype=torch.float32,
            device=segmenter.device,
        )
        logits = segmenter.model(x)[0].transpose(0, 1)
        probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
    return probs.astype(np.float32)


def make_raw_records(segmenter: OfflineSegmenter, items: list[dict[str, Any]]) -> list[tuple[dict[str, Any], np.ndarray]]:
    return [(item, raw_probabilities(segmenter, item)) for item in items]


def apply_postprocess(
    raw_records: list[tuple[dict[str, Any], np.ndarray]],
    config: PostprocessConfig,
) -> list[tuple[dict[str, Any], np.ndarray]]:
    return [(item, postprocess_probabilities(probs, config)) for item, probs in raw_records]


def config_grid() -> list[PostprocessConfig]:
    configs = []
    for prob_smooth in (1, 5, 9):
        for min_segment in (1, 4, 8, 12):
            for merge_gap in (0, 2, 6, 10):
                for confidence_threshold in (0.0, 0.3, 0.45):
                    configs.append(
                        PostprocessConfig(
                            prob_smooth=prob_smooth,
                            min_segment=min_segment,
                            merge_gap=merge_gap,
                            confidence_threshold=confidence_threshold,
                        )
                    )
    return configs


def tune_postprocess(raw_val_records: list[tuple[dict[str, Any], np.ndarray]]) -> tuple[PostprocessConfig, dict[str, Any], list[dict[str, Any]]]:
    best_config: PostprocessConfig | None = None
    best_metrics: dict[str, Any] | None = None
    rows = []
    for config in config_grid():
        records = apply_postprocess(raw_val_records, config)
        metrics = evaluate_segment_records(records)
        score = objective_score(metrics)
        row = {
            "config": config.as_dict(),
            "objective": score,
            "accuracy": metrics["frame"]["accuracy"],
            "frame_f1": metrics["frame"]["target_macro_frame_f1"],
            "f1@0.25": metrics["segment"]["target_macro_segment_f1@0.25"],
            "f1@0.5": metrics["segment"]["target_macro_segment_f1@0.5"],
            "gt_segments": metrics["segment_quality"]["total_gt_segments"],
            "pred_segments": metrics["segment_quality"]["total_pred_segments"],
            "count_abs_error": metrics["segment_quality"]["total_count_abs_error"],
            "boundary_error_mean_frames": metrics["segment_quality"]["boundary_error_mean_frames"],
        }
        rows.append(row)
        if best_metrics is None or score > objective_score(best_metrics):
            best_config = config
            best_metrics = metrics
    assert best_config is not None and best_metrics is not None
    rows.sort(key=lambda row: row["objective"], reverse=True)
    return best_config, best_metrics, rows[:20]


def save_predictions(
    raw_records: list[tuple[dict[str, Any], np.ndarray]],
    config: PostprocessConfig,
    out_dir: Path,
    model_name: str,
    split_name: str,
) -> list[tuple[dict[str, Any], np.ndarray]]:
    records = apply_postprocess(raw_records, config)
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    for item, pred in records:
        np.savez_compressed(
            pred_dir / f"{model_name}_{split_name}_task_{item['task_id']}_postprocessed_labels.npz",
            task_id=np.array([item["task_id"]]),
            split=np.array([split_name]),
            video_ref=np.array([item.get("video_ref", "")]),
            predicted_labels=pred.astype(np.int64),
            class_names=np.array(CLASSES),
            postprocess_config=np.array([json.dumps(config.as_dict(), ensure_ascii=False)]),
        )
    return records


def enrich_checkpoint(path: Path, metadata: dict[str, Any]) -> None:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    checkpoint.update(metadata)
    torch.save(checkpoint, path)


def train_one(
    model_name: str,
    base_items: list[dict[str, Any]],
    rgb_cache_index: dict[tuple[str, str], Path],
    rgb_dim: int,
    epochs: int,
    device: torch.device,
    out_dir: Path,
) -> dict[str, Any]:
    set_seed()
    recipe = V3_RECIPES[model_name]
    items = make_model_items(base_items, rgb_cache_index, model_name, rgb_dim)
    split = split_by_declared_split(items, seed=SEED)
    train_items = split.train if recipe["train_mode"] == "full_sequence" else make_windows(split.train)

    feature_dim = int(items[0]["features"].shape[1])
    segmenter = OfflineSegmenter(model_name, feature_dim, len(CLASSES), device)
    train_info = segmenter.fit(train_items, epochs=epochs)

    raw_val = make_raw_records(segmenter, split.val)
    raw_test = make_raw_records(segmenter, split.test)
    best_config, val_metrics, top_configs = tune_postprocess(raw_val)
    val_records = save_predictions(raw_val, best_config, out_dir, model_name, "val")
    test_records = save_predictions(raw_test, best_config, out_dir, model_name, "test")
    metrics = {
        "val": evaluate_segment_records(val_records),
        "test": evaluate_segment_records(test_records),
    }

    model_path = out_dir / "models" / f"image_v3_{model_name}_offline_segmenter.pt"
    segmenter.save(model_path, items[0]["feature_names"], items[0].get("feature_version", ""))
    enrich_checkpoint(
        model_path,
        {
            "image_train_version": "v3",
            "recipe": copy.deepcopy(recipe),
            "epochs": epochs,
            "feature_dim": feature_dim,
            "rgb_feature_dim": rgb_dim if recipe["use_rgb"] else 0,
            "postprocess_config": best_config.as_dict(),
            "validation_postprocess_metrics": val_metrics,
            "metrics": metrics,
            "train_sequences": len(split.train),
            "train_samples": len(train_items),
            "val_sequences": len(split.val),
            "test_sequences": len(split.test),
        },
    )

    return {
        "model": model_name,
        "model_path": str(model_path),
        "feature_method": recipe["feature_method"],
        "base_feature_method": recipe["base_feature_method"],
        "use_rgb": recipe["use_rgb"],
        "train_mode": recipe["train_mode"],
        "reason": recipe["reason"],
        "feature_version": items[0].get("feature_version", ""),
        "feature_dim": feature_dim,
        "rgb_feature_dim": rgb_dim if recipe["use_rgb"] else 0,
        "epochs": epochs,
        "train_sequences": len(split.train),
        "train_samples": len(train_items),
        "val_sequences": len(split.val),
        "test_sequences": len(split.test),
        "last_loss": train_info["history"][-1]["loss"] if train_info.get("history") else None,
        "postprocess_config": best_config.as_dict(),
        "postprocess_top_configs": top_configs,
        "train": train_info,
        "metrics": metrics,
    }


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def dataset_stats(items: list[dict[str, Any]]) -> dict[str, Any]:
    stats: dict[str, dict[str, int]] = {}
    for item in items:
        split = str(item.get("split", "unknown"))
        entry = stats.setdefault(split, {"sequences": 0, "frames": 0})
        entry["sequences"] += 1
        entry["frames"] += int(item["features"].shape[0])
    return stats


def metric_row(row: dict[str, Any], split_name: str) -> str:
    metric = row["metrics"].get(split_name, {})
    frame = metric.get("frame", {})
    seg = metric.get("segment", {})
    quality = metric.get("segment_quality", {})
    return (
        "| "
        f"`{row['model']}` | `{row['feature_method']}` | `{row['train_mode']}` | "
        f"{fmt(frame.get('accuracy'))} | {fmt(frame.get('target_macro_frame_f1'))} | "
        f"{fmt(seg.get('target_macro_segment_f1@0.25'))} | {fmt(seg.get('target_macro_segment_f1@0.5'))} | "
        f"{quality.get('total_gt_segments', '-')} | {quality.get('total_pred_segments', '-')} | "
        f"{quality.get('total_count_abs_error', '-')} | {fmt(quality.get('boundary_error_mean_frames'))} |"
    )


def add_per_class_segment_table(lines: list[str], row: dict[str, Any], split_name: str) -> None:
    metric = row["metrics"].get(split_name, {})
    if not metric:
        return
    lines += [
        f"#### {split_name}",
        "",
        "| 动作类别 | support(帧) | predicted(帧) | Precision | Recall | Frame-F1 | Seg-F1@0.25 | Seg-F1@0.5 | GT段数 | Pred段数 | 段数误差 | 边界误差均值(帧) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in CLASSES:
        frame_cls = metric["frame"]["per_class"][name]
        seg_cls = metric["segment"]["per_class"].get(name, {})
        quality_cls = metric["segment_quality"]["per_class"].get(name, {})
        lines.append(
            "| "
            f"`{name}` | {frame_cls['support']} | {frame_cls['predicted']} | "
            f"{fmt(frame_cls['precision'])} | {fmt(frame_cls['recall'])} | {fmt(frame_cls['frame_f1'])} | "
            f"{fmt(seg_cls.get('segment_f1@0.25', 0.0))} | {fmt(seg_cls.get('segment_f1@0.5', 0.0))} | "
            f"{quality_cls.get('gt_segments', 0)} | {quality_cls.get('pred_segments', 0)} | "
            f"{quality_cls.get('count_abs_error', 0)} | {fmt(quality_cls.get('boundary_error_mean_frames'))} |"
        )


def load_previous(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def compare_with_previous(results: list[dict[str, Any]], previous: dict[str, Any] | None, label: str) -> list[dict[str, Any]]:
    if not previous:
        return []
    prev_rows = {row["model"]: row for row in previous.get("results", [])}
    comparison = []
    for row in results:
        old = prev_rows.get(row["model"])
        if not old:
            continue
        entry = {"model": row["model"], "reference": label}
        for split_name in ("val", "test"):
            for key, metric_path in {
                "f1@0.25": ("segment", "target_macro_segment_f1@0.25"),
                "f1@0.5": ("segment", "target_macro_segment_f1@0.5"),
            }.items():
                old_value = old["metrics"][split_name][metric_path[0]][metric_path[1]]
                new_value = row["metrics"][split_name][metric_path[0]][metric_path[1]]
                entry[f"{split_name}_{key}_old"] = old_value
                entry[f"{split_name}_{key}_new"] = new_value
                entry[f"{split_name}_{key}_delta"] = new_value - old_value
        comparison.append(entry)
    return comparison


def build_analysis(results: list[dict[str, Any]], v2_comparison: list[dict[str, Any]]) -> list[str]:
    analysis = []
    for split_name in ("val", "test"):
        ranked = sorted(
            results,
            key=lambda row: (
                row["metrics"][split_name]["segment"]["target_macro_segment_f1@0.5"],
                row["metrics"][split_name]["segment"]["target_macro_segment_f1@0.25"],
                -row["metrics"][split_name]["segment_quality"]["total_count_abs_error"],
            ),
            reverse=True,
        )
        best = ranked[0]
        seg = best["metrics"][split_name]["segment"]
        quality = best["metrics"][split_name]["segment_quality"]
        analysis.append(
            f"{split_name} 上按段级目标最好的模型是 `{best['model']}`：F1@0.5={fmt(seg['target_macro_segment_f1@0.5'])}，F1@0.25={fmt(seg['target_macro_segment_f1@0.25'])}，预测段数/真值段数={quality['total_pred_segments']}/{quality['total_gt_segments']}，段数绝对误差={quality['total_count_abs_error']}。"
        )
    if v2_comparison:
        best_test = sorted(v2_comparison, key=lambda row: row["test_f1@0.5_delta"], reverse=True)[0]
        analysis.append(
            f"相对 v2，test F1@0.5 提升最大的是 `{best_test['model']}`，变化为 {fmt(best_test['test_f1@0.5_delta'])}。"
        )
        dropped = [
            f"`{row['model']}`({fmt(row['test_f1@0.5_delta'])})"
            for row in v2_comparison
            if row["test_f1@0.5_delta"] < 0
        ]
        if dropped:
            analysis.append(
                "但 test 上存在泛化退化：" + "、".join(dropped) + "。这说明只在单一 val split 搜索后处理参数容易过拟合，后续应使用交叉验证或更保守的参数选择。"
            )
    analysis.append("v3 的改进重点不是改变网络结构，而是在验证集上选择段级后处理参数；这直接服务于“段数更接近、边界更接近”的目标。")
    analysis.append("当前数据集较小，val 上搜索后处理参数可能过拟合，因此报告同时列出 test 结果作为外部参考。")
    return analysis


def write_markdown(report: dict[str, Any], path: Path) -> None:
    rows = report["results"]
    lines = [
        "# image_train_v3：面向动作时间段划分的训练报告",
        "",
        "## 1. v3 目标",
        "",
        "v3 不再把 ACC 作为主要优化目标，而是优先让预测动作段和标注动作段更接近：段数尽量一致，F1@0.25/F1@0.5 更高，匹配段的起止边界误差更小。ACC 仍保留在报告中，但只作为参考指标。",
        "",
        "## 2. 文件夹组织",
        "",
        "```text",
        "image_train/",
        "  common/",
        "    roi_cache.py                 # v2/v3 共享 ROI manifest 和 RGB cache",
        "    segment_postprocess.py       # v3 新增，段级后处理和边界评估",
        "  v3/",
        "    train_image_v3.py            # v3 实验入口",
        "  output_v3/",
        "    feature_store_v2/",
        "    roi_manifest_p15/",
        "    roi_rgb_v2_cache/",
        "    models/",
        "    predictions/",
        "    image_train_v3.json",
        "  image_train_v3.md",
        "```",
        "",
        "三种时序模型主体仍统一复用根目录 `segmenter/*.py`，版本目录只管理特征组合、缓存和后处理策略。",
        "",
        "## 3. v3 做了什么修改",
        "",
        f"- 数据集：`{report['dataset_root']}`",
        f"- 输出目录：`{report['out_dir']}`",
        f"- 训练轮数：`{report['epochs']}`",
        f"- 设备：`{report['device']}`",
        f"- ROI manifest：`{report['roi_manifest_dir']}`",
        f"- RGB cache：`{report['rgb_cache_dir']}`",
        "",
        "### 3.1 特征策略",
        "",
        "| 模型 | 特征组合 | 是否使用 RGB | 训练方式 | dim |",
        "|---|---|---|---|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['model']}` | `{row['feature_method']}` | `{row['use_rgb']}` | `{row['train_mode']}` | {row['feature_dim']} |"
        )

    lines += [
        "",
        "### 3.2 段级后处理搜索",
        "",
        "每个模型训练后先输出 raw softmax 概率，然后在 val split 上搜索：",
        "",
        "- `prob_smooth`：概率时间平滑窗口。",
        "- `min_segment`：最短非 idle 动作段长度，短段会被删除。",
        "- `merge_gap`：同类动作段之间的短 idle 间隔合并阈值。",
        "- `confidence_threshold`：低置信帧置为 idle。",
        "",
        "搜索目标函数优先 `F1@0.5`，其次 `F1@0.25`，再惩罚预测段数和真值段数差距、匹配段边界误差。选出的同一套参数再用于 test。",
        "",
        "## 4. 数据概况",
        "",
        "| split | 序列数 | 帧数 |",
        "|---|---:|---:|",
    ]
    for split, stat in sorted(report["dataset_stats"].items()):
        lines.append(f"| `{split}` | {stat['sequences']} | {stat['frames']} |")

    for split_name in ("val", "test"):
        lines += [
            "",
            f"## 5. {split_name} 段级整体结果",
            "",
            "| 模型 | 特征 | 训练方式 | ACC | Frame-F1 | F1@0.25 | F1@0.5 | GT段数 | Pred段数 | 段数误差 | 边界误差均值(帧) |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in rows:
            lines.append(metric_row(row, split_name))

    lines += [
        "",
        "## 6. 每个模型选出的后处理参数",
        "",
        "| 模型 | prob_smooth | min_segment | merge_gap | confidence_threshold |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        cfg = row["postprocess_config"]
        lines.append(
            f"| `{row['model']}` | {cfg['prob_smooth']} | {cfg['min_segment']} | {cfg['merge_gap']} | {cfg['confidence_threshold']} |"
        )

    comparisons = report.get("v2_comparison", [])
    if comparisons:
        lines += [
            "",
            "## 7. v2 到 v3 的段级变化",
            "",
            "| 模型 | val F1@0.25变化 | val F1@0.5变化 | test F1@0.25变化 | test F1@0.5变化 |",
            "|---|---:|---:|---:|---:|",
        ]
        for row in comparisons:
            lines.append(
                "| "
                f"`{row['model']}` | {fmt(row['val_f1@0.25_delta'])} | {fmt(row['val_f1@0.5_delta'])} | "
                f"{fmt(row['test_f1@0.25_delta'])} | {fmt(row['test_f1@0.5_delta'])} |"
            )

    lines += ["", "## 8. 每个模型逐动作段识别情况", ""]
    for row in rows:
        lines += [
            f"### {row['model']}",
            "",
            f"- 设计理由：{row['reason']}",
            f"- 特征版本：`{row['feature_version']}`",
            f"- 训练样本：`{row['train_samples']}`，最后一轮 loss：`{fmt(row['last_loss'])}`",
            "",
        ]
        add_per_class_segment_table(lines, row, "val")
        lines.append("")
        add_per_class_segment_table(lines, row, "test")
        lines.append("")

    lines += [
        "## 9. 结果分析",
        "",
        *[f"- {item}" for item in report["analysis"]],
        "",
        "## 10. 后续建议",
        "",
        *[f"- {item}" for item in report["next_steps"]],
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train ActionMixed image feature v3 segment-oriented models.")
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "input" / "modelscope" / "lhh010__cleansight-ActionMixed" / "cleansight-ActionMixed")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "image_train" / "output_v3")
    parser.add_argument("--feature-store-dir", type=Path, default=None, help="reuse an existing FeatureStore directory")
    parser.add_argument("--roi-manifest-dir", type=Path, default=None, help="reuse or write an ROI manifest directory")
    parser.add_argument("--rgb-cache-dir", type=Path, default=None, help="reuse or write an ROI RGB v2 cache directory")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--padding", type=float, default=0.15)
    parser.add_argument("--models", nargs="+", default=["ms_tcn", "asformer", "bigru"], choices=sorted(V3_RECIPES))
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument("--report-path", type=Path, default=None)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    set_seed()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    feature_dir = args.feature_store_dir or (args.out_dir / "feature_store_v2")
    if args.feature_store_dir is None:
        clear_npz_dir(feature_dir)
        actionmixed_to_feature_store(args.dataset_root, feature_dir)
    base_items = FeatureStore(feature_dir).load_all()
    if not base_items:
        raise RuntimeError(f"No feature sequences found in {feature_dir}")

    roi_manifest_dir = args.roi_manifest_dir or (args.out_dir / "roi_manifest_p15")
    rgb_cache_dir = args.rgb_cache_dir or (args.out_dir / "roi_rgb_v2_cache")
    manifest_index = make_roi_manifest(args.dataset_root, roi_manifest_dir, padding=args.padding, force=args.force_cache)
    rgb_cache_index = make_roi_rgb_v2_cache(
        manifest_index,
        rgb_cache_dir,
        slots=CRITICAL_SLOTS,
        smooth_window=5,
        force=args.force_cache,
    )
    rgb_dim = len(CRITICAL_SLOTS) * 10 * 3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []
    for model_name in args.models:
        print(f"training image v3 model: {model_name}", flush=True)
        results.append(train_one(model_name, base_items, rgb_cache_index, rgb_dim, args.epochs, device, args.out_dir))

    v2_summary = load_previous(ROOT / "image_train" / "output_v2" / "image_train_v2.json")
    v2_comparison = compare_with_previous(results, v2_summary, "v2")
    report = {
        "status": "completed",
        "version": "v3",
        "script": str(Path(__file__).resolve()),
        "dataset_root": str(args.dataset_root),
        "out_dir": str(args.out_dir),
        "roi_manifest_dir": str(roi_manifest_dir),
        "rgb_cache_dir": str(rgb_cache_dir),
        "device": str(device),
        "epochs": args.epochs,
        "padding": args.padding,
        "classes": CLASSES,
        "dataset_stats": dataset_stats(base_items),
        "results": results,
        "v2_reference": v2_summary,
        "v2_comparison": v2_comparison,
        "analysis": build_analysis(results, v2_comparison),
        "next_steps": [
            "v3 已证明后处理参数对段级结果影响很大；下一步应把后处理搜索从单一 val split 扩展到交叉验证，降低过拟合。",
            "如果目标继续是边界接近，建议新增 boundary head：从标注段起止点生成边界标签，让模型显式学习动作起止。",
            "当前后处理仍是全类别共享参数，后续可做每类 min_segment/merge_gap，特别是 long_brush 与 flush 的持续时长差异很大。",
            "保留 ROI manifest 作为稳定输入层；视觉特征下一版可换成冻结 DINOv2 embedding，但仍应以段级 F1 和边界误差作为主指标。",
            "报告中的 test 结果比 val 更重要；如果某模型 val 提升但 test 退化，应优先认为后处理搜索过拟合。",
        ],
    }
    json_path = args.out_dir / "image_train_v3.json"
    md_path = args.report_path or (ROOT / "image_train" / "image_train_v3.md")
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(report, md_path)
    print(json.dumps({"status": "completed", "report": str(md_path), "json": str(json_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
