"""ActionMixed image-feature v2 experiment.

Goal: improve frame accuracy while keeping the three temporal model
architectures shared with the existing offline-model code.
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
from run_optimization_experiments import (
    SEED,
    add_business_priors,
    add_centered_window_stats,
    evaluate_predictions,
    make_windows,
)
from run_pipeline import OfflineSegmenter


V2_RECIPES: dict[str, dict[str, str]] = {
    "ms_tcn": {
        "feature_method": "window_stats+business_priors+roi_rgb_v2",
        "base_feature_method": "window_stats+business_priors",
        "train_mode": "sliding_window",
        "reason": "v2 以准确率为目标，给 ms_tcn 增加中心窗口统计和滑窗样本数，缓解原 full_sequence 样本过少。",
    },
    "asformer": {
        "feature_method": "business_priors+roi_rgb_v2",
        "base_feature_method": "business_priors",
        "train_mode": "full_sequence",
        "reason": "ASFormer 保留当前更稳的 full_sequence + business_priors，只加入低维质量感知 ROI RGB 特征。",
    },
    "bigru": {
        "feature_method": "window_stats+business_priors+roi_rgb_v2",
        "base_feature_method": "window_stats+business_priors",
        "train_mode": "sliding_window",
        "reason": "BiGRU 延续当前准确率最高的滑窗训练和增强 bbox 特征，再加入 v2 ROI RGB 特征。",
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
    recipe = V2_RECIPES[model_name]
    out = []
    fallback_names = []
    for name in roi_rgb_v2_names(CRITICAL_SLOTS):
        fallback_names.append(name)
    fallback_names = (
        fallback_names
        + [f"{name}_center_mean_w5" for name in fallback_names]
        + [f"{name}_delta" for name in fallback_names]
    )

    for item in base_items:
        base = apply_base_feature_method(item, recipe["base_feature_method"])
        rgb, rgb_names = load_cached_rgb_features(rgb_cache_index, item, rgb_dim)
        out.append(add_rgb_to_item(base, rgb, rgb_names or fallback_names))
    return out


def predict_records(segmenter: OfflineSegmenter, items: list[dict[str, Any]], out_dir: Path, model_name: str, split_name: str) -> list[tuple[dict[str, Any], np.ndarray]]:
    records = []
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    for item in items:
        pred, probs = segmenter.predict(item)
        records.append((item, pred))
        np.savez_compressed(
            pred_dir / f"{model_name}_{split_name}_task_{item['task_id']}_soft_labels.npz",
            task_id=np.array([item["task_id"]]),
            split=np.array([split_name]),
            video_ref=np.array([item.get("video_ref", "")]),
            predicted_labels=pred.astype(np.int64),
            probabilities=probs.astype(np.float32),
            class_names=np.array(CLASSES),
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
    recipe = V2_RECIPES[model_name]
    items = make_model_items(base_items, rgb_cache_index, model_name, rgb_dim)
    split = split_by_declared_split(items, seed=SEED)
    train_items = split.train if recipe["train_mode"] == "full_sequence" else make_windows(split.train)

    feature_dim = int(items[0]["features"].shape[1])
    segmenter = OfflineSegmenter(model_name, feature_dim, len(CLASSES), device)
    train_info = segmenter.fit(train_items, epochs=epochs)

    metrics: dict[str, Any] = {}
    for split_name, eval_items in {"val": split.val, "test": split.test}.items():
        records = predict_records(segmenter, eval_items, out_dir, model_name, split_name)
        metrics[split_name] = evaluate_predictions(records) if records else {}

    model_path = out_dir / "models" / f"image_v2_{model_name}_offline_segmenter.pt"
    segmenter.save(model_path, items[0]["feature_names"], items[0].get("feature_version", ""))
    enrich_checkpoint(
        model_path,
        {
            "image_train_version": "v2",
            "recipe": copy.deepcopy(recipe),
            "epochs": epochs,
            "feature_dim": feature_dim,
            "rgb_feature_dim": rgb_dim,
            "rgb_slots": CRITICAL_SLOTS,
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
        "train_mode": recipe["train_mode"],
        "reason": recipe["reason"],
        "feature_version": items[0].get("feature_version", ""),
        "feature_dim": feature_dim,
        "rgb_feature_dim": rgb_dim,
        "epochs": epochs,
        "train_sequences": len(split.train),
        "train_samples": len(train_items),
        "val_sequences": len(split.val),
        "test_sequences": len(split.test),
        "last_loss": train_info["history"][-1]["loss"] if train_info.get("history") else None,
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
    return (
        "| "
        f"`{row['model']}` | `{row['feature_method']}` | `{row['train_mode']}` | {row['feature_dim']} | "
        f"{fmt(frame.get('accuracy'))} | {fmt(frame.get('target_macro_precision'))} | "
        f"{fmt(frame.get('target_macro_recall'))} | {fmt(frame.get('target_macro_frame_f1'))} | "
        f"{fmt(seg.get('target_macro_segment_f1@0.25'))} | {fmt(seg.get('target_macro_segment_f1@0.5'))} | "
        f"`{row['model_path']}` |"
    )


def add_per_class_table(lines: list[str], row: dict[str, Any], split_name: str) -> None:
    metric = row["metrics"].get(split_name, {})
    if not metric:
        lines.append(f"`{split_name}` split 没有样本。")
        return
    lines += [
        f"#### {split_name}",
        "",
        "| 动作类别 | support(帧) | predicted(帧) | TP | FP | FN | Precision | Recall | Frame-F1 | Seg-F1@0.25 | Seg-F1@0.5 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in CLASSES:
        cls = metric["frame"]["per_class"][name]
        support = int(cls["support"])
        predicted = int(cls["predicted"])
        precision = float(cls["precision"])
        recall = float(cls["recall"])
        tp = int(round(recall * support)) if support else 0
        fp = int(round(tp / precision - tp)) if precision > 0 else max(0, predicted - tp)
        fn = max(0, support - tp)
        seg_cls = metric["segment"]["per_class"].get(name, {})
        lines.append(
            "| "
            f"`{name}` | {support} | {predicted} | {tp} | {fp} | {fn} | "
            f"{fmt(precision)} | {fmt(recall)} | {fmt(cls['frame_f1'])} | "
            f"{fmt(seg_cls.get('segment_f1@0.25', 0.0))} | {fmt(seg_cls.get('segment_f1@0.5', 0.0))} |"
        )


def load_baseline_summary() -> dict[str, Any] | None:
    path = ROOT / "output_actionmixed_best_models" / "best_model_report.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def load_v1_summary() -> dict[str, Any] | None:
    path = ROOT / "image_train" / "output_v1" / "image_train_v1.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def build_v1_comparison(results: list[dict[str, Any]], v1: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not v1:
        return []
    v1_rows = {row["model"]: row for row in v1.get("results", [])}
    comparison = []
    for row in results:
        old = v1_rows.get(row["model"])
        if not old:
            continue
        entry = {"model": row["model"]}
        for split_name in ("val", "test"):
            old_acc = old["metrics"][split_name]["frame"]["accuracy"]
            new_acc = row["metrics"][split_name]["frame"]["accuracy"]
            entry[f"v1_{split_name}_acc"] = old_acc
            entry[f"v2_{split_name}_acc"] = new_acc
            entry[f"{split_name}_acc_delta"] = new_acc - old_acc
        comparison.append(entry)
    return comparison


def build_analysis(results: list[dict[str, Any]], baseline: dict[str, Any] | None, v1_comparison: list[dict[str, Any]]) -> list[str]:
    analysis = []
    for split_name in ("val", "test"):
        ranked = sorted(
            results,
            key=lambda row: (
                row["metrics"].get(split_name, {}).get("frame", {}).get("accuracy", -1.0),
                row["metrics"].get(split_name, {}).get("frame", {}).get("target_macro_frame_f1", -1.0),
            ),
            reverse=True,
        )
        if ranked:
            best = ranked[0]
            frame = best["metrics"][split_name]["frame"]
            seg = best["metrics"][split_name]["segment"]
            analysis.append(
                f"{split_name} 上按准确率最高的是 `{best['model']}`：ACC={fmt(frame['accuracy'])}，Frame-F1={fmt(frame['target_macro_frame_f1'])}，F1@0.25={fmt(seg['target_macro_segment_f1@0.25'])}。"
            )
    if baseline:
        base_rows = {row["model"]: row for row in baseline.get("results", [])}
        for row in results:
            base = base_rows.get(row["model"])
            if not base:
                continue
            old_acc = base["metrics"]["frame"]["accuracy"]
            new_acc = row["metrics"]["val"]["frame"]["accuracy"]
            analysis.append(
                f"`{row['model']}` 与历史 best_model_report 的 val ACC 粗略对比：baseline={fmt(old_acc)}，v2={fmt(new_acc)}，差值={fmt(new_acc - old_acc)}。注意历史报告对应当时的数据划分/样本数，主要作参考。"
            )
    analysis.append("v2 把 v1 的 190 维原始 RGB 直方图替换为 7 个关键 ROI 的质量感知低维特征，并追加中心窗口均值与 delta，目标是降低 RGB 噪声对小数据训练的干扰。")
    analysis.append("三种时序模型主体没有复制到 image_train；v2 继续通过 `OfflineSegmenter` 复用 `segmenter/ms_tcn.py`、`segmenter/asformer.py`、`segmenter/bigru.py`。")
    if v1_comparison:
        best_test = sorted(v1_comparison, key=lambda row: row["test_acc_delta"], reverse=True)[0]
        analysis.append(
            f"相对 v1，test ACC 提升最大的是 `{best_test['model']}`，变化为 {fmt(best_test['test_acc_delta'])}。"
        )
    return analysis


def write_markdown(report: dict[str, Any], path: Path) -> None:
    rows = report["results"]
    lines = [
        "# image_train_v2：ActionMixed ROI RGB 质量感知特征训练报告",
        "",
        "## 1. v2 目标",
        "",
        "v2 的主目标是提高帧级准确率，同时把 ROI 预处理抽成可复用层，方便后续 v3/v4 继续使用同一套 ROI 对齐结果。模型架构不在每个版本里复制，三种模型仍统一复用仓库根目录下的 `segmenter/*.py`。",
        "",
        "## 2. 文件夹组织设计",
        "",
        "当前建议的 `image_train` 组织方式如下：",
        "",
        "```text",
        "image_train/",
        "  common/",
        "    roi_cache.py              # 通用 ROI manifest / RGB feature cache",
        "  v2/",
        "    train_image_v2.py         # v2 实验入口，只放本版本 feature recipe 和报告逻辑",
        "  output_v2/",
        "    feature_store_v2/         # 复用原 bbox FeatureStore",
        "    roi_manifest_p15/         # 所有帧的 ROI 槽位、bbox、图片路径、置信度",
        "    roi_rgb_v2_cache/         # 基于 manifest 的 v2 RGB 特征缓存",
        "    models/                   # 三种模型权重",
        "    predictions/              # val/test soft label 输出",
        "    image_train_v2.json       # 结构化实验结果",
        "  image_train_v2.md           # 人读报告",
        "```",
        "",
        "这样后续版本只需要新增 `v3/train_image_v3.py` 和新的 feature cache，不需要复制三种模型实现。ROI manifest 是稳定层，DINOv2、VideoMAE、gated fusion 都可以继续基于它生成不同版本缓存。",
        "",
        "## 3. v2 具体改进",
        "",
        f"- 数据集：`{report['dataset_root']}`",
        f"- 输出目录：`{report['out_dir']}`",
        f"- 训练轮数：`{report['epochs']}`",
        f"- 设备：`{report['device']}`",
        f"- ROI padding：`{report['padding']}`",
        f"- ROI manifest：`{report['roi_manifest_dir']}`",
        f"- RGB v2 cache：`{report['rgb_cache_dir']}`",
        "",
        "### 3.1 ROI manifest",
        "",
        "v2 先对所有 train/val/test 帧预处理 ROI manifest。每个视频片段一个 `.npz`，包含：`frame_numbers`、`image_paths`、`yolo_paths`、`slots`、`boxes[T,S,4]`、`valid[T,S]`、`conf[T,S]`。slot 规则仍是 `hand_top1/hand_top2` 和其它对象 top-1。",
        "",
        "### 3.2 ROI RGB v2 特征",
        "",
        f"v2 只对关键 ROI 槽位提取图像特征：`{', '.join(report['rgb_slots'])}`。",
        "",
        "每个 ROI 原始特征 10 维：`valid`、`conf`、`area`、`aspect`、`r_mean`、`g_mean`、`b_mean`、`brightness_mean`、`brightness_std`、`saturation_proxy`。",
        "",
        "随后追加两类时序增强：中心窗口均值 `center_mean_w5` 和相邻帧差分 `delta`。因此 RGB v2 维度为 `7 slots * 10 dims * 3 blocks = 210`。",
        "",
        "## 4. 三个模型当前特征",
        "",
        "| 模型 | base 特征 | 新增图像特征 | 总 dim | 训练方式 |",
        "|---|---|---|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['model']}` | `{row['base_feature_method']}` | `roi_rgb_v2_quality_smooth` ({row['rgb_feature_dim']}维) | {row['feature_dim']} | `{row['train_mode']}` |"
        )

    lines += [
        "",
        "特征分类：",
        "",
        "- bbox/检测结构特征：present、confidence、center、area、speed、missing_age、imputed、对象距离和距离变化。",
        "- 离线窗口统计：对关键检测列做中心窗口均值，利用前后帧上下文提高稳定性。",
        "- 业务先验：根据短刷/针筒/气枪/长刷与内镜部位的关系构造弱先验分数。",
        "- ROI RGB v2：关键 ROI 的低维颜色、亮度、饱和度和质量特征，并追加局部时间平滑与变化量。",
        "",
        "## 5. 数据概况",
        "",
        "| split | 序列数 | 帧数 |",
        "|---|---:|---:|",
    ]
    for split, stat in sorted(report["dataset_stats"].items()):
        lines.append(f"| `{split}` | {stat['sequences']} | {stat['frames']} |")

    for idx, split_name in enumerate(("val", "test"), start=1):
        lines += [
            "",
            f"## 6.{idx}. {split_name} 整体结果",
            "",
            "| 模型 | 特征 | 训练方式 | dim | ACC | Precision | Recall | Frame-F1 | F1@0.25 | F1@0.5 | 权重 |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
        for row in rows:
            lines.append(metric_row(row, split_name))

    v1_comparison = report.get("v1_comparison", [])
    if v1_comparison:
        lines += [
            "",
            "## 7. v1 到 v2 的准确率变化",
            "",
            "| 模型 | v1 val ACC | v2 val ACC | val 变化 | v1 test ACC | v2 test ACC | test 变化 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for row in v1_comparison:
            lines.append(
                "| "
                f"`{row['model']}` | {fmt(row['v1_val_acc'])} | {fmt(row['v2_val_acc'])} | {fmt(row['val_acc_delta'])} | "
                f"{fmt(row['v1_test_acc'])} | {fmt(row['v2_test_acc'])} | {fmt(row['test_acc_delta'])} |"
            )
        lines += [
            "",
            "从准确率目标看，v2 对 `bigru` 和 `asformer` 有明显收益；`ms_tcn` 对当前高维窗口增强更敏感，本版不建议作为主线。",
        ]

    lines += ["", "## 8. 每个模型逐动作识别情况（以帧数为单位）", ""]
    for row in rows:
        lines += [
            f"### {row['model']}",
            "",
            f"- 设计理由：{row['reason']}",
            f"- 使用特征版本：`{row['feature_version']}`",
            f"- 训练样本：`{row['train_samples']}`，train 序列：`{row['train_sequences']}`，val 序列：`{row['val_sequences']}`，test 序列：`{row['test_sequences']}`",
            f"- 最后一轮 loss：`{fmt(row['last_loss'])}`",
            "",
        ]
        add_per_class_table(lines, row, "val")
        lines.append("")
        add_per_class_table(lines, row, "test")
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
    parser = argparse.ArgumentParser(description="Train ActionMixed image feature v2 models.")
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "input" / "modelscope" / "lhh010__cleansight-ActionMixed" / "cleansight-ActionMixed")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "image_train" / "output_v2")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--padding", type=float, default=0.15)
    parser.add_argument("--models", nargs="+", default=["ms_tcn", "asformer", "bigru"], choices=sorted(V2_RECIPES))
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument("--report-path", type=Path, default=None)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    set_seed()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    feature_dir = args.out_dir / "feature_store_v2"
    clear_npz_dir(feature_dir)
    actionmixed_to_feature_store(args.dataset_root, feature_dir)
    base_items = FeatureStore(feature_dir).load_all()
    if not base_items:
        raise RuntimeError(f"No feature sequences found in {feature_dir}")

    roi_manifest_dir = args.out_dir / "roi_manifest_p15"
    rgb_cache_dir = args.out_dir / "roi_rgb_v2_cache"
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
        print(f"training image v2 model: {model_name}", flush=True)
        results.append(train_one(model_name, base_items, rgb_cache_index, rgb_dim, args.epochs, device, args.out_dir))

    baseline = load_baseline_summary()
    v1_summary = load_v1_summary()
    v1_comparison = build_v1_comparison(results, v1_summary)
    report = {
        "status": "completed",
        "version": "v2",
        "script": str(Path(__file__).resolve()),
        "dataset_root": str(args.dataset_root),
        "out_dir": str(args.out_dir),
        "roi_manifest_dir": str(roi_manifest_dir),
        "rgb_cache_dir": str(rgb_cache_dir),
        "device": str(device),
        "epochs": args.epochs,
        "padding": args.padding,
        "classes": CLASSES,
        "rgb_slots": CRITICAL_SLOTS,
        "rgb_feature_dim": rgb_dim,
        "dataset_stats": dataset_stats(base_items),
        "results": results,
        "baseline_reference": baseline,
        "v1_reference": v1_summary,
        "v1_comparison": v1_comparison,
        "analysis": build_analysis(results, baseline, v1_comparison),
        "next_steps": [
            "继续保留 `common/roi_cache.py` 的 ROI manifest，v3 可以直接在此基础上预提取 DINOv2 ROI embedding，不再重复解析 YOLO 和图片路径。",
            "如果以准确率为主，可继续尝试更长 epoch、多个 seed 和按 val ACC 选择 checkpoint；当前 `OfflineSegmenter.fit` 只保存最后一轮。",
            "对 ASFormer 单独调小输入维度或增加 dropout；它对小数据和额外 RGB 特征更敏感。",
            "对 `flush`/`air_injection` 增加 syringe 与 air_gun 的专门二分类外观差异特征，减少两类互相误报。",
            "如果后续目标转回动作段边界，建议以 F1@0.25/F1@0.5 为主，并加入 boundary head 或边界后处理。",
        ],
    }
    json_path = args.out_dir / "image_train_v2.json"
    md_path = args.report_path or (ROOT / "image_train" / "image_train_v2.md")
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(report, md_path)
    print(json.dumps({"status": "completed", "report": str(md_path), "json": str(json_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
