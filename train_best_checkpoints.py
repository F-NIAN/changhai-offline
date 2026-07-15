"""Train best-known ActionMixed checkpoints for each offline segmenter.

This script is intentionally separate from ``run_pipeline.py``:

Input:
    - ActionMixed dataset under ``input/modelscope/lhh010__cleansight-ActionMixed``.
    - YOLO detection txt files + frame-level action labels from that dataset.

Process:
    1. Convert ActionMixed to the base FeatureStore-like npz format.
    2. Apply the best feature recipe found in the previous experiment for each model.
    3. Train the model with the matching training mode.
    4. Evaluate on the declared validation split.

Output:
    - ``output_actionmixed_best_models/models/best_<model>_offline_segmenter.pt``
    - ``output_actionmixed_best_models/best_model_report.json``
    - ``output_actionmixed_best_models/best_model_report.md``

The saved checkpoint contains feature_names, feature_version, normalizer
parameters and the selected feature/training recipe. Downstream inference must
use the same feature recipe before loading the corresponding weight file.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from data_transfer import CLASSES, FeatureStore
from dataset import actionmixed_to_feature_store, split_by_declared_split
from run_optimization_experiments import (
    SEED,
    apply_feature_method,
    evaluate_predictions,
    make_windows,
)
from run_pipeline import OfflineSegmenter


BEST_RECIPES: dict[str, dict[str, str]] = {
    # Best MS-TCN row from the optimization report.
    "ms_tcn": {
        "feature_method": "v2",
        "train_mode": "full_sequence",
        "reason": "MS-TCN 在当前实验表中使用 v2 + full_sequence 时片段指标最高。",
    },
    # ASFormer sliding-window run was skipped due runtime, so use best full-sequence ASFormer row.
    "asformer": {
        "feature_method": "business_priors",
        "train_mode": "full_sequence",
        "reason": "ASFormer 在当前实验表中使用 business_priors + full_sequence 时最好。",
    },
    # Overall best row from the optimization report.
    "bigru": {
        "feature_method": "window_stats+business_priors",
        "train_mode": "sliding_window",
        "reason": "BiGRU 使用 window_stats+business_priors + sliding_window 时为当前整体最优。",
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


def clear_npz_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for file in path.glob("*.npz"):
        file.unlink()


def enrich_checkpoint(path: Path, metadata: dict[str, Any]) -> None:
    """Append recipe and validation metadata to a checkpoint saved by OfflineSegmenter."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    checkpoint.update(metadata)
    torch.save(checkpoint, path)


def train_one(
    model_name: str,
    base_items: list[dict[str, Any]],
    epochs: int,
    device: torch.device,
    output_dir: Path,
) -> dict[str, Any]:
    recipe = BEST_RECIPES[model_name]
    set_seed()

    items = apply_feature_method(base_items, recipe["feature_method"])
    split = split_by_declared_split(items, seed=SEED)
    train_items = split.train if recipe["train_mode"] == "full_sequence" else make_windows(split.train)
    feature_dim = int(items[0]["features"].shape[1])
    feature_version = str(items[0].get("feature_version", "unknown"))

    segmenter = OfflineSegmenter(model_name, feature_dim, len(CLASSES), device)
    train_info = segmenter.fit(train_items, epochs=epochs)

    val_records: list[tuple[dict[str, Any], np.ndarray]] = []
    for item in split.val:
        pred, _ = segmenter.predict(item)
        val_records.append((item, pred))
    metrics = evaluate_predictions(val_records)

    model_dir = output_dir / "models"
    model_path = model_dir / f"best_{model_name}_offline_segmenter.pt"
    segmenter.save(model_path, items[0]["feature_names"], feature_version)
    enrich_checkpoint(
        model_path,
        {
            "best_recipe": copy.deepcopy(recipe),
            "epochs": epochs,
            "feature_dim": feature_dim,
            "validation_metrics": metrics,
            "train_sequences": len(split.train),
            "train_samples": len(train_items),
            "val_sequences": len(split.val),
        },
    )

    return {
        "model": model_name,
        "model_path": str(model_path),
        "feature_method": recipe["feature_method"],
        "train_mode": recipe["train_mode"],
        "reason": recipe["reason"],
        "feature_version": feature_version,
        "feature_dim": feature_dim,
        "epochs": epochs,
        "train_sequences": len(split.train),
        "train_samples": len(train_items),
        "val_sequences": len(split.val),
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


def write_markdown(report: dict[str, Any], path: Path) -> None:
    rows = report["results"]
    lines = [
        "# ActionMixed 三模型最佳特征组合权重训练报告",
        "",
        f"- 数据集：`{report['dataset_root']}`",
        f"- 输出目录：`{report['out_dir']}`",
        f"- 训练轮数：`{report['epochs']}`",
        f"- 设备：`{report['device']}`",
        "",
        "## 总览",
        "",
        "| 模型 | 最佳特征组合 | 训练方式 | dim | ACC | Precision | Recall | Frame-F1 | F1@0.25 | F1@0.5 | 权重 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        frame = row["metrics"]["frame"]
        seg = row["metrics"]["segment"]
        lines.append(
            "| "
            f"`{row['model']}` | `{row['feature_method']}` | `{row['train_mode']}` | {row['feature_dim']} | "
            f"{fmt(frame['accuracy'])} | {fmt(frame['target_macro_precision'])} | "
            f"{fmt(frame['target_macro_recall'])} | {fmt(frame['target_macro_frame_f1'])} | "
            f"{fmt(seg['target_macro_segment_f1@0.25'])} | "
            f"{fmt(seg['target_macro_segment_f1@0.5'])} | `{row['model_path']}` |"
        )

    for row in rows:
        lines += [
            "",
            f"## {row['model']}",
            "",
            f"- 选择理由：{row['reason']}",
            f"- 特征版本：`{row['feature_version']}`",
            f"- 训练样本：`{row['train_samples']}`，验证序列：`{row['val_sequences']}`",
            "",
            "| 类别 | support | predicted | Precision | Recall | Frame-F1 | Seg-F1@0.25 | Seg-F1@0.5 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for name in CLASSES:
            frame_cls = row["metrics"]["frame"]["per_class"][name]
            seg_cls = row["metrics"]["segment"]["per_class"].get(name, {})
            lines.append(
                "| "
                f"`{name}` | {frame_cls['support']} | {frame_cls['predicted']} | "
                f"{fmt(frame_cls['precision'])} | {fmt(frame_cls['recall'])} | "
                f"{fmt(frame_cls['frame_f1'])} | "
                f"{fmt(seg_cls.get('segment_f1@0.25', 0.0))} | "
                f"{fmt(seg_cls.get('segment_f1@0.5', 0.0))} |"
            )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train best checkpoints for ActionMixed offline models.")
    parser.add_argument("--dataset-root", type=Path, default=Path("input/modelscope/lhh010__cleansight-ActionMixed"))
    parser.add_argument("--out-dir", type=Path, default=Path("output_actionmixed_best_models"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--models", nargs="+", default=["ms_tcn", "asformer", "bigru"], choices=sorted(BEST_RECIPES))
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []
    for model_name in args.models:
        print(f"training best checkpoint: {model_name}", flush=True)
        results.append(train_one(model_name, base_items, args.epochs, device, args.out_dir))

    report = {
        "status": "completed",
        "dataset_root": str(args.dataset_root),
        "out_dir": str(args.out_dir),
        "device": str(device),
        "epochs": args.epochs,
        "class_names": CLASSES,
        "results": results,
    }
    json_path = args.out_dir / "best_model_report.json"
    md_path = args.out_dir / "best_model_report.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(report, md_path)
    print(json.dumps({"status": "completed", "report": str(md_path), "results": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
