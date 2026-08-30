"""ActionMixed image-feature v4 experiment.

v4 trains for frame-wise classification accuracy first. Segment smoothing is
handled after model output through a tunable post-processing stage.
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
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_transfer import CLASSES, FeatureStore
from dataset import actionmixed_to_feature_store, class_weights, make_normalizer, split_by_declared_split
from image_train.common.proxy_roi import (
    PROXY_SLOTS,
    PROXY_SUFFIXES,
    load_cached_proxy_features,
    make_proxy_rgb_v4_cache,
    proxy_rgb_v4_names,
)
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
from image_train.v3.train_image_v3 import clear_npz_dir, clone_item, dataset_stats, fmt
from run_optimization_experiments import SEED, add_business_priors, add_centered_window_stats, make_windows
from segmenter import MODEL_REGISTRY


V4_RECIPES: dict[str, dict[str, Any]] = {
    "ms_tcn": {
        "base_feature_method": "business_priors",
        "use_rgb": True,
        "use_proxy": True,
        "reason": "MS-TCN 保留多阶段逐帧 refinement，v4 给它补充 object ROI 与 proxy ROI，但训练选择以 raw frame accuracy 为主。",
    },
    "asformer": {
        "base_feature_method": "business_priors",
        "use_rgb": True,
        "use_proxy": True,
        "reason": "ASFormer 继续使用完整序列 attention 建模长上下文，输入与 MS-TCN 保持同一套 v4 低维视觉特征。",
    },
    "bigru": {
        "base_feature_method": "window_stats+business_priors",
        "use_rgb": True,
        "use_proxy": True,
        "reason": "BiGRU 保留窗口统计特征以增强局部稳定性，同时使用 ROI/proxy RGB 弥补短刷和长刷弱检测。",
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


def split_csv_ints(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def split_csv_floats(value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


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


def proxy_feature_names_v4(smooth_window: int) -> list[str]:
    names = proxy_rgb_v4_names(PROXY_SLOTS)
    return names + [f"{name}_center_mean_w{smooth_window}" for name in names] + [f"{name}_delta" for name in names]


def add_feature_block(item: dict[str, Any], block: np.ndarray, names: list[str], source: str) -> dict[str, Any]:
    out = clone_item(item)
    out["features"] = np.concatenate([out["features"], block], axis=1).astype(np.float32)
    out["feature_names"] = list(out["feature_names"]) + list(names)
    out["feature_version"] = f"{out.get('feature_version', 'unknown')}+{source}"
    out["sources"] = list(out.get("sources", ["bbox", "geometry", "motion"])) + [source]
    return out


def make_model_items(
    base_items: list[dict[str, Any]],
    rgb_cache_index: dict[tuple[str, str], Path],
    proxy_cache_index: dict[tuple[str, str], Path],
    model_name: str,
    rgb_dim: int,
    proxy_dim: int,
    proxy_smooth_window: int,
) -> list[dict[str, Any]]:
    recipe = V4_RECIPES[model_name]
    items = []
    for item in base_items:
        current = apply_base_feature_method(item, str(recipe["base_feature_method"]))
        if recipe["use_rgb"]:
            rgb, rgb_names = load_cached_rgb_features(rgb_cache_index, item, rgb_dim)
            current = add_feature_block(current, rgb, rgb_names or rgb_feature_names_v2(), "roi_rgb_v2_quality_smooth")
        if recipe["use_proxy"]:
            proxy, proxy_names = load_cached_proxy_features(proxy_cache_index, item, proxy_dim)
            current = add_feature_block(current, proxy, proxy_names or proxy_feature_names_v4(proxy_smooth_window), "proxy_rgb_v4")
        items.append(current)
    return items


class AccuracyTrainer:
    def __init__(self, model_name: str, in_dim: int, class_count: int, device: torch.device, args: argparse.Namespace):
        self.model_name = model_name
        self.device = device
        self.model = MODEL_REGISTRY[model_name](in_dim, class_count).to(device)
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        self.args = args
        self.class_count = class_count

    def _item_logits(self, item: dict[str, Any]) -> torch.Tensor:
        if self.mean is None or self.std is None:
            raise RuntimeError("normalizer is not fitted")
        x = torch.tensor(
            ((item["features"] - self.mean) / self.std)[None, :, :],
            dtype=torch.float32,
            device=self.device,
        )
        return self.model(x)

    def _loss_on_items(self, items: list[dict[str, Any]], loss_fn: nn.Module) -> tuple[float, float, float]:
        self.model.eval()
        losses = []
        gt_all = []
        pred_all = []
        with torch.no_grad():
            for item in items:
                logits = self._item_logits(item)
                y = torch.tensor(item["labels"][None, :], dtype=torch.long, device=self.device)
                loss = loss_fn(logits.transpose(1, 2).reshape(-1, self.class_count), y.reshape(-1))
                losses.append(float(loss.detach().cpu()))
                probs = torch.softmax(logits[0].transpose(0, 1), dim=-1).detach().cpu().numpy()
                gt_all.append(item["labels"].astype(np.int64))
                pred_all.append(probs.argmax(axis=1).astype(np.int64))
        gt = np.concatenate(gt_all)
        pred = np.concatenate(pred_all)
        accuracy = float((gt == pred).mean()) if len(gt) else 0.0
        target_f1s = []
        for class_id, name in enumerate(CLASSES):
            if name == "idle":
                continue
            tp = int(((gt == class_id) & (pred == class_id)).sum())
            fp = int(((gt != class_id) & (pred == class_id)).sum())
            fn = int(((gt == class_id) & (pred != class_id)).sum())
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            target_f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
        return float(np.mean(losses)) if losses else 0.0, accuracy, float(np.mean(target_f1s)) if target_f1s else 0.0

    def fit(self, train_items: list[dict[str, Any]], val_items: list[dict[str, Any]]) -> dict[str, Any]:
        self.mean, self.std = make_normalizer(train_items)
        weights, support = class_weights(train_items, self.device)
        if not self.args.use_class_weights:
            weights = None
        loss_fn = nn.CrossEntropyLoss(weight=weights, label_smoothing=self.args.label_smoothing)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)

        scheduler = None
        if self.args.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, self.args.epochs),
                eta_min=self.args.min_lr,
            )
        elif self.args.scheduler == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="max",
                factor=self.args.plateau_factor,
                patience=self.args.plateau_patience,
                min_lr=self.args.min_lr,
            )

        best_score = -1e9
        best_state: dict[str, torch.Tensor] | None = None
        best_epoch = 0
        stale_epochs = 0
        history = []

        for epoch in range(1, self.args.epochs + 1):
            random.shuffle(train_items)
            self.model.train()
            losses = []
            for item in train_items:
                logits = self._item_logits(item)
                y = torch.tensor(item["labels"][None, :], dtype=torch.long, device=self.device)
                loss = loss_fn(logits.transpose(1, 2).reshape(-1, self.class_count), y.reshape(-1))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))

            val_loss, val_acc, val_target_f1 = self._loss_on_items(val_items, loss_fn)
            lr = float(optimizer.param_groups[0]["lr"])
            train_loss = float(np.mean(losses)) if losses else 0.0
            score = val_acc if self.args.early_metric == "val_accuracy" else -val_loss
            improved = score > best_score + self.args.min_delta
            if improved:
                best_score = score
                best_epoch = epoch
                stale_epochs = 0
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                stale_epochs += 1

            history.append(
                {
                    "epoch": epoch,
                    "lr": round(lr, 8),
                    "train_loss": round(train_loss, 6),
                    "val_loss": round(val_loss, 6),
                    "val_accuracy": round(val_acc, 6),
                    "val_target_macro_frame_f1": round(val_target_f1, 6),
                    "best_epoch": best_epoch,
                }
            )

            if scheduler is not None:
                if self.args.scheduler == "plateau":
                    scheduler.step(val_acc)
                else:
                    scheduler.step()

            if self.args.early_stopping and stale_epochs >= self.args.patience:
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return {
            "history": history,
            "best_epoch": best_epoch,
            "best_score": round(float(best_score), 6),
            "class_support_frames": support,
            "stopped_epoch": history[-1]["epoch"] if history else 0,
        }

    def probabilities(self, item: dict[str, Any]) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            logits = self._item_logits(item)[0].transpose(0, 1)
            return torch.softmax(logits, dim=-1).detach().cpu().numpy().astype(np.float32)

    def save(self, path: Path, feature_names: list[str], feature_version: str, metadata: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "model_name": self.model_name,
                "class_names": CLASSES,
                "feature_names": feature_names,
                "feature_version": feature_version,
                "normalizer_mean": self.mean,
                "normalizer_std": self.std,
                **metadata,
            },
            path,
        )


def raw_records(trainer: AccuracyTrainer, items: list[dict[str, Any]]) -> list[tuple[dict[str, Any], np.ndarray]]:
    return [(item, trainer.probabilities(item).argmax(axis=1).astype(np.int64)) for item in items]


def probability_records(trainer: AccuracyTrainer, items: list[dict[str, Any]]) -> list[tuple[dict[str, Any], np.ndarray]]:
    return [(item, trainer.probabilities(item)) for item in items]


def apply_postprocess(records: list[tuple[dict[str, Any], np.ndarray]], config: PostprocessConfig) -> list[tuple[dict[str, Any], np.ndarray]]:
    return [(item, postprocess_probabilities(probs, config)) for item, probs in records]


def postprocess_grid(args: argparse.Namespace) -> list[PostprocessConfig]:
    configs = []
    for prob_smooth in split_csv_ints(args.post_prob_smooth):
        for min_segment in split_csv_ints(args.post_min_segment):
            for merge_gap in split_csv_ints(args.post_merge_gap):
                for confidence_threshold in split_csv_floats(args.post_confidence_threshold):
                    configs.append(
                        PostprocessConfig(
                            prob_smooth=prob_smooth,
                            min_segment=min_segment,
                            merge_gap=merge_gap,
                            confidence_threshold=confidence_threshold,
                        )
                    )
    return configs or [PostprocessConfig()]


def postprocess_selection_score(metrics: dict[str, Any], args: argparse.Namespace) -> float:
    if args.postprocess_objective == "segment":
        return objective_score(metrics)
    frame = metrics["frame"]
    segment = metrics["segment"]
    quality = metrics["segment_quality"]
    gt_segments = max(1, int(quality.get("total_gt_segments", 1)))
    count_penalty = float(quality.get("total_count_abs_error", gt_segments)) / gt_segments
    return (
        5.0 * float(frame["accuracy"])
        + 1.0 * float(frame["target_macro_frame_f1"])
        + 0.15 * float(segment["target_macro_segment_f1@0.25"])
        + 0.10 * float(segment["target_macro_segment_f1@0.5"])
        - 0.05 * count_penalty
    )


def tune_postprocess(prob_val_records: list[tuple[dict[str, Any], np.ndarray]], args: argparse.Namespace) -> tuple[PostprocessConfig, dict[str, Any], list[dict[str, Any]]]:
    best_config: PostprocessConfig | None = None
    best_metrics: dict[str, Any] | None = None
    rows = []
    for config in postprocess_grid(args):
        records = apply_postprocess(prob_val_records, config)
        metrics = evaluate_segment_records(records)
        score = postprocess_selection_score(metrics, args)
        rows.append(
            {
                "config": config.as_dict(),
                "score": score,
                "accuracy": metrics["frame"]["accuracy"],
                "target_macro_frame_f1": metrics["frame"]["target_macro_frame_f1"],
                "f1@0.25": metrics["segment"]["target_macro_segment_f1@0.25"],
                "f1@0.5": metrics["segment"]["target_macro_segment_f1@0.5"],
                "count_abs_error": metrics["segment_quality"]["total_count_abs_error"],
            }
        )
        if best_metrics is None or score > postprocess_selection_score(best_metrics, args):
            best_config = config
            best_metrics = metrics
    assert best_config is not None and best_metrics is not None
    rows.sort(key=lambda row: row["score"], reverse=True)
    return best_config, best_metrics, rows[:20]


def save_prediction_files(
    prob_records: list[tuple[dict[str, Any], np.ndarray]],
    post_records: list[tuple[dict[str, Any], np.ndarray]],
    out_dir: Path,
    model_name: str,
    split_name: str,
    config: PostprocessConfig,
) -> None:
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    post_by_id = {str(item["task_id"]): pred for item, pred in post_records}
    for item, probs in prob_records:
        raw_pred = probs.argmax(axis=1).astype(np.int64)
        post_pred = post_by_id[str(item["task_id"])]
        np.savez_compressed(
            pred_dir / f"{model_name}_{split_name}_task_{item['task_id']}_v4_predictions.npz",
            task_id=np.array([item["task_id"]]),
            split=np.array([split_name]),
            video_ref=np.array([item.get("video_ref", "")]),
            probabilities=probs.astype(np.float32),
            raw_predicted_labels=raw_pred,
            postprocessed_labels=post_pred.astype(np.int64),
            class_names=np.array(CLASSES),
            postprocess_config=np.array([json.dumps(config.as_dict(), ensure_ascii=False)]),
        )


def train_one(
    model_name: str,
    base_items: list[dict[str, Any]],
    rgb_cache_index: dict[tuple[str, str], Path],
    proxy_cache_index: dict[tuple[str, str], Path],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    set_seed(args.seed)
    items = make_model_items(
        base_items,
        rgb_cache_index,
        proxy_cache_index,
        model_name,
        rgb_dim=len(CRITICAL_SLOTS) * 10 * 3,
        proxy_dim=len(PROXY_SLOTS) * len(PROXY_SUFFIXES) * 3,
        proxy_smooth_window=args.proxy_smooth_window,
    )
    split = split_by_declared_split(items, seed=args.seed)
    train_items = split.train if args.train_mode == "full_sequence" else make_windows(split.train, window=args.window, stride=args.stride)

    feature_dim = int(items[0]["features"].shape[1])
    trainer = AccuracyTrainer(model_name, feature_dim, len(CLASSES), device, args)
    train_info = trainer.fit(train_items, split.val)

    val_probs = probability_records(trainer, split.val)
    test_probs = probability_records(trainer, split.test)
    raw_val_records = [(item, probs.argmax(axis=1).astype(np.int64)) for item, probs in val_probs]
    raw_test_records = [(item, probs.argmax(axis=1).astype(np.int64)) for item, probs in test_probs]
    raw_val_metrics = evaluate_segment_records(raw_val_records)
    raw_test_metrics = evaluate_segment_records(raw_test_records)

    best_config, val_post_metrics, top_configs = tune_postprocess(val_probs, args)
    val_post_records = apply_postprocess(val_probs, best_config)
    test_post_records = apply_postprocess(test_probs, best_config)
    test_post_metrics = evaluate_segment_records(test_post_records)
    save_prediction_files(val_probs, val_post_records, args.out_dir, model_name, "val", best_config)
    save_prediction_files(test_probs, test_post_records, args.out_dir, model_name, "test", best_config)

    model_path = args.out_dir / "models" / f"image_v4_{model_name}_offline_segmenter.pt"
    trainer.save(
        model_path,
        items[0]["feature_names"],
        str(items[0].get("feature_version", "unknown")),
        {
            "image_train_version": "v4",
            "train_config": vars(args),
            "postprocess_config": best_config.as_dict(),
            "best_epoch": train_info["best_epoch"],
        },
    )

    return {
        "model": model_name,
        "feature_dim": feature_dim,
        "feature_method": f"{V4_RECIPES[model_name]['base_feature_method']}+roi_rgb_v2+proxy_rgb_v4",
        "feature_version": str(items[0].get("feature_version", "unknown")),
        "train_mode": args.train_mode,
        "use_rgb": V4_RECIPES[model_name]["use_rgb"],
        "use_proxy": V4_RECIPES[model_name]["use_proxy"],
        "reason": V4_RECIPES[model_name]["reason"],
        "train_samples": len(train_items),
        "model_path": str(model_path),
        "train": train_info,
        "postprocess_config": best_config.as_dict(),
        "top_postprocess_configs": top_configs,
        "metrics": {
            "val_raw": raw_val_metrics,
            "test_raw": raw_test_metrics,
            "val_post": val_post_metrics,
            "test_post": test_post_metrics,
        },
    }


def metric_summary_row(row: dict[str, Any], split_name: str) -> str:
    raw = row["metrics"][f"{split_name}_raw"]
    post = row["metrics"][f"{split_name}_post"]
    return (
        f"| `{row['model']}` | {row['feature_dim']} | "
        f"{fmt(raw['frame']['accuracy'])} | {fmt(raw['frame']['target_macro_precision'])} | "
        f"{fmt(raw['frame']['target_macro_recall'])} | {fmt(raw['frame']['target_macro_frame_f1'])} | "
        f"{fmt(post['frame']['accuracy'])} | {fmt(post['frame']['target_macro_frame_f1'])} | "
        f"{fmt(post['segment']['target_macro_segment_f1@0.25'])} | "
        f"{fmt(post['segment']['target_macro_segment_f1@0.5'])} | "
        f"{post['segment_quality']['total_pred_segments']}/{post['segment_quality']['total_gt_segments']} | "
        f"{post['segment_quality']['total_count_abs_error']} | "
        f"{fmt(post['segment_quality']['boundary_error_mean_frames'])} |"
    )


def add_per_class_frame_table(lines: list[str], row: dict[str, Any], split_name: str, stage: str) -> None:
    metrics = row["metrics"][f"{split_name}_{stage}"]["frame"]["per_class"]
    lines.extend(
        [
            f"#### {split_name} {stage} 逐动作帧级识别",
            "",
            "| 动作 | GT帧数 | Pred帧数 | Precision | Recall | Frame-F1 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name in CLASSES:
        data = metrics[name]
        lines.append(
            f"| `{name}` | {data['support']} | {data['predicted']} | "
            f"{fmt(data['precision'])} | {fmt(data['recall'])} | {fmt(data['frame_f1'])} |"
        )


def add_per_class_segment_table(lines: list[str], row: dict[str, Any], split_name: str) -> None:
    seg = row["metrics"][f"{split_name}_post"]["segment"]["per_class"]
    qual = row["metrics"][f"{split_name}_post"]["segment_quality"]["per_class"]
    lines.extend(
        [
            f"#### {split_name} post 动作段识别",
            "",
            "| 动作 | GT段数 | Pred段数 | 段数误差 | F1@0.25 | F1@0.5 | 边界误差均值(帧) |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name in CLASSES:
        if name == "idle":
            continue
        s = seg[name]
        q = qual[name]
        lines.append(
            f"| `{name}` | {q['gt_segments']} | {q['pred_segments']} | {q['count_abs_error']} | "
            f"{fmt(s['segment_f1@0.25'])} | {fmt(s['segment_f1@0.5'])} | {fmt(q['boundary_error_mean_frames'])} |"
        )


def build_analysis(results: list[dict[str, Any]]) -> list[str]:
    analysis = []
    for split_name in ("val", "test"):
        best_raw = max(results, key=lambda row: row["metrics"][f"{split_name}_raw"]["frame"]["accuracy"])
        best_post = max(results, key=lambda row: row["metrics"][f"{split_name}_post"]["frame"]["accuracy"])
        analysis.append(
            f"{split_name} raw frame accuracy 最好的模型是 `{best_raw['model']}`，ACC={fmt(best_raw['metrics'][f'{split_name}_raw']['frame']['accuracy'])}。"
        )
        analysis.append(
            f"{split_name} post frame accuracy 最好的模型是 `{best_post['model']}`，ACC={fmt(best_post['metrics'][f'{split_name}_post']['frame']['accuracy'])}；post 结果用于后续动作段输出。"
        )
    best_test = max(results, key=lambda row: row["metrics"]["test_raw"]["frame"]["accuracy"])
    analysis.append(
        f"按本轮主目标 test raw frame accuracy 排序，`{best_test['model']}` 最好，ACC={fmt(best_test['metrics']['test_raw']['frame']['accuracy'])}。"
    )
    withdraw_notes = []
    for row in results:
        cls = row["metrics"]["test_raw"]["frame"]["per_class"].get("long_brush_withdraw")
        if cls and int(cls["support"]) == 0 and int(cls["predicted"]) > 0:
            withdraw_notes.append(f"`{row['model']}` 误报 {int(cls['predicted'])} 帧")
    if withdraw_notes:
        analysis.append(
            "`long_brush_withdraw` 在当前 test split 中 GT 帧数为 0，但模型仍有预测：" + "、".join(withdraw_notes) + "；这说明当前 split 类别分布不均衡，且长刷插入/拔出方向证据仍不稳定。"
        )
    analysis.append("v4 训练选择 raw frame accuracy 做主目标，避免在切分片段数据上过度追逐段级后处理参数。")
    analysis.append("postprocess 仍会影响 ACC 和段级 F1；本轮默认按验证集 post frame accuracy 选择参数，segment F1/段数/边界误差作为附带约束。")
    analysis.append("新增 proxy RGB 特征维度较多，如果 test 明显低于 val，需要优先做特征组消融，而不是继续加复杂模型。")
    return analysis


def write_markdown(report: dict[str, Any], path: Path) -> None:
    rows = report["results"]
    lines = [
        "# image_train_v4：逐帧准确率优先的 RGB/proxy ROI 训练报告",
        "",
        "## 1. v4 目标",
        "",
        "v4 按 2026-07-28 讨论后的策略执行：训练阶段以逐帧动作分类准确率为主要目标，动作段切分、段内平滑、短段删除和相邻同类段合并作为模型输出后的后处理。报告同时列出 raw 输出和 postprocessed 输出。",
        "",
        "## 2. 参考方法依据",
        "",
        "- 本地 `F:/暑期实习/参考文献/CS-TCN.pdf`：MS-TCN 以逐帧交叉熵为核心分类损失，平滑损失用于减少过分割，因此 v4 把帧分类和段平滑拆开处理。",
        "- 本地 `Garcia-Hernando_First-Person_Hand_Action_CVPR_2018_paper.pdf`：第一人称手部动作识别中，手部和手-物交互线索对细粒度动作有效，因此 v4 新增 expanded/ring hand proxy ROI。",
        "- 本地 `s11263-022-01594-9.pdf`：综述指出真实视频动作识别受背景、光照、相似动作和标注不足影响，因此 v4 把 ROI、训练超参和后处理参数全部参数化。",
        "",
        "## 3. 文件组织",
        "",
        "```text",
        "image_train/",
        "  common/",
        "    roi_cache.py",
        "    proxy_roi.py",
        "    segment_postprocess.py",
        "  v4/",
        "    configs/image_train_v4.json",
        "    train_image_v4.py",
        "  output_v4/",
        "    feature_store_v2/",
        "    roi_manifest_p15/",
        "    roi_rgb_v2_cache/",
        "    proxy_rgb_v4_cache/",
        "    models/",
        "    predictions/",
        "    image_train_v4.json",
        "  image_train_v4.md",
        "```",
        "",
        "## 4. 训练和调参参数",
        "",
        "| 参数 | 当前值 | 说明 |",
        "|---|---:|---|",
    ]
    for name in [
        "epochs",
        "lr",
        "min_lr",
        "weight_decay",
        "label_smoothing",
        "grad_clip",
        "scheduler",
        "early_stopping",
        "patience",
        "early_metric",
        "train_mode",
        "proxy_smooth_window",
        "hand_expand",
        "postprocess_objective",
    ]:
        lines.append(f"| `{name}` | `{report['config'].get(name)}` | 命令行参数，可在 v4 config 中集中修改 |")

    lines.extend(
        [
            "",
            "## 5. 数据概况",
            "",
            "| split | 序列数 | 帧数 |",
            "|---|---:|---:|",
        ]
    )
    for split, stat in sorted(report["dataset_stats"].items()):
        lines.append(f"| `{split}` | {stat['sequences']} | {stat['frames']} |")

    lines.extend(
        [
            "",
            "## 6. 特征设计",
            "",
            "| 模型 | 特征组合 | dim | 说明 |",
            "|---|---|---:|---|",
        ]
    )
    for row in rows:
        lines.append(f"| `{row['model']}` | `{row['feature_method']}` | {row['feature_dim']} | {row['reason']} |")
    lines.extend(
        [
            "",
            "新增 proxy ROI 包括 `wash_tank`、三段 `wash_tank_strip`、`scope_channel`、`hand_top1/2_expanded`、`hand_top1/2_ring`、`hand_control_union`。每个 ROI 提取 `valid/conf/area/aspect/RGB mean/brightness mean/std/saturation/edge_energy/motion_energy`，再追加中心 5 帧均值和 delta。",
            "",
        ]
    )

    for section_number, split_name in ((7, "val"), (8, "test")):
        lines.extend(
            [
                f"## {section_number}. {split_name} 整体结果",
                "",
                "| 模型 | dim | Raw ACC | Raw P | Raw R | Raw Frame-F1 | Post ACC | Post Frame-F1 | Post F1@0.25 | Post F1@0.5 | Pred/GT段数 | 段数误差 | 边界误差均值(帧) |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in rows:
            lines.append(metric_summary_row(row, split_name))
        lines.append("")

    lines.extend(
        [
            "## 9. 后处理参数",
            "",
            "| 模型 | prob_smooth | min_segment | merge_gap | confidence_threshold |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        cfg = row["postprocess_config"]
        lines.append(f"| `{row['model']}` | {cfg['prob_smooth']} | {cfg['min_segment']} | {cfg['merge_gap']} | {cfg['confidence_threshold']} |")

    lines.extend(["", "## 10. 逐模型逐动作结果", ""])
    for row in rows:
        lines.extend(
            [
                f"### {row['model']}",
                "",
                f"- 最佳 epoch：`{row['train']['best_epoch']}`；停止 epoch：`{row['train']['stopped_epoch']}`；验证最佳分数：`{fmt(row['train']['best_score'])}`。",
                f"- 权重文件：`{row['model_path']}`。",
                "",
            ]
        )
        add_per_class_frame_table(lines, row, "val", "raw")
        lines.append("")
        add_per_class_frame_table(lines, row, "test", "raw")
        lines.append("")
        add_per_class_segment_table(lines, row, "val")
        lines.append("")
        add_per_class_segment_table(lines, row, "test")
        lines.append("")

    lines.extend(["## 11. 结果分析", ""])
    lines.extend([f"- {item}" for item in report["analysis"]])
    lines.extend(
        [
            "",
            "## 12. 后续建议",
            "",
            "- 先做 proxy 特征组消融：只开 object ROI、只开 hand proxy、只开 wash/channel proxy、全部开启，确认哪些组真实提升 test raw ACC。",
            "- 当前 postprocess 默认按验证集 frame accuracy 搜索；若后续再次切回动作段目标，可把 `--postprocess-objective segment` 打开。",
            "- 若完整视频级人工标注准备好，应以整段视频作为 sequence 重新训练和测试；当前切分片段数据仍不适合最终判断真实动作段划分能力。",
            "- 如果 proxy ROI 带来 val 高但 test 低，应先收紧固定 ROI 或提高正则，而不是继续加 DINOv2/VideoMAE 高维特征。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def load_config_defaults(config_path: Path | None) -> dict[str, Any]:
    if config_path is None or not config_path.exists():
        return {}
    return json.loads(config_path.read_text(encoding="utf-8"))


def make_parser() -> argparse.ArgumentParser:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=ROOT / "image_train" / "v4" / "configs" / "image_train_v4.json")
    pre_args, _ = pre.parse_known_args()
    defaults = load_config_defaults(pre_args.config)

    parser = argparse.ArgumentParser(description="Train ActionMixed image feature v4 frame-accuracy-oriented models.", parents=[pre])
    parser.set_defaults(**defaults)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "input" / "modelscope" / "lhh010__cleansight-ActionMixed" / "cleansight-ActionMixed")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "image_train" / "output_v4")
    parser.add_argument("--feature-store-dir", type=Path, default=None)
    parser.add_argument("--roi-manifest-dir", type=Path, default=None)
    parser.add_argument("--rgb-cache-dir", type=Path, default=None)
    parser.add_argument("--proxy-cache-dir", type=Path, default=None)
    parser.add_argument("--models", nargs="+", default=["ms_tcn", "asformer", "bigru"], choices=sorted(V4_RECIPES))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.03)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--scheduler", choices=["none", "cosine", "plateau"], default="cosine")
    parser.add_argument("--plateau-factor", type=float, default=0.5)
    parser.add_argument("--plateau-patience", type=int, default=2)
    parser.add_argument("--early-stopping", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--early-metric", choices=["val_accuracy", "val_loss"], default="val_accuracy")
    parser.add_argument("--use-class-weights", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train-mode", choices=["full_sequence", "sliding_window"], default="full_sequence")
    parser.add_argument("--window", type=int, default=128)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--padding", type=float, default=0.15)
    parser.add_argument("--rgb-smooth-window", type=int, default=5)
    parser.add_argument("--proxy-smooth-window", type=int, default=5)
    parser.add_argument("--hand-expand", type=float, default=1.6)
    parser.add_argument("--postprocess-objective", choices=["accuracy", "segment"], default="accuracy")
    parser.add_argument("--post-prob-smooth", type=str, default="1,3,5,7")
    parser.add_argument("--post-min-segment", type=str, default="1,3,5,8")
    parser.add_argument("--post-merge-gap", type=str, default="0,2,4,8")
    parser.add_argument("--post-confidence-threshold", type=str, default="0.0,0.25,0.4")
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument("--report-path", type=Path, default=None)
    parser.set_defaults(**defaults)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    set_seed(args.seed)
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
    proxy_cache_dir = args.proxy_cache_dir or (args.out_dir / "proxy_rgb_v4_cache")
    manifest_index = make_roi_manifest(args.dataset_root, roi_manifest_dir, padding=args.padding, force=args.force_cache)
    rgb_cache_index = make_roi_rgb_v2_cache(
        manifest_index,
        rgb_cache_dir,
        slots=CRITICAL_SLOTS,
        smooth_window=args.rgb_smooth_window,
        force=args.force_cache,
    )
    proxy_cache_index = make_proxy_rgb_v4_cache(
        manifest_index,
        proxy_cache_dir,
        smooth_window=args.proxy_smooth_window,
        hand_expand=args.hand_expand,
        force=args.force_cache,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []
    for model_name in args.models:
        print(f"training image v4 model: {model_name}", flush=True)
        results.append(train_one(model_name, base_items, rgb_cache_index, proxy_cache_index, args, device))

    report = {
        "status": "completed",
        "version": "v4",
        "script": str(Path(__file__).resolve()),
        "dataset_root": str(args.dataset_root),
        "out_dir": str(args.out_dir),
        "roi_manifest_dir": str(roi_manifest_dir),
        "rgb_cache_dir": str(rgb_cache_dir),
        "proxy_cache_dir": str(proxy_cache_dir),
        "device": str(device),
        "config": copy.deepcopy(vars(args)),
        "classes": CLASSES,
        "dataset_stats": dataset_stats(base_items),
        "results": results,
        "analysis": build_analysis(results),
    }

    json_path = args.out_dir / "image_train_v4.json"
    md_path = args.report_path or (ROOT / "image_train" / "image_train_v4.md")
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    write_markdown(report, md_path)
    print(json.dumps({"status": "completed", "report": str(md_path), "json": str(json_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
