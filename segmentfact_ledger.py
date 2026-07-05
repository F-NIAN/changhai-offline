"""
SegmentFact 与 FactLedger 转换模块。

输入：
    模型输出的逐帧 labels，以及可选逐帧类别概率 probs。

输出：
    SegmentFact：业务可读的动作片段事实。
    FactLedger 行：带 idem_key 的幂等 upsert 记录。
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from data_transfer import CLASSES


def labels_to_segment_facts(labels: np.ndarray, fps: float, task_id: int, step_id: int, source: str, probs: np.ndarray | None = None) -> list[dict[str, Any]]:
    """把逐帧标签合并成连续动作片段；background 不输出。"""
    if len(labels) == 0:
        return []
    facts: list[dict[str, Any]] = []
    start = 0
    current = int(labels[0])
    for idx in range(1, len(labels) + 1):
        next_value = int(labels[idx]) if idx < len(labels) else None
        if next_value != current:
            if current != 0:
                label = CLASSES[current]
                confidence = 1.0 if probs is None else float(np.mean(probs[start:idx, current]))
                facts.append(
                    {
                        "fact_id": f"{task_id}:{step_id}:{source}:{start + 1}:{idx}:{label}",
                        "task_id": int(task_id),
                        "step_id": int(step_id),
                        "label": label,
                        "start_frame": int(start + 1),
                        "end_frame": int(idx),
                        "start_ms": int(round(start * 1000.0 / fps)),
                        "end_ms": int(round(idx * 1000.0 / fps)),
                        "confidence": round(confidence, 5),
                        "source": source,
                    }
                )
            start = idx
            current = next_value if next_value is not None else 0
    return facts


def segment_facts_to_fact_ledger(facts: list[dict[str, Any]], model_version: str) -> list[dict[str, Any]]:
    """把 SegmentFact 包装成可幂等 upsert 的 ledger 行。"""
    run_id = f"offline-segmenter-{int(time.time())}"
    return [
        {
            "ledger_op": "upsert",
            "idem_key": f"{fact['fact_id']}:{model_version}",
            "run_id": run_id,
            "model_version": model_version,
            "fact": fact,
        }
        for fact in facts
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """写 JSON Lines，便于后续按行入库或排查。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_segment_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """写 CSV，方便人工快速检查动作片段。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["fact_id", "task_id", "step_id", "label", "start_frame", "end_frame", "start_ms", "end_ms", "confidence", "source"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})

