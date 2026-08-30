"""用真实 YOLO 预测框统一复跑 image_train v1-v4。

脚本先调用 ``common/prepare_yolo_actionmixed.py`` 生成统一 8 类检测结果，再调用各版本
训练入口。所有默认路径和训练轮数集中在文件顶部，命令行参数可覆盖。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


# ============================ 集中参数区 ============================
DEFAULT_SOURCE_DATASET_ROOT = ROOT / "input" / "modelscope" / "lhh010__cleansight-ActionMixed" / "cleansight-ActionMixed"
DEFAULT_GENERATED_DATASET_ROOT = ROOT / "yolo_image_train" / "generated" / "actionmixed_yolo_v03"
DEFAULT_OUTPUT_ROOT = ROOT / "yolo_image_train"
DEFAULT_YOLO_DIR = ROOT / "yolo"
DEFAULT_EXTRACT_DIR = ROOT / "yolo_image_train" / "weights"
DEFAULT_VERSIONS = ["v1", "v2", "v3", "v4"]
DEFAULT_YOLO_CONF = 0.25
DEFAULT_YOLO_IOU = 0.7
DEFAULT_IMAGE_SIZE = 640
DEFAULT_BATCH_SIZE = 8
DEFAULT_CHUNK_SIZE = 256

VERSION_SCRIPTS = {
    "v1": ROOT / "image_train" / "train_image_v1.py",
    "v2": ROOT / "image_train" / "v2" / "train_image_v2.py",
    "v3": ROOT / "image_train" / "v3" / "train_image_v3.py",
    "v4": ROOT / "image_train" / "v4" / "train_image_v4.py",
}

DEFAULT_EPOCHS = {
    "v1": 3,
    "v2": 8,
    "v3": 8,
    "v4": 12,
}


def run_command(command: list[str], cwd: Path, dry_run: bool = False) -> int:
    print(" ".join(f'"{part}"' if " " in part else part for part in command), flush=True)
    if dry_run:
        return 0
    completed = subprocess.run(command, cwd=str(cwd), check=False)
    return int(completed.returncode)


def make_prepare_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "yolo_image_train" / "common" / "prepare_yolo_actionmixed.py"),
        "--source-dataset-root",
        str(args.source_dataset_root),
        "--out-dataset-root",
        str(args.generated_dataset_root),
        "--yolo-dir",
        str(args.yolo_dir),
        "--extract-dir",
        str(args.extract_dir),
        "--conf",
        str(args.yolo_conf),
        "--iou",
        str(args.yolo_iou),
        "--imgsz",
        str(args.imgsz),
        "--batch",
        str(args.batch),
        "--chunk-size",
        str(args.chunk_size),
    ]
    if args.device:
        command.extend(["--device", args.device])
    if args.limit_frames is not None:
        command.extend(["--limit-frames", str(args.limit_frames)])
    if args.force_layout:
        command.append("--force-layout")
    if args.force_extract:
        command.append("--force-extract")
    if args.skip_inference:
        command.append("--skip-inference")
    return command


def make_train_command(version: str, args: argparse.Namespace) -> list[str]:
    out_dir = args.output_root / f"output_{version}"
    report_path = args.output_root / f"yolo_image_train_{version}.md"
    command = [
        sys.executable,
        str(VERSION_SCRIPTS[version]),
        "--dataset-root",
        str(args.generated_dataset_root),
        "--out-dir",
        str(out_dir),
        "--epochs",
        str(args.epochs if args.epochs is not None else DEFAULT_EPOCHS[version]),
        "--report-path",
        str(report_path),
    ]
    if args.models:
        command.extend(["--models", *args.models])
    if version in {"v2", "v3", "v4"} and args.force_cache:
        command.append("--force-cache")
    return command


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replicate image_train v1-v4 using real YOLO detections.")
    parser.add_argument("--source-dataset-root", type=Path, default=DEFAULT_SOURCE_DATASET_ROOT)
    parser.add_argument("--generated-dataset-root", type=Path, default=DEFAULT_GENERATED_DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--yolo-dir", type=Path, default=DEFAULT_YOLO_DIR)
    parser.add_argument("--extract-dir", type=Path, default=DEFAULT_EXTRACT_DIR)
    parser.add_argument("--versions", nargs="+", default=DEFAULT_VERSIONS, choices=sorted(VERSION_SCRIPTS))
    parser.add_argument("--models", nargs="+", default=None, help="optional subset: ms_tcn asformer bigru")
    parser.add_argument("--epochs", type=int, default=None, help="override all version defaults")
    parser.add_argument("--yolo-conf", type=float, default=DEFAULT_YOLO_CONF)
    parser.add_argument("--yolo-iou", type=float, default=DEFAULT_YOLO_IOU)
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--limit-frames", type=int, default=None)
    parser.add_argument("--force-layout", action="store_true")
    parser.add_argument("--force-extract", action="store_true")
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--skip-inference", action="store_true", help="prepare labels/images only; training normally requires frames txt")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = make_parser().parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    summary = {
        "generated_dataset_root": str(args.generated_dataset_root),
        "output_root": str(args.output_root),
        "versions": args.versions,
        "commands": [],
    }

    if not args.skip_prepare:
        command = make_prepare_command(args)
        summary["commands"].append(command)
        code = run_command(command, ROOT, dry_run=args.dry_run)
        if code != 0:
            raise SystemExit(code)

    if args.prepare_only:
        (args.output_root / "yolo_image_train_run_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return

    for version in args.versions:
        command = make_train_command(version, args)
        summary["commands"].append(command)
        code = run_command(command, ROOT, dry_run=args.dry_run)
        if code != 0:
            raise SystemExit(code)

    (args.output_root / "yolo_image_train_run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
