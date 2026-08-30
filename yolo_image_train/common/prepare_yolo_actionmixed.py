"""Build an ActionMixed-style dataset using real YOLO model inference.

The generated dataset keeps ActionMixed temporal action labels and images, but
replaces `frames/{split}/*.txt` with detections produced by the two YOLO models
under the repository `yolo/` folder.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset import ACTIONMIXED_DETECTION_CLASSES

GLOBAL_CLASS_ID = {name: idx for idx, name in ACTIONMIXED_DETECTION_CLASSES.items()}


def safe_link_or_copy(src: Path, dst: Path, force: bool = False) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if force:
            dst.unlink()
        else:
            return "exists"
    try:
        dst.hardlink_to(src)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def mirror_tree_files(src_dir: Path, dst_dir: Path, patterns: tuple[str, ...], force: bool = False) -> dict[str, int]:
    stats = {"exists": 0, "hardlink": 0, "copy": 0}
    for pattern in patterns:
        for src in src_dir.rglob(pattern):
            if not src.is_file():
                continue
            rel = src.relative_to(src_dir)
            mode = safe_link_or_copy(src, dst_dir / rel, force=force)
            stats[mode] = stats.get(mode, 0) + 1
    return stats


def write_detection_yaml(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["nc: 8", "names:"]
    for idx, name in sorted(ACTIONMIXED_DETECTION_CLASSES.items()):
        lines.append(f"  {idx}: {name}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def find_zip_member(zf: zipfile.ZipFile, suffix: str) -> str | None:
    matches = [name for name in zf.namelist() if name.endswith(suffix)]
    return sorted(matches)[0] if matches else None


def extract_yolo_archives(yolo_dir: Path, extract_dir: Path, force: bool = False) -> dict[str, dict[str, Any]]:
    extract_dir.mkdir(parents=True, exist_ok=True)
    models: dict[str, dict[str, Any]] = {}
    for zip_path in sorted(yolo_dir.glob("*.zip")):
        key = "large" if "large" in zip_path.stem else "small" if "small" in zip_path.stem else zip_path.stem
        target = extract_dir / zip_path.stem
        if force and target.exists():
            shutil.rmtree(target)
        if not target.exists():
            target.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(target)

        pt_files = sorted(target.rglob("*.pt"))
        meta_files = sorted(target.rglob("*.pt.meta.json"))
        if not pt_files:
            raise RuntimeError(f"No .pt weights found in {zip_path}")

        preferred = [p for p in pt_files if p.name.startswith("clean-") and p.name.endswith(".pt")]
        weight_path = preferred[0] if preferred else [p for p in pt_files if p.name == "best.pt"][0] if any(p.name == "best.pt" for p in pt_files) else pt_files[0]
        meta_path = meta_files[0] if meta_files else None
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path else {}
        names = {int(k): v for k, v in (meta.get("names") or {}).items()}
        models[key] = {
            "archive": str(zip_path),
            "weight": str(weight_path),
            "meta": str(meta_path) if meta_path else None,
            "names": names,
        }
    required = {"large", "small"}
    missing = required - set(models)
    if missing:
        raise RuntimeError(f"Missing YOLO archives for: {sorted(missing)}")
    return models


def image_to_frame_txt_name(image_path: Path) -> str:
    return image_path.with_suffix(".txt").name


def load_ultralytics():
    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency `ultralytics`. Install it before running YOLO inference, "
            "for example: python -m pip install ultralytics"
        ) from exc
    return YOLO


def rows_from_result(model: Any, result: Any, local_names: dict[int, str]) -> list[tuple[int, float, float, float, float, float]]:
    rows: list[tuple[int, float, float, float, float, float]] = []
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return rows
    xywhn = boxes.xywhn.detach().cpu().numpy()
    cls = boxes.cls.detach().cpu().numpy()
    score = boxes.conf.detach().cpu().numpy()
    for box, local_class, confidence in zip(xywhn, cls, score):
        local_id = int(local_class)
        name = local_names.get(local_id)
        if name is None:
            name = str(getattr(model, "names", {}).get(local_id, ""))
        global_id = GLOBAL_CLASS_ID.get(name)
        if global_id is None:
            continue
        cx, cy, width, height = [float(x) for x in box]
        rows.append((global_id, cx, cy, width, height, float(confidence)))
    return rows


def predict_one_model(model: Any, image_path: Path, local_names: dict[int, str], conf: float, iou: float, imgsz: int, device: str | None) -> list[tuple[int, float, float, float, float, float]]:
    kwargs: dict[str, Any] = {
        "source": str(image_path),
        "conf": conf,
        "iou": iou,
        "imgsz": imgsz,
        "verbose": False,
    }
    if device:
        kwargs["device"] = device
    results = model.predict(**kwargs)
    for result in results:
        return rows_from_result(model, result, local_names)
    return []


def chunked(values: list[Path], size: int) -> list[list[Path]]:
    return [values[idx : idx + size] for idx in range(0, len(values), size)]


def predict_model_batch(
    model: Any,
    image_paths: list[Path],
    local_names: dict[int, str],
    conf: float,
    iou: float,
    imgsz: int,
    device: str | None,
    batch: int,
) -> dict[Path, list[tuple[int, float, float, float, float, float]]]:
    kwargs: dict[str, Any] = {
        "source": [str(path) for path in image_paths],
        "conf": conf,
        "iou": iou,
        "imgsz": imgsz,
        "batch": batch,
        "verbose": False,
    }
    if device:
        kwargs["device"] = device
    results = model.predict(**kwargs)
    out: dict[Path, list[tuple[int, float, float, float, float, float]]] = {}
    for image_path, result in zip(image_paths, results):
        out[image_path] = rows_from_result(model, result, local_names)
    return out


def run_yolo_inference(args: argparse.Namespace, models_info: dict[str, dict[str, Any]]) -> dict[str, Any]:
    YOLO = load_ultralytics()
    loaded = {
        key: YOLO(info["weight"])
        for key, info in models_info.items()
    }
    counts: dict[str, Any] = {}
    for split in args.splits:
        image_dir = args.source_dataset_root / "images" / split
        frame_dir = args.out_dataset_root / "frames" / split
        frame_dir.mkdir(parents=True, exist_ok=True)
        images = sorted(image_dir.glob("*.jpg"))
        if args.limit_frames is not None:
            images = images[: args.limit_frames]
        split_stats = {"images": 0, "detections": 0, "empty_frames": 0}
        split_rows = {image_path: [] for image_path in images}
        for chunk in chunked(images, args.chunk_size):
            for key, model in loaded.items():
                batch_rows = predict_model_batch(
                    model,
                    chunk,
                    models_info[key]["names"],
                    conf=args.conf,
                    iou=args.iou,
                    imgsz=args.imgsz,
                    device=args.device,
                    batch=args.batch,
                )
                for image_path, rows in batch_rows.items():
                    split_rows[image_path].extend(rows)
        for image_path in images:
            rows = split_rows[image_path]
            rows.sort(key=lambda row: (row[0], -row[5]))
            out_path = frame_dir / image_to_frame_txt_name(image_path)
            out_path.write_text(
                "\n".join(
                    f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {score:.6f}"
                    for cls, cx, cy, w, h, score in rows
                )
                + ("\n" if rows else ""),
                encoding="utf-8",
            )
            split_stats["images"] += 1
            split_stats["detections"] += len(rows)
            if not rows:
                split_stats["empty_frames"] += 1
        counts[split] = split_stats
    return counts


def prepare_layout(args: argparse.Namespace) -> dict[str, Any]:
    args.out_dataset_root.mkdir(parents=True, exist_ok=True)
    label_stats = mirror_tree_files(args.source_dataset_root / "labels", args.out_dataset_root / "labels", ("*.txt", "*.yaml"), force=args.force_layout)
    image_stats = mirror_tree_files(args.source_dataset_root / "images", args.out_dataset_root / "images", ("*.jpg",), force=args.force_layout)
    write_detection_yaml(args.out_dataset_root / "frames" / "data.yaml")
    return {
        "labels": label_stats,
        "images": image_stats,
        "frames_yaml": str(args.out_dataset_root / "frames" / "data.yaml"),
    }


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate ActionMixed frames from real YOLO inference.")
    parser.add_argument("--source-dataset-root", type=Path, default=ROOT / "input" / "modelscope" / "lhh010__cleansight-ActionMixed" / "cleansight-ActionMixed")
    parser.add_argument("--out-dataset-root", type=Path, default=ROOT / "yolo_image_train" / "generated" / "actionmixed_yolo_v03")
    parser.add_argument("--yolo-dir", type=Path, default=ROOT / "yolo")
    parser.add_argument("--extract-dir", type=Path, default=ROOT / "yolo_image_train" / "weights")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--limit-frames", type=int, default=None)
    parser.add_argument("--force-layout", action="store_true")
    parser.add_argument("--force-extract", action="store_true")
    parser.add_argument("--skip-inference", action="store_true", help="prepare labels/images/layout only")
    return parser


def main() -> None:
    args = make_parser().parse_args()
    layout = prepare_layout(args)
    models_info = extract_yolo_archives(args.yolo_dir, args.extract_dir, force=args.force_extract)
    inference = None if args.skip_inference else run_yolo_inference(args, models_info)
    summary = {
        "status": "layout_only" if args.skip_inference else "completed",
        "source_dataset_root": str(args.source_dataset_root),
        "out_dataset_root": str(args.out_dataset_root),
        "models": models_info,
        "layout": layout,
        "inference": inference,
        "yolo_conf": args.conf,
        "yolo_iou": args.iou,
        "imgsz": args.imgsz,
        "splits": args.splits,
    }
    summary_path = args.out_dataset_root / "yolo_generation_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": summary["status"], "summary": str(summary_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
