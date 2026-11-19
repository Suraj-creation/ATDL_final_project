"""Run validation-only model-transfer experiments for the ATDL project.

The first stage compares COCO-pretrained YOLO checkpoints on the unchanged
Phase 2 training/validation split. It deliberately does not read the test set.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import shutil
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]
MINORITY_CLASSES = ["crazing", "pitted_surface", "rolled-in_scale", "scratches"]
SEEDS = [42, 43, 44]
MODEL_WEIGHTS = {
    "yolov8n": ROOT / "yolov8n.pt",
    "yolo11n": ROOT / "yolo11n.pt",
    "yolo26n": ROOT / "yolo26n.pt",
    "yolo11s": ROOT / "yolo11s.pt",
}
EPOCHS = 50
IMAGE_SIZE = 640
BATCH_SIZE = 16
PATIENCE = 12
DATA_YAML = ROOT / "configs" / "neu_det_yolo.yaml"
SPLIT_MANIFEST = ROOT / "data" / "splits" / "phase2_image_split.csv"
CONVERSION_MANIFEST = ROOT / "results" / "phase2" / "yolo_conversion_manifest.csv"
OUTPUT_ROOT = ROOT / "outputs" / "checkpoints" / "phase8" / "model_transfer"
VALIDATION_ROOT = ROOT / "outputs" / "logs" / "phase8" / "model_transfer"
RESULTS_ROOT = ROOT / "results" / "comprehensive"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def load_environment() -> tuple[Any, Any, dict[str, Any]]:
    import torch
    import ultralytics
    from ultralytics import YOLO

    if not torch.cuda.is_available():
        raise RuntimeError("The comprehensive training stage requires the configured CUDA GPU.")
    env = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "ultralytics": ultralytics.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
    }
    return torch, YOLO, env


def validate_inputs() -> None:
    for path in (DATA_YAML, SPLIT_MANIFEST, CONVERSION_MANIFEST):
        if not path.is_file():
            raise FileNotFoundError(f"Required Phase 2 input is missing: {path}")
    for name, path in MODEL_WEIGHTS.items():
        if not path.is_file():
            raise FileNotFoundError(
                f"COCO-pretrained weights for {name} are missing at {path}. "
                "Load the official Ultralytics checkpoint once to download it."
            )
    split_count = sum(1 for _ in csv.DictReader(SPLIT_MANIFEST.open(encoding="utf-8")))
    if split_count != 1800:
        raise ValueError(f"Expected 1,800 source rows in the grouped split manifest, found {split_count}.")


def architecture_signature(model_name: str, seed: int) -> dict[str, Any]:
    return {
        "stage": "pretrained_model_transfer",
        "model_name": model_name,
        "pretrained_checkpoint": relative(MODEL_WEIGHTS[model_name]),
        "pretrained_sha256": sha256_file(MODEL_WEIGHTS[model_name]),
        "data_yaml_sha256": sha256_file(DATA_YAML),
        "split_manifest_sha256": sha256_file(SPLIT_MANIFEST),
        "conversion_manifest_sha256": sha256_file(CONVERSION_MANIFEST),
        "epochs": EPOCHS,
        "imgsz": IMAGE_SIZE,
        "batch": BATCH_SIZE,
        "patience": PATIENCE,
        "optimizer": "auto",
        "seed": seed,
        "workers": 4,
        "deterministic": True,
    }


def extract_metrics(result: Any, model_name: str, seed: int, checkpoint: Path,
                    pretrained_sha256: str, elapsed_seconds: float,
                    params: int, epochs_completed: int | None) -> dict[str, Any]:
    box = result.box
    per_class: dict[str, dict[str, float | None]] = {}
    for position, class_id in enumerate(list(getattr(box, "ap_class_index", []))):
        cid = int(class_id)
        if cid not in range(len(CLASS_NAMES)):
            continue
        name = CLASS_NAMES[cid]
        precision = float(box.p[position]) if len(box.p) > position else None
        recall = float(box.r[position]) if len(box.r) > position else None
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision is not None and recall is not None and precision + recall > 0
            else 0.0
        )
        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "ap50": float(box.ap50[position]) if len(box.ap50) > position else None,
            "ap50_95": float(box.ap[position].mean()) if len(box.ap) > position else None,
        }

    missing = sorted(set(CLASS_NAMES) - set(per_class))
    if missing:
        raise ValueError(f"Validation metrics are missing classes for {model_name}, seed {seed}: {missing}")
    global_metrics = {str(k): float(v) for k, v in result.results_dict.items()}
    global_p = global_metrics.get("metrics/precision(B)")
    global_r = global_metrics.get("metrics/recall(B)")
    global_f1 = (
        2 * global_p * global_r / (global_p + global_r)
        if global_p is not None and global_r is not None and global_p + global_r > 0
        else 0.0
    )
    global_metrics["metrics/F1(B)"] = global_f1
    tail_ap50 = [per_class[name]["ap50"] for name in MINORITY_CLASSES]
    tail_recall = [per_class[name]["recall"] for name in MINORITY_CLASSES]
    tail_f1 = [per_class[name]["f1"] for name in MINORITY_CLASSES]
    return {
        "experiment": f"{model_name}_seed{seed}",
        "stage": "pretrained_model_transfer",
        "model_name": model_name,
        "pretrained_source": "Ultralytics COCO pretrained checkpoint",
        "pretrained_checkpoint": relative(MODEL_WEIGHTS[model_name]),
        "pretrained_sha256": pretrained_sha256,
        "seed": seed,
        "epochs_requested": EPOCHS,
        "epochs_completed": epochs_completed,
        "imgsz": IMAGE_SIZE,
        "batch": BATCH_SIZE,
        "optimizer": "auto",
        "dataset_split": "fixed Phase 2 70/15/15 image-group split",
        "train_images": 1257,
        "validation_images": 266,
        "checkpoint": relative(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "parameters": params,
        "training_seconds": round(elapsed_seconds, 3),
        "validation_speed_ms_per_image": {
            str(k): float(v) for k, v in getattr(result, "speed", {}).items()
        },
        "global_metrics": global_metrics,
        "macro_f1": statistics.mean(float(per_class[name]["f1"]) for name in CLASS_NAMES),
        "minority_macro_ap50": statistics.mean(float(v) for v in tail_ap50),
        "minority_macro_recall": statistics.mean(float(v) for v in tail_recall),
        "minority_macro_f1": statistics.mean(float(v) for v in tail_f1),
        "per_class": per_class,
        "evaluated_split": "validation",
        "selected_on": "validation only",
    }


def _baseline_seed42_record() -> dict[str, Any]:
    metrics_path = ROOT / "results" / "phase3_baseline_metrics.json"
    checkpoint = ROOT / "outputs" / "checkpoints" / "phase3_baseline" / "E0_baseline" / "weights" / "best.pt"
    if not metrics_path.is_file() or not checkpoint.is_file():
        raise FileNotFoundError("The executed YOLOv8n seed 42 baseline is required for the transfer comparison.")
    previous = json.loads(metrics_path.read_text(encoding="utf-8"))
    pre_sha = sha256_file(MODEL_WEIGHTS["yolov8n"])
    per_class = {item["class_name"]: {
        key: item.get(key) for key in ("precision", "recall", "f1", "ap50", "ap50_95")
    } for item in previous.get("per_class", [])}
    result = {
        "experiment": "yolov8n_seed42",
        "stage": "pretrained_model_transfer",
        "model_name": "yolov8n",
        "pretrained_source": "Ultralytics COCO pretrained checkpoint",
        "pretrained_checkpoint": relative(MODEL_WEIGHTS["yolov8n"]),
        "pretrained_sha256": pre_sha,
        "seed": 42,
        "epochs_requested": 50,
        "epochs_completed": 50,
        "imgsz": 640,
        "batch": 16,
        "optimizer": "auto",
        "dataset_split": "fixed Phase 2 70/15/15 image-group split",
        "train_images": 1257,
        "validation_images": 266,
        "checkpoint": relative(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "parameters": 3157200,
        "training_seconds": 227.095,
        "validation_speed_ms_per_image": previous.get("inference_speed_ms_per_image", {}),
        "global_metrics": previous.get("metrics", {}),
        "macro_f1": previous.get("macro_F1"),
        "per_class": per_class,
        "evaluated_split": "validation",
        "selected_on": "validation only",
        "historical_run_reused": True,
    }
    global_p = result["global_metrics"].get("metrics/precision(B)")
    global_r = result["global_metrics"].get("metrics/recall(B)")
    result["global_metrics"]["metrics/F1(B)"] = (
        2 * global_p * global_r / (global_p + global_r)
        if global_p is not None and global_r is not None and global_p + global_r > 0
        else 0.0
    )
    result["minority_macro_ap50"] = statistics.mean(per_class[n]["ap50"] for n in MINORITY_CLASSES)
    result["minority_macro_recall"] = statistics.mean(per_class[n]["recall"] for n in MINORITY_CLASSES)
    result["minority_macro_f1"] = statistics.mean(per_class[n]["f1"] for n in MINORITY_CLASSES)
    return result


def _read_epoch_count(run_dir: Path) -> int | None:
    path = run_dir / "results.csv"
    if not path.is_file():
        return None
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    # Ultralytics writes completed epochs as 1..N in results.csv.
    return int(float(rows[-1]["epoch"])) if rows else None


def _read_elapsed(run_dir: Path, fallback: float) -> float:
    path = run_dir / "results.csv"
    if not path.is_file():
        return fallback
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if rows and "time" in rows[-1]:
        return float(rows[-1]["time"])
    return fallback


def run_architecture(model_name: str, seed: int, YOLO: Any, torch: Any,
                     environment: dict[str, Any]) -> dict[str, Any]:
    if model_name == "yolov8n" and seed == 42:
        record = _baseline_seed42_record()
    else:
        run_dir = OUTPUT_ROOT / model_name / f"seed_{seed}"
        checkpoint = run_dir / "weights" / "best.pt"
        signature_path = run_dir / "run_signature.json"
        metrics_path = RESULTS_ROOT / "runs" / f"{model_name}_seed{seed}.json"
        signature = architecture_signature(model_name, seed)
        if checkpoint.is_file() and signature_path.is_file():
            saved_signature = json.loads(signature_path.read_text(encoding="utf-8"))
            if saved_signature != signature:
                raise RuntimeError(
                    f"Stale checkpoint for {model_name} seed {seed}. Preserve it, then rerun with --force."
                )
        else:
            run_dir.mkdir(parents=True, exist_ok=True)
            started = time.perf_counter()
            model = YOLO(str(MODEL_WEIGHTS[model_name]))
            model.train(
                data=str(DATA_YAML), epochs=EPOCHS, imgsz=IMAGE_SIZE, batch=BATCH_SIZE,
                patience=PATIENCE, seed=seed, project=str(run_dir.parent), name=run_dir.name,
                exist_ok=True, device=0, workers=4, deterministic=True,
                optimizer="auto", plots=True, save=True, val=True,
            )
            elapsed = time.perf_counter() - started
            if not checkpoint.is_file():
                raise FileNotFoundError(f"Ultralytics training did not create {checkpoint}")
            signature_path.write_text(json.dumps(signature, indent=2) + "\n", encoding="utf-8")
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if not metrics_path.is_file():
            started = time.perf_counter()
            evaluated_model = YOLO(str(checkpoint))
            result = evaluated_model.val(
                data=str(DATA_YAML), split="val", imgsz=IMAGE_SIZE, batch=BATCH_SIZE,
                project=str(VALIDATION_ROOT), name=f"{model_name}_seed{seed}",
                exist_ok=True, device=0, plots=True, save_txt=False,
            )
            elapsed = _read_elapsed(run_dir, time.perf_counter() - started)
            params = sum(parameter.numel() for parameter in evaluated_model.model.parameters())
            record = extract_metrics(
                result, model_name, seed, checkpoint, signature["pretrained_sha256"],
                elapsed, params, _read_epoch_count(run_dir),
            )
            record["environment"] = environment
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            metrics_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
            del evaluated_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            record = json.loads(metrics_path.read_text(encoding="utf-8"))
            if record.get("checkpoint_sha256") != sha256_file(checkpoint):
                raise ValueError(f"Saved metrics do not match checkpoint: {checkpoint}")
            record["epochs_completed"] = _read_epoch_count(run_dir)
            record["training_seconds"] = round(_read_elapsed(run_dir, record.get("training_seconds", 0.0)), 3)
            record["checkpoint_bytes"] = checkpoint.stat().st_size
            metrics_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    record["environment"] = environment
    record["metrics_path"] = relative(RESULTS_ROOT / "runs" / f"{model_name}_seed{seed}.json")
    return record


def write_architecture_outputs(records: list[dict[str, Any]]) -> None:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for record in records:
        all_rows.append({
            "model_name": record["model_name"],
            "seed": record["seed"],
            "pretrained_sha256": record["pretrained_sha256"],
            "checkpoint": record["checkpoint"],
            "checkpoint_sha256": record["checkpoint_sha256"],
            "checkpoint_bytes": record["checkpoint_bytes"],
            "parameters": record["parameters"],
            "epochs_completed": record["epochs_completed"],
            "training_seconds": record["training_seconds"],
            "precision": record["global_metrics"].get("metrics/precision(B)"),
            "recall": record["global_metrics"].get("metrics/recall(B)"),
            "validation_inference_ms_per_image": record.get("validation_speed_ms_per_image", {}).get("inference"),
            "mAP50": record["global_metrics"].get("metrics/mAP50(B)"),
            "mAP50_95": record["global_metrics"].get("metrics/mAP50-95(B)"),
            "global_F1": record["global_metrics"].get("metrics/F1(B)"),
            "macro_F1": record["macro_f1"],
            "minority_macro_AP50": record["minority_macro_ap50"],
            "minority_macro_recall": record["minority_macro_recall"],
            "minority_macro_F1": record["minority_macro_f1"],
            "AP50_crazing": record["per_class"]["crazing"]["ap50"],
            "AP50_pitted_surface": record["per_class"]["pitted_surface"]["ap50"],
            "AP50_rolled-in_scale": record["per_class"]["rolled-in_scale"]["ap50"],
            "AP50_scratches": record["per_class"]["scratches"]["ap50"],
            "speed_preprocess_ms": record.get("validation_speed_ms_per_image", {}).get("preprocess"),
            "speed_inference_ms": record.get("validation_speed_ms_per_image", {}).get("inference"),
            "speed_postprocess_ms": record.get("validation_speed_ms_per_image", {}).get("postprocess"),
            "evaluated_split": record["evaluated_split"],
        })
    with (RESULTS_ROOT / "model_transfer_runs.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=all_rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(all_rows)
    with (RESULTS_ROOT / "model_transfer_runs.json").open("w", encoding="utf-8") as stream:
        json.dump(records, stream, indent=2)
        stream.write("\n")

    summary_rows = []
    for model_name in MODEL_WEIGHTS:
        model_rows = [row for row in all_rows if row["model_name"] == model_name]
        if not model_rows:
            continue
        metric_names = [
            "precision", "recall", "mAP50", "mAP50_95", "global_F1", "macro_F1",
            "minority_macro_AP50", "minority_macro_recall", "minority_macro_F1",
            "parameters", "checkpoint_bytes", "training_seconds", "validation_inference_ms_per_image",
        ]
        summary = {"model_name": model_name, "seeds": ",".join(str(row["seed"]) for row in model_rows)}
        for metric in metric_names:
            values = [float(row[metric]) for row in model_rows if row[metric] is not None]
            summary[f"{metric}_mean"] = statistics.mean(values) if values else None
            summary[f"{metric}_sd"] = statistics.stdev(values) if len(values) > 1 else 0.0
        summary_rows.append(summary)
    summary_rows.sort(
        key=lambda row: (row.get("minority_macro_AP50_mean") or -math.inf,
                         row.get("mAP50_mean") or -math.inf), reverse=True
    )
    with (RESULTS_ROOT / "model_transfer_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary_rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary_rows)
    if summary_rows:
        winner = summary_rows[0]["model_name"]
        selection = {
            "selection_split": "validation",
            "selection_rule": "highest mean minority macro AP50 across seeds 42, 43, 44; tie-break by mean overall mAP50",
            "minority_classes": MINORITY_CLASSES,
            "selected_model": winner,
            "selected_pretrained_checkpoint": relative(MODEL_WEIGHTS[winner]),
            "selected_pretrained_sha256": sha256_file(MODEL_WEIGHTS[winner]),
            "mean_minority_macro_AP50": summary_rows[0]["minority_macro_AP50_mean"],
            "mean_validation_mAP50": summary_rows[0]["mAP50_mean"],
            "test_split_read": False,
        }
        (RESULTS_ROOT / "model_transfer_selection.json").write_text(
            json.dumps(selection, indent=2) + "\n", encoding="utf-8"
        )


def run_architecture_stage(force: bool = False) -> list[dict[str, Any]]:
    validate_inputs()
    torch, YOLO, environment = load_environment()
    records = []
    for model_name in MODEL_WEIGHTS:
        for seed in SEEDS:
            print(f"\n[{datetime.now(timezone.utc).isoformat()}] {model_name} seed {seed}", flush=True)
            metric_path = RESULTS_ROOT / "runs" / f"{model_name}_seed{seed}.json"
            run_dir = OUTPUT_ROOT / model_name / f"seed_{seed}"
            if force:
                if not (model_name == "yolov8n" and seed == 42):
                    if run_dir.exists():
                        shutil.rmtree(run_dir)
                    if metric_path.exists():
                        metric_path.unlink()
            record = run_architecture(model_name, seed, YOLO, torch, environment)
            records.append(record)
            print(
                f"completed {model_name} seed {seed}: "
                f"tail AP50={record['minority_macro_ap50']:.4f}, "
                f"mAP50={record['global_metrics'].get('metrics/mAP50(B)', float('nan')):.4f}",
                flush=True,
            )
            write_architecture_outputs(records if len(records) == 12 else _load_all_architecture_records())
    records = _load_all_architecture_records()
    write_architecture_outputs(records)
    return records


def _load_all_architecture_records() -> list[dict[str, Any]]:
    records = []
    for model_name in MODEL_WEIGHTS:
        for seed in SEEDS:
            if model_name == "yolov8n" and seed == 42:
                record = _baseline_seed42_record()
            else:
                path = RESULTS_ROOT / "runs" / f"{model_name}_seed{seed}.json"
                if not path.is_file():
                    continue
                record = json.loads(path.read_text(encoding="utf-8"))
            records.append(record)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("architectures",), default="architectures")
    parser.add_argument("--force", action="store_true", help="remove and retrain generated architecture runs; the historical seed-42 baseline is retained")
    args = parser.parse_args()
    if args.stage == "architectures":
        records = run_architecture_stage(force=args.force)
        selected = (RESULTS_ROOT / "model_transfer_selection.json").read_text(encoding="utf-8")
        print(f"\nSaved {len(records)} validation records.")
        print(selected)


if __name__ == "__main__":
    main()
