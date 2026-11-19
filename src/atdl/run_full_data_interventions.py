"""Run full-data, equal-exposure imbalance interventions after model selection.

This stage uses validation only. It preserves every original training image in
all treatments and adds deterministic image-level repeat sampling and/or
label-correct geometric augmentation for tail-class images.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import statistics
import time
from pathlib import Path
from typing import Any

from PIL import Image, ImageEnhance

ROOT = Path(__file__).resolve().parents[2]
CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
TAIL = ["crazing", "pitted_surface", "rolled-in_scale", "scratches"]
TAIL_IDS = {CLASSES.index(name) for name in TAIL}
SEEDS = [42, 43, 44]
BASE_YAML = ROOT / "configs" / "neu_det_yolo.yaml"
BASE_TRAIN_IMAGES = ROOT / "data/processed/images/train"
BASE_TRAIN_LABELS = ROOT / "data/processed/labels/train"
VAL_IMAGES = ROOT / "data/processed/images/val"
VAL_LABELS = ROOT / "data/processed/labels/val"
TEST_IMAGES = ROOT / "data/processed/images/test"
TEST_LABELS = ROOT / "data/processed/labels/test"
DATA_ROOT = ROOT / "data/phase8_generated"
RUN_ROOT = ROOT / "outputs/checkpoints/phase8/interventions"
VAL_ROOT = ROOT / "outputs/logs/phase8/interventions"
RESULTS = ROOT / "results/comprehensive"
TARGET_EXPOSURES = 1257 * 50


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def read_labels(path: Path) -> list[list[float]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[list[float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = [float(part) for part in line.split()]
        if len(fields) != 5 or fields[0] != int(fields[0]):
            raise ValueError(f"Invalid YOLO label row in {path}: {line}")
        rows.append(fields)
    return rows


def link_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        target.unlink()
    try:
        target.symlink_to(source.resolve())
    except OSError:
        shutil.copy2(source, target)


def class_instance_counts(label_paths: list[Path]) -> list[int]:
    counts = [0] * len(CLASSES)
    for path in label_paths:
        for row in read_labels(path):
            counts[int(row[0])] += 1
    return counts


def tail_image_contains(rows: list[list[float]]) -> bool:
    return any(int(row[0]) in TAIL_IDS for row in rows)


def repeat_rate_for_image(rows: list[list[float]], rates: list[float]) -> float:
    found = [rates[int(row[0])] for row in rows if int(row[0]) in TAIL_IDS]
    return max(found, default=1.0)


def stochastic_repeats(rate: float, seed: int, image_name: str) -> int:
    base = math.floor(rate)
    fractional = rate - base
    digest = hashlib.sha256(f"{seed}:{image_name}".encode()).digest()
    coin = int.from_bytes(digest[:8], "big") / 2**64
    return min(3, base + int(coin < fractional))


def transform_labels(rows: list[list[float]], operation: str) -> list[list[float]]:
    transformed = []
    for class_id, x, y, w, h in rows:
        if operation == "horizontal_flip":
            x = 1 - x
        elif operation == "vertical_flip":
            y = 1 - y
        elif operation == "rotate_90_clockwise":
            x, y, w, h = 1 - y, x, h, w
        else:
            raise ValueError(operation)
        transformed.append([class_id, x, y, w, h])
    return transformed


def make_augmented(source_image: Path, source_label: Path, out_image: Path,
                   out_label: Path, operation: str, brightness: float) -> None:
    with Image.open(source_image) as opened:
        image = opened.convert("RGB")
    if operation == "horizontal_flip":
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    elif operation == "vertical_flip":
        image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    elif operation == "rotate_90_clockwise":
        image = image.transpose(Image.Transpose.ROTATE_270)
    image = ImageEnhance.Brightness(image).enhance(brightness)
    image.save(out_image, quality=95, optimize=True)
    rows = transform_labels(read_labels(source_label), operation)
    out_label.write_text("".join(" ".join(f"{v:.8f}" if i else str(int(v)) for i, v in enumerate(row)) + "\n" for row in rows), encoding="utf-8")


def save_manifest_provenance(destination: Path, treatment: str) -> None:
    """Keep the compact generated manifest and recipe summary in Git-tracked results."""
    provenance = RESULTS / "dataset_generation"
    provenance.mkdir(parents=True, exist_ok=True)
    shutil.copy2(destination / "manifest.csv", provenance / f"{treatment}_manifest.csv")
    shutil.copy2(destination / "dataset_summary.json", provenance / f"{treatment}_summary.json")


def add_expanded_class_counts(destination: Path, summary: dict[str, Any]) -> dict[str, Any]:
    """Count class-bearing image presentations and boxes after repeats/augmentation."""
    if summary.get("expanded_counts_deduplicated") is True:
        return summary
    image_counts = [0] * len(CLASSES)
    instance_counts = [0] * len(CLASSES)
    with (destination / "manifest.csv").open(newline="", encoding="utf-8") as stream:
        manifest_rows = list(csv.DictReader(stream))
    for entry in manifest_rows:
        source_rows = read_labels(Path(entry["source_label"]))
        rows = list({tuple(round(value, 8) for value in row) for row in source_rows})
        present = set()
        for row in rows:
            class_id = int(row[0])
            instance_counts[class_id] += 1
            present.add(class_id)
        for class_id in present:
            image_counts[class_id] += 1
    summary["expanded_train_images_with_class"] = dict(zip(CLASSES, image_counts))
    summary["expanded_train_instances_by_class"] = dict(zip(CLASSES, instance_counts))
    summary["expanded_manifest_rows"] = len(manifest_rows)
    summary["expanded_counts_deduplicated"] = True
    summary_path = destination / "dataset_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def make_treatment(treatment: str, force: bool = False) -> dict[str, Any]:
    if treatment not in {"full_repeat", "full_targeted_aug", "full_combined"}:
        raise ValueError(f"Unknown intervention: {treatment}")
    destination = DATA_ROOT / treatment
    manifest_path = destination / "manifest.csv"
    if manifest_path.is_file() and not force:
        summary = json.loads((destination / "dataset_summary.json").read_text(encoding="utf-8"))
        summary = add_expanded_class_counts(destination, summary)
        save_manifest_provenance(destination, treatment)
        return summary
    if destination.exists():
        shutil.rmtree(destination)
    image_out = destination / "images/train"
    label_out = destination / "labels/train"
    image_out.mkdir(parents=True)
    label_out.mkdir(parents=True)

    images = sorted(p for p in BASE_TRAIN_IMAGES.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if len(images) != 1257:
        raise ValueError(f"Expected 1,257 Phase 2 training images, found {len(images)}")
    label_paths = [BASE_TRAIN_LABELS / f"{p.stem}.txt" for p in images]
    counts = class_instance_counts(label_paths)
    max_count = max(counts)
    # LVIS-style inverse-frequency repeat rate: square-root attenuation and an
    # image-level maximum over tail classes. Frequency uses box instances.
    rates = [max(1.0, math.sqrt(max_count / max(1, count))) for count in counts]
    image_counts = [0] * len(CLASSES)
    manifest: list[dict[str, Any]] = []
    repeat_enabled = treatment in {"full_repeat", "full_combined"}
    aug_enabled = treatment in {"full_targeted_aug", "full_combined"}
    for image in images:
        label = BASE_TRAIN_LABELS / f"{image.stem}.txt"
        rows = read_labels(label)
        for row in rows:
            image_counts[int(row[0])] += 1
        base_image = image_out / image.name
        base_label = label_out / label.name
        link_file(image, base_image)
        link_file(label, base_label)
        manifest.append({"image": str(base_image), "label": str(base_label), "source_image": str(image), "source_label": str(label), "operation": "original"})

        if repeat_enabled:
            repetitions = stochastic_repeats(repeat_rate_for_image(rows, rates), 1207, image.name)
            for repeat_no in range(1, repetitions):
                repeat_image = image_out / f"{image.stem}_repeat{repeat_no}{image.suffix}"
                repeat_label = label_out / f"{image.stem}_repeat{repeat_no}.txt"
                link_file(image, repeat_image)
                link_file(label, repeat_label)
                manifest.append({"image": str(repeat_image), "label": str(repeat_label), "source_image": str(image), "source_label": str(label), "operation": f"repeat_{repeat_no}"})

        if aug_enabled and tail_image_contains(rows):
            # Two fixed, label-preserving transforms per tail-containing image.
            for aug_no, operation in enumerate(("horizontal_flip", "rotate_90_clockwise"), start=1):
                aug_image = image_out / f"{image.stem}_tailaug{aug_no}.jpg"
                aug_label = label_out / f"{image.stem}_tailaug{aug_no}.txt"
                make_augmented(image, label, aug_image, aug_label, operation, 0.92 if aug_no == 1 else 1.08)
                manifest.append({"image": str(aug_image), "label": str(aug_label), "source_image": str(image), "source_label": str(label), "operation": f"{operation}_brightness_{0.92 if aug_no == 1 else 1.08}"})

    manifest_fields = ["image", "label", "source_image", "source_label", "operation"]
    with manifest_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=manifest_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(manifest)

    dataset_yaml = destination / "data.yaml"
    yaml_content = (
        f"path: {destination.resolve().as_posix()}\n"
        "train: images/train\n"
        f"val: {VAL_IMAGES.resolve().as_posix()}\n"
        f"test: {TEST_IMAGES.resolve().as_posix()}\n"
        "names:\n" + "".join(f"  {i}: {name}\n" for i, name in enumerate(CLASSES))
    )
    dataset_yaml.write_text(yaml_content, encoding="utf-8")
    summary = {
        "treatment": treatment,
        "base_train_images": len(images),
        "expanded_train_images": len(manifest),
        "train_image_counts_by_class": dict(zip(CLASSES, image_counts)),
        "train_instance_counts_by_class": dict(zip(CLASSES, counts)),
        "repeat_rate_by_class": dict(zip(CLASSES, [round(value, 5) for value in rates])),
        "tail_classes": TAIL,
        "repetition_formula": "r_c=max(1,sqrt(max_class_box_count/class_box_count)); image repeat rate=max r_c over tail classes present; deterministic stochastic rounding, max 3 copies",
        "augmentation": "All originals retained; each tail-containing image adds horizontal flip + brightness 0.92 and 90-degree clockwise rotation + brightness 1.08; boxes transformed analytically.",
        "dataset_yaml": relative(dataset_yaml),
        "manifest": relative(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "held_out_splits_modified": False,
    }
    (destination / "dataset_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    summary = add_expanded_class_counts(destination, summary)
    save_manifest_provenance(destination, treatment)
    return summary


def get_environment() -> tuple[Any, Any, dict[str, Any]]:
    import torch
    import ultralytics
    from ultralytics import YOLO
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for this study.")
    return torch, YOLO, {
        "python": __import__("platform").python_version(),
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "gpu": torch.cuda.get_device_name(0),
    }


def extract_record(treatment: str, seed: int, model_name: str, checkpoint: Path,
                   pretrained: Path, dataset_summary: dict[str, Any], result: Any,
                   params: int, elapsed: float, epochs: int, environment: dict[str, Any]) -> dict[str, Any]:
    per_class: dict[str, dict[str, float]] = {}
    box = result.box
    for i, class_id in enumerate(list(box.ap_class_index)):
        name = CLASSES[int(class_id)]
        p, r = float(box.p[i]), float(box.r[i])
        per_class[name] = {
            "precision": p,
            "recall": r,
            "f1": 2 * p * r / (p + r) if p + r else 0.0,
            "ap50": float(box.ap50[i]),
            "ap50_95": float(box.ap[i].mean()),
        }
    if set(per_class) != set(CLASSES):
        raise ValueError(f"Missing class metrics for {treatment}: {set(CLASSES)-set(per_class)}")
    global_metrics = {str(k): float(v) for k, v in result.results_dict.items()}
    gp, gr = global_metrics.get("metrics/precision(B)", 0), global_metrics.get("metrics/recall(B)", 0)
    global_metrics["metrics/F1(B)"] = 2 * gp * gr / (gp + gr) if gp + gr else 0.0
    return {
        "experiment": f"{treatment}_seed{seed}",
        "stage": "full_data_equal_exposure_interventions",
        "treatment": treatment,
        "model_name": model_name,
        "seed": seed,
        "pretrained_checkpoint": relative(pretrained),
        "pretrained_sha256": sha256_file(pretrained),
        "checkpoint": relative(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "parameters": params,
        "epochs_requested": epochs,
        "epochs_completed": epochs,
        "imgsz": 640,
        "batch": 16,
        "image_exposure_budget": TARGET_EXPOSURES,
        "planned_image_exposures": dataset_summary["expanded_train_images"] * epochs,
        "training_seconds": round(elapsed, 3),
        "global_metrics": global_metrics,
        "validation_speed_ms_per_image": {str(k): float(v) for k, v in getattr(result, "speed", {}).items()},
        "minority_macro_ap50": statistics.mean(per_class[c]["ap50"] for c in TAIL),
        "minority_macro_recall": statistics.mean(per_class[c]["recall"] for c in TAIL),
        "minority_macro_f1": statistics.mean(per_class[c]["f1"] for c in TAIL),
        "macro_f1": statistics.mean(per_class[c]["f1"] for c in CLASSES),
        "per_class": per_class,
        "evaluated_split": "validation",
        "selected_on": "validation only",
        "dataset": dataset_summary,
        "environment": environment,
    }


def run_one(treatment: str, seed: int, model_name: str, pretrained: Path,
            dataset_summary: dict[str, Any], YOLO: Any, torch: Any,
            environment: dict[str, Any], force: bool = False) -> dict[str, Any]:
    dataset_root = DATA_ROOT / treatment
    dataset_yaml = dataset_root / "data.yaml"
    train_count = int(dataset_summary["expanded_train_images"])
    epochs = max(1, int(round(TARGET_EXPOSURES / train_count)))
    run_dir = RUN_ROOT / treatment / f"seed_{seed}"
    checkpoint = run_dir / "weights/best.pt"
    signature_path = run_dir / "run_signature.json"
    metrics_path = RESULTS / "runs" / f"{treatment}_seed{seed}.json"
    signature = {
        "treatment": treatment,
        "dataset_manifest_sha256": dataset_summary["manifest_sha256"],
        "dataset_yaml_sha256": sha256_file(dataset_yaml),
        "model": model_name,
        "pretrained_sha256": sha256_file(pretrained),
        "seed": seed,
        "epochs": epochs,
        "image_exposures": train_count * epochs,
        "imgsz": 640,
        "batch": 16,
        "optimizer": "auto",
        "deterministic": True,
        "patience": epochs + 1,
    }
    if force and run_dir.exists():
        shutil.rmtree(run_dir)
        metrics_path.unlink(missing_ok=True)
    if checkpoint.is_file() and signature_path.is_file():
        saved_signature = json.loads(signature_path.read_text(encoding="utf-8"))
        if saved_signature != signature:
            raise ValueError(f"Stale run checkpoint {checkpoint}; rerun with --force.")
    else:
        if run_dir.exists():
            shutil.rmtree(run_dir)
        run_dir.parent.mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()
        model = YOLO(str(pretrained))
        model.train(
            data=str(dataset_yaml), epochs=epochs, imgsz=640, batch=16,
            patience=epochs + 1, seed=seed, project=str(run_dir.parent),
            name=run_dir.name, exist_ok=True, device=0, workers=4,
            deterministic=True, optimizer="auto", plots=True, save=True, val=True,
        )
        elapsed = time.perf_counter() - start
        if not checkpoint.is_file():
            raise FileNotFoundError(f"No best checkpoint created at {checkpoint}")
        signature_path.write_text(json.dumps(signature, indent=2) + "\n", encoding="utf-8")
        del model
        torch.cuda.empty_cache()
    if metrics_path.is_file():
        record = json.loads(metrics_path.read_text(encoding="utf-8"))
        if record["checkpoint_sha256"] != sha256_file(checkpoint):
            raise ValueError(f"Metrics/checkpoint mismatch at {metrics_path}")
        return record
    start = time.perf_counter()
    model = YOLO(str(checkpoint))
    result = model.val(
        data=str(dataset_yaml), split="val", imgsz=640, batch=16,
        project=str(VAL_ROOT / treatment), name=f"seed_{seed}", exist_ok=True,
        device=0, plots=True, save_txt=False,
    )
    elapsed = 0.0
    results_csv = run_dir / "results.csv"
    if results_csv.is_file():
        with results_csv.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        if rows and "time" in rows[-1]:
            elapsed = float(rows[-1]["time"])
    if elapsed <= 0:
        elapsed = time.perf_counter() - start
    params = sum(parameter.numel() for parameter in model.model.parameters())
    record = extract_record(treatment, seed, model_name, checkpoint, pretrained,
                            dataset_summary, result, params, elapsed, epochs, environment)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    del model
    torch.cuda.empty_cache()
    return record


def write_summary(records: list[dict[str, Any]]) -> None:
    path = RESULTS / "full_data_intervention_runs.json"
    path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    rows = []
    for treatment in sorted({r["treatment"] for r in records}):
        subset = [r for r in records if r["treatment"] == treatment]
        row: dict[str, Any] = {"treatment": treatment, "seeds": ",".join(str(r["seed"]) for r in subset)}
        metrics = ["minority_macro_ap50", "minority_macro_recall", "minority_macro_f1", "macro_f1", "training_seconds"]
        metrics.extend(["metrics/mAP50(B)", "metrics/mAP50-95(B)", "metrics/F1(B)"])
        for metric in metrics:
            values = [float(r["global_metrics"].get(metric)) if metric.startswith("metrics/") else float(r[metric]) for r in subset]
            row[f"{metric}_mean"] = statistics.mean(values)
            row[f"{metric}_sd"] = statistics.stdev(values) if len(values) > 1 else 0.0
        for metric in ("metrics/precision(B)", "metrics/recall(B)"):
            values = [float(r["global_metrics"][metric]) for r in subset]
            row[f"{metric}_mean"] = statistics.mean(values)
            row[f"{metric}_sd"] = statistics.stdev(values) if len(values) > 1 else 0.0
        latency = [float(r["validation_speed_ms_per_image"].get("inference", 0.0)) for r in subset]
        row["validation_inference_ms_mean"] = statistics.mean(latency)
        row["validation_inference_ms_sd"] = statistics.stdev(latency) if len(latency) > 1 else 0.0
        rows.append(row)
    rows.sort(key=lambda r: (r["minority_macro_ap50_mean"], r["metrics/mAP50(B)_mean"]), reverse=True)
    with (RESULTS / "full_data_intervention_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="rebuild generated datasets and retrain cached runs")
    parser.add_argument("--treatments", nargs="+", default=["full_repeat", "full_targeted_aug", "full_combined"], choices=["full_repeat", "full_targeted_aug", "full_combined"])
    args = parser.parse_args()
    selection_path = RESULTS / "model_transfer_selection.json"
    if not selection_path.is_file():
        raise FileNotFoundError("Run run_comprehensive_study.py --stage architectures first.")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    model_name = selection["selected_model"]
    pretrained = ROOT / selection["selected_pretrained_checkpoint"]
    if not pretrained.is_file():
        raise FileNotFoundError(pretrained)
    torch, YOLO, environment = get_environment()
    all_records: list[dict[str, Any]] = []
    for treatment in args.treatments:
        print(f"\nBuilding {treatment} (all Phase 2 training images retained)", flush=True)
        dataset_summary = make_treatment(treatment, force=args.force)
        print(f"{dataset_summary['expanded_train_images']} training image presentations per epoch", flush=True)
        for seed in SEEDS:
            print(f"\n[{treatment}] seed {seed}: {model_name}", flush=True)
            record = run_one(treatment, seed, model_name, pretrained, dataset_summary,
                             YOLO, torch, environment, force=args.force)
            all_records.append(record)
            print(f"tail AP50={record['minority_macro_ap50']:.4f}; "
                  f"mAP50={record['global_metrics'].get('metrics/mAP50(B)', 0):.4f}", flush=True)
            write_summary(all_records)
    # Resume-safe summary includes records from previous invocations.
    all_existing = []
    for treatment in args.treatments:
        for seed in SEEDS:
            p = RESULTS / "runs" / f"{treatment}_seed{seed}.json"
            if p.is_file():
                all_existing.append(json.loads(p.read_text(encoding="utf-8")))
    write_summary(all_existing)


if __name__ == "__main__":
    main()
