"""Read the versioned study artifacts used by the lab."""

import csv
import json
from collections import Counter
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "results" / "comprehensive"
DATASET = ROOT / "data" / "processed"
CLASSES = ("crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches")
TAIL = ("crazing", "pitted_surface", "rolled-in_scale", "scratches")
DEFAULT_MODEL = "combined"
MODES = {
    "combined": ("Combined YOLO26n", "full_combined_seed42"),
    "standard": ("Standard YOLO26n", "yolo26n_seed42"),
    "standard320": ("Standard YOLO26n · 320", "yolo26n_imgsz320_seed42"),
}
TREATMENTS = ("full_repeat", "full_targeted_aug", "full_combined")
FIGURES = RESULTS / "figures"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


@lru_cache(maxsize=1)
def model_registry() -> dict:
    selection = read_json(RESULTS / "final_model_selection.json")
    registry = {}
    for key, (label, run_id) in MODES.items():
        record = read_json(RESULTS / "runs" / f"{run_id}.json")
        registry[key] = {
            "id": key, "label": label, "run": run_id, "checkpoint": record["checkpoint"],
            "sha256": record["checkpoint_sha256"], "size": record["imgsz"],
            "seed": record["seed"], "parameters": record["parameters"],
            "batch": record["batch"], "epochs": record["epochs_completed"],
        }
    selected = registry[DEFAULT_MODEL]
    if (selected["checkpoint"], selected["sha256"], selected["seed"]) != (
        selection["checkpoint"], selection["checkpoint_sha256"], selection["selected_seed"]
    ):
        raise RuntimeError("The lab's finalist does not match the Phase 8 selection record.")
    return registry


def figure_path(name: str) -> Path | None:
    path = FIGURES / name
    return path if path.parent == FIGURES and path.suffix == ".png" and path.is_file() else None


def validation_case(name: str) -> dict | None:
    if name != Path(name).name or Path(name).suffix.lower() != ".jpg":
        return None
    image = DATASET / "images" / "val" / name
    label = DATASET / "labels" / "val" / f"{image.stem}.txt"
    return {"image": image, "label": label} if image.is_file() and label.is_file() else None


def annotations(label: Path) -> list[dict]:
    entries = []
    for index, row in enumerate(dict.fromkeys(label.read_text(encoding="utf-8").splitlines())):
        if not row.strip():
            continue
        category, x, y, width, height = map(float, row.split())
        entries.append({
            "id": f"truth-{index}",
            "class_name": CLASSES[int(category)],
            "xywh": [x, y, width, height],
        })
    return entries


@lru_cache(maxsize=1)
def study_data() -> dict:
    models = model_registry()
    selection = read_json(RESULTS / "final_model_selection.json")
    test = read_json(RESULTS / "test_evaluations" / "full_combined" / "seed_42_test_metrics.json")
    split_counts = Counter(row["split"] for row in read_rows(ROOT / "results/phase2/yolo_conversion_manifest.csv"))
    distributions = {}
    for treatment in TREATMENTS:
        summary = read_json(RESULTS / "dataset_generation" / f"{treatment}_summary.json")
        distributions[treatment] = {
            "images": summary["expanded_train_images"],
            "boxes": summary["expanded_train_instances_by_class"],
            "repeat_rates": summary["repeat_rate_by_class"],
        }
    combined = read_json(RESULTS / "runs" / "full_combined_seed42.json")
    distributions["original"] = {
        "images": combined["dataset"]["base_train_images"],
        "boxes": combined["dataset"]["train_instance_counts_by_class"],
    }
    training = {}
    runs = {}
    for key, entry in models.items():
        checkpoint_folder = (ROOT / entry["checkpoint"]).parents[1]
        training[key] = [
            {column.strip(): float(value) for column, value in row.items() if value.strip()}
            for row in read_rows(checkpoint_folder / "results.csv")
        ]
        record = read_json(RESULTS / "runs" / f'{entry["run"]}.json')
        signature = read_json(checkpoint_folder / "run_signature.json")
        runs[key] = {
            "optimizer": signature.get("optimizer", "auto"),
            "training_seconds": record["training_seconds"],
            "image_exposures": signature.get("image_exposures", entry["epochs"] * distributions["original"]["images"]),
            "validation": record["global_metrics"],
            "per_class": record["per_class"],
        }
    metrics = test["global_metrics"]
    groups = selection["candidate_validation_summary"]
    group_runs = {group["candidate"]: [] for group in groups}
    for row in read_rows(RESULTS / "model_transfer_runs.csv"):
        group_runs[row["model_name"]].append({
            "seed": int(row["seed"]), "tail_ap50": float(row["minority_macro_AP50"]), "mAP50": float(row["mAP50"]),
        })
    for treatment in TREATMENTS:
        for seed in (42, 43, 44):
            record = read_json(RESULTS / "runs" / f"{treatment}_seed{seed}.json")
            group_runs[treatment].append({
                "seed": seed, "tail_ap50": record["minority_macro_ap50"],
                "mAP50": record["global_metrics"]["metrics/mAP50(B)"],
            })
    return {
        "classes": CLASSES, "tail_classes": TAIL, "models": models, "default_model": DEFAULT_MODEL,
        "dataset": {"image_counts": dict(split_counts), "distributions": distributions},
        "training": training, "runs": runs,
        "candidates": groups, "candidate_runs": group_runs,
        "resolution": {
            key: float(value) for key, value in read_rows(RESULTS / "resolution_ablation_summary.csv")[0].items()
            if key != "model_name"
        },
        "thresholds": {name: round(row["confidence_threshold"], 2) for name, row in test["per_class_operating_metrics"].items()},
        "test": {
            "mAP50": metrics["metrics/mAP50(B)"],
            "mAP50_95": metrics["metrics/mAP50-95(B)"],
            "precision": metrics["metrics/precision(B)"],
            "recall": metrics["metrics/recall(B)"],
            "F1": metrics["metrics/F1(B)"],
            "images": test["test_images_evaluated"],
            "per_class_ap": test["per_class_ap_metrics"],
            "per_class_operating": test["per_class_operating_metrics"],
        },
    }


def project_data() -> dict:
    cases = read_rows(RESULTS / "validation_error_cases.csv")[:12]
    return {
        **study_data(),
        "validation_cases": [
            {
                "name": row["image_name"],
                "false_positives": int(row["false_positives"]),
                "missed_boxes": int(row["missed_boxes"]),
                "available": validation_case(row["image_name"]) is not None,
            }
            for row in cases
        ],
    }



@lru_cache(maxsize=9)
def training_run(model: str, seed: int) -> dict | None:
    if model not in MODES or seed not in (42, 43, 44):
        return None
    _, default_id = MODES[model]
    run_id = f"{default_id.rsplit('seed', 1)[0]}seed{seed}"
    record = read_json(RESULTS / "runs" / f"{run_id}.json")
    folder = (ROOT / record["checkpoint"]).parents[1]
    signature = read_json(folder / "run_signature.json")
    return {
        "model": record["model_name"], "seed": seed, "run": run_id,
        "size": record["imgsz"], "batch": record["batch"],
        "optimizer": signature.get("optimizer", "auto"),
        "image_exposures": signature.get("image_exposures", record["epochs_completed"] * 1257),
        "training_seconds": record["training_seconds"], "validation": record["global_metrics"],
        "history": [
            {column.strip(): float(value) for column, value in row.items() if value.strip()}
            for row in read_rows(folder / "results.csv")
        ],
    }
