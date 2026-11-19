"""Select by validation, run one final held-out test, and build figures/tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/comprehensive"
FIGURES = RESULTS / "figures"
TEST_EVAL_ROOT = RESULTS / "test_evaluations"
CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
TAIL = ["crazing", "pitted_surface", "rolled-in_scale", "scratches"]
TAIL_IDS = {CLASSES.index(name) for name in TAIL}
BASELINE_TEST = ROOT / "results/final_test_metrics.json"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def metric(record: dict[str, Any], key: str) -> float:
    value = record["global_metrics"].get(key)
    if value is not None:
        return float(value)
    if key == "metrics/F1(B)":
        precision = record["global_metrics"].get("metrics/precision(B)", 0.0)
        recall = record["global_metrics"].get("metrics/recall(B)", 0.0)
        return 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return 0.0


def candidate_groups() -> dict[str, list[dict[str, Any]]]:
    arch_path = RESULTS / "model_transfer_runs.json"
    if not arch_path.is_file():
        raise FileNotFoundError("Run the architecture benchmark first.")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in read_json(arch_path):
        groups[record["model_name"]].append(record)
    intervention_path = RESULTS / "full_data_intervention_runs.json"
    if intervention_path.is_file():
        for record in read_json(intervention_path):
            groups[record["treatment"]].append(record)
    return groups


def score_groups(groups: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows = []
    for name, records in groups.items():
        if len(records) < 3:
            continue
        tail_ap = [float(record["minority_macro_ap50"]) for record in records]
        map50 = [metric(record, "metrics/mAP50(B)") for record in records]
        map95 = [metric(record, "metrics/mAP50-95(B)") for record in records]
        tail_recall = [float(record["minority_macro_recall"]) for record in records]
        tail_f1 = [float(record["minority_macro_f1"]) for record in records]
        precision = [metric(record, "metrics/precision(B)") for record in records]
        recall = [metric(record, "metrics/recall(B)") for record in records]
        global_f1 = [metric(record, "metrics/F1(B)") for record in records]
        macro_f1 = [float(record.get("macro_f1", record.get("global_metrics", {}).get("macro_F1", 0.0))) for record in records]
        latency = [float(record.get("validation_speed_ms_per_image", record.get("inference_speed_ms_per_image", {})).get("inference", 0.0)) for record in records]
        rows.append({
            "candidate": name,
            "n_seeds": len(records),
            "seeds": ",".join(str(record["seed"]) for record in records),
            "minority_macro_AP50_mean": statistics.mean(tail_ap),
            "minority_macro_AP50_sd": statistics.stdev(tail_ap),
            "minority_macro_recall_mean": statistics.mean(tail_recall),
            "minority_macro_recall_sd": statistics.stdev(tail_recall),
            "minority_macro_F1_mean": statistics.mean(tail_f1),
            "minority_macro_F1_sd": statistics.stdev(tail_f1),
            "precision_mean": statistics.mean(precision),
            "precision_sd": statistics.stdev(precision),
            "recall_mean": statistics.mean(recall),
            "recall_sd": statistics.stdev(recall),
            "global_F1_mean": statistics.mean(global_f1),
            "global_F1_sd": statistics.stdev(global_f1),
            "macro_F1_mean": statistics.mean(macro_f1),
            "macro_F1_sd": statistics.stdev(macro_f1),
            "mAP50_mean": statistics.mean(map50),
            "mAP50_sd": statistics.stdev(map50),
            "mAP50_95_mean": statistics.mean(map95),
            "mAP50_95_sd": statistics.stdev(map95),
            "parameters": records[0].get("parameters"),
            "checkpoint_bytes": records[0].get("checkpoint_bytes"),
            "validation_inference_ms_mean": statistics.mean(latency),
            "validation_inference_ms_sd": statistics.stdev(latency),
            "mean_training_seconds": statistics.mean(float(record["training_seconds"]) for record in records),
            "training_seconds_sd": statistics.stdev(float(record["training_seconds"]) for record in records),
            "validation_only_selection": True,
        })
    rows.sort(key=lambda row: (row["minority_macro_AP50_mean"], row["mAP50_mean"]), reverse=True)
    return rows


def select_final(groups: dict[str, list[dict[str, Any]]], rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    if not rows:
        raise RuntimeError("No complete three-seed validation candidate groups are available.")
    winner_name = rows[0]["candidate"]
    records = groups[winner_name]
    # Fix a single seed-specific checkpoint using validation only: highest tail AP50,
    # followed by highest overall mAP50. Never inspect the test split here.
    selected = max(records, key=lambda record: (record["minority_macro_ap50"], metric(record, "metrics/mAP50(B)")))
    if "checkpoint" not in selected:
        raise ValueError("Selected validation record has no checkpoint path.")
    selection = {
        "selected_candidate_group": winner_name,
        "selected_seed": selected["seed"],
        "checkpoint": selected["checkpoint"],
        "checkpoint_sha256": selected["checkpoint_sha256"],
        "selection_split": "validation",
        "selection_rule": "highest mean minority macro AP50 across three seeds; tie-break by mean overall mAP50; choose the seed-specific checkpoint by validation minority AP50 then mAP50",
        "candidate_validation_summary": rows,
        "test_split_read": False,
    }
    (RESULTS / "final_model_selection.json").write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
    with (RESULTS / "candidate_comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return selected, selection


def iou(box_a: list[float], box_b: list[float]) -> float:
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    a = (ax - aw / 2, ay - ah / 2, ax + aw / 2, ay + ah / 2)
    b = (bx - bw / 2, by - bh / 2, bx + bw / 2, by + bh / 2)
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def read_yolo_rows(path: Path, prediction: bool) -> list[tuple[int, list[float], float]]:
    if not path.is_file():
        return []
    output = []
    seen_ground_truth: set[tuple[int, tuple[float, ...]]] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        values = [float(value) for value in line.split()]
        if prediction:
            if len(values) < 6:
                continue
            cls_id, x, y, w, h, confidence = values[:6]
        else:
            if len(values) != 5:
                continue
            cls_id, x, y, w, h = values
            confidence = 1.0
            key = (int(cls_id), tuple(round(value, 8) for value in (x, y, w, h)))
            if key in seen_ground_truth:
                continue
            seen_ground_truth.add(key)
        output.append((int(cls_id), [x, y, w, h], confidence))
    return output


def prediction_label_dir(save_dir: Path) -> Path:
    candidates = [save_dir / "labels", *save_dir.glob("**/labels")]
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.glob("*.txt")):
            return candidate
    raise FileNotFoundError(f"No save_txt prediction labels found beneath {save_dir}")


def match_class(predictions: list[tuple[int, list[float], float]],
                ground_truth: list[tuple[int, list[float], float]],
                class_id: int, threshold: float, iou_threshold: float = 0.5) -> tuple[int, int, int]:
    gt = [box for cls, box, _ in ground_truth if cls == class_id]
    preds = sorted(((box, conf) for cls, box, conf in predictions if cls == class_id and conf >= threshold), key=lambda item: item[1], reverse=True)
    matched: set[int] = set()
    tp = fp = 0
    for box, _ in preds:
        options = [(iou(box, target), idx) for idx, target in enumerate(gt) if idx not in matched]
        best_iou, best_idx = max(options, default=(0.0, -1))
        if best_iou >= iou_threshold:
            matched.add(best_idx)
            tp += 1
        else:
            fp += 1
    return tp, fp, len(gt) - len(matched)


def read_split_predictions(prediction_dir: Path, labels_root: Path) -> list[tuple[list[tuple[int, list[float], float]], list[tuple[int, list[float], float]]]]:
    pairs = []
    for gt_path in sorted(labels_root.glob("*.txt")):
        prediction_path = prediction_dir / f"{gt_path.stem}.txt"
        pairs.append((read_yolo_rows(prediction_path, True), read_yolo_rows(gt_path, False)))
    return pairs


def aggregate_counts(pairs: list[tuple[list[tuple[int, list[float], float]], list[tuple[int, list[float], float]]]],
                     class_id: int, threshold: float) -> tuple[int, int, int]:
    counts = [0, 0, 0]
    for predictions, truth in pairs:
        result = match_class(predictions, truth, class_id, threshold)
        counts = [a + b for a, b in zip(counts, result)]
    return tuple(counts)


def calibrate_thresholds(pairs: list[Any]) -> dict[str, dict[str, float]]:
    thresholds: dict[str, dict[str, float]] = {}
    threshold_grid = np.unique(np.concatenate((np.linspace(0.01, 0.99, 99), np.array([0.001]))))
    for class_id, name in enumerate(CLASSES):
        best = (-1.0, 0.001, (0, 0, 0))
        for threshold in threshold_grid:
            tp, fp, fn = aggregate_counts(pairs, class_id, float(threshold))
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            if f1 > best[0]:
                best = (f1, float(threshold), (tp, fp, fn))
        f1, threshold, (tp, fp, fn) = best
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        thresholds[name] = {"confidence_threshold": threshold, "validation_precision_at_iou50": precision, "validation_recall_at_iou50": recall, "validation_f1_at_iou50": f1}
    return thresholds


def qualitative_validation_errors(prediction_dir: Path, thresholds: dict[str, dict[str, float]]) -> Path | None:
    """Save validation-only FP/FN records and a four-image annotated overview."""
    import matplotlib.patches as patches
    from PIL import Image

    image_root = ROOT / "data/processed/images/val"
    label_root = ROOT / "data/processed/labels/val"
    errors = []
    for label_path in sorted(label_root.glob("*.txt")):
        prediction_path = prediction_dir / f"{label_path.stem}.txt"
        image_path = next((image_root / f"{label_path.stem}{suffix}" for suffix in (".jpg", ".jpeg", ".png") if (image_root / f"{label_path.stem}{suffix}").is_file()), None)
        if image_path is None or not label_path.is_file():
            continue
        predictions = read_yolo_rows(prediction_path, True)
        truth = read_yolo_rows(label_path, False)
        false_positive = false_negative = 0
        for class_id, name in enumerate(CLASSES):
            tp, fp, fn = match_class(predictions, truth, class_id, thresholds[name]["confidence_threshold"])
            false_positive += fp
            false_negative += fn
        if false_positive + false_negative:
            errors.append({
                "image": image_path,
                "image_name": image_path.name,
                "false_positives": false_positive,
                "missed_boxes": false_negative,
                "total_errors": false_positive + false_negative,
            })
    if not errors:
        return None
    errors.sort(key=lambda row: (row["total_errors"], row["missed_boxes"], row["false_positives"]), reverse=True)
    selected_cases = errors[:4]
    FIGURES.mkdir(parents=True, exist_ok=True)
    with (RESULTS / "validation_error_cases.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["image_name", "false_positives", "missed_boxes", "total_errors"]
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows([{key: row[key] for key in fields} for row in errors])

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, row in zip(axes.flat, selected_cases):
        with Image.open(row["image"]) as opened:
            image = opened.convert("RGB")
        ax.imshow(image)
        prediction_path = prediction_dir / f"{row['image'].stem}.txt"
        truth_path = label_root / f"{row['image'].stem}.txt"
        preds = read_yolo_rows(prediction_path, True)
        gts = read_yolo_rows(truth_path, False)
        width, height = image.size
        for cls_id, box, _ in gts:
            x, y, w, h = box
            rect = patches.Rectangle(((x - w/2)*width, (y-h/2)*height), w*width, h*height,
                                     linewidth=1.5, edgecolor="#39c16c", facecolor="none")
            ax.add_patch(rect)
        for cls_id, box, conf in preds:
            if conf < thresholds[CLASSES[cls_id]]["confidence_threshold"]:
                continue
            x, y, w, h = box
            rect = patches.Rectangle(((x - w/2)*width, (y-h/2)*height), w*width, h*height,
                                     linewidth=1.3, edgecolor="#ee5533", facecolor="none", linestyle="--")
            ax.add_patch(rect)
        ax.set_title(f"{row['image_name']}  |  FN {row['missed_boxes']}  FP {row['false_positives']}", fontsize=8)
        ax.axis("off")
    fig.suptitle("Validation error examples: green = ground truth, dashed red = prediction", fontsize=11)
    fig.tight_layout()
    output = FIGURES / "validation_error_examples.png"
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output


def final_test(selected: dict[str, Any], selection: dict[str, Any]) -> dict[str, Any]:
    import torch
    from ultralytics import YOLO

    candidate_name = selection["selected_candidate_group"]
    seed = int(selection["selected_seed"])
    checkpoint = ROOT / selected["checkpoint"]
    model = YOLO(str(checkpoint))
    output_root = ROOT / "outputs/logs/phase8/final_evaluation" / candidate_name / f"seed_{seed}"
    output_root.mkdir(parents=True, exist_ok=True)
    state_path = TEST_EVAL_ROOT / candidate_name / f"seed_{seed}_test_run_state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    val_result = model.val(
        data=str(ROOT / "configs/neu_det_yolo.yaml"),
        split="val", imgsz=640, batch=16, project=str(output_root),
        name="validation_calibration", exist_ok=True, device=0,
        plots=True, save_txt=True, save_conf=True,
    )
    val_dir = prediction_label_dir(Path(val_result.save_dir))
    val_pairs = read_split_predictions(val_dir, ROOT / "data/processed/labels/val")
    if not val_pairs:
        raise RuntimeError(f"Could not align validation predictions at {val_dir}")
    thresholds = calibrate_thresholds(val_pairs)
    validation_error_figure = qualitative_validation_errors(val_dir, thresholds)

    # Save a durable start marker and refuse to run test again if a previous
    # process started it but did not finish. This keeps retries from silently
    # turning one planned test pass into repeated model selection on test data.
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("checkpoint_sha256") != selected["checkpoint_sha256"]:
            raise RuntimeError("A different checkpoint already used this test split; create a new holdout before another final test.")
        if not state.get("test_inference_completed"):
            raise RuntimeError(f"A test run was started but not recorded as complete; inspect {state_path} and saved logs instead of rerunning test.")
        test_save_dir = Path(state["test_save_dir"])
        test_global_metrics = state["global_metrics"]
        per_class_ap = state["per_class_ap_metrics"]
        test_speed = state.get("test_speed_ms_per_image", {})
    else:
        state = {
            "candidate": candidate_name,
            "seed": seed,
            "checkpoint_sha256": selected["checkpoint_sha256"],
            "test_inference_started": True,
            "test_inference_completed": False,
        }
        state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
        # This is the sole test-split evaluation for the newly selected candidate.
        test_result = model.val(
            data=str(ROOT / "configs/neu_det_yolo.yaml"), split="test", imgsz=640,
            batch=16, project=str(output_root), name="final_test", exist_ok=True,
            device=0, plots=True, save_txt=True, save_conf=True,
        )
        test_save_dir = Path(test_result.save_dir)
        per_class_ap = {}
        box = test_result.box
        for i, class_id in enumerate(list(box.ap_class_index)):
            per_class_ap[CLASSES[int(class_id)]] = {
                "precision": float(box.p[i]), "recall": float(box.r[i]),
                "ap50": float(box.ap50[i]), "ap50_95": float(box.ap[i].mean()),
            }
        test_global_metrics = {str(k): float(v) for k, v in test_result.results_dict.items()}
        global_p = test_global_metrics.get("metrics/precision(B)", 0.0)
        global_r = test_global_metrics.get("metrics/recall(B)", 0.0)
        test_global_metrics["metrics/F1(B)"] = 2 * global_p * global_r / (global_p + global_r) if global_p + global_r else 0.0
        test_speed = {str(k): float(v) for k, v in getattr(test_result, "speed", {}).items()}
        state.update({
            "test_inference_completed": True,
            "test_save_dir": str(test_save_dir),
            "global_metrics": test_global_metrics,
            "per_class_ap_metrics": per_class_ap,
            "test_speed_ms_per_image": test_speed,
        })
        state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    test_dir = prediction_label_dir(test_save_dir)
    test_pairs = read_split_predictions(test_dir, ROOT / "data/processed/labels/test")
    if not test_pairs:
        raise RuntimeError(f"Could not align test predictions at {test_dir}")
    class_results = {}
    for class_id, name in enumerate(CLASSES):
        threshold = thresholds[name]["confidence_threshold"]
        tp, fp, fn = aggregate_counts(test_pairs, class_id, threshold)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        class_results[name] = {
            **thresholds[name], "tp": tp, "fp": fp, "fn": fn,
            "test_precision_at_iou50": precision,
            "test_recall_at_iou50": recall,
            "test_f1_at_iou50": f1,
        }
    evaluation = {
        "experiment": f"{candidate_name}_seed{seed}_test_once",
        "candidate": candidate_name,
        "seed": seed,
        "checkpoint": selected["checkpoint"],
        "checkpoint_sha256": selected["checkpoint_sha256"],
        "selection_split": "validation",
        "evaluated_split": "test",
        "test_evaluation_count": 1,
        "test_split_was_read_for_selection": False,
        "global_metrics": test_global_metrics,
        "test_speed_ms_per_image": test_speed,
        "per_class_ap_metrics": per_class_ap,
        "per_class_operating_metrics": class_results,
        "macro_test_operating_f1_iou50": statistics.mean(row["test_f1_at_iou50"] for row in class_results.values()),
        "minority_macro_test_operating_f1_iou50": statistics.mean(class_results[name]["test_f1_at_iou50"] for name in TAIL),
        "minority_macro_test_ap50": statistics.mean(per_class_ap[name]["ap50"] for name in TAIL),
        "validation_prediction_dir": str(val_dir),
        "validation_error_cases_csv": str(RESULTS / "validation_error_cases.csv"),
        "validation_error_examples_figure": str(validation_error_figure) if validation_error_figure else None,
        "test_prediction_dir": str(test_dir),
        "validation_thresholds_source": "per-class F1 maximization at IoU 0.50 using validation detections",
        "threshold_note": "Per-class operating point metrics use validation-selected confidence thresholds; AP metrics use the Ultralytics validator protocol.",
        "validation_images_used_for_thresholds": len(val_pairs),
        "test_images_evaluated": len(test_pairs),
        "test_confusion_matrix": str(test_save_dir / "confusion_matrix.png"),
        "environment": {
            "torch": torch.__version__,
            "ultralytics": __import__("ultralytics").__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        },
    }
    output = TEST_EVAL_ROOT / candidate_name / f"seed_{seed}_test_metrics.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(evaluation, indent=2) + "\n", encoding="utf-8")
    with (output.parent / f"seed_{seed}_per_class_test.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["class_name", *class_results[CLASSES[0]].keys()], lineterminator="\n")
        writer.writeheader()
        for name, values in class_results.items():
            writer.writerow({"class_name": name, **values})
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return evaluation


def dataset_figures() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    split_path = ROOT / "results/phase2/split_class_distribution.csv"
    if split_path.is_file():
        rows = list(csv.DictReader(split_path.open(encoding="utf-8")))
        split_names = sorted(set(row.get("split", "") for row in rows))
        fig, ax = plt.subplots(figsize=(11, 5.8))
        x = np.arange(len(CLASSES))
        width = 0.25
        for j, split in enumerate(split_names):
            values = []
            for name in CLASSES:
                match_row = next((r for r in rows if r.get("split") == split and r.get("class_name") == name), None)
                values.append(int(float(match_row.get("object_count", match_row.get("objects", 0)))) if match_row else 0)
            ax.bar(x + (j - (len(split_names) - 1) / 2) * width, values, width, label=split.title())
        ax.set_xticks(x, [name.replace("_", "\n") for name in CLASSES])
        ax.set_ylabel("Annotated defect boxes")
        ax.set_title("NEU-DET object counts by split and defect class")
        ax.legend(frameon=False)
        ax.grid(axis="y", alpha=0.22)
        fig.tight_layout()
        fig.savefig(FIGURES / "dataset_split_class_distribution.png", dpi=180)
        plt.close(fig)

    areas: dict[str, list[float]] = defaultdict(list)
    for path in sorted((ROOT / "data/processed/labels/train").glob("*.txt")):
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = [float(v) for v in line.split()]
            if len(parts) == 5:
                cls, _, _, w, h = parts
                areas[CLASSES[int(cls)]].append(w * h * 100)
    if areas:
        fig, ax = plt.subplots(figsize=(10, 5.4))
        ax.boxplot([areas[name] for name in CLASSES], tick_labels=[n.replace("_", "\n") for n in CLASSES], showfliers=False)
        ax.set_ylabel("Bounding-box area (% of image area)")
        ax.set_title("Training-set defect box area by class")
        ax.grid(axis="y", alpha=0.22)
        fig.tight_layout()
        fig.savefig(FIGURES / "training_box_area_by_class.png", dpi=180)
        plt.close(fig)


def result_figures(groups: dict[str, list[dict[str, Any]]], rows: list[dict[str, Any]], selected: dict[str, Any], test_record: dict[str, Any] | None) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    checkpoint = ROOT / selected["checkpoint"]
    run_dir = checkpoint.parent.parent
    for source_name, output_name in (("results.png", "selected_training_curves.png"), ("BoxPR_curve.png", "selected_validation_pr_curve.png"), ("confusion_matrix.png", "selected_validation_confusion_matrix.png")):
        source = run_dir / source_name
        if source.is_file():
            shutil.copy2(source, FIGURES / output_name)
    names = [row["candidate"] for row in rows]
    tail_means = [row["minority_macro_AP50_mean"] for row in rows]
    tail_sds = [row["minority_macro_AP50_sd"] for row in rows]
    map_means = [row["mAP50_mean"] for row in rows]
    map_sds = [row["mAP50_sd"] for row in rows]
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(names))
    ax.errorbar(x, tail_means, yerr=tail_sds, fmt="o", capsize=4, label="Tail-class macro AP50")
    ax.errorbar(x, map_means, yerr=map_sds, fmt="s", capsize=4, label="Overall mAP50")
    ax.set_xticks(x, names, rotation=28, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Validation score (mean ± SD across seeds)")
    ax.set_title("Validation-only candidate comparison")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "validation_candidate_comparison.png", dpi=180)
    plt.close(fig)

    base_group = groups.get("yolov8n", [])
    final_group = groups.get(selected["experiment"].rsplit("_seed", 1)[0], [])
    baseline_per_class = defaultdict(list)
    final_per_class = defaultdict(list)
    for record in base_group:
        for name, values in record["per_class"].items():
            baseline_per_class[name].append(float(values["ap50"]))
    for record in final_group:
        for name, values in record["per_class"].items():
            final_per_class[name].append(float(values["ap50"]))
    if baseline_per_class and final_per_class:
        fig, ax = plt.subplots(figsize=(11, 5.8))
        x = np.arange(len(CLASSES))
        w = 0.36
        ax.bar(x - w / 2, [statistics.mean(baseline_per_class[n]) for n in CLASSES], w, label="YOLOv8n baseline (val mean)")
        ax.bar(x + w / 2, [statistics.mean(final_per_class[n]) for n in CLASSES], w, label=f"Selected: {selected['experiment'].rsplit('_seed', 1)[0]} (val mean)")
        ax.set_xticks(x, [n.replace("_", "\n") for n in CLASSES])
        ax.set_ylim(0, 1)
        ax.set_ylabel("AP50")
        ax.set_title("Per-class validation AP50: baseline vs selected candidate")
        ax.legend(frameon=False)
        ax.grid(axis="y", alpha=0.22)
        fig.tight_layout()
        fig.savefig(FIGURES / "per_class_ap50_baseline_vs_selected.png", dpi=180)
        plt.close(fig)

    if test_record:
        confusion = Path(test_record.get("test_confusion_matrix", ""))
        if confusion.is_file():
            shutil.copy2(confusion, FIGURES / "final_test_confusion_matrix.png")
        validation_dir = Path(test_record.get("validation_prediction_dir", ""))
        validation_pr = validation_dir.parent / "BoxPR_curve.png"
        if validation_pr.is_file():
            shutil.copy2(validation_pr, FIGURES / "selected_validation_pr_curve.png")
        names = CLASSES
        val = [test_record["per_class_ap_metrics"][name]["ap50"] for name in names]
        fig, ax = plt.subplots(figsize=(10, 5.2))
        colors = ["#d95f02" if name in TAIL else "#1b9e77" for name in names]
        ax.bar(range(len(names)), val, color=colors)
        ax.set_xticks(range(len(names)), [n.replace("_", "\n") for n in names])
        ax.set_ylim(0, 1)
        ax.set_ylabel("AP50")
        ax.set_title("Single final held-out test: per-class AP50")
        ax.grid(axis="y", alpha=0.22)
        fig.tight_layout()
        fig.savefig(FIGURES / "final_test_per_class_ap50.png", dpi=180)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--select-only", action="store_true", help="write validation selection and figures without reading test data")
    parser.add_argument("--skip-test", action="store_true", help="alias for --select-only")
    args = parser.parse_args()
    groups = candidate_groups()
    rows = score_groups(groups)
    selected, selection = select_final(groups, rows)
    dataset_figures()
    test_record = None
    if not args.select_only and not args.skip_test:
        output = TEST_EVAL_ROOT / selection["selected_candidate_group"] / f"seed_{selection['selected_seed']}_test_metrics.json"
        if output.is_file():
            test_record = read_json(output)
            if test_record.get("checkpoint_sha256") != selected["checkpoint_sha256"]:
                raise RuntimeError("Saved final test metrics belong to another checkpoint; the current test split cannot be reused for a new candidate.")
        else:
            test_record = final_test(selected, selection)
    result_figures(groups, rows, selected, test_record)
    print(json.dumps({"selection": selection, "test_evaluation": test_record}, indent=2))


if __name__ == "__main__":
    main()
