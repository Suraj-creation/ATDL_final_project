"""Build the Markdown research report from saved Phase 1–8 experiment records."""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/comprehensive"
REPORTS = ROOT / "reports"
CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
TAIL = ["crazing", "pitted_surface", "rolled-in_scale", "scratches"]
MANIFEST_COLUMNS = [
    "experiment", "stage", "seed", "model_name", "checkpoint_path",
    "checkpoint_sha256", "size_bytes", "evaluated_on_test",
    "minority_macro_AP50_val", "mAP50_val", "mAP50_95_val", "macro_F1_val",
]


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def f(value: Any, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def fs(value: Any, digits: int = 3) -> str:
    try:
        return f"{float(value):+.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def read_csv(path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(path.open(encoding="utf-8")))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_duplicate_label_rows() -> list[dict[str, Any]]:
    duplicates: list[dict[str, Any]] = []
    for label_path in sorted((ROOT / "data/processed/labels").glob("*/*.txt")):
        counts: dict[tuple[int, tuple[float, ...]], int] = {}
        for line in label_path.read_text(encoding="utf-8").splitlines():
            values = [float(part) for part in line.split()]
            if len(values) != 5:
                continue
            key = (int(values[0]), tuple(round(value, 8) for value in values[1:]))
            counts[key] = counts.get(key, 0) + 1
        for (class_id, box), count in counts.items():
            if count > 1:
                duplicates.append({
                    "split": label_path.parent.name,
                    "label_file": label_path.relative_to(ROOT).as_posix(),
                    "class_id": class_id,
                    "duplicate_rows": count - 1,
                    "x_center": box[0], "y_center": box[1], "width": box[2], "height": box[3],
                })
    output = ROOT / "results/tables/exact_duplicate_yolo_label_rows.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        fields = ["split", "label_file", "class_id", "duplicate_rows", "x_center", "y_center", "width", "height"]
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(duplicates)
    return duplicates


def update_checkpoint_manifest(architecture: list[dict[str, Any]], resolution: list[dict[str, Any]],
                               interventions: list[dict[str, Any]], selected: dict[str, Any]) -> None:
    manifest_path = ROOT / "results/checkpoint_manifest.csv"
    rows: list[dict[str, Any]] = []
    if manifest_path.is_file():
        old_rows = read_csv(manifest_path)
        for old in old_rows:
            is_baseline = old["experiment"] == "E0_baseline"
            rows.append({
                "experiment": old["experiment"],
                "stage": "phase3_baseline" if is_baseline else "phase5_exploratory_single_seed",
                "seed": 42,
                "model_name": "yolov8n",
                "checkpoint_path": old["checkpoint_path"],
                "checkpoint_sha256": old["checkpoint_sha256"],
                "size_bytes": old["size_bytes"],
                "evaluated_on_test": "historical_baseline" if is_baseline else "no",
                "minority_macro_AP50_val": old["minority_macro_AP50_val"],
                "mAP50_val": old["mAP50_val"],
                "mAP50_95_val": old["mAP50_95_val"],
                "macro_F1_val": old["macro_F1_val"],
            })
    for record in architecture + resolution + interventions:
        rows.append({
            "experiment": record["experiment"],
            "stage": record["stage"],
            "seed": record["seed"],
            "model_name": record.get("model_name", record.get("treatment")),
            "checkpoint_path": record["checkpoint"],
            "checkpoint_sha256": record["checkpoint_sha256"],
            "size_bytes": record["checkpoint_bytes"],
            "evaluated_on_test": "phase8_final_candidate" if record["checkpoint_sha256"] == selected["checkpoint_sha256"] else "no",
            "minority_macro_AP50_val": record["minority_macro_ap50"],
            "mAP50_val": record["global_metrics"].get("metrics/mAP50(B)"),
            "mAP50_95_val": record["global_metrics"].get("metrics/mAP50-95(B)"),
            "macro_F1_val": record.get("macro_f1"),
        })
    deduplicated: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        deduplicated[(row["experiment"], row["checkpoint_sha256"])] = row
    with manifest_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(deduplicated.values())
    pretrained: dict[str, dict[str, Any]] = {}
    for record in architecture:
        model_name = record["model_name"]
        checkpoint_path = record["pretrained_checkpoint"]
        full_path = ROOT / checkpoint_path
        pretrained[model_name] = {
            "model_name": model_name,
            "checkpoint_path": checkpoint_path,
            "checkpoint_sha256": record["pretrained_sha256"],
            "size_bytes": full_path.stat().st_size if full_path.is_file() else "missing",
        }
    with (ROOT / "results/comprehensive/pretrained_weights_manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["model_name", "checkpoint_path", "checkpoint_sha256", "size_bytes"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(pretrained.values())


def build() -> Path:
    phase2 = load(ROOT / "results/phase2/phase2_dataset_summary.json")
    duplicate_label_rows = audit_duplicate_label_rows()
    duplicate_row_count = sum(row["duplicate_rows"] for row in duplicate_label_rows)
    archive_path = ROOT / "dataset/dataset.zip"
    archive_sha = sha256_file(archive_path) if archive_path.is_file() else "archive unavailable in this workspace"
    split_sha = sha256_file(ROOT / "data/splits/phase2_image_split.csv")
    baseline = load(ROOT / "results/phase3_baseline_metrics.json")
    phase5 = load(ROOT / "results/phase5_summary.json")
    architecture = load(RESULTS / "model_transfer_runs.json")
    interventions = load(RESULTS / "full_data_intervention_runs.json")
    resolution = load(RESULTS / "resolution_ablation_summary.json")
    resolution_runs = load(RESULTS / "resolution_ablation_runs.json")
    comparison = read_csv(RESULTS / "candidate_comparison.csv")
    final_selection = load(RESULTS / "final_model_selection.json")
    update_checkpoint_manifest(architecture, resolution_runs, interventions, final_selection)
    test_path = RESULTS / "test_evaluations" / final_selection["selected_candidate_group"] / f"seed_{final_selection['selected_seed']}_test_metrics.json"
    test = load(test_path)
    phase5_compare = read_csv(ROOT / "results/phase5_validation_comparison.csv")
    phase4_distribution = read_csv(ROOT / "results/phase4_final_distribution_report.csv")
    phase5_distribution = read_csv(ROOT / "results/phase5_method_distribution.csv")
    treatment_summaries = {
        name: load(RESULTS / "dataset_generation" / f"{name}_summary.json")
        for name in ("full_repeat", "full_targeted_aug", "full_combined")
    }
    baseline_test = load(ROOT / "results/final_test_metrics.json")
    split_rows = read_csv(ROOT / "results/phase2/split_class_distribution.csv")
    split_counts = {}
    for row in split_rows:
        split_counts.setdefault(row["split"], {"images": int(row["total_images"]), "boxes": 0})
        split_counts[row["split"]]["boxes"] += int(row["object_count"])

    train_class_counts = {row["class_name"]: int(row["object_count"]) for row in split_rows if row["split"] == "train"}
    phase5_train_counts = {
        name: info.get("train_images", 0)
        for name, info in phase5.get("experiments", {}).items()
    }
    legacy_table = [
        "| Phase 5 condition | Train presentations | Tail macro AP50 | mAP50 | mAP50–95 | Macro F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in phase5_compare:
        tail_values = [float(row[f"AP50_{name}"]) for name in TAIL]
        legacy_table.append(
            f"| {row['experiment']} | {phase5_train_counts.get(row['experiment'], '—')} | "
            f"{f(statistics.mean(tail_values))} | {f(row['metrics/mAP50(B)'])} | "
            f"{f(row['metrics/mAP50-95(B)'])} | {f(row['macro_F1'])} |"
        )
    phase4_distribution_table = [
        "| Phase 4 profile | Class | Train images with class | Train boxes |",
        "|---|---|---:|---:|",
    ]
    for row in phase4_distribution:
        phase4_distribution_table.append(
            f"| {row['variant']} | {row['class_name']} | {int(row['actual_train_images']):,} | {int(row['actual_train_objects']):,} |"
        )
    phase5_distribution_table = [
        "| Phase 5 experiment | Class | Train images with class | Train boxes | Total train presentations |",
        "|---|---|---:|---:|---:|",
    ]
    for row in phase5_distribution:
        phase5_distribution_table.append(
            f"| {row['experiment']} | {row['class_name']} | {int(row['train_images_containing_class']):,} | "
            f"{int(row['train_objects']):,} | {int(row['total_train_images']):,} |"
        )
    phase8_distribution_table = [
        "| Phase 8 treatment | Class | Train presentations with class | Train boxes presented | Total train presentations/epoch |",
        "|---|---|---:|---:|---:|",
    ]
    for treatment, summary in treatment_summaries.items():
        for name in CLASSES:
            phase8_distribution_table.append(
                f"| {treatment} | {name} | {summary['expanded_train_images_with_class'][name]:,} | "
                f"{summary['expanded_train_instances_by_class'][name]:,} | {summary['expanded_train_images']:,} |"
            )
    arch_groups: dict[str, list[dict[str, Any]]] = {}
    for row in architecture:
        arch_groups.setdefault(row["model_name"], []).append(row)
    intervention_groups: dict[str, list[dict[str, Any]]] = {}
    for row in interventions:
        intervention_groups.setdefault(row["treatment"], []).append(row)

    def group_summary(records: list[dict[str, Any]]) -> tuple[float, float, float, float, float, float]:
        tail = [float(record["minority_macro_ap50"]) for record in records]
        map50 = [float(record["global_metrics"]["metrics/mAP50(B)"]) for record in records]
        map95 = [float(record["global_metrics"]["metrics/mAP50-95(B)"]) for record in records]
        return (statistics.mean(tail), statistics.stdev(tail), statistics.mean(map50),
                statistics.stdev(map50), statistics.mean(map95), statistics.stdev(map95))

    arch_table = ["| Model | Seeds | Tail macro AP50 | Overall mAP50 | mAP50–95 | Params | Time / run (s) |", "|---|---:|---:|---:|---:|---:|---:|"]
    for model in ["yolov8n", "yolo11n", "yolo26n", "yolo11s"]:
        records = arch_groups.get(model, [])
        if not records:
            continue
        tail, tail_sd, map50, map50_sd, map95, map95_sd = group_summary(records)
        arch_table.append(f"| {model} | {len(records)} | {f(tail)} ± {f(tail_sd)} | {f(map50)} ± {f(map50_sd)} | {f(map95)} ± {f(map95_sd)} | {int(records[0]['parameters']):,} | {f(statistics.mean(float(r['training_seconds']) for r in records), 1)} |")

    ablation_table = [
        "| Input size | Seeds | Tail macro AP50 | Overall mAP50 | mAP50–95 | Inference (ms/image) | Train time (s) |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for size, label in ((640, "imgsz_640_reference"), (320, "imgsz_320")):
        values = resolution[label]
        count = 3 if size == 640 else int(values["n_seeds"])
        ablation_table.append(
            f"| {size} | {count} | {f(values['mean_tail_macro_AP50'])} ± {f(values['sd_tail_macro_AP50'])} | "
            f"{f(values['mean_mAP50'])} ± {f(values['sd_mAP50'])} | "
            f"{f(values['mean_mAP50_95'])} ± {f(values['sd_mAP50_95'])} | "
            f"{f(values['mean_inference_ms_per_image'], 2)} ± {f(values['sd_inference_ms_per_image'], 2)} | "
            f"{f(values['mean_training_seconds'], 1)} |"
        )
    tail_delta = resolution["paired_deltas"]["minority_macro_ap50"]
    map_delta = resolution["paired_deltas"]["metrics/mAP50(B)"]
    speed_delta = resolution["paired_deltas"]["validation_inference_ms_per_image"]

    intervention_table = ["| Treatment | Seeds | Tail macro AP50 | Overall mAP50 | mAP50–95 | Epochs | Expanded train images |", "|---|---:|---:|---:|---:|---:|---:|"]
    for treatment in ["full_repeat", "full_targeted_aug", "full_combined"]:
        records = intervention_groups.get(treatment, [])
        if not records:
            continue
        tail, tail_sd, map50, map50_sd, map95, map95_sd = group_summary(records)
        first = records[0]
        intervention_table.append(f"| {treatment} | {len(records)} | {f(tail)} ± {f(tail_sd)} | {f(map50)} ± {f(map50_sd)} | {f(map95)} ± {f(map95_sd)} | {first['epochs_requested']} | {first['dataset']['expanded_train_images']:,} |")

    candidate_table = [
        "| Candidate | Seeds | Tail macro AP50 | mAP50 | mAP50–95 | Tail recall | Tail F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    operating_comparison_table = [
        "| Candidate | Overall precision | Overall recall | Global F1 | Macro F1 | Inference ms/image |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    efficiency_table = [
        "| Candidate | Parameters | Checkpoint size (MB) | Mean training time (s) |",
        "|---|---:|---:|---:|",
    ]
    for row in comparison:
        candidate_table.append(
            f"| {row['candidate']} | {row['n_seeds']} | {f(row['minority_macro_AP50_mean'])} ± {f(row['minority_macro_AP50_sd'])} | "
            f"{f(row['mAP50_mean'])} ± {f(row['mAP50_sd'])} | {f(row['mAP50_95_mean'])} ± {f(row['mAP50_95_sd'])} | "
            f"{f(row['minority_macro_recall_mean'])} ± {f(row['minority_macro_recall_sd'])} | "
            f"{f(row['minority_macro_F1_mean'])} ± {f(row['minority_macro_F1_sd'])} |"
        )
        operating_comparison_table.append(
            f"| {row['candidate']} | {f(row['precision_mean'])} ± {f(row['precision_sd'])} | "
            f"{f(row['recall_mean'])} ± {f(row['recall_sd'])} | {f(row['global_F1_mean'])} ± {f(row['global_F1_sd'])} | "
            f"{f(row['macro_F1_mean'])} ± {f(row['macro_F1_sd'])} | "
            f"{f(row['validation_inference_ms_mean'], 2)} ± {f(row['validation_inference_ms_sd'], 2)} |"
        )
        efficiency_table.append(
            f"| {row['candidate']} | {int(row['parameters'] or 0):,} | {f(float(row['checkpoint_bytes'] or 0) / 1_000_000, 1)} | "
            f"{f(row['mean_training_seconds'], 1)} ± {f(row['training_seconds_sd'], 1)} |"
        )

    per_class_table = ["| Class | Train boxes | Baseline val AP50 | Selected val AP50 | Final test AP50 |", "|---|---:|---:|---:|---:|"]
    operating_table = ["| Class | Confidence threshold (validation) | Test precision | Test recall | Test F1 |", "|---|---:|---:|---:|---:|"]
    selected_group = final_selection["selected_candidate_group"]
    selected_records = arch_groups.get(selected_group, intervention_groups.get(selected_group, []))
    selected_seed_record = next(record for record in selected_records if int(record["seed"]) == int(final_selection["selected_seed"]))
    baseline_val = {item["class_name"]: item for item in baseline["per_class"]}
    for name in CLASSES:
        t = test["per_class_operating_metrics"][name]
        per_class_table.append(
            f"| {name} | {train_class_counts.get(name, 0)} | {f(baseline_val[name]['ap50'])} | "
            f"{f(selected_seed_record['per_class'][name]['ap50'])} | {f(test['per_class_ap_metrics'][name]['ap50'])} |"
        )
        operating_table.append(
            f"| {name} | {f(t['confidence_threshold'], 2)} | {f(t['test_precision_at_iou50'])} | "
            f"{f(t['test_recall_at_iou50'])} | {f(t['test_f1_at_iou50'])} |"
        )

    old_test_metrics = baseline_test.get("metrics", baseline_test.get("global_metrics", {}))
    old_test_map50 = old_test_metrics.get("metrics/mAP50(B)")
    old_test_map95 = old_test_metrics.get("metrics/mAP50-95(B)")
    old_test_f1 = old_test_metrics.get("metrics/F1(B)")
    winning_group = comparison[0]
    baseline_group = next((row for row in comparison if row["candidate"] == "yolov8n"), None)
    delta = None if baseline_group is None else float(winning_group["minority_macro_AP50_mean"]) - float(baseline_group["minority_macro_AP50_mean"])
    improvement_sentence = (
        f"The validation-selected group ({winning_group['candidate']}) changed mean tail-class macro AP50 by {delta:+.3f} versus YOLOv8n across the three-seed groups."
        if delta is not None else "The validation comparison is not complete enough to compute a full-data baseline delta."
    )

    lines = [
        "# Final Experimental Report on Class Imbalance and Steel Surface Defect Detection",
        "",
        f"**Project:** ATDL Final Year Project\n\n**Generated:** {datetime.now().strftime('%Y-%m-%d')}\n\n**Dataset:** NEU-DET, six steel surface defect classes",
        "",
        "## Abstract",
        "",
        f"This study evaluates COCO-pretrained YOLO detectors and class-imbalance interventions for six-class steel surface defect detection. The cleaned dataset contains {phase2['image_count_converted']:,} unique images and {phase2['object_count_converted'] - duplicate_row_count:,} unique boxes after removing {duplicate_row_count} exact repeated label rows. The controlled Phase 8 benchmark used {split_counts.get('train', {}).get('images', 0):,} training images, {split_counts.get('val', {}).get('images', 0):,} validation images, and {split_counts.get('test', {}).get('images', 0):,} test images. Four pretrained model families were compared across three random seeds; full-data repeat sampling, targeted augmentation, and combined training were then compared with matched image-exposure budgets. Candidate selection used validation only. {improvement_sentence} The final checkpoint was evaluated once on the test set after selection.",
        "",
        "## 1 Research Objective and Hypotheses",
        "",
        "The research question is whether imbalance-aware training can increase detection quality for the long-tail / error-prone defect categories while retaining overall detection performance. The pre-registered tail group follows the project's existing Phase 3–5 definition: `crazing`, `pitted_surface`, `rolled-in_scale`, and `scratches`. The primary validation endpoint is macro AP50 over those four classes; overall mAP50 is the tie-break and a safeguard metric. Secondary metrics are per-class precision/recall/F1, mAP50–95, training time, parameter count, checkpoint size, and per-image validation speed.",
        "",
        "The project hypothesis is treated as a testable claim, not an assumed result. AP improvements are only reported when the saved validation/test results show them. Means and standard deviations across three seeds describe run variability; n=3 is not enough for strong inferential or deployment claims.",
        "",
        "## 2 Dataset and Split Integrity",
        "",
        f"Phase 1 matched {phase2['raw_image_count_matched']:,} raw images and removed one image from a duplicated exact-image group, leaving {phase2['image_count_converted']:,} unique image/annotation pairs and {phase2['object_count_converted']:,} box-label rows. A second audit found {duplicate_row_count} exact repeated YOLO label rows across {len(duplicate_label_rows)} images; Ultralytics removes these duplicate rows when it loads labels, and the Phase 8 operating-threshold evaluator applies the same deduplication. This leaves {phase2['object_count_converted'] - duplicate_row_count:,} unique box instances. The fixed grouped split contains {split_counts.get('train', {}).get('images', 0):,} train, {split_counts.get('val', {}).get('images', 0):,} validation, and {split_counts.get('test', {}).get('images', 0):,} test images. Split rows below count annotation rows, not images.",
        "",
        "| Split | Images | Annotated boxes |", "|---|---:|---:|",
        *[f"| {name} | {entry['images']:,} | {entry['boxes']:,} |" for name, entry in split_counts.items()],
        "",
        "### Training Boxes by Class",
        "",
        "| Class | Train boxes |", "|---|---:|",
        *[f"| {name} | {train_class_counts.get(name, 0):,} |" for name in CLASSES],
        "",
        "![Class/split box counts](../results/comprehensive/figures/dataset_split_class_distribution.png)",
        "",
        "![Training box area by class](../results/comprehensive/figures/training_box_area_by_class.png)",
        "",
        "Box-area percentages are normalized to the source image area (the source images are 200×200). The box-area view describes object scale but does not by itself establish model difficulty.",
        "",
        "Exact repeated annotation rows: `results/tables/exact_duplicate_yolo_label_rows.csv`. Historical Phase 4/5 distribution tables report source label rows. Phase 8 expanded treatment counts deduplicate exact box rows to match Ultralytics label loading.",
        "",
        f"Dataset archive SHA-256: `{archive_sha}`. Fixed split manifest SHA-256: `{split_sha}`. The raw archive and processed images are excluded from the Git repository; obtain the dataset from its cited source and verify it against the archive hash when available.",
        "",
        "## 3 Experimental Design",
        "",
        "The project runs from a raw-data audit and duplicate-group check, through annotation conversion and a fixed grouped train/validation/test split, then the full-data baseline, controlled imbalance-severity variants, single-seed moderate-subset methods, multi-seed pretrained-model transfer, full-data equal-exposure interventions, validation-only selection, one final test evaluation, and report generation. The Phase 8 notebook orchestrates the final stages; prior notebooks remain the record of the earlier phases.",
        "",
        "### Pretrained Architecture Transfer",
        "",
        "COCO-pretrained YOLOv8n, YOLO11n, YOLO26n, and YOLO11s were trained on the complete train split. Training used 640-pixel inputs, batch size 16, 50 requested epochs, optimizer `auto`, deterministic seeds 42/43/44, and the same validation split. The already completed YOLOv8n seed-42 baseline was reused; it was not retrained. Model family selection was based on validation mean tail-class AP50, then validation overall mAP50.",
        "",
        *arch_table,
        "",
        "![Validation candidate comparison](../results/comprehensive/figures/validation_candidate_comparison.png)",
        "",
        "### Input-size ablation for the selected model family",
        "",
        f"The 640-pixel transfer winner (`{resolution['model_name']}`) was then compared with the same pretrained model at 320 pixels using paired seeds 42/43/44. Split, optimizer, batch, epoch cap, and all other training settings stayed fixed. This ablation is reported separately and was not added to the architecture/intervention candidate ranking. The 320-pixel input changed tail macro AP50 by {fs(tail_delta['mean_delta_320_minus_640'])} ± {f(tail_delta['sd_paired_delta'])} and mAP50 by {fs(map_delta['mean_delta_320_minus_640'])} ± {f(map_delta['sd_paired_delta'])}; inference time changed by {f(speed_delta['mean_delta_320_minus_640'], 2)} ms/image.",
        "",
        *ablation_table,
        "",
        "![Paired input-size ablation](../results/comprehensive/figures/resolution_ablation_comparison.png)",
        "",
        "### Full Data Imbalance Interventions",
        "",
        "The selected architecture was fixed before intervention training. Each treatment restarts from the same COCO-pretrained checkpoint for that architecture, rather than continuing from a validation-selected fine-tuned checkpoint. Every treatment includes all 1,257 original training images; validation and test are never augmented or resampled. Repeat sampling uses square-root inverse box-instance frequency, with the per-image factor equal to the maximum factor among tail classes present and deterministic stochastic rounding. Targeted augmentation adds two transforms to each training image with a tail-class object: horizontal flip with brightness 0.92 and 90-degree clockwise rotation with brightness 1.08. Bounding boxes are transformed analytically. The combined method applies both operations.",
        "",
        "To control optimization exposure, all interventions target 62,850 image presentations (1,257 × 50 baseline presentations). Epoch count is rounded from this target using the expanded manifest size; actual planned exposures are recorded per run. Early stopping is disabled by setting patience above the epoch count. The baseline was the standard Phase 3 schedule; this exposure matching is for the intervention contrasts.",
        "",
        *intervention_table,
        "",
        "### Candidate Selection and Test",
        "",
        "The final candidate group was selected among all complete three-seed architecture and intervention groups using mean validation tail macro AP50, with overall validation mAP50 as the tie-break. One seed-specific checkpoint was then fixed from that group using validation tail AP50 followed by overall mAP50. The Phase 8 selection scripts did not read test metrics. The existing test split had already been used for the Phase 6 YOLOv8n baseline evaluation before this extension; therefore, the new candidate's one-pass test results are descriptive and are not a fully blind independent comparison. A fresh, genuinely untouched holdout would require a new dataset or a complete retraining protocol under a newly reserved split.",
        "",
        "## 4 Results",
        "",
        "### Historical Phase 5 Comparison",
        "",
        "These seed-42 results are useful context for the original imbalance-severity and method experiments. Train presentations differ across rows, and methods use the moderate subset, so this table is exploratory rather than a causal or matched-exposure comparison.",
        "",
        *legacy_table,
        "",
        "### Overall Validation Comparison",
        "",
        "All group-level metrics below are mean ± standard deviation over the three seeds. Primary accuracy and tail-operating metrics:",
        "",
        *candidate_table,
        "",
        "Overall detection operating metrics and validation inference timing:",
        "",
        *operating_comparison_table,
        "",
        "Model-size and training-cost metrics:",
        "",
        *efficiency_table,
        "",
        "![Per-class baseline and selected candidate validation AP50](../results/comprehensive/figures/per_class_ap50_baseline_vs_selected.png)",
        "",
        "![Selected run training and validation curves](../results/comprehensive/figures/selected_training_curves.png)",
        "",
        "![Selected validation precision recall curve](../results/comprehensive/figures/selected_validation_pr_curve.png)",
        "",
        "### Per Class Results for the Selected Checkpoint",
        "",
        "The validation AP50 column below is the specific seed selected for the final test. Test AP50 comes from the Ultralytics validation protocol at its confidence sweep. The second table reports per-class operating metrics at thresholds selected on validation predictions by maximizing F1 at IoU 0.50, then applied unchanged to test.",
        "",
        *per_class_table,
        "",
        *operating_table,
        "",
        "### Validation Error Analysis",
        "",
        "The following examples come from validation predictions only. Green boxes show ground truth; dashed red boxes show predictions after applying the confidence thresholds selected on validation. The complete ranked file list is `results/comprehensive/validation_error_cases.csv`. These images illustrate errors; they are not a substitute for the aggregate metrics.",
        "",
        "![Validation false-positive and missed-box examples](../results/comprehensive/figures/validation_error_examples.png)",
        "",
        "![Final test AP50 by class](../results/comprehensive/figures/final_test_per_class_ap50.png)",
        "",
        "![Final test confusion matrix](../results/comprehensive/figures/final_test_confusion_matrix.png)",
        "",
        "### Final Test Summary",
        "",
        f"Selected group: **{final_selection['selected_candidate_group']}**, seed **{final_selection['selected_seed']}**. Test precision = **{f(test['global_metrics'].get('metrics/precision(B)'))}**, recall = **{f(test['global_metrics'].get('metrics/recall(B)'))}**, global F1 = **{f(test['global_metrics'].get('metrics/F1(B)'))}**, mAP50 = **{f(test['global_metrics'].get('metrics/mAP50(B)'))}**, and mAP50–95 = **{f(test['global_metrics'].get('metrics/mAP50-95(B)'))}**. Test macro AP50 for the four tail classes = **{f(test['minority_macro_test_ap50'])}**, and tail macro operating-point F1 = **{f(test['minority_macro_test_operating_f1_iou50'])}**. The test split contained {test['test_images_evaluated']} images in the single final evaluation; validation-reported inference speed was {f(test.get('test_speed_ms_per_image', {}).get('inference'), 2)} ms/image on the recorded GPU.",
        "",
        f"Historical YOLOv8n seed-42 test record: mAP50 = {f(old_test_map50)}, mAP50–95 = {f(old_test_map95)}, global F1 = {f(old_test_f1)}. This record predates the Phase 8 model search. The Phase 8 selection itself is validation-only, but a reader already had access to the earlier baseline test result, so any baseline-to-finalist test difference is descriptive.",
        "",
        "## 5 Interpretation and Findings",
        "",
        f"1. **Best validation candidate:** {winning_group['candidate']} ranked first under the validation-only objective at tail macro AP50 {f(winning_group['minority_macro_AP50_mean'])} ± {f(winning_group['minority_macro_AP50_sd'])}; overall mAP50 was {f(winning_group['mAP50_mean'])} ± {f(winning_group['mAP50_sd'])}.",
        f"2. **Change versus the multi-seed YOLOv8n baseline:** {improvement_sentence}",
        f"3. **Tail-class spread:** per-class results remain important because a macro score can hide a class-specific regression. In the original YOLOv8n validation record, `crazing` AP50 was {f(baseline_val['crazing']['ap50'])} and `rolled-in_scale` AP50 was {f(baseline_val['rolled-in_scale']['ap50'])}; compare these with each selected candidate's per-class table and plots.",
        "4. **Legacy Phase 5 evidence:** the Phase 5 variants were single-seed and trained from a reduced 808-image moderate subset; repeated samples and augmentations also changed epoch-level exposure. Their results remain exploratory and should not be treated as a controlled proof that imbalance handling beats full-data training.",
        "5. **Generalization:** the final test evaluation is one selected checkpoint on one fixed split. Three training seeds, a single dataset, and one test set do not establish statistical significance or production readiness.",
        "",
        "## 6 Study Limits",
        "",
        "- The NEU-DET dataset is small and images are 200×200; high-resolution deployment behavior and domain shift are not measured here.",
        "- Three seeds improve on a single run but produce wide uncertainty estimates; no confirmatory power analysis or hypothesis test is claimed.",
        "- The transfer benchmark varies model family and capacity together. Results answer which tested checkpoint performed best under this fixed budget, not which architecture mechanism caused the difference.",
        "- Offline flips and right-angle rotation are plausible texture-preserving transforms but may not represent every steel-inspection camera orientation or process.",
        "- Thresholds were selected from validation detections. The reported operating point is useful for error analysis but will require recalibration for different cameras, prevalence, or costs.",
        "- The previously completed baseline test result used this same test split. Although Phase 8 selection is validation-only and the new candidate is tested once, the holdout is not fully blind across the entire project history; treat comparisons as descriptive.",
        "",
        "## 7 Training-Data Distribution Appendix",
        "",
        "Image counts indicate the number of image presentations containing each class; an image with multiple labels contributes to multiple rows. Historical Phase 4/5 box counts are source annotation rows; three exact repeated rows are removed by Ultralytics during loading. Phase 5 methods operate on a reduced moderate subset and have unmatched exposure, while Phase 8 keeps all originals and reports expanded box counts after exact-row deduplication.",
        "",
        "### Phase 4 severity profiles",
        "",
        *phase4_distribution_table,
        "",
        "### Phase 5 moderate-subset methods",
        "",
        *phase5_distribution_table,
        "",
        "### Phase 8 full-data treatments",
        "",
        *phase8_distribution_table,
        "",
        "## 8 Reproducibility and Project Workflow",
        "",
        "Run from the repository root using the project `.venv` Jupyter kernel or `.venv/bin/python`:",
        "",
        "```bash",
        ".venv/bin/python -m src.atdl.run_comprehensive_study --stage architectures",
        ".venv/bin/python -m src.atdl.run_resolution_ablation",
        ".venv/bin/python -m src.atdl.run_full_data_interventions",
        ".venv/bin/python -m src.atdl.finalize_evaluation --select-only",
        ".venv/bin/python -m src.atdl.finalize_evaluation",
        ".venv/bin/python -m src.atdl.build_final_report",
        "```",
        "",
        "Main artifacts:",
        "- `notebook/atdl_phase8_comprehensive_evaluation.ipynb` — VS Code/Jupyter end-to-end runbook.",
        "- `src/atdl/run_comprehensive_study.py` — pretrained model comparison.",
        "- `src/atdl/run_full_data_interventions.py` — generated treatment datasets and equal-exposure training.",
        "- `src/atdl/finalize_evaluation.py` — validation selection, single test evaluation, threshold/error analysis, and figures.",
        "- `results/comprehensive/` — run JSON, CSV summaries, test metrics, selection record, and plots.",
        "- `outputs/checkpoints/phase8/` — best checkpoints and run configuration signatures.",
        "",
        "## References",
        "",
        "1. Lv et al., ‘Deep Metallic Surface Defect Detection: The New Benchmark and Detection Network,’ *Sensors*, 2020. [https://doi.org/10.3390/s20061562](https://doi.org/10.3390/s20061562)",
        "2. ‘Comparison of Pre-Trained YOLO Models on Steel Surface Defects Detector Based on Transfer Learning with GPU-Based Embedded Devices,’ *Sensors*, 2022. [https://www.mdpi.com/1424-8220/22/24/9926](https://www.mdpi.com/1424-8220/22/24/9926)",
        "3. Ultralytics, official model and detection task documentation. [https://docs.ultralytics.com/models/](https://docs.ultralytics.com/models/) · [https://docs.ultralytics.com/tasks/detect/](https://docs.ultralytics.com/tasks/detect/)",
        "4. Yaman et al., ‘Instance and Repeat Factor Sampling for Long-Tailed Object Detection,’ arXiv:2305.08069, 2023. [https://arxiv.org/abs/2305.08069](https://arxiv.org/abs/2305.08069)",
        "",
        "---",
        "Generated from versioned result JSON and CSV files. If the training matrix is rerun, rebuild this report and the slide deck so they match the current artifacts.",
    ]
    REPORTS.mkdir(parents=True, exist_ok=True)
    output = REPORTS / "FINAL_EXPERIMENT_REPORT.md"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


if __name__ == "__main__":
    print(build())
