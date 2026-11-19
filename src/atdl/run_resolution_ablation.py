"""Compare 320 px and 640 px inputs for the validation-selected model family."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import statistics
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from src.atdl.run_comprehensive_study import (
    BATCH_SIZE,
    DATA_YAML,
    EPOCHS,
    IMAGE_SIZE,
    MODEL_WEIGHTS,
    PATIENCE,
    SEEDS,
    _read_epoch_count,
    _read_elapsed,
    extract_metrics,
    load_environment,
    relative,
    sha256_file,
)

ROOT = Path(__file__).resolve().parents[2]
SIZE = 320
RUN_ROOT = ROOT / "outputs/checkpoints/phase8/resolution_ablation"
VALIDATION_ROOT = ROOT / "outputs/logs/phase8/resolution_ablation"
RESULTS = ROOT / "results/comprehensive"


def _path_metric(record: dict[str, Any], key: str) -> float:
    return float(record.get("global_metrics", {}).get(key, 0.0))


def _write_records(records: list[dict[str, Any]], model_name: str) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    records_path = RESULTS / "resolution_ablation_runs.json"
    records_path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    csv_rows = [{
        "model_name": r["model_name"], "seed": r["seed"], "imgsz": r["imgsz"],
        "epochs_requested": r["epochs_requested"], "epochs_completed": r["epochs_completed"],
        "minority_macro_AP50": r["minority_macro_ap50"],
        "minority_macro_recall": r["minority_macro_recall"],
        "minority_macro_F1": r["minority_macro_f1"],
        "mAP50": _path_metric(r, "metrics/mAP50(B)"),
        "mAP50_95": _path_metric(r, "metrics/mAP50-95(B)"),
        "precision": _path_metric(r, "metrics/precision(B)"),
        "recall": _path_metric(r, "metrics/recall(B)"),
        "global_F1": _path_metric(r, "metrics/F1(B)"),
        "macro_F1": r["macro_f1"],
        "inference_ms_per_image": r.get("validation_speed_ms_per_image", {}).get("inference"),
        "training_seconds": r["training_seconds"],
        "parameters": r["parameters"], "checkpoint_bytes": r["checkpoint_bytes"],
        "checkpoint": r["checkpoint"], "checkpoint_sha256": r["checkpoint_sha256"],
    } for r in records]
    with (RESULTS / "resolution_ablation_runs.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=csv_rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(csv_rows)

    baseline_path = RESULTS / "model_transfer_runs.json"
    baseline_records = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline = [r for r in baseline_records if r["model_name"] == model_name]
    by_seed = {int(r["seed"]): r for r in baseline}
    paired_deltas = {}
    metrics = [
        ("minority_macro_ap50", "minority_macro_ap50"),
        ("minority_macro_recall", "minority_macro_recall"),
        ("minority_macro_f1", "minority_macro_f1"),
        ("metrics/mAP50(B)", "metrics/mAP50(B)"),
        ("metrics/mAP50-95(B)", "metrics/mAP50-95(B)"),
        ("metrics/precision(B)", "metrics/precision(B)"),
        ("metrics/recall(B)", "metrics/recall(B)"),
        ("metrics/F1(B)", "metrics/F1(B)"),
        ("validation_inference_ms_per_image", "validation_inference_ms_per_image"),
        ("training_seconds", "training_seconds"),
    ]
    for label, key in metrics:
        deltas = []
        for record in records:
            base = by_seed[int(record["seed"])]
            if key.startswith("metrics/"):
                delta = _path_metric(record, key) - _path_metric(base, key)
            elif key == "validation_inference_ms_per_image":
                delta = float(record.get("validation_speed_ms_per_image", {}).get("inference", 0.0)) - float(base.get("validation_speed_ms_per_image", {}).get("inference", 0.0))
            else:
                reference_key = "training_seconds" if key == "training_seconds" else key
                delta = float(record[reference_key]) - float(base[reference_key])
            deltas.append(delta)
        paired_deltas[label] = {"mean_delta_320_minus_640": statistics.mean(deltas),
                                "sd_paired_delta": statistics.stdev(deltas) if len(deltas) > 1 else 0.0,
                                "per_seed_delta": deltas}
    summary = {
        "model_name": model_name,
        "selection_split_for_model_family": "validation",
        "comparison": "paired seeds; same COCO pretrained weights, dataset split, epochs, batch, optimizer and patience; only imgsz changes from 640 to 320",
        "imgsz_640_reference": {"mean_tail_macro_AP50": statistics.mean(float(r["minority_macro_ap50"]) for r in baseline),
                                 "sd_tail_macro_AP50": statistics.stdev(float(r["minority_macro_ap50"]) for r in baseline),
                                 "mean_mAP50": statistics.mean(_path_metric(r, "metrics/mAP50(B)") for r in baseline),
                                 "sd_mAP50": statistics.stdev(_path_metric(r, "metrics/mAP50(B)") for r in baseline),
                                 "mean_mAP50_95": statistics.mean(_path_metric(r, "metrics/mAP50-95(B)") for r in baseline),
                                 "sd_mAP50_95": statistics.stdev(_path_metric(r, "metrics/mAP50-95(B)") for r in baseline),
                                 "mean_inference_ms_per_image": statistics.mean(float(r.get("validation_speed_ms_per_image", {}).get("inference", 0.0)) for r in baseline),
                                 "sd_inference_ms_per_image": statistics.stdev(float(r.get("validation_speed_ms_per_image", {}).get("inference", 0.0)) for r in baseline),
                                 "mean_training_seconds": statistics.mean(float(r["training_seconds"]) for r in baseline)},
        "imgsz_320": {"n_seeds": len(records),
                      "mean_tail_macro_AP50": statistics.mean(float(r["minority_macro_ap50"]) for r in records),
                      "sd_tail_macro_AP50": statistics.stdev(float(r["minority_macro_ap50"]) for r in records),
                      "mean_mAP50": statistics.mean(_path_metric(r, "metrics/mAP50(B)") for r in records),
                      "sd_mAP50": statistics.stdev(_path_metric(r, "metrics/mAP50(B)") for r in records),
                      "mean_mAP50_95": statistics.mean(_path_metric(r, "metrics/mAP50-95(B)") for r in records),
                      "sd_mAP50_95": statistics.stdev(_path_metric(r, "metrics/mAP50-95(B)") for r in records),
                      "mean_inference_ms_per_image": statistics.mean(float(r.get("validation_speed_ms_per_image", {}).get("inference", 0.0)) for r in records),
                      "sd_inference_ms_per_image": statistics.stdev(float(r.get("validation_speed_ms_per_image", {}).get("inference", 0.0)) for r in records),
                      "mean_training_seconds": statistics.mean(float(r["training_seconds"]) for r in records)},
        "paired_deltas": paired_deltas,
        "test_split_read": False,
        "records": [r["experiment"] for r in records],
    }
    (RESULTS / "resolution_ablation_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    flat = {"model_name": model_name, "n_seeds": len(records)}
    for label, value in summary["imgsz_640_reference"].items():
        flat[f"imgsz640_{label}"] = value
    for label, value in summary["imgsz_320"].items():
        flat[f"imgsz320_{label}"] = value
    for label, value in paired_deltas.items():
        flat[f"delta_{label}"] = value["mean_delta_320_minus_640"]
        flat[f"delta_sd_{label}"] = value["sd_paired_delta"]
    with (RESULTS / "resolution_ablation_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=flat.keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerow(flat)

    figure_dir = RESULTS / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    labels = ["Tail macro AP50", "Overall mAP50"]
    base_vals = [summary["imgsz_640_reference"]["mean_tail_macro_AP50"], summary["imgsz_640_reference"]["mean_mAP50"]]
    small_vals = [summary["imgsz_320"]["mean_tail_macro_AP50"], summary["imgsz_320"]["mean_mAP50"]]
    base_err = [summary["imgsz_640_reference"]["sd_tail_macro_AP50"], summary["imgsz_640_reference"]["sd_mAP50"]]
    small_err = [summary["imgsz_320"]["sd_tail_macro_AP50"], summary["imgsz_320"]["sd_mAP50"]]
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    x = [0, 1]
    width = 0.32
    ax.bar([i - width/2 for i in x], base_vals, width, yerr=base_err, capsize=4, color="#526477", label="640 px")
    ax.bar([i + width/2 for i in x], small_vals, width, yerr=small_err, capsize=4, color="#087E8B", label="320 px")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Validation score (mean across paired seeds)")
    ax.set_title(f"Input-size ablation: {model_name}")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figure_dir / "resolution_ablation_comparison.png", dpi=180)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


def run_one(model_name: str, seed: int, YOLO: Any, torch: Any,
            environment: dict[str, Any], force: bool = False) -> dict[str, Any]:
    pretrained = MODEL_WEIGHTS[model_name]
    run_dir = RUN_ROOT / model_name / f"imgsz_{SIZE}" / f"seed_{seed}"
    checkpoint = run_dir / "weights" / "best.pt"
    signature_path = run_dir / "run_signature.json"
    metrics_path = RESULTS / "runs" / f"{model_name}_imgsz{SIZE}_seed{seed}.json"
    signature = {
        "stage": "resolution_ablation", "model_name": model_name,
        "pretrained_sha256": sha256_file(pretrained),
        "data_yaml_sha256": sha256_file(DATA_YAML),
        "split_manifest_sha256": sha256_file(ROOT / "data/splits/phase2_image_split.csv"),
        "imgsz": SIZE, "reference_imgsz": IMAGE_SIZE,
        "epochs": EPOCHS, "batch": BATCH_SIZE, "patience": PATIENCE,
        "optimizer": "auto", "seed": seed, "deterministic": True,
    }
    if force:
        if run_dir.exists():
            shutil.rmtree(run_dir)
        metrics_path.unlink(missing_ok=True)
    if checkpoint.is_file() and signature_path.is_file():
        if json.loads(signature_path.read_text(encoding="utf-8")) != signature:
            raise RuntimeError(f"Stale resolution checkpoint; rerun with --force: {checkpoint}")
    else:
        if run_dir.exists():
            shutil.rmtree(run_dir)
        run_dir.parent.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        model = YOLO(str(pretrained))
        model.train(data=str(DATA_YAML), epochs=EPOCHS, imgsz=SIZE, batch=BATCH_SIZE,
                    patience=PATIENCE, seed=seed, project=str(run_dir.parent), name=run_dir.name,
                    exist_ok=True, device=0, workers=4, deterministic=True,
                    optimizer="auto", plots=True, save=True, val=True)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Training did not create {checkpoint}")
        signature_path.write_text(json.dumps(signature, indent=2) + "\n", encoding="utf-8")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elapsed_fallback = time.perf_counter() - started
    if metrics_path.is_file():
        record = json.loads(metrics_path.read_text(encoding="utf-8"))
        if record["checkpoint_sha256"] != sha256_file(checkpoint):
            raise RuntimeError(f"Saved resolution metrics do not match checkpoint: {metrics_path}")
        record["epochs_completed"] = _read_epoch_count(run_dir)
        record["training_seconds"] = round(_read_elapsed(run_dir, record.get("training_seconds", 0.0)), 3)
        record["checkpoint_bytes"] = checkpoint.stat().st_size
        metrics_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        return record
    if "elapsed_fallback" not in locals():
        elapsed_fallback = 0.0
    start = time.perf_counter()
    model = YOLO(str(checkpoint))
    result = model.val(data=str(DATA_YAML), split="val", imgsz=SIZE, batch=BATCH_SIZE,
                       project=str(VALIDATION_ROOT / model_name),
                       name=f"seed_{seed}", exist_ok=True, device=0,
                       plots=True, save_txt=False)
    params = sum(p.numel() for p in model.model.parameters())
    elapsed = _read_elapsed(run_dir, elapsed_fallback or (time.perf_counter() - start))
    record = extract_metrics(result, model_name, seed, checkpoint,
                             signature["pretrained_sha256"], elapsed, params,
                             _read_epoch_count(run_dir))
    record.update({
        "experiment": f"{model_name}_imgsz{SIZE}_seed{seed}",
        "stage": "resolution_ablation", "imgsz": SIZE,
        "reference_imgsz": IMAGE_SIZE,
        "resolution_selection_candidate": False,
        "environment": environment,
        "signature": signature,
        "metrics_path": relative(metrics_path),
    })
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return record


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="retrain the 320 px runs")
    args = parser.parse_args()
    selection_path = RESULTS / "model_transfer_selection.json"
    if not selection_path.is_file():
        raise FileNotFoundError("Complete the architecture transfer stage first.")
    model_name = json.loads(selection_path.read_text(encoding="utf-8"))["selected_model"]
    torch, YOLO, environment = load_environment()
    records = []
    for seed in SEEDS:
        print(f"\nResolution ablation: {model_name}, imgsz={SIZE}, seed={seed}", flush=True)
        record = run_one(model_name, seed, YOLO, torch, environment, force=args.force)
        records.append(record)
        print(f"tail AP50={record['minority_macro_ap50']:.4f}; "
              f"mAP50={record['global_metrics'].get('metrics/mAP50(B)', 0):.4f}; "
              f"training={record['training_seconds']:.1f}s", flush=True)
    _write_records(_load_all(model_name), model_name)


def _load_all(model_name: str) -> list[dict[str, Any]]:
    rows = []
    for seed in SEEDS:
        path = RESULTS / "runs" / f"{model_name}_imgsz{SIZE}_seed{seed}.json"
        if path.is_file():
            rows.append(json.loads(path.read_text(encoding="utf-8")))
    return rows


if __name__ == "__main__":
    main()
