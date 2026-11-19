# Final-Year Project Readiness Audit

**Project:** Class-Imbalance-Aware Steel Surface Defect Detection Using Pretrained YOLO Models<br>
**Audit date:** 19 November 2026<br>
**Workspace:** VS Code notebook project<br>
**Status:** End-to-end experiment, repeated-seed evaluation, error analysis, notebook execution, final report, and presentation are complete. Test-set comparisons remain descriptive because the historical baseline already used the same split.

## 1. What changed and was executed

1. Corrected the dataset root handling and recorded exact image hashes, 64-bit perceptual hashes, image/XML matching, and annotation counts.
2. Found byte-identical `train/images/patches/patches_101.jpg` and `patches_105.jpg`. Their XML annotations disagree slightly. The raw pair remains intact; the lower relative path (`patches_101.jpg`) is the deterministic canonical image converted for modeling, and the sibling is reported as excluded. Both source paths are assigned to the same split group. The prior split-dependent models, processed data, logs, metrics, and demo outputs were preserved under `archive/pre_duplicate_group_split/` and marked invalid for final reporting.
3. Rebuilt the model dataset with 1,799 unique image contents and 4,186 converted annotation rows. A later exact-row audit found three duplicate box rows across three labels; loaders and the Phase 8 threshold evaluator remove those rows, giving 4,183 unique boxes.
4. Added run-signature checks so Phase 3/5 checkpoints are reused only when their data and training settings match. Final-test reuse checks the checkpoint and split hashes.
5. Added F1, speed, confusion-matrix, per-class AP, and checkpoint hash fields to evaluation records.
6. Added and trained the previously missing mild and severe standard-training comparisons.
7. Added a root README, dependency pin for Ultralytics, a `.gitignore`, a unified validation table, and this audit.
8. Executed every code cell in the Phase 1–7 research notebooks and the Phase 8 comprehensive notebook without notebook errors. The project-explorer notebook distinguishes raw source files from unique image contents.
9. Completed a four-family pretrained-model sweep (three seeds per family), paired 320/640 input-size ablation, and three full-data imbalance interventions (three seeds per treatment). Candidate selection used validation only; the selected checkpoint received one final test pass.
10. Built the final Markdown, Word, and PDF reports, editable 15-slide PowerPoint, validation error-case figures, per-class metrics, and checkpoint/run manifests. The outputs are versioned in this repository.

## 2. Dataset audit

The official NEU page describes 1,800 200×200 images across six steel-surface defect classes. In the local archive, all 1,800 images are readable JPEGs and are stored as 3-channel arrays with identical channel values. The local files are therefore grayscale information encoded as RGB. The audit found one exact duplicate image pair, `patches_101.jpg` and `patches_105.jpg`, whose XML boxes differ slightly. Raw files remain intact; both paths share a split group, and only the deterministic canonical path is modeled. Separately, three exact repeated YOLO label rows were found in three processed labels and are excluded from evaluation counts. The source `train` and `validation` folders are pooled, then repartitioned into a group-aware 70/15/15 split.

| Class | Images containing class | Boxes in converted unique-image dataset |
|---|---:|---:|
| crazing | 300 | 689 |
| inclusion | 382 | 1,011 |
| patches | 341 | 878 |
| pitted_surface | 301 | 432 |
| rolled-in_scale | 300 | 628 |
| scratches | 300 | 548 |

Image presence counts exceed 300 for some categories because annotated images can contain multiple defect classes. The box counts in this table are converted annotation rows before exact repeated-row deduplication.

| Model split | Images |
|---|---:|
| Train | 1,257 |
| Validation | 266 |
| Test | 276 |
| **Total unique image contents** | **1,799** |

The converted labels contain 4,186 rows. Three exact duplicate rows are removed by Ultralytics on load and by the Phase 8 operating-point evaluator, leaving 4,183 unique boxes.

The seeded iterative multilabel split groups identical SHA-256 image contents before assignment. A 64-bit perceptual-hash audit at Hamming distance ≤5 found only the exact duplicate pair. This makes exact/perceptual leakage unlikely in this copy; the perceptual threshold is a screening rule, not a universal guarantee for every notion of a near-duplicate.

## 3. Experimental design and results

### Earlier Phase 3–5 experiments (seed 42; exploratory)

All conditions use `yolov8n.pt`, up to 50 epochs, 640 input size, batch size 16, patience 12, and seed 42. The same Phase 2 validation and test paths are used. Phase 4 varies training-image retention only. Phase 5 applies whole-image repeats or training-only transformations. No validation/test images are copied into the training variants.

| Experiment | Train images | Minority macro AP50 (val) | mAP50 (val) | mAP50–95 (val) | Macro F1 (val) |
|---|---:|---:|---:|---:|---:|
| E0 baseline | 1,257 | 0.6933 | 0.7578 | 0.4348 | 0.6834 |
| E1 mild standard | 1,109 | 0.6725 | 0.7420 | 0.4192 | 0.6853 |
| E1 moderate standard | 808 | 0.5881 | 0.6852 | 0.3860 | 0.6240 |
| E1 severe standard | 597 | 0.5694 | 0.6647 | 0.3721 | 0.5750 |
| E2 repeat factor | 1,459 | 0.6014 | 0.6855 | 0.3820 | 0.6337 |
| E3 targeted augmentation | 2,056 | 0.6070 | 0.6872 | 0.3831 | 0.6455 |
| E4 combined | 2,707 | 0.6041 | 0.6899 | 0.3741 | 0.6667 |

Minority macro AP50 is the average AP50 of `crazing`, `pitted_surface`, `rolled-in_scale`, and `scratches`. The validation-only selection rule chose the baseline (0.6933 minority macro AP50; 0.7578 overall mAP50). The severity experiment has a consistent downward trend for minority macro AP50 and overall mAP50 from baseline to mild, moderate, and severe. Under the moderate profile, repeat factor, targeted augmentation, and combined training recover some performance over the moderate standard model, but do not exceed the baseline. E1 mild has slightly higher macro F1 than E0 (0.6853 vs. 0.6834) while scoring lower on the predeclared minority AP50 objective.

The historical baseline's one-time test result is mAP50 0.7746, mAP50–95 0.4506, global F1 0.7191, and per-class macro F1 0.7127. This test split was already viewed before Phase 8. Test per-class metrics:

| Class | Precision | Recall | F1 | AP50 | AP50–95 |
|---|---:|---:|---:|---:|---:|
| crazing | 0.5200 | 0.2900 | 0.3724 | 0.3821 | 0.1547 |
| inclusion | 0.7188 | 0.8081 | 0.7609 | 0.8460 | 0.4612 |
| patches | 0.8570 | 0.8731 | 0.8650 | 0.9381 | 0.6311 |
| pitted_surface | 0.8183 | 0.7766 | 0.7969 | 0.8541 | 0.5678 |
| rolled-in_scale | 0.6516 | 0.6000 | 0.6248 | 0.6795 | 0.3034 |
| scratches | 0.8122 | 0.9055 | 0.8563 | 0.9479 | 0.5851 |

The historical baseline's main weakness is `crazing`. Phase 8 added a class-wise error review, validation-calibrated operating points, a confusion matrix, and validation-only examples; these are included in the final report and `results/comprehensive/`.

### Phase 8 repeated-seed benchmark

Four COCO-pretrained YOLO families were compared at 640-pixel input, batch 16, and 50 requested epochs. The existing YOLOv8n seed-42 baseline was reused; the other family/seed runs were recorded separately. Entries below are mean ± standard deviation over three seeds.

| Candidate group | Tail macro AP50 (val) | Overall mAP50 (val) | mAP50–95 (val) |
|---|---:|---:|---:|
| YOLOv8n | 0.6638 ± 0.0256 | 0.7358 ± 0.0190 | 0.4235 ± 0.0105 |
| YOLO11n | 0.6653 ± 0.0053 | 0.7379 ± 0.0048 | 0.4337 ± 0.0040 |
| YOLO26n | 0.6790 ± 0.0121 | 0.7489 ± 0.0080 | 0.4386 ± 0.0004 |
| YOLO11s | 0.6505 ± 0.0102 | 0.7298 ± 0.0055 | 0.4262 ± 0.0064 |
| Full-data repeat sampling | 0.6580 ± 0.0112 | 0.7345 ± 0.0041 | 0.4342 ± 0.0067 |
| Full-data targeted augmentation | 0.6865 ± 0.0129 | 0.7521 ± 0.0090 | 0.4362 ± 0.0061 |
| Full-data combined | **0.6984 ± 0.0080** | **0.7572 ± 0.0053** | **0.4402 ± 0.0037** |

The combined treatment led on the predefined validation objective. Its three-seed mean tail AP50 was 0.0345 higher and mean mAP50 0.0214 higher than the three-seed YOLOv8n group. Group selection used mean validation tail macro AP50, with overall mAP50 as the tie-break; one checkpoint was then selected using validation scores. These are small-sample validation results, not statistical-significance claims.

The three full-data methods target 62,850 image presentations (1,257 × 50). Whole epochs make exact matching impossible: planned exposures were 63,263 for repeat sampling (41 epochs), 61,677 for targeted augmentation (21 epochs), and 64,460 for the combined treatment (20 epochs), about +0.7%, −1.9%, and +2.6% from target. All originals remain in each treatment; validation/test images are not transformed or resampled.

### Input-resolution ablation and final test

For the selected YOLO26n family, paired 320-pixel runs reduced mean validation inference time from 1.564 to 0.451 ms/image and mean training time from 284.8 to 179.2 seconds, while reducing tail macro AP50 by 0.0311, mAP50 by 0.0273, and mAP50–95 by 0.0192. This efficiency ablation did not enter candidate selection.

The validation-selected candidate was `full_combined`, seed 42. Its one final test run on 276 images returned precision 0.7290, recall 0.7452, global F1 0.7370, mAP50 0.7842, mAP50–95 0.4710, tail macro AP50 0.7286, and tail macro operating-point F1 0.6998. Test AP50 was 0.396 for `crazing`, 0.843 for `inclusion`, 0.948 for `patches`, 0.869 for `pitted_surface`, 0.725 for `rolled-in_scale`, and 0.925 for `scratches`.

The historical YOLOv8n test mAP50 was 0.7746. The apparent +0.0096 mAP50 difference is descriptive only because the same test split had already been used in Phase 6. Phase 8 used validation only for candidate selection and ran one final test pass, but the project-wide comparison is not a blind independent result. One dataset and three seeds also do not establish production generalization.

## 4. Remaining work and project boundaries

The planned work for the available dataset and compute budget is complete. The final report, editable presentation, executed Phase 8 notebook, run signatures, best checkpoints, split/class summaries, validation error cases, and checkpoint manifests are included. Remaining items are follow-on validation rather than unrun steps in the current protocol:

- Evaluate on a genuinely untouched external dataset or newly reserved holdout under a fresh, pre-registered split. If the split changes, retrain every compared method consistently.
- Use more seeds or multiple grouped splits if a confirmatory estimate or formal inference is required. Three seeds describe run variability but provide limited power.
- Measure latency with deployment-specific batch size, image transfer, preprocessing, postprocessing, and target hardware before making production-line speed claims. Current Ultralytics figures are per-image validation timing on the RTX 4000 Ada.
- Recalibrate confidence thresholds if camera conditions, class prevalence, or error costs change. Saved operating-point thresholds were chosen on validation only.
- Keep the dataset archive out of Git unless its source terms explicitly permit redistribution. Code, checkpoints, metrics, and reports are pushed; raw and generated image data remain excluded.

## 5. Source and evidence files

- Dataset/image/XML audit: `results/tables/phase1_dataset_audit.json`, `results/tables/exact_duplicate_groups.csv`, `results/tables/perceptual_duplicate_candidates.csv`, `results/tables/xml_annotation_audit.json`.
- Split and label-conversion evidence: `data/splits/phase2_image_split.csv`, `results/phase2/yolo_conversion_manifest.csv`, `results/phase2/phase2_dataset_summary.json`.
- Model comparisons: `results/validation_model_comparison.csv`, `results/phase5_validation_comparison.csv`, `results/phase3_baseline_metrics.json`, `results/final_test_metrics.json`.
- Training and runtime records: `results/phase3_runtime.json`, `outputs/checkpoints/`, `outputs/logs/`.
- Phase 8 records and report figures: `results/comprehensive/`.
- Final outputs: `reports/FINAL_EXPERIMENT_REPORT.md`, `reports/FINAL_EXPERIMENT_REPORT.docx`, `reports/FINAL_EXPERIMENT_REPORT.pdf`, `reports/ATDL_Final_Project_Results.pptx`.
- Archived invalid pre-audit artifacts: `archive/pre_duplicate_group_split/README.md`.

## References

1. Northeastern University, “NEU surface defect database,” https://faculty.neu.edu.cn/songkechen/zh_CN/zhym/263269/
2. X. Lv, F. Duan, J. Jiang, X. Fu, and L. Gan, “Deep Metallic Surface Defect Detection: The New Benchmark and Detection Network,” *Sensors*, 20(6), 1562, 2020, https://doi.org/10.3390/s20061562.
3. Ultralytics, “Model Validation with Ultralytics YOLO,” https://docs.ultralytics.com/modes/val.
