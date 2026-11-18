# Final-Year Project Readiness Audit

**Project:** Class-Imbalance-Aware Steel Surface Defect Detection Using Pretrained YOLO Models<br>
**Audit date:** 16 November 2026<br>
**Workspace:** VS Code notebook project<br>
**Status:** Notebook pipeline rerun; academic submission still needs repeated-seed evidence, error analysis, literature review, and final report/presentation.

## 1. What changed and was executed

1. Corrected the dataset root handling and recorded exact image hashes, 64-bit perceptual hashes, image/XML matching, and annotation counts.
2. Found byte-identical `train/images/patches/patches_101.jpg` and `patches_105.jpg`. Their XML annotations disagree slightly. The raw pair remains intact; the lower relative path (`patches_101.jpg`) is the deterministic canonical image converted for modeling, and the sibling is reported as excluded. Both source paths are assigned to the same split group. The prior split-dependent models, processed data, logs, metrics, and demo outputs were preserved under `archive/pre_duplicate_group_split/` and marked invalid for final reporting.
3. Rebuilt the model dataset with 1,799 unique image contents and 4,186 boxes. Phase 2 validates every converted YOLO label and writes split/class-count tables.
4. Added run-signature checks so Phase 3/5 checkpoints are reused only when their data and training settings match. Final-test reuse checks the checkpoint and split hashes.
5. Added F1, speed, confusion-matrix, per-class AP, and checkpoint hash fields to evaluation records.
6. Added and trained the previously missing mild and severe standard-training comparisons.
7. Added a root README, dependency pin for Ultralytics, a `.gitignore`, a unified validation table, and this audit.
8. Executed every code cell in the seven research notebooks (Phases 1–7) without notebook errors. The project-explorer notebook was rerun after documentation updates and its split summary now distinguishes raw source files from unique image contents.

## 2. Dataset audit

The official NEU page describes 1,800 200×200 images across six steel-surface defect classes. In the local archive, all 1,800 images are readable JPEGs and are stored as 3-channel arrays with identical channel values. The local files are therefore grayscale information encoded as RGB. The project contains images under the original `train` and `validation` folders; the documented protocol pools those folders and creates a new image-level 70/15/15 split. This choice is stated explicitly so the original folder names are not mistaken for the final evaluation split.

| Class | Images containing class | Boxes in converted unique-image dataset |
|---|---:|---:|
| crazing | 300 | 689 |
| inclusion | 382 | 1,011 |
| patches | 341 | 878 |
| pitted_surface | 301 | 432 |
| rolled-in_scale | 300 | 628 |
| scratches | 300 | 548 |

Image presence counts exceed 300 for some categories because annotated images can contain multiple defect classes. One duplicate group's slightly different box annotations account for three removed repeated boxes in the canonical modeling set.

| Model split | Images |
|---|---:|
| Train | 1,257 |
| Validation | 266 |
| Test | 276 |
| **Total unique image contents** | **1,799** |

The seeded iterative multilabel split groups identical SHA-256 image contents before assignment. A 64-bit perceptual-hash audit at Hamming distance ≤5 found only the exact duplicate pair. This makes exact/perceptual leakage unlikely in this copy; the perceptual threshold is a screening rule, not a universal guarantee for every notion of a near-duplicate.

## 3. Experimental design and results

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

The baseline's one-time held-out test result is mAP50 0.7746, mAP50–95 0.4506, global F1 0.7191, and per-class macro F1 0.7127. Test per-class metrics:

| Class | Precision | Recall | F1 | AP50 | AP50–95 |
|---|---:|---:|---:|---:|---:|
| crazing | 0.5200 | 0.2900 | 0.3724 | 0.3821 | 0.1547 |
| inclusion | 0.7188 | 0.8081 | 0.7609 | 0.8460 | 0.4612 |
| patches | 0.8570 | 0.8731 | 0.8650 | 0.9381 | 0.6311 |
| pitted_surface | 0.8183 | 0.7766 | 0.7969 | 0.8541 | 0.5678 |
| rolled-in_scale | 0.6516 | 0.6000 | 0.6248 | 0.6795 | 0.3034 |
| scratches | 0.8122 | 0.9055 | 0.8563 | 0.9479 | 0.5851 |

The model's main weakness is `crazing`: test recall is 0.29 and AP50 is 0.382. `rolled-in_scale` is the next concern, with recall 0.60 and AP50 0.680. The current demo saves one validation-image prediction. The final test metrics JSON also saves the confusion matrix and its class/background order; Ultralytics validation logs contain PR/F1 curves, normalized confusion matrices, and sampled prediction/label panels.

## 4. Remaining work ranked for a strong final-year submission

### P0 — needed before making a strong research claim

- **Repeat training with multiple seeds.** Current comparisons use only seed 42. Run at least three training seeds for the baseline and all planned methods (keep the split fixed for training-randomness analysis), then report mean ± standard deviation. If you also vary the split, keep that as a separately stated split-robustness experiment.
- **Do not claim the imbalance-aware methods improve over baseline.** The current validation table does not support that statement. A defensible current result is that controlled training imbalance reduces validation performance on this run and the tested methods partly recover the moderate-profile score but not the baseline.
- **Complete error analysis.** Review missed `crazing` and `rolled-in_scale` boxes, false positives by category, confidence thresholds, and representative predictions. Select thresholds only on validation; leave the saved test record untouched. Add examples and a per-class discussion to the final report.
- **Write the formal report and presentation.** The existing Word document is a methodology/proposal, not the final report. Include the research question, related work, dataset provenance, split/duplicate protocol, exact experiment matrix, results, limitations, and citations.

### P1 — improve the experimental argument

- Explain the multi-label selection rule for the long-tail subsets and show exact image/object distributions by split and experiment. These are already available in `results/phase4_final_distribution_report.csv` and `results/phase5_method_distribution.csv`; include them in the report.
- Add a compute-aware input-size ablation (for example 320 versus 640) because source images are only 200×200. Keep model and split fixed and choose using validation.
- Add a quantitative model-size/latency/throughput table with batch size, GPU, image size, warm-up, and timing method. Current Ultralytics speed is batched validation timing on the RTX 4000 Ada; it is not end-to-end production-line latency.
- Add qualitative failure examples to the demo or report and explain whether the issue is missed small texture defects, annotation variation, or class similarity.

### P2 — polish and reproducibility

- Add a bibliography and cite the official NEU database page and the dataset-related research papers. Record dataset archive SHA-256, model checkpoint SHA-256, seeds, package versions, and the exact split manifest in the final report.
- Keep the `.venv` kernel selected in VS Code. When changing the split, rebuild Phase 4 variants and Phase 5 method data before training; do not reuse old split-dependent metrics or checkpoints.
- If redistributing the dataset archive with the final submission, verify the source's permitted use and redistribution terms.

## 5. Source and evidence files

- Dataset/image/XML audit: `results/tables/phase1_dataset_audit.json`, `results/tables/exact_duplicate_groups.csv`, `results/tables/perceptual_duplicate_candidates.csv`, `results/tables/xml_annotation_audit.json`.
- Split and label-conversion evidence: `data/splits/phase2_image_split.csv`, `results/phase2/yolo_conversion_manifest.csv`, `results/phase2/phase2_dataset_summary.json`.
- Model comparisons: `results/validation_model_comparison.csv`, `results/phase5_validation_comparison.csv`, `results/phase3_baseline_metrics.json`, `results/final_test_metrics.json`.
- Training and runtime records: `results/phase3_runtime.json`, `outputs/checkpoints/`, `outputs/logs/`.
- Archived invalid pre-audit artifacts: `archive/pre_duplicate_group_split/README.md`.

## References

1. Northeastern University, “NEU surface defect database,” https://faculty.neu.edu.cn/songkechen/zh_CN/zhym/263269/
2. X. Lv, F. Duan, J. Jiang, X. Fu, and L. Gan, “Deep Metallic Surface Defect Detection: The New Benchmark and Detection Network,” *Sensors*, 20(6), 1562, 2020, https://doi.org/10.3390/s20061562.
3. Ultralytics, “Model Validation with Ultralytics YOLO,” https://docs.ultralytics.com/modes/val.
