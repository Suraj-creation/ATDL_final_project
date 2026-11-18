# Class-Imbalance-Aware Steel Surface Defect Detection

This final-year project studies object detection for the six NEU-DET hot-rolled steel surface defects and measures how controlled class imbalance affects a pretrained YOLOv8n detector. The current code is organized as VS Code Jupyter notebooks. It does not claim that imbalance-aware training improves the baseline: on the executed seed, the baseline scored best under the predeclared minority-class AP50 selection rule.

## Dataset and provenance

The dataset used here is NEU-DET from Northeastern University. The dataset page describes 1,800 200×200 images across six defect categories and provides Pascal VOC-style detection annotations. In this local copy, all 1,800 files are readable JPEGs at 200×200; they are stored as three-channel RGB with identical channels. The audit found one exact duplicate image pair, `patches_101.jpg` and `patches_105.jpg`. Their XML boxes differ slightly. The preprocessing keeps the raw files unchanged, records the disagreement, assigns both paths to one split group, and converts only the deterministic canonical path `patches_101.jpg` into the modeling dataset.

The generated dataset contains 1,799 unique image contents and 4,186 boxes. The original `train` and `validation` source folders are pooled, then repartitioned with seed 42 into a group-aware multilabel split. This follows the project methodology's proposed 70/15/15 image-level protocol; it does not preserve the source folder names as official holdouts. The model split is 1,257 train, 266 validation, and 276 test images. The raw source folders remain untouched.

Source: [NEU surface defect database](https://faculty.neu.edu.cn/songkechen/zh_CN/zhym/263269/). Cite the dataset and related literature in the final report. The local archive is `dataset/dataset.zip` and is not committed; after downloading it, extract the source folders under `data/raw/NEU-DET/NEU-DET/` so the audit and preprocessing notebooks can find the `train` and `validation` image/annotation folders. Do not redistribute the archive unless its source terms permit that.

## Research design

- Baseline: pretrained `yolov8n.pt`, trained on the full Phase 2 training split.
- Severity study: standard training on mild, moderate, and severe long-tail subsets.
- Imbalance-aware methods on the moderate subset: repeat-factor sampling by whole-image copies, targeted horizontal-flip/90°-rotation/brightness augmentation, and their combination.
- Common settings: image size 640, batch 16, up to 50 epochs, patience 12, seed 42, deterministic training. Validation and test paths stay fixed across these experiments.
- Model selection: highest validation macro AP50 over `crazing`, `pitted_surface`, `rolled-in_scale`, and `scratches`; tie-break by overall validation mAP50. The selected checkpoint is evaluated once on the test split.

## Executed results (seed 42)

| Experiment | Training images | Minority macro AP50 (val) | mAP50 (val) | mAP50–95 (val) | Macro F1 (val) |
|---|---:|---:|---:|---:|---:|
| E0 baseline | 1,257 | 0.6933 | 0.7578 | 0.4348 | 0.6834 |
| E1 mild standard | 1,109 | 0.6725 | 0.7420 | 0.4192 | 0.6853 |
| E1 moderate standard | 808 | 0.5881 | 0.6852 | 0.3860 | 0.6240 |
| E1 severe standard | 597 | 0.5694 | 0.6647 | 0.3721 | 0.5750 |
| E2 repeat factor | 1,459 | 0.6014 | 0.6855 | 0.3820 | 0.6337 |
| E3 targeted augmentation | 2,056 | 0.6070 | 0.6872 | 0.3831 | 0.6455 |
| E4 combined | 2,707 | 0.6041 | 0.6899 | 0.3741 | 0.6667 |

The selected E0 baseline test results are mAP50 0.7746, mAP50–95 0.4506, global F1 0.7191, and per-class macro F1 0.7127. Test AP50 is strongest for `scratches` (0.9479), `patches` (0.9381), and `pitted_surface` (0.8541), and weakest for `crazing` (0.3821). Test recall for `crazing` is 0.2900, making it the clearest error-analysis target. These are single-seed results and should be described as exploratory until repeated across seeds. Full records are in `results/validation_model_comparison.csv` and `results/final_test_metrics.json`.

## VS Code notebook workflow

Open this folder as the VS Code workspace. In each notebook, select the project's `.venv` Python kernel, then use **Run All** in this order:

1. `notebook/atdl_cuda_detection.ipynb` — raw dataset/image/XML audit, exact and perceptual duplicate candidates.
2. `notebook/atdl_phase2_annotation_preprocessing.ipynb` — VOC-to-YOLO conversion, grouped split, label validation, dataset YAML.
3. `notebook/atdl_phase3_baseline_training.ipynb` — baseline training and validation.
4. `notebook/atdl_phase4_imbalance_generation.ipynb` — mild, moderate, and severe training-only subsets.
5. `notebook/atdl_phase5_imbalance_methods.ipynb` — six controlled comparisons (mild/moderate/severe standard, repeat factor, targeted augmentation, combined).
6. `notebook/atdl_phase6_evaluationS.ipynb` — validation selection and the one-time final test evaluation.
7. `notebook/atdl_phase7_final_demoS.ipynb` — single-image inference demo using a validation image.
8. `notebook/atdl_vscode_project_explorer.ipynb` — workspace/output inventory.

The test notebook checks checkpoint and split hashes before reusing the saved test result. If you change the split or train a new final candidate after the test has been opened, establish a new untouched holdout for a fresh final evaluation; do not silently overwrite the current record. If you intentionally change the Phase 2 split, also set the Phase 4 `FORCE_REBUILD` and Phase 5 `FORCE_REBUILD_METHODS` flags before rerunning those phases.

## Environment

The executed environment was Python 3.14.9, PyTorch 2.15.0 with CUDA 13.2, Ultralytics 8.4.215, and an NVIDIA RTX 4000 Ada GPU. The PyTorch wheel must match the local CUDA driver. Once that is installed in the selected virtual environment, install the project packages with `python -m pip install -r requirements-local.txt`.

The repository includes the raw dataset inventory and split manifest, project YAML, executed notebooks, all experiment `best.pt` checkpoints, per-run arguments/results/fingerprints, evaluation plots, metric CSV/JSON files, and the final demo prediction. See `results/checkpoint_manifest.csv` for checkpoint hashes and validation metrics; `results/selected_model.json` identifies the model chosen without using the test split. Duplicate last-epoch checkpoints and batch-preview images are omitted.

The raw dataset archive, extracted images, and generated processed/imbalance datasets are excluded from Git. Obtain the dataset from its cited source and place it at the documented path before running the notebooks. This keeps the repository portable and avoids redistributing the dataset without permission.

## Important interpretation and remaining work

The validation trend declines from baseline to mild, moderate, and severe standard training on this seed. E3 and E4 recover some performance relative to the moderate standard run, but none surpass the baseline on the stated selection criterion. Mild standard training has a slightly higher macro F1 than baseline, while its minority AP50 is lower; this is a metric trade-off, not a contradiction.

Before submission, run at least three training seeds for each key condition and report mean and standard deviation; inspect false positives and missed defects, especially for `crazing`; complete the literature review and formal report/presentation; and state that NEU-DET is a small benchmark dataset rather than evidence of production-line generalization. See `reports/FINAL_YEAR_PROJECT_READINESS_AUDIT.md` for the in-depth audit and prioritized checklist.

## References

1. Northeastern University, “NEU surface defect database,” dataset description and download page: https://faculty.neu.edu.cn/songkechen/zh_CN/zhym/263269/
2. X. Lv, F. Duan, J. Jiang, X. Fu, and L. Gan, “Deep Metallic Surface Defect Detection: The New Benchmark and Detection Network,” *Sensors*, 20(6), 1562, 2020. https://doi.org/10.3390/s20061562
3. Ultralytics, “Model Validation with Ultralytics YOLO,” https://docs.ultralytics.com/modes/val
