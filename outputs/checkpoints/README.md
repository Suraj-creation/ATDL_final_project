# Trained checkpoints

Each experiment retains its best validation-selected checkpoint at `weights/best.pt`, plus its training arguments, epoch metrics, run fingerprint, and evaluation curves. Phase 3/5 contain the original YOLOv8n single-seed study. Phase 8 contains repeated-seed transfer experiments for YOLOv8n, YOLO11n, YOLO26n, and YOLO11s, a paired 320/640 input-size ablation, and full-data imbalance interventions on the architecture chosen by validation. Matching hashes and validation metrics are catalogued in `results/checkpoint_manifest.csv`.

`E0_baseline` is the historical Phase 3 model selected by validation and evaluated on test in Phase 6. Phase 8 selects a new final candidate using validation only and records its one-pass test evaluation separately under `results/comprehensive/test_evaluations/`. The existing test split has already been seen in the historical baseline evaluation, so the Phase 8 test comparison is descriptive rather than a fully blind final holdout.

Duplicate `weights/last.pt` files and batch-level training/validation preview images are omitted from version control. The pretrained `yolov8n.pt`, `yolo11n.pt`, `yolo26n.pt`, and `yolo11s.pt` initializations are kept at the repository root with their hashes in `results/comprehensive/pretrained_weights_manifest.csv`.
