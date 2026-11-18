# Trained checkpoints

Each experiment retains its best validation-selected YOLOv8n checkpoint at `weights/best.pt`, plus its training arguments, epoch metrics, run fingerprint, and evaluation curves. The matching hashes and validation metrics are catalogued in `results/checkpoint_manifest.csv`.

`E0_baseline` is the model selected by the predeclared validation rule and evaluated once on the test split. Its final test metrics are in `results/final_test_metrics.json`. Other checkpoints are comparison models; they were not selected for the final test.

Duplicate `weights/last.pt` files and batch-level training/validation preview images are omitted from version control. The base pretrained `yolov8n.pt` initialization is kept at the repository root.
