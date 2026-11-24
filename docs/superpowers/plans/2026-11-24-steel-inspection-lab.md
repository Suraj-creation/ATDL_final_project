# ATDL Steel Inspection Lab Implementation Plan

**Goal**

Run the selected steel-defect model from the browser and let a viewer inspect, compare, and understand results for the same frame.

**Architecture**

Serve the React build and Python API from one FastAPI service. Use the same verified PyTorch checkpoints for CPU and CUDA, and let the browser submit one sampled frame at a time. Read study evidence from the existing CSV and JSON artifacts.

**Stack**

FastAPI, Uvicorn, PyTorch, Ultralytics, Pillow, React, TypeScript and Vite.

**Approved scope**

Recorded footage first, webcam and still images supported. Inspect, Experiments, and Results remain the three workspaces. The model and benchmarks come from Phase 8. The demo works locally and has a documented path to required CUDA on the NVIDIA host.

## 1. Checkpoint and service

- [x] Define the contract in `tests/test_web_lab.py` for project artifacts, bad inputs, detection shape, video-frame correlation, and the chosen device.
- [x] In `src/atdl/web/evidence.py`, assemble the existing metrics, configurations, study cases, and run histories. Resolve assets relative to the repo. Serve restored validation examples only from the validation split.
- [x] In `src/atdl/web/inference.py`, load the hash-checked Phase 8 finalist by default. Share one serialized, warmed model worker. Use real letterbox previews, candidate coordinates, timings, and opt-in activation capture.
- [x] In `src/atdl/web/app.py`, implement `GET /api/health`, `GET /api/project`, `POST /api/predict`, and the `/api/stream` WebSocket. Use bounded image payloads and return structured input and device errors. Mount the production build after API routes.
- [x] Run `python -m pytest tests/test_web_lab.py -q` with the real CPU checkpoint and a generated image.

## 2. Inspection workspace

- [x] In `web/src`, implement the shell, large viewport, keyboard-operable timeline, and detections rail. Follow the visual decisions in `web/DESIGN.md`.
- [x] Use still-image HTTP inference and browser-decoded video frames over WebSocket. Keep one active request per playback session and drop requests that would draw stale overlays after a seek, model change, or source change. Show analyzed rate and skipped samples.
- [x] Add one-frame trace and activations, class and global threshold controls, same-frame model comparison, and a result export. Separate recorded sample playback, webcam, and deliberate offline sampling of every decoded frame.
- [x] Add a deterministic synthetic fixture as a first-run interaction and document how to add real footage without writing user uploads to disk. Verify recorded playback, webcam via browser fake capture, upload, pause, and seek through the built app.

## 3. Research workspaces

- [x] In `web/src/Study.tsx`, implement the XML and YOLO box explorer, imbalance distributions, augmentation preview, saved training epoch controls, model comparison, and paired resolution ablation.
- [x] Implement Results using the recorded validation ranking, final test metrics, class scores, curves, and error cases. Open restorable labeled cases in Inspect. Keep published metrics tied to the original run and split.
- [x] Verify both workspaces in the browser at desktop and phone widths and check that their controls update the view.

## 4. CPU verification and GPU handoff

- [x] Build with `npm --prefix web run build`; serve the build through Uvicorn and exercise the normal browser entry point.
- [x] Document CPU and required-CUDA startup, sample data, webcam permissions, external evaluation, and performance limits in the README. Provide one repeatable GPU preflight and compare two or more models on the same frames. Report measured CPU results and distinguish host-side CUDA measurements that still need to run.


## Verification record

The production build, nine API tests, and five browser tests passed locally with the report's Ultralytics 8.4.215 pin. The browser suite used the real checkpoints and a simulated Chromium webcam. The published NEU-DET archive restored all 12 validation cases after checking image hashes against the converted dataset manifest.

The running CPU app passed the WebSocket preflight with 30 synthetic frames and all three registered checkpoints. CUDA host execution and throughput on representative steel footage are pending access to the NVIDIA machine and its footage.
