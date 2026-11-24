# ATDL Steel Inspection Lab

Run the research checkpoint in a browser. Inspect a still or video frame, pause and compare checkpoints, and walk through the recorded training and validation evidence.

## Start on CPU

Run these commands from the ATDL repository root. Use Python 3.12 or a Python version supported by the chosen PyTorch wheel and Node.js 22.12+.

1. Create and activate a Python virtual environment. Install the CPU builds of PyTorch and torchvision using the [official installer selector](https://pytorch.org/get-started/locally/).
2. Install the lab and build the browser assets.

   ```sh
   python -m pip install -r requirements-web.txt
   npm --prefix web ci
   npm --prefix web run build
   ```

3. Start the server in PowerShell and open [the inspection lab](http://127.0.0.1:8765).

   ```powershell
   $env:ATDL_DEVICE = 'cpu'
   python -m uvicorn src.atdl.web.app:app --host 127.0.0.1 --port 8765 --ws-max-size 12582912
   ```

   On Linux or macOS use `ATDL_DEVICE=cpu python -m uvicorn src.atdl.web.app:app --host 127.0.0.1 --port 8765 --ws-max-size 12582912`. The default device setting is `auto`. Adjust `ATDL_CPU_THREADS` if profiling suggests a better setting for the host.

The included five-second coupon is a deterministic synthetic clip. Upload an actual steel image or a browser-decodable clip via **Open media**. The webcam works on localhost or HTTPS with camera permission. For live playback the browser sends one frame at a time and samples up to 24 fps. Frame size for browser video requests is limited to a 1,280 px longest side. Discarded samples are counted. **Analyze every frame** uses the server decoder and the original video dimensions, processes each decoded frame sequentially, and accepts clips up to 64 MiB. The offline summary counts detections across frames. Uploaded stills and live frames stay in memory, and the offline decoder uses a temporary video file that is removed when the job ends.

The finalist uses the recorded per-class validation thresholds. Other checkpoints start at a shared 0.25 cutoff unless a global override is active. A fixed 0.01 candidate floor and up to 300 returned candidates bound the browser controls. Browser round trip includes frame capture and transfer for streamed frames and is measured through receipt of the prediction, before rendering.

## Restore labeled validation cases

The application can display the saved error montage in a clean checkout. To open the individual cases, obtain NEU-DET from the [dataset publisher](https://faculty.neu.edu.cn/songkc/en/zdylm/263265/) and restore the examples from the local zip:

```sh
python -m src.atdl.web.restore_validation path/to/NEU-DET.zip
```

The tool checks the source-image hashes against the Phase 2 conversion manifest and writes the 12 advertised examples under the ignored `data/processed` folder. The rest of the dataset can be restored through the original notebooks if needed. Reload Results after the files are available.

## Run on the NVIDIA machine

Use the same repo, build, and checkpoint files on the NVIDIA host. Select a CUDA-enabled PyTorch and torchvision pair for that host's installed driver, then install the lab dependencies. Check `nvidia-smi` and `python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"` before starting.

```powershell
$env:ATDL_DEVICE = 'cuda'
python -m uvicorn src.atdl.web.app:app --host 127.0.0.1 --port 8765 --ws-max-size 12582912
```

On Linux prefix the same command with `ATDL_DEVICE=cuda`. Use one server worker. A required CUDA startup failure reports the reason in the device status and leaves the research views accessible. The API reports the actual selected device.

In another terminal, verify all three checkpoints, actual tensor previews, feature activations, and correlated stream responses:

```sh
python -m src.atdl.web.verify --require-cuda --frames 60
```

Add `--image path/to/steel.jpg` to exercise the same paths on local steel imagery. This command reports request and model timings. Measure visible-overlay latency and analyzed throughput on the actual browser and network when commissioning a live source. The first target is one 720p clip at 10 analyzed fps with p95 overlay latency below 150 ms. Use HTTPS and a private access boundary if the GPU host is accessed remotely, especially for webcam use.

For an accuracy evaluation, reserve labeled footage from distinct coils or capture sessions before tuning. Keep analysis and evaluation sources distinct, record the checkpoint and thresholds, and include clean samples when counting false positives. The saved Results tab shows NEU-DET study results, and the synthetic clip is a workflow fixture.

## Verification commands

```sh
python -m pip install pytest httpx
python -m pytest tests/test_web_lab.py -q
npm --prefix web run build
npm --prefix web run test:e2e
```

The browser tests use installed Chrome and Chromium's fake camera. Set `ATDL_BROWSER=msedge` for Edge or `ATDL_PYTHON` to the intended Python executable. Stop any server on port 8765 before running the browser suite on the GPU, then set `ATDL_DEVICE=cuda` and rerun the browser suite. A test checks the required device.
