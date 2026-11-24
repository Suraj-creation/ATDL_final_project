"""Verify a running lab and measure warm prediction round trips."""

import argparse
import json
import statistics
from pathlib import Path
from time import perf_counter
from urllib.parse import urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen

from websockets.sync.client import connect

from .evidence import DEFAULT_MODEL
from .samples import FRAMES, sample_image


def request_json(url: str, body: bytes | None = None) -> dict:
    with urlopen(Request(url, data=body), timeout=30) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--image", type=Path, help="Optional steel image to send in place of the synthetic fixture.")
    options = parser.parse_args()
    if options.frames < 2:
        parser.error("--frames must be at least 2")
    base_url = options.url.rstrip("/")
    status = request_json(f"{base_url}/api/health")
    if not status["ready"]:
        raise SystemExit(status["error"] or "The engine is unavailable.")
    if options.require_cuda and not status["device"].startswith("cuda"):
        raise SystemExit("The running lab did not select CUDA.")
    project = request_json(f"{base_url}/api/project")
    if project["default_model"] != DEFAULT_MODEL:
        raise SystemExit("The running lab did not select the Phase 8 finalist.")
    image = options.image.read_bytes() if options.image else sample_image()
    for model in project["models"]:
        query = urlencode({"model": model, "trace": "true", "activations": "true"})
        result = request_json(f"{base_url}/api/predict?{query}", image)
        if result["model"] != model or result["device"] != status["device"]:
            raise SystemExit(f"Unexpected model or device in the prediction for {model}.")
        if not result["input"]["preview"] or not result["activations"]:
            raise SystemExit(f"Missing trace artifacts for {model}.")

    parts = urlsplit(base_url)
    ws_url = urlunsplit(("wss" if parts.scheme == "https" else "ws", parts.netloc, "/api/stream", "", ""))
    payloads = [image if options.image else sample_image(i % FRAMES) for i in range(options.frames)]
    round_trips, model_ms = [], []
    with connect(ws_url, max_size=12 * 1024 * 1024, open_timeout=15) as socket:
        for i, payload in enumerate(payloads):
            start = perf_counter()
            socket.send(json.dumps({"id": i, "source_time": i / 24, "model": DEFAULT_MODEL}))
            socket.send(payload)
            reply = json.loads(socket.recv(timeout=30))
            round_trips.append((perf_counter() - start) * 1000)
            if reply.get("id") != i or reply.get("error"):
                raise SystemExit(f"Stream verification failed for frame {i}: {reply.get('error')}")
            model_ms.append(reply["result"]["timing"]["inference"])
    print(json.dumps({
        "device": status["device"], "device_name": status["device_name"],
        "torch": status["torch"], "ultralytics": status["ultralytics"],
        "verified_checkpoints": list(project["models"]), "stream_frames": len(round_trips),
        "frames_source": "provided still image" if options.image else "synthetic fixture",
        "socket_round_trip_ms": {
            "p50": round(statistics.median(round_trips), 2),
            "p95": round(statistics.quantiles(round_trips, n=20, method="inclusive")[-1], 2),
        },
        "model_inference_p50_ms": round(statistics.median(model_ms), 2),
    }, indent=2))


if __name__ == "__main__":
    main()
