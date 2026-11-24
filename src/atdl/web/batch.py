"""Stream an inspection summary for each frame decoded in sequence."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2

from .inference import InvalidInput, MAX_IMAGE_PIXELS

MAX_VIDEO_BYTES = 64 * 1024 * 1024


def analyze_video(engine, payload: bytes, model: str, thresholds: dict):
    with TemporaryDirectory(prefix="atdl-inspection-") as folder:
        path = Path(folder) / "source.video"
        path.write_bytes(payload)
        video = cv2.VideoCapture(str(path))
        try:
            if not video.isOpened():
                raise InvalidInput("The server could not decode this video.")
            expected = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            yield json.dumps({"kind": "start", "expected": max(0, expected)}) + "\n"
            index = 0
            while True:
                success, frame = video.read()
                if not success:
                    break
                if frame.shape[0] * frame.shape[1] > MAX_IMAGE_PIXELS:
                    raise InvalidInput("Use video frames at or below 25 megapixels.")
                result = engine.predict_frame(frame, model)
                counts = {name: 0 for name in thresholds}
                for box in result["boxes"]:
                    name = box["class_name"]
                    if box["score"] >= thresholds[name]:
                        counts[name] += 1
                index += 1
                yield json.dumps({
                    "kind": "frame", "index": index, "detections": counts,
                    "inference_ms": result["timing"].get("inference"),
                }) + "\n"
            if not index:
                raise InvalidInput("The video did not contain a decodable frame.")
            yield json.dumps({"kind": "complete", "frames": index, "expected": max(0, expected)}) + "\n"
        finally:
            video.release()
