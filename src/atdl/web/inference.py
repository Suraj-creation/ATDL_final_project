"""Load, run, and inspect the project's torch checkpoints."""

import base64
import hashlib
import io
import os
from threading import Lock
from time import perf_counter

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from PIL.Image import open as open_image

from .evidence import DEFAULT_MODEL, ROOT, model_registry

CANDIDATE_FLOOR = 0.01
MAX_BOXES = 300
MAX_IMAGE_BYTES = 12 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000


class InvalidInput(ValueError):
    """The frame or model selection could not be processed."""


def decode_image(data: bytes) -> np.ndarray:
    if not data or len(data) > MAX_IMAGE_BYTES:
        raise InvalidInput("Choose an image under 12 MiB.")
    try:
        with open_image(io.BytesIO(data), formats=("PNG", "JPEG", "WEBP")) as incoming:
            if incoming.width * incoming.height > MAX_IMAGE_PIXELS:
                raise InvalidInput("Use a frame at or below 25 megapixels.")
            image = ImageOps.exif_transpose(incoming).convert("RGB")
            return np.ascontiguousarray(np.asarray(image)[..., ::-1])
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise InvalidInput("Choose a valid PNG, JPEG, or WebP image.") from error


def png_url(array: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


class InspectionEngine:
    """One worker lock covers model initialization and predictor state."""

    def __init__(self, device: str):
        if device not in {"auto", "cpu", "cuda"}:
            raise ValueError("ATDL_DEVICE must be auto, cpu, or cuda.")
        self.requested_device = device
        self.device = ""
        self.device_name = ""
        self.models = {}
        self.lock = Lock()
        self.torch = None
        self.ultralytics_version = ""

    def start(self) -> None:
        os.environ.setdefault("YOLO_AUTOINSTALL", "false")
        import torch
        from ultralytics import __version__

        self.torch = torch
        self.ultralytics_version = __version__
        has_cuda = torch.cuda.is_available()
        if self.requested_device == "cuda" and not has_cuda:
            raise RuntimeError("CUDA was requested. Install a CUDA-enabled PyTorch build and verify the NVIDIA driver.")
        self.device = "cuda:0" if has_cuda and self.requested_device != "cpu" else "cpu"
        if self.device == "cpu":
            threads = int(os.environ.get("ATDL_CPU_THREADS", "4"))
            torch.set_num_threads(max(1, threads))
            self.device_name = "CPU"
        else:
            self.device_name = torch.cuda.get_device_name(0)
        with self.lock:
            self._load(DEFAULT_MODEL)

    def status(self) -> dict:
        with self.lock:
            return {
                "device": self.device or self.requested_device,
                "device_name": self.device_name,
                "torch": str(self.torch.__version__) if self.torch else None,
                "ultralytics": self.ultralytics_version,
                "loaded_models": list(self.models),
            }

    def _load(self, key: str):
        from ultralytics import YOLO
        from ultralytics.models.yolo.detect import DetectionPredictor

        class TracingPredictor(DetectionPredictor):
            capture_input = False
            capture_activations = False
            input_meta = None
            activations = None

            def preprocess(self, images):
                tensor = super().preprocess(images)
                source_h, source_w = images[0].shape[:2]
                tensor_h, tensor_w = tensor.shape[-2:]
                scale = min(tensor_w / source_w, tensor_h / source_h)
                scaled_w, scaled_h = round(source_w * scale), round(source_h * scale)
                pad_x, pad_y = (tensor_w - scaled_w) / 2, (tensor_h - scaled_h) / 2
                preview = None
                if self.capture_input:
                    rgb = (tensor[0].detach().permute(1, 2, 0).float().cpu().numpy() * 255).round().astype(np.uint8)
                    preview = png_url(rgb)
                self.input_meta = {
                    "shape": list(tensor.shape), "dtype": str(tensor.dtype).removeprefix("torch."),
                    "scale": round(scale, 6), "padding": [round(pad_x - 0.1), round(pad_y - 0.1),
                                                          round(pad_x + 0.1), round(pad_y + 0.1)],
                    "preview": preview, "rect": bool(self.args.rect),
                }
                return tensor

            def inference(self, images, *args, **kwargs):
                self.activations = []
                if not self.capture_activations:
                    return super().inference(images, *args, **kwargs)
                modules = self.model.model.model
                taps = [(0, "Early backbone"), (len(modules) // 3, "Backbone features"),
                        (len(modules) - 2, "Before detection head")]
                handles = []
                for index, label in taps:
                    def capture(_module, _inputs, output, label=label, index=index):
                        if not isinstance(output, self.torch_type) or output.ndim != 4:
                            return
                        response = output[0].detach().float().abs().mean(dim=0)
                        response -= response.min()
                        response /= response.max().clamp_min(1e-8)
                        heat = (response.cpu().numpy() * 255).round().astype(np.uint8)
                        self.activations.append({
                            "layer": index, "label": label, "shape": list(output.shape),
                            "image": png_url(heat),
                        })
                    handles.append(modules[index].register_forward_hook(capture))
                try:
                    return super().inference(images, *args, **kwargs)
                finally:
                    for handle in handles:
                        handle.remove()

        TracingPredictor.torch_type = self.torch.Tensor

        if key in self.models:
            return self.models[key]
        entry = model_registry().get(key)
        if entry is None:
            raise InvalidInput("Choose a registered Phase 8 checkpoint.")
        path = ROOT / entry["checkpoint"]
        with path.open("rb") as checkpoint:
            digest = hashlib.file_digest(checkpoint, "sha256").hexdigest()
        if digest != entry["sha256"]:
            raise RuntimeError(f"Checkpoint verification failed for {entry['label']}.")
        detector = YOLO(str(path))
        detector.predict(
            np.zeros((200, 200, 3), dtype=np.uint8), predictor=TracingPredictor,
            **self._predict_options(key),
        )
        self.models[key] = detector
        return detector

    def _predict_options(self, key: str) -> dict:
        return {
            "imgsz": model_registry()[key]["size"], "device": self.device,
            "conf": CANDIDATE_FLOOR, "max_det": MAX_BOXES, "save": False,
            "verbose": False, "rect": True,
        }

    def predict(self, data: bytes, model: str, trace: bool = False, activations: bool = False) -> dict:
        start = perf_counter()
        frame = decode_image(data)
        return self.predict_frame(frame, model, trace, activations, started=start)

    def predict_frame(self, frame: np.ndarray, model: str, trace: bool = False,
                      activations: bool = False, started: float | None = None) -> dict:
        if model not in model_registry():
            raise InvalidInput("Choose a registered Phase 8 checkpoint.")
        decoded = perf_counter()
        start = decoded if started is None else started
        with self.lock:
            acquired = perf_counter()
            detector = self._load(model)
            model_ready = perf_counter()
            predictor = detector.predictor
            predictor.capture_input = trace
            predictor.capture_activations = activations
            try:
                detection = detector.predict(frame, **self._predict_options(model))[0]
                input_meta = predictor.input_meta
                captured = predictor.activations
            finally:
                predictor.capture_input = False
                predictor.capture_activations = False
                predictor.input_meta = None
                predictor.activations = None
            end_to_end = bool(getattr(predictor.model, "end2end", False))
            boxes = [
                {"id": index, "class_name": detection.names[int(category)], "score": score, "xyxy": coordinates}
                for index, (coordinates, score, category) in enumerate(zip(
                    detection.boxes.xyxy.cpu().tolist(),
                    detection.boxes.conf.cpu().tolist(),
                    detection.boxes.cls.cpu().tolist(),
                ))
            ]
        return {
            "model": model, "device": self.device,
            "source_size": [frame.shape[1], frame.shape[0]], "input": input_meta,
            "boxes": boxes, "activations": captured, "candidate_floor": CANDIDATE_FLOOR,
            "max_candidates": MAX_BOXES,
            "postprocess": {"nms": not end_to_end, "iou": None if end_to_end else predictor.args.iou},
            "timing": {
                "decode": (decoded - start) * 1000, "queue": (acquired - decoded) * 1000,
                "load": (model_ready - acquired) * 1000, **detection.speed,
                "request": (perf_counter() - start) * 1000,
            },
        }
