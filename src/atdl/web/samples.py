"""A deterministic synthetic playback fixture, independent of NEU-DET."""

import io
from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
from PIL import Image, ImageEnhance

WIDTH, HEIGHT, FRAMES, FPS = 640, 360, 120, 24
SPEED = 3
DEFECTS = (
    ("scratches", (175, 88, 420, 143)),
    ("pitted_surface", (363, 171, 480, 275)),
    ("patches", (51, 208, 214, 287)),
    ("scratches", (707, 72, 952, 127)),
    ("pitted_surface", (850, 182, 967, 291)),
)


@lru_cache(maxsize=1)
def strip() -> np.ndarray:
    rng = np.random.default_rng(42)
    width = WIDTH + SPEED * (FRAMES - 1)
    fine = rng.normal(0, 13, (HEIGHT, width)).astype(np.float32)
    grains = cv2.blur(fine, (15, 2))
    weathering = cv2.resize(rng.normal(0, 12, (34, 68)).astype(np.float32), (width, HEIGHT), interpolation=cv2.INTER_CUBIC)
    cross_section = 40 * np.sin(np.linspace(0.15, np.pi - 0.15, HEIGHT))[:, None]
    base = np.clip(86 + 0.52 * fine + 1.5 * grains + 0.65 * weathering + cross_section, 0, 255).astype(np.uint8)
    relief = np.zeros_like(base, dtype=np.float32)
    for kind, (x1, y1, x2, y2) in DEFECTS:
        if kind == "scratches":
            for line in range(6):
                x = x1 + int(rng.integers(0, 25))
                y = y1 + line * 6
                cv2.line(relief, (x, y), (x2 - 4, y2 - 38 + line * 5), float(-32 + line), 1, cv2.LINE_AA)
                cv2.line(relief, (x, y + 2), (x2 - 4, y2 - 36 + line * 5), 15.0, 1, cv2.LINE_AA)
        elif kind == "pitted_surface":
            for _ in range(38):
                x = int(rng.normal((x1 + x2) / 2, 18))
                y = int(rng.normal((y1 + y2) / 2, 21))
                if x1 + 4 <= x < x2 - 4 and y1 + 4 <= y < y2 - 4:
                    r = int(rng.integers(1, 4))
                    cv2.circle(relief, (x + 1, y + 1), r, 31.0, -1)
                    cv2.circle(relief, (x, y), r, float(rng.integers(-70, -24)), -1)
        else:
            polygon = np.array([[x1 + 8, y1 + 25], [x1 + 46, y1 + 11], [x1 + 80, y1],
                                [x2 - 19, y1 + 7], [x2, y2 - 18], [x1 + 88, y2], [x1, y2 - 15]])
            cv2.fillPoly(relief, [polygon], -29.0)
    return np.clip(base.astype(np.float32) + cv2.GaussianBlur(relief, (5, 5), 0.65), 0, 255).astype(np.uint8)


def image_at(position: int) -> np.ndarray:
    offset = position * SPEED
    return strip()[:, offset:offset + WIDTH]


def sample_annotations(position: int) -> list[dict]:
    offset = position * SPEED
    boxes = []
    for index, (name, (left, top, right, bottom)) in enumerate(DEFECTS):
        left = max(0, left - offset)
        right = min(WIDTH, right - offset)
        if right > left:
            boxes.append({
                "id": f"sample-{index}", "class_name": name,
                "xywh": [(left + right) / (2 * WIDTH), (top + bottom) / (2 * HEIGHT),
                         (right - left) / WIDTH, (bottom - top) / HEIGHT],
            })
    return boxes


def sample_image(position: int = 0, transform: str = "original") -> bytes:
    buffer = io.BytesIO()
    image = Image.fromarray(image_at(position))
    if transform == "flip":
        image = ImageEnhance.Brightness(image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)).enhance(0.92)
    elif transform == "rotate":
        image = ImageEnhance.Brightness(image.transpose(Image.Transpose.ROTATE_270)).enhance(1.08)
    image.save(buffer, format="PNG")
    return buffer.getvalue()


@lru_cache(maxsize=1)
def sample_clip() -> bytes:
    with TemporaryDirectory(prefix="atdl-demo-") as folder:
        path = Path(folder) / "steel-coupon.webm"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"VP80"), FPS, (WIDTH, HEIGHT))
        if not writer.isOpened():
            raise RuntimeError("The OpenCV installation could not encode the sample clip.")
        try:
            for position in range(FRAMES):
                writer.write(cv2.cvtColor(image_at(position), cv2.COLOR_GRAY2BGR))
        finally:
            writer.release()
        return path.read_bytes()
