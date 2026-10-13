from pathlib import Path

import numpy as np
from PIL import Image


def preprocessing_config(size: int) -> dict:
    return {
        "frame": {
            "size": size,
            "color": "RGB",
            "resize": "aspect-preserving bilinear",
            "padding": "centered black",
            "tensor_layout": "CHW",
            "dtype": "float32",
            "value_range": [0.0, 1.0],
        }
    }


def preprocess_image(image: Image.Image, size: int = 96) -> np.ndarray:
    """Convert to RGB, preserve aspect ratio, and center-pad to a float32 CHW image."""
    if size <= 0:
        raise ValueError("size must be positive")
    image = image.convert("RGB")
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError("image must have positive dimensions")

    scale = min(size / width, size / height)
    resized = image.resize(
        (max(1, round(width * scale)), max(1, round(height * scale))),
        Image.Resampling.BILINEAR,
    )
    canvas = Image.new("RGB", (size, size), color=(0, 0, 0))
    left = (size - resized.width) // 2
    top = (size - resized.height) // 2
    canvas.paste(resized, (left, top))
    pixels = np.asarray(canvas, dtype=np.float32) / 255.0
    return np.transpose(pixels, (2, 0, 1)).copy()


def preprocess_path(path: Path, size: int = 96) -> np.ndarray:
    with Image.open(path) as image:
        return preprocess_image(image, size=size)
