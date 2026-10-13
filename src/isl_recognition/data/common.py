from pathlib import Path

from isl_recognition.config import STATIC_DATA_ROOT
from isl_recognition.labels import STATIC_LABELS

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def locate_image_root(dataset_root: Path = STATIC_DATA_ROOT) -> Path:
    for candidate in (dataset_root / "original_images", dataset_root):
        if candidate.is_dir() and any(
            (candidate / label).is_dir() for label in STATIC_LABELS
        ):
            return candidate
    raise FileNotFoundError(
        f"No alphabet image folders found under {dataset_root}; expected "
        "original_images/A through original_images/Z."
    )


def image_paths_by_label(image_root: Path, labels: tuple[str, ...] = STATIC_LABELS):
    directories: dict[str, Path] = {}
    for child in image_root.iterdir():
        if child.is_dir() and child.name.upper() in labels:
            label = child.name.upper()
            if label in directories:
                raise ValueError(f"Multiple directories map to class {label}")
            directories[label] = child
    return {
        label: sorted(
            path for path in directories[label].glob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if label in directories
        else []
        for label in labels
    }
