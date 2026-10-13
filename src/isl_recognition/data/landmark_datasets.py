"""Cached MediaPipe features derived from the static labeled image corpus."""

import json
import atexit
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from isl_recognition.config import ARTIFACT_ROOT, STATIC_DATA_ROOT, PROJECT_ROOT
from isl_recognition.inference.landmarks import (
    FEATURE_SCHEMA, FEATURES_PER_HAND, LANDMARK_FEATURES, create_hands_tracker, extract_image_landmarks,
)


CACHE_PATH = ARTIFACT_ROOT / "static_landmark_features.npz"
_WORKER_HANDS = None


def _initialize_worker():
    global _WORKER_HANDS
    import cv2
    cv2.setNumThreads(1)
    _WORKER_HANDS = create_hands_tracker()
    atexit.register(_WORKER_HANDS.close)


def _extract_chunk(task):
    start, relative_paths, root_text = task
    root = Path(root_text)
    output = []
    for offset, relative_path in enumerate(relative_paths):
        values = extract_image_landmarks(root / relative_path, _WORKER_HANDS)
        if values is not None:
            output.append((start + offset, values))
    return output


def load_or_extract_features(
    manifest_path: Path | None = None,
    dataset_root: Path | None = None,
    cache_path: Path | None = None,
) -> dict:
    manifest_path = (manifest_path or PROJECT_ROOT / "data/manifests/static_splits.json").resolve()
    dataset_root = (dataset_root or STATIC_DATA_ROOT).resolve()
    cache_path = cache_path or CACHE_PATH
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    paths = sorted({row["path"] for rows in manifest["splits"].values() for row in rows})
    if cache_path.is_file():
        archive = np.load(cache_path, allow_pickle=False)
        if (str(archive["schema"].item()) == FEATURE_SCHEMA
                and str(archive["manifest"].item()) == str(manifest_path)
                and np.array_equal(archive["paths"], np.asarray(paths))):
            features = archive["features"].astype(np.float32)
            valid = archive["valid"].astype(bool)
            if features.shape == (len(paths), LANDMARK_FEATURES) and valid.shape == (len(paths),):
                return {"manifest": manifest, "paths": paths, "features": features, "valid": valid}

    features = np.zeros((len(paths), LANDMARK_FEATURES), dtype=np.float32)
    valid = np.zeros(len(paths), dtype=bool)
    index_by_path = {path: index for index, path in enumerate(paths)}
    chunk_size = 128
    tasks = [(start, paths[start:start + chunk_size], str(dataset_root))
             for start in range(0, len(paths), chunk_size)]
    workers = min(4, max(1, os.cpu_count() or 1))
    print(f"Extracting MediaPipe landmarks from {len(paths)} images with {workers} workers.", flush=True)
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context,
                             initializer=_initialize_worker) as pool:
        completed = 0
        for extracted in pool.map(_extract_chunk, tasks, chunksize=1):
            for index, values in extracted:
                features[index] = values
                valid[index] = True
            completed += chunk_size
            if completed % 2048 == 0 or completed >= len(paths):
                print(f"MediaPipe feature extraction: {min(completed, len(paths))}/{len(paths)} images; "
                      f"{int(valid.sum())} with hands", flush=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        schema=np.asarray(FEATURE_SCHEMA),
        manifest=np.asarray(str(manifest_path)),
        paths=np.asarray(paths),
        features=features,
        valid=valid,
    )
    print(f"Saved landmark feature cache: {cache_path}")
    return {"manifest": manifest, "paths": paths, "features": features, "valid": valid}


class StaticLandmarkDataset(Dataset):
    def __init__(self, manifest_path: Path, split: str, dataset_root: Path,
                 labels: tuple[str, ...], cache_path: Path | None = None):
        cached = load_or_extract_features(manifest_path, dataset_root, cache_path)
        self.labels = labels
        self.label_to_index = {label: i for i, label in enumerate(labels)}
        path_to_index = {path: i for i, path in enumerate(cached["paths"])}
        split_rows = cached["manifest"]["splits"][split]
        self.records = [row for row in split_rows
                        if cached["valid"][path_to_index[row["path"]]]]
        self.features = cached["features"]
        self.path_to_index = path_to_index
        if not self.records:
            raise ValueError(f"MediaPipe did not detect hands in the {split} split")
        self.dropped = len(split_rows) - len(self.records)
        self.detected_counts = {
            label: sum(row["label"] == label for row in self.records)
            for label in labels
        }
        self.hand_count_counts = {label: {"1": 0, "2": 0} for label in labels}
        for row in self.records:
            feature_index = self.path_to_index[row["path"]]
            vector = self.features[feature_index]
            hand_count = int(vector[FEATURES_PER_HAND - 1] > 0) + int(
                vector[2 * FEATURES_PER_HAND - 1] > 0
            )
            if hand_count in (1, 2):
                self.hand_count_counts[row["label"]][str(hand_count)] += 1

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        feature_index = self.path_to_index[row["path"]]
        features = torch.from_numpy(self.features[feature_index])
        label_index = self.label_to_index[row["label"]]
        return features, label_index
