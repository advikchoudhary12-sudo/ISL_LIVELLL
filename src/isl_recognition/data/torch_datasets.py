import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from isl_recognition.preprocessing import preprocess_path


class StaticImageDataset(Dataset):
    def __init__(self, manifest_path: Path, split: str, dataset_root: Path, labels: tuple[str, ...], image_size: int):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if split not in ("train", "validation", "test"):
            raise ValueError(f"Unsupported split: {split}")
        self.records = manifest["splits"][split]
        self.dataset_root = dataset_root
        self.labels = labels
        self.label_to_index = {label: index for index, label in enumerate(labels)}
        self.image_size = image_size
        if not self.records:
            raise ValueError(f"The {split} split is empty")
        if any(row["label"] not in self.label_to_index for row in self.records):
            raise ValueError(f"The {split} manifest contains an unsupported label")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        row = self.records[index]
        image_path = self.dataset_root / row["path"]
        image = torch.from_numpy(preprocess_path(image_path, size=self.image_size))
        return image, self.label_to_index[row["label"]]
