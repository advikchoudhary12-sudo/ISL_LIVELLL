import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from isl_recognition.config import PROJECT_ROOT, STATIC_DATA_ROOT, load_config
from isl_recognition.data.common import image_paths_by_label, locate_image_root
from isl_recognition.data.validate import inspect_dataset
from isl_recognition.labels import STATIC_LABELS
from isl_recognition.preprocessing import preprocessing_config

DATASET_ID = "atharvadumbre/indian-sign-language-islrtc-referred"


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def difference_hash(path: Path) -> int:
    with Image.open(path) as image:
        gray = image.convert("L").resize((9, 8), Image.Resampling.BILINEAR)
        values = np.asarray(gray, dtype=np.uint8)
    bits = values[:, 1:] > values[:, :-1]
    result = 0
    for bit in bits.flat:
        result = (result << 1) | int(bit)
    return result


def _split_groups(records: list[dict], seed: int) -> dict[str, list[dict]]:
    groups_by_label: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        groups_by_label[record["label"]][record["group_id"]].append(record)

    output: dict[str, list[dict]] = {"train": [], "validation": [], "test": []}
    for label in STATIC_LABELS:
        group_ids = sorted(groups_by_label[label])
        if len(group_ids) < 5:
            raise ValueError(f"Class {label} has only {len(group_ids)} independent image groups; at least 5 are required.")
        shuffled = random.Random(seed + ord(label)).sample(group_ids, len(group_ids))
        test_n = max(1, round(len(shuffled) * 0.15))
        test_ids = set(shuffled[:test_n])
        remainder = shuffled[test_n:]
        validation_n = max(1, round(len(group_ids) * 0.15))
        validation_ids = set(remainder[:validation_n])
        for group_id, group_records in groups_by_label[label].items():
            part = "test" if group_id in test_ids else "validation" if group_id in validation_ids else "train"
            output[part].extend(group_records)
    for records_in_split in output.values():
        records_in_split.sort(key=lambda row: (row["label"], row["path"]))
    return output


def build_manifest(dataset_root: Path, seed: int) -> dict:
    validation = inspect_dataset(dataset_root)
    if validation["status"] != "ok":
        raise ValueError(
            "Dataset validation must pass before splitting; "
            f"missing={validation['missing_source_classes']}, "
            f"invalid={len(validation['invalid_or_conflicting_samples'])}"
        )
    image_root = locate_image_root(dataset_root)
    paths_by_label = image_paths_by_label(image_root)
    missing = [label for label, paths in paths_by_label.items() if not paths]
    if missing:
        raise ValueError(f"Static dataset has no images for: {', '.join(missing)}")

    records: list[dict] = []
    for label in STATIC_LABELS:
        paths = paths_by_label[label]
        image_hashes = [difference_hash(path) for path in paths]
        offset = len(records)
        records.extend({
            "path": path.relative_to(dataset_root).as_posix(),
            "label": label,
            "group_id": "",
        } for path in paths)
        uf = UnionFind(len(paths))
        for current in range(len(paths)):
            current_hash = image_hashes[current]
            for previous in range(current):
                if (current_hash ^ image_hashes[previous]).bit_count() <= 3:
                    uf.union(current, previous)
        for index in range(len(paths)):
            records[offset + index]["group_id"] = f"{label}-{uf.find(index)}"

    splits = _split_groups(records, seed)
    group_sets = {part: {row["group_id"] for row in rows} for part, rows in splits.items()}
    if any(group_sets[left] & group_sets[right] for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise RuntimeError("Duplicate-image groups leaked across splits")
    provenance_path = PROJECT_ROOT / "data/provenance/static_dataset.json"
    provenance = (
        json.loads(provenance_path.read_text(encoding="utf-8"))
        if provenance_path.is_file()
        else {
            "dataset_id": DATASET_ID,
            "dataset_url": f"https://www.kaggle.com/datasets/{DATASET_ID}",
            "dataset_version": 1,
            "dataset_version_url": f"https://www.kaggle.com/datasets/{DATASET_ID}/versions/1",
            "version_updated": "2022-05-10",
            "creator": "Atharva Dumbre",
            "license": "CC BY-NC-SA 4.0",
            "archive_sha256": None,
            "note": "Archive hash unavailable because this dataset was not obtained through isl-download-static.",
        }
    )
    return {
        "dataset": provenance,
        "dataset_id": DATASET_ID,
        "license": "CC BY-NC-SA 4.0",
        "seed": seed,
        "preprocessing": preprocessing_config(load_config()["image_size"]),
        "split_method": "per-class deterministic 70/15/15 group split; group exact/near duplicate difference hashes (Hamming distance <= 3)",
        "known_capture_group_limitation": "The source does not document signer/session identifiers. Near-duplicate groups are separated, but signer/session independence cannot be guaranteed.",
        "validated_input_counts": validation["class_counts"],
        "static_labels": list(STATIC_LABELS),
        "counts": {part: {label: sum(row["label"] == label for row in rows) for label in STATIC_LABELS} for part, rows in splits.items()},
        "splits": splits,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create group-aware static train/validation/test manifests.")
    parser.add_argument("--dataset", type=Path, default=STATIC_DATA_ROOT)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data/manifests/static_splits.json")
    parser.add_argument("--seed", type=int, default=load_config()["seed"])
    args = parser.parse_args()
    manifest = build_manifest(args.dataset.expanduser().resolve(), args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Group-aware split manifest written to {args.output}")
    for split, counts in manifest["counts"].items():
        print(f"{split}: {sum(counts.values())} images")
    print("Note: source signer/session IDs are absent; only visual near-duplicate grouping is possible.")


if __name__ == "__main__":
    main()
