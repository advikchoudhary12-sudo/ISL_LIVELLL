import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from isl_recognition.config import STATIC_DATA_ROOT, PROJECT_ROOT
from isl_recognition.data.common import image_paths_by_label, locate_image_root
from isl_recognition.labels import ALL_LABELS, SOURCE_LABELS, STATIC_LABELS


def inspect_dataset(dataset_root: Path) -> dict:
    image_root = locate_image_root(dataset_root)
    paths_by_label = image_paths_by_label(image_root, SOURCE_LABELS)
    invalid: list[dict[str, str]] = []
    hashes: dict[str, str] = {}
    counts = Counter()
    exact_duplicates = 0
    duplicate_samples: list[dict[str, str]] = []
    for label, paths in paths_by_label.items():
        for path in paths:
            try:
                with Image.open(path) as image:
                    image.verify()
                with path.open("rb") as sample:
                    digest = hashlib.sha256(sample.read()).hexdigest()
            except (OSError, UnidentifiedImageError) as error:
                invalid.append({"path": str(path), "error": str(error)})
                continue
            previous = hashes.get(digest)
            if previous:
                if Path(previous).parent.name.upper() != label:
                    invalid.append({
                        "path": str(path),
                        "error": f"Exact duplicate has a conflicting class label: {previous}",
                    })
                    continue
                else:
                    exact_duplicates += 1
                    if len(duplicate_samples) < 100:
                        duplicate_samples.append({"path": str(path), "duplicate_of": previous})
            else:
                hashes[digest] = str(path)
            counts[label] += 1

    missing = [label for label in SOURCE_LABELS if counts[label] == 0]
    return {
        "dataset_root": str(dataset_root),
        "image_root": str(image_root),
        "source_labels": list(SOURCE_LABELS),
        "alphabet_labels": list(ALL_LABELS),
        "static_model_labels": list(STATIC_LABELS),
        "class_counts": {label: counts[label] for label in SOURCE_LABELS},
        "total_valid_source_images": sum(counts.values()),
        "missing_source_classes": missing,
        "exact_duplicate_count": exact_duplicates,
        "exact_duplicate_examples": duplicate_samples,
        "invalid_or_conflicting_samples": invalid,
        "status": "ok" if not missing and not invalid else "needs_attention",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate dataset layout, labels, and readable images.")
    parser.add_argument("--dataset", type=Path, default=STATIC_DATA_ROOT)
    parser.add_argument("--report", type=Path, default=PROJECT_ROOT / "reports/generated/dataset_validation.json")
    args = parser.parse_args()
    report = inspect_dataset(args.dataset.expanduser().resolve())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for label in SOURCE_LABELS:
        print(f"{label}: {report['class_counts'].get(label, 0)}")
    print(f"Static alphabet classes: {', '.join(STATIC_LABELS)}")
    print(f"Valid source images: {report['total_valid_source_images']}")
    print(f"Same-class exact duplicate images: {report['exact_duplicate_count']}")
    print(f"Malformed/conflicting images: {len(report['invalid_or_conflicting_samples'])}")
    print(f"Validation report: {args.report}")
    if report["status"] != "ok":
        raise SystemExit("Dataset validation failed; see report for missing source classes or malformed files.")


if __name__ == "__main__":
    main()
