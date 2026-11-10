import argparse
import hashlib
import json
from pathlib import Path

from isl_recognition.config import ARTIFACT_ROOT, STATIC_DATA_ROOT, PROJECT_ROOT, load_config
from isl_recognition.data.landmark_datasets import StaticLandmarkDataset
from isl_recognition.labels import STATIC_LABELS
from isl_recognition.models.landmark_mlp import LandmarkSignMLP
from isl_recognition.inference.landmarks import FEATURE_SCHEMA, LANDMARK_FEATURES
from isl_recognition.training.common import run_training, set_seed


def main() -> None:
    defaults = load_config()
    parser = argparse.ArgumentParser(description="Extract MediaPipe hand landmarks and train the 36-class static landmark model (A-Z and 0-9).")
    parser.add_argument("--dataset", type=Path, default=STATIC_DATA_ROOT)
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "data/manifests/static_splits.json")
    parser.add_argument("--output", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument("--epochs", type=int, default=defaults["static_epochs"])
    parser.add_argument("--batch-size", type=int, default=defaults["static_batch_size"])
    parser.add_argument("--seed", type=int, default=defaults["seed"])
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    dataset_root = args.dataset.expanduser().resolve()
    manifest = args.manifest.expanduser().resolve()
    output = args.output.expanduser().resolve()
    set_seed(args.seed)
    train_data = StaticLandmarkDataset(manifest, "train", dataset_root, STATIC_LABELS)
    validation_data = StaticLandmarkDataset(manifest, "validation", dataset_root, STATIC_LABELS)
    print(f"Train landmarks: {len(train_data):,} kept, {train_data.dropped:,} images without detected hands")
    print(f"Validation landmarks: {len(validation_data):,} kept, {validation_data.dropped:,} images without detected hands")
    print("Detected train examples per class: " + ", ".join(
        f"{label}={train_data.detected_counts[label]}" for label in STATIC_LABELS
    ))
    split_metadata = json.loads(manifest.read_text(encoding="utf-8"))
    run_training(
        kind="static",
        model=LandmarkSignMLP(num_classes=len(STATIC_LABELS), input_features=LANDMARK_FEATURES),
        train_dataset=train_data,
        validation_dataset=validation_data,
        labels=STATIC_LABELS,
        image_size=LANDMARK_FEATURES,
        preprocessing={
            "input": "MediaPipe Hands landmark geometry",
            "feature_schema": FEATURE_SCHEMA,
            "landmarks_per_hand": 21,
            "maximum_hands": 2,
            "features": LANDMARK_FEATURES,
            "normalization": "wrist-relative xyz normalized by hand span, plus wrist center, scale and presence",
        },
        batch_size=args.batch_size,
        epochs=args.epochs,
        seed=args.seed,
        output_dir=output,
        resume_path=args.resume.expanduser().resolve() if args.resume else None,
        provenance={
            "dataset": split_metadata["dataset"],
            "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "split_method": split_metadata["split_method"],
            "known_capture_group_limitation": split_metadata["known_capture_group_limitation"],
            "training_classes": list(STATIC_LABELS),
        },
    )


if __name__ == "__main__":
    main()
