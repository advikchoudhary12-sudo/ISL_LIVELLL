import argparse
from pathlib import Path

from isl_recognition.config import ARTIFACT_ROOT, STATIC_DATA_ROOT, PROJECT_ROOT
from isl_recognition.data.torch_datasets import StaticImageDataset
from isl_recognition.data.landmark_datasets import StaticLandmarkDataset
from isl_recognition.models.landmark_mlp import LandmarkSignMLP
from isl_recognition.evaluation.common import evaluate_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the static classifier only on its held-out test split.")
    parser.add_argument("--dataset", type=Path, default=STATIC_DATA_ROOT)
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "data/manifests/static_splits.json")
    parser.add_argument("--checkpoint", type=Path, default=ARTIFACT_ROOT / "static_model.pt")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "reports/generated")
    args = parser.parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("architecture") == LandmarkSignMLP.architecture_name:
        dataset = StaticLandmarkDataset(
            args.manifest.expanduser().resolve(), "test", args.dataset.expanduser().resolve(),
            tuple(payload["labels"]),
        )
    else:
        dataset = StaticImageDataset(
            args.manifest.expanduser().resolve(), "test", args.dataset.expanduser().resolve(),
            tuple(payload["labels"]), int(payload["image_size"]),
        )
    evaluate_checkpoint(
        kind="static",
        checkpoint_path=checkpoint,
        dataset=dataset,
        output_dir=args.output.expanduser().resolve(),
    )


if __name__ == "__main__":
    main()
