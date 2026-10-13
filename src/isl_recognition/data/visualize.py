import argparse
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from isl_recognition.config import PROJECT_ROOT, STATIC_DATA_ROOT
from isl_recognition.data.common import image_paths_by_label, locate_image_root
from isl_recognition.labels import ALL_LABELS


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a reproducible class sample contact sheet.")
    parser.add_argument("--dataset", type=Path, default=STATIC_DATA_ROOT)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "reports/generated/static_samples.png")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    paths = image_paths_by_label(
        locate_image_root(args.dataset.expanduser().resolve()), ALL_LABELS
    )
    if any(not paths[label] for label in ALL_LABELS):
        absent = [label for label in ALL_LABELS if not paths[label]]
        raise ValueError(f"Missing classes: {', '.join(absent)}")

    rng = random.Random(args.seed)
    fig, axes = plt.subplots(5, 6, figsize=(16, 13))
    for label, axis in zip(ALL_LABELS, axes.flat):
        sample = rng.choice(paths[label])
        with Image.open(sample) as image:
            axis.imshow(image.convert("RGB"))
        axis.set_title(f"{label}: {sample.name}")
        axis.axis("off")
    for axis in axes.flat[len(ALL_LABELS):]:
        axis.axis("off")
    fig.suptitle("ISLRTC-referred static alphabet samples (A–Z)")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=140)
    plt.close(fig)
    print(f"Sample contact sheet saved to {args.output}")


if __name__ == "__main__":
    main()
