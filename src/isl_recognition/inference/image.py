import argparse
import json
from pathlib import Path

from PIL import Image

from isl_recognition.config import ARTIFACT_ROOT, load_config
from isl_recognition.inference.predict import predict_image


def main() -> None:
    defaults = load_config()
    parser = argparse.ArgumentParser(description="Run one image through the trained static model (no webcam).")
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path, default=ARTIFACT_ROOT / "static_model.pt")
    parser.add_argument("--confidence", type=float, default=defaults["confidence_threshold"])
    parser.add_argument("--margin", type=float, default=defaults["margin_threshold"])
    args = parser.parse_args()
    with Image.open(args.image.expanduser()) as image:
        result = predict_image(
            image, args.checkpoint.expanduser().resolve(),
            confidence_threshold=args.confidence, margin_threshold=args.margin,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

