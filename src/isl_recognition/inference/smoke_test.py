import torch

from isl_recognition.labels import STATIC_LABELS
from isl_recognition.inference.landmarks import LANDMARK_FEATURES
from isl_recognition.models.landmark_mlp import LandmarkSignMLP


def main() -> None:
    static_input = torch.zeros((1, LANDMARK_FEATURES), dtype=torch.float32)
    static_output = LandmarkSignMLP(
        num_classes=len(STATIC_LABELS), input_features=LANDMARK_FEATURES
    )(static_input)
    if static_output.shape != (1, len(STATIC_LABELS)):
        raise RuntimeError(f"Unexpected static model output shape: {static_output.shape}")
    if not torch.isfinite(static_output).all():
        raise RuntimeError("Smoke-test output contains non-finite values")
    print("Landmark feature/model shape smoke test passed.")
    print("WARNING: models have random initialization; this is not a recognition or accuracy test.")


if __name__ == "__main__":
    main()
