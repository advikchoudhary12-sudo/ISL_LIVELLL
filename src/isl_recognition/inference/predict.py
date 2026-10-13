from pathlib import Path

import numpy as np
import torch
from PIL import Image

from isl_recognition.evaluation.common import load_model
from isl_recognition.inference.landmarks import (
    LANDMARK_FEATURES, create_hands_tracker, hand_count_compatible_scores,
    landmarks_to_features,
)
from isl_recognition.preprocessing import preprocess_image


def predict_image(
    image: Image.Image,
    checkpoint_path: Path,
    confidence_threshold: float = 0.75,
    margin_threshold: float = 0.15,
    device: torch.device | None = None,
) -> dict:
    model, checkpoint, device = load_model(checkpoint_path, "static", device)
    if checkpoint.get("architecture") == "static_landmark_mlp":
        rgb = np.asarray(image.convert("RGB"))
        tracker = create_hands_tracker()
        try:
            result = tracker.process(rgb)
            hand_count = len(result.multi_hand_landmarks or [])
            features = landmarks_to_features(result.multi_hand_landmarks, rgb.shape)
        finally:
            tracker.close()
        if features is None:
            return {"status": "no_hand", "label": None, "top_candidate": None,
                    "confidence": 0.0, "margin": 0.0, "top_scores": [],
                    "thresholds": {"confidence": confidence_threshold, "margin": margin_threshold}}
        tensor = torch.from_numpy(features).reshape(1, LANDMARK_FEATURES).to(device)
    else:
        pixels = preprocess_image(image, int(checkpoint["image_size"]))
        tensor = torch.from_numpy(pixels).unsqueeze(0).to(device)
    with torch.inference_mode():
        probabilities = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()
    if checkpoint.get("architecture") == "static_landmark_mlp":
        probabilities = hand_count_compatible_scores(probabilities, checkpoint, hand_count)
    order = np.argsort(probabilities)[::-1]
    top = int(order[0])
    margin = float(probabilities[top] - probabilities[int(order[1])]) if len(order) > 1 else 1.0
    confidence = float(probabilities[top])
    confirmed = confidence >= confidence_threshold and margin >= margin_threshold
    return {
        "status": "prediction" if confirmed else "uncertain",
        "label": checkpoint["labels"][top] if confirmed else None,
        "top_candidate": checkpoint["labels"][top],
        "confidence": confidence,
        "margin": margin,
        "top_scores": [
            {"label": checkpoint["labels"][int(index)], "score": float(probabilities[index])}
            for index in order[:3]
        ],
        "thresholds": {
            "confidence": confidence_threshold,
            "margin": margin_threshold,
        },
    }
