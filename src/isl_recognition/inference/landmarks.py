"""MediaPipe landmark extraction and normalization for static ISL signs."""

from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


LANDMARKS_PER_HAND = 21
FEATURES_PER_HAND = LANDMARKS_PER_HAND * 3 + 4
LANDMARK_FEATURES = FEATURES_PER_HAND * 2
FEATURE_SCHEMA = "two-hand-wrist-relative-xyz-v1"
MIN_HAND_COUNT_CLASS_SHARE = 0.25


def hand_count_compatible_scores(scores, checkpoint: dict, hand_count: int) -> np.ndarray:
    """Suppress classes rarely represented with the number of visible hands.

    The class-specific support is learned from the training split. A class is
    eligible only when at least a quarter of its detected training examples
    had this hand count; otherwise it is likely to be a missed-hand variant.
    """
    values = np.asarray(scores, dtype=np.float64).copy()
    counts = checkpoint.get("training_hand_count_counts")
    if hand_count not in (1, 2) or not isinstance(counts, dict):
        return values
    labels = checkpoint.get("labels", [])
    compatible = []
    for label in labels:
        class_counts = counts.get(label, {})
        one = int(class_counts.get("1", 0))
        two = int(class_counts.get("2", 0))
        total = one + two
        share = (one if hand_count == 1 else two) / total if total else 0.0
        compatible.append(share >= MIN_HAND_COUNT_CLASS_SHARE)
    compatible = np.asarray(compatible, dtype=bool)
    if compatible.shape != values.shape or not compatible.any():
        return values
    values[~compatible] = 0.0
    total = values.sum()
    return values / total if total > 0 else scores


def landmarks_to_features(hand_landmarks, frame_shape) -> np.ndarray | None:
    """Encode up to two hands; return None when no hand landmarks were found.

    Each hand contributes 21 wrist-relative 3D points plus normalized wrist
    position and hand scale. Hands are consistently ordered left-to-right.
    Missing second hands are represented by zeros and an explicit presence bit.
    """
    if not hand_landmarks:
        return None
    height, width = frame_shape[:2]
    if width <= 0 or height <= 0:
        return None

    encoded = []
    for hand in hand_landmarks[:2]:
        points = np.asarray([[p.x, p.y, p.z] for p in hand.landmark], dtype=np.float32)
        if points.shape != (LANDMARKS_PER_HAND, 3):
            continue
        wrist = points[0].copy()
        pixel_x = points[:, 0] * width
        pixel_y = points[:, 1] * height
        hand_scale = max(float(np.ptp(pixel_x)), float(np.ptp(pixel_y)), 1.0)
        relative = points - wrist
        relative[:, 0] *= width / hand_scale
        relative[:, 1] *= height / hand_scale
        relative[:, 2] *= width / hand_scale
        center_x = float((pixel_x.min() + pixel_x.max()) * 0.5)
        center_y = float((pixel_y.min() + pixel_y.max()) * 0.5)
        metadata = np.asarray([
            (center_x / width) * 2.0 - 1.0,
            (center_y / height) * 2.0 - 1.0,
            hand_scale / min(width, height),
            1.0,
        ], dtype=np.float32)
        encoded.append((center_x, np.concatenate((relative.reshape(-1), metadata))))

    if not encoded:
        return None
    encoded.sort(key=lambda item: item[0])
    result = np.zeros(LANDMARK_FEATURES, dtype=np.float32)
    for index, (_, values) in enumerate(encoded[:2]):
        start = index * FEATURES_PER_HAND
        result[start:start + FEATURES_PER_HAND] = values
    return result


def extract_image_landmarks(image_path: Path, hands) -> np.ndarray | None:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return None
    height, width = image.shape[:2]
    scale = max(1.0, min(3.0, 512.0 / max(height, width)))
    if scale > 1:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    result = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return landmarks_to_features(result.multi_hand_landmarks, image.shape)


def create_hands_tracker():
    return mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        model_complexity=0,
        min_detection_confidence=0.05,
    )
