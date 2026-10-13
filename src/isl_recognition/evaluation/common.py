import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from isl_recognition.models.static_cnn import StaticSignCNN
from isl_recognition.models.landmark_mlp import LandmarkSignMLP
from isl_recognition.inference.landmarks import FEATURES_PER_HAND, hand_count_compatible_scores


def load_model(checkpoint_path: Path, expected_kind: str, device: torch.device | None = None):
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Trained checkpoint is missing: {checkpoint_path}")
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("kind") != expected_kind:
        raise ValueError(f"Expected {expected_kind} checkpoint, found {checkpoint.get('kind')!r}")
    if expected_kind == "static":
        if checkpoint.get("architecture") == LandmarkSignMLP.architecture_name:
            model = LandmarkSignMLP(
                num_classes=len(checkpoint["labels"]),
                input_features=int(checkpoint["input_features"]),
            )
        else:
            model = StaticSignCNN(num_classes=len(checkpoint["labels"]))
    else:
        raise ValueError(f"Unsupported model kind: {expected_kind}")
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, checkpoint, device


def evaluate_checkpoint(
    *,
    kind: str,
    checkpoint_path: Path,
    dataset,
    output_dir: Path,
) -> dict:
    model, checkpoint, device = load_model(checkpoint_path, kind)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    true_values: list[int] = []
    predicted_values: list[int] = []
    sample_records: list[dict] = []
    labels = checkpoint["labels"]
    sample_offset = 0
    for inputs, targets in loader:
        inputs = inputs.to(device)
        with torch.inference_mode():
            probabilities = torch.softmax(model(inputs), dim=1).cpu().numpy()
        if checkpoint.get("architecture") == LandmarkSignMLP.architecture_name:
            for row_index, vector in enumerate(inputs.cpu().numpy()):
                hand_count = int(vector[FEATURES_PER_HAND - 1] > 0) + int(
                    vector[2 * FEATURES_PER_HAND - 1] > 0
                )
                probabilities[row_index] = hand_count_compatible_scores(
                    probabilities[row_index], checkpoint, hand_count
                )
        probabilities = torch.as_tensor(probabilities)
        scores, predictions = probabilities.max(dim=1)
        true_values.extend(targets.tolist())
        predicted_values.extend(predictions.cpu().tolist())
        for offset, (expected, predicted, confidence) in enumerate(
            zip(targets.tolist(), predictions.cpu().tolist(), scores.cpu().tolist())
        ):
            if expected != predicted:
                sample_records.append({
                    "test_record": sample_offset + offset,
                    "expected": labels[expected],
                    "predicted": labels[predicted],
                    "confidence": float(confidence),
                    "source_path": dataset.records[sample_offset + offset]["path"],
                })
        sample_offset += len(targets)

    if not true_values:
        raise ValueError("Held-out test split contains no samples")
    report_dict = classification_report(
        true_values, predicted_values, labels=list(range(len(labels))),
        target_names=labels, output_dict=True, zero_division=0,
    )
    report_text = classification_report(
        true_values, predicted_values, labels=list(range(len(labels))),
        target_names=labels, zero_division=0,
    )
    matrix = confusion_matrix(true_values, predicted_values, labels=list(range(len(labels))))
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = output_dir / f"{kind}_test_confusion_matrix.png"
    figure, axis = plt.subplots(figsize=(max(8, len(labels) * 0.42), max(7, len(labels) * 0.38)))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        xticks=np.arange(len(labels)), yticks=np.arange(len(labels)),
        xticklabels=labels, yticklabels=labels,
        xlabel="Predicted label", ylabel="True label",
        title=f"{kind.title()} held-out test confusion matrix",
    )
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    figure.tight_layout()
    figure.savefig(matrix_path, dpi=150)
    plt.close(figure)

    batch = next(iter(DataLoader(dataset, batch_size=1, shuffle=False)))
    sample = batch[0].to(device)
    with torch.inference_mode():
        for _ in range(5):
            model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        repetitions = 30
        for _ in range(repetitions):
            model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    latency_ms = (time.perf_counter() - start) * 1000 / repetitions
    counts = {
        label: int(sum(value == index for value in true_values))
        for index, label in enumerate(labels)
    }
    dropped_samples = int(getattr(dataset, "dropped", 0))
    source_test_samples = len(true_values) + dropped_samples
    detection_coverage = len(true_values) / source_test_samples if source_test_samples else 0.0
    result = {
        "kind": kind,
        "checkpoint": str(checkpoint_path),
        "test_samples": len(true_values),
        "test_accuracy": float(report_dict["accuracy"]),
        "source_test_samples": source_test_samples,
        "landmark_dropped_samples": dropped_samples,
        "landmark_detection_coverage": detection_coverage,
        "coverage_adjusted_accuracy": float(report_dict["accuracy"]) * detection_coverage,
        "per_class": {
            label: {
                "precision": float(report_dict[label]["precision"]),
                "recall": float(report_dict[label]["recall"]),
                "f1_score": float(report_dict[label]["f1-score"]),
                "test_samples": counts[label],
            }
            for label in labels
        },
        "classification_report": report_text,
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_image": str(matrix_path),
        "model_size_bytes": checkpoint_path.stat().st_size,
        "batch_1_inference_latency_ms": latency_ms,
        "measurement_device": str(device),
        "known_failure_cases": {
            "basis": "Misclassified held-out samples; images/videos still need human review.",
            "examples": sample_records[:100],
        },
        "training_config": {
            key: checkpoint.get(key)
            for key in (
                "seed", "image_size", "pretrained_weights",
                "training_class_counts", "environment", "provenance",
            )
        },
    }
    (output_dir / f"{kind}_test_report.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / f"{kind}_classification_report.txt").write_text(
        f"Held-out test accuracy: {result['test_accuracy']:.4f}\n"
        f"Test samples: {result['test_samples']}\n"
        f"MediaPipe coverage: {detection_coverage:.4f} ({result['test_samples']}/{source_test_samples})\n"
        f"Accuracy including images with no detected hand as misses: {result['coverage_adjusted_accuracy']:.4f}\n"
        f"Batch-1 inference latency: {latency_ms:.2f} ms ({device})\n\n"
        f"{report_text}\n",
        encoding="utf-8",
    )
    print(f"Held-out test accuracy: {result['test_accuracy']:.4f}")
    print(f"Test samples: {result['test_samples']}")
    print(f"MediaPipe coverage: {detection_coverage:.4f} ({result['test_samples']}/{source_test_samples})")
    print(f"Coverage-adjusted accuracy: {result['coverage_adjusted_accuracy']:.4f}")
    print(f"Batch-1 latency: {latency_ms:.2f} ms ({device})")
    print(report_text)
    print(f"Report: {output_dir / f'{kind}_test_report.json'}")
    return result
