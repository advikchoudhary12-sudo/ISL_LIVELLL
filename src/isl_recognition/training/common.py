import json
import os
import platform
import random
import sys
from pathlib import Path
from importlib.metadata import PackageNotFoundError, version

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))


def class_weights(dataset, num_classes: int, device: torch.device) -> torch.Tensor:
    counts = torch.zeros(num_classes, dtype=torch.float32)
    for row in dataset.records:
        counts[dataset.label_to_index[row["label"]]] += 1
    if torch.any(counts == 0):
        missing = [dataset.labels[i] for i, count in enumerate(counts.tolist()) if count == 0]
        raise ValueError(f"Training split is missing classes: {', '.join(missing)}")
    weights = counts.sum() / (num_classes * counts)
    return weights.to(device)


def _evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum = 0.0
    correct = 0
    count = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            loss_sum += criterion(logits, targets).item() * targets.size(0)
            correct += (logits.argmax(dim=1) == targets).sum().item()
            count += targets.size(0)
    return loss_sum / count, correct / count


def _write_curves(history: dict, output_dir: Path, kind: str) -> None:
    for metric, title in (("accuracy", "Accuracy"), ("loss", "Loss")):
        fig, axis = plt.subplots(figsize=(8, 5))
        axis.plot(history[f"train_{metric}"], label=f"Training {title}")
        axis.plot(history[f"validation_{metric}"], label=f"Validation {title}")
        axis.set_xlabel("Epoch")
        axis.set_ylabel(title)
        axis.set_title(f"{kind.title()} model {title.lower()}")
        axis.grid(True, alpha=0.25)
        axis.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"{kind}_{metric}.png", dpi=140)
        plt.close(fig)


def run_training(
    *,
    kind: str,
    model: nn.Module,
    train_dataset,
    validation_dataset,
    labels: tuple[str, ...],
    image_size: int,
    preprocessing: dict,
    batch_size: int,
    epochs: int,
    seed: int,
    output_dir: Path,
    resume_path: Path | None,
    provenance: dict,
) -> Path:
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch size must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, generator=generator, num_workers=0
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    weights = class_weights(train_dataset, len(labels), device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    history = {"train_loss": [], "validation_loss": [], "train_accuracy": [], "validation_accuracy": []}
    start_epoch = 0
    best_loss = float("inf")
    best_accuracy = 0.0
    stale_epochs = 0
    best_updated = False
    patience = 6
    resume_checkpoint = None

    if resume_path:
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        if checkpoint.get("kind") != kind or checkpoint.get("labels") != list(labels):
            raise ValueError("Resume checkpoint type/labels do not match this training run")
        if checkpoint.get("image_size") != image_size:
            raise ValueError("Resume checkpoint preprocessing configuration does not match")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        history = checkpoint["history"]
        start_epoch = checkpoint["epoch"] + 1
        best_loss = checkpoint["best_validation_loss"]
        best_accuracy = checkpoint["best_validation_accuracy"]
        stale_epochs = checkpoint["stale_epochs"]
        generator.set_state(checkpoint["train_generator_state"].cpu())
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        random.setstate(checkpoint["python_rng_state"])
        np.random.set_state(checkpoint["numpy_rng_state"])
        if torch.cuda.is_available() and checkpoint.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["cuda_rng_state"]])
        resume_checkpoint = checkpoint
        print(f"Resuming {kind} training from epoch {start_epoch + 1}: {resume_path}")
        if start_epoch >= epochs:
            raise ValueError("Requested total epoch count must exceed the saved checkpoint epoch")

    best_path = output_dir / f"{kind}_best.pt"
    last_path = output_dir / f"{kind}_last.pt"
    config = {
        "kind": kind,
        "architecture": getattr(model, "architecture_name", "static_cnn"),
        "input_features": getattr(model, "input_features", None),
        "labels": list(labels),
        "image_size": image_size,
        "preprocessing": preprocessing,
        "seed": seed,
        "batch_size": batch_size,
        "optimizer": "AdamW(lr=0.001, weight_decay=0.0001)",
        "loss": "class-weighted CrossEntropyLoss",
        "pretrained_weights": False,
        "provenance": provenance,
        "training_class_counts": {
            label: sum(row["label"] == label for row in train_dataset.records)
            for label in labels
        },
        "training_hand_count_counts": getattr(train_dataset, "hand_count_counts", None),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": str(torch.__version__),
            "numpy": np.__version__,
            "torch_cuda_runtime": torch.version.cuda,
        },
    }
    for package in ("Pillow", "opencv-python", "scikit-learn", "matplotlib", "kaggle", "pytest"):
        try:
            config["environment"][package] = version(package)
        except PackageNotFoundError:
            config["environment"][package] = "not-installed"
    if resume_checkpoint is not None:
        for key in ("seed", "batch_size", "provenance", "training_class_counts", "training_hand_count_counts", "architecture", "input_features"):
            if resume_checkpoint.get(key) != config[key]:
                raise ValueError(f"Cannot resume with changed {key}; use the original training configuration")
    for epoch in range(start_epoch, epochs):
        model.train()
        loss_sum = 0.0
        correct = 0
        sample_count = 0
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * targets.size(0)
            correct += (logits.argmax(dim=1) == targets).sum().item()
            sample_count += targets.size(0)

        train_loss = loss_sum / sample_count
        train_accuracy = correct / sample_count
        validation_loss, validation_accuracy = _evaluate(model, validation_loader, criterion, device)
        history["train_loss"].append(train_loss)
        history["validation_loss"].append(validation_loss)
        history["train_accuracy"].append(train_accuracy)
        history["validation_accuracy"].append(validation_accuracy)
        improved = validation_loss < best_loss
        if improved:
            best_loss = validation_loss
            best_accuracy = validation_accuracy
            stale_epochs = 0
            best_updated = True
        else:
            stale_epochs += 1

        checkpoint = {
            **config,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "best_validation_loss": best_loss,
            "best_validation_accuracy": best_accuracy,
            "stale_epochs": stale_epochs,
            "history": history,
            "train_generator_state": generator.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        torch.save(checkpoint, last_path)
        if improved:
            torch.save(checkpoint, best_path)
        (output_dir / f"{kind}_history.json").write_text(
            json.dumps({"config": config, "history": history}, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            f"{kind} epoch {epoch + 1}/{epochs}: "
            f"train loss={train_loss:.4f}, train acc={train_accuracy:.4f}, "
            f"val loss={validation_loss:.4f}, val acc={validation_accuracy:.4f}"
        )
        if stale_epochs >= patience:
            print(f"Early stopping after {epoch + 1} epochs (validation loss did not improve).")
            break

    _write_curves(history, output_dir, kind)
    if not best_path.is_file() or (resume_path is None and not best_updated):
        raise RuntimeError("Training ended without producing a best-validation checkpoint")
    best_checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
    deployment_path = output_dir / f"{kind}_model.pt"
    torch.save(
        {
            key: best_checkpoint[key]
            for key in (
                "kind", "architecture", "input_features", "labels", "image_size", "preprocessing", "seed",
                "pretrained_weights", "provenance", "environment",
                "training_class_counts", "training_hand_count_counts", "model_state",
            )
        },
        deployment_path,
    )
    print(f"Best validation loss: {best_loss:.4f}; accuracy at best loss: {best_accuracy:.4f}")
    print(f"Resumable best checkpoint: {best_path}")
    print(f"Deployment model: {deployment_path}")
    return best_path
