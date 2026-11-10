#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"
TRAIN_CMD="$VENV_DIR/bin/isl-train-static"
WEBCAM_LAUNCHER="$PROJECT_DIR/launch-isl-webcam.sh"

show_usage() {
    cat <<'EOF'
Usage: ./ui startup/isl-launcher.sh [train|webcam [camera-index]]

  train   Extract MediaPipe landmarks and train the static A-Z/0-9 model (25 epochs, batch size 32)
  webcam  Launch the webcam app (default camera index: 0)
  (none)  Show an interactive menu

Pass a camera index after "webcam" or choose one in the interactive menu.
Training can be configured with ISL_EPOCHS and ISL_BATCH_SIZE.
EOF
}

check_environment() {
    if [[ ! -x "$TRAIN_CMD" ]]; then
        echo "The project environment is not set up. Follow README.md first." >&2
        exit 1
    fi
}

train_model() {
    check_environment
    if [[ ! -f "$PROJECT_DIR/data/manifests/static_splits.json" ]]; then
        echo "The static data manifest is missing. Activate the project environment and run isl-prepare-static first." >&2
        exit 1
    fi
    local epochs="${ISL_EPOCHS:-25}"
    local batch_size="${ISL_BATCH_SIZE:-32}"
    echo "Extracting MediaPipe landmarks and training the static A-Z/0-9 model (up to $epochs epochs, batch size $batch_size)."
    exec "$TRAIN_CMD" --epochs "$epochs" --batch-size "$batch_size"
}

launch_webcam() {
    if [[ ! -x "$WEBCAM_LAUNCHER" ]]; then
        echo "Webcam launcher not found: $WEBCAM_LAUNCHER" >&2
        exit 1
    fi
    local camera_index="${1:-${ISL_CAMERA:-0}}"
    if [[ ! "$camera_index" =~ ^[0-9]+$ ]]; then
        echo "Camera index must be a non-negative integer." >&2
        exit 2
    fi
    exec "$WEBCAM_LAUNCHER" "$camera_index"
}

case "${1:-menu}" in
    train)
        train_model
        ;;
    webcam)
        launch_webcam "${2:-${ISL_CAMERA:-0}}"
        ;;
    -h|--help|help)
        show_usage
        ;;
    menu)
        echo "ISL Sign Recognition"
        echo "1) Train model"
        echo "2) Open webcam"
        read -r -p "Choose 1 or 2: " choice
        case "$choice" in
            1) train_model ;;
            2)
                read -r -p "Camera index (0 is usually the default; press Enter to use 0): " camera_index
                launch_webcam "${camera_index:-0}"
                ;;
            *) echo "Choose 1 or 2. Run "ui startup/isl-launcher.sh" to try again." >&2; exit 2 ;;
        esac
        ;;
    *)
        show_usage >&2
        exit 2
        ;;
esac
