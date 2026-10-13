#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"
TRAIN_CMD="$VENV_DIR/bin/isl-train-static"
WEBCAM_LAUNCHER="$PROJECT_DIR/launch-isl-webcam.sh"

show_usage() {
    cat <<'EOF'
Usage: ./ui startup/isl-launcher.sh [train|webcam]

  train   Extract MediaPipe landmarks and train the static A-Z model (25 epochs, batch size 32)
  webcam  Launch the webcam app with the trained model
  (none)  Show an interactive menu

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
    echo "Extracting MediaPipe landmarks and training the static A-Z model (up to $epochs epochs, batch size $batch_size)."
    exec "$TRAIN_CMD" --epochs "$epochs" --batch-size "$batch_size"
}

launch_webcam() {
    if [[ ! -x "$WEBCAM_LAUNCHER" ]]; then
        echo "Webcam launcher not found: $WEBCAM_LAUNCHER" >&2
        exit 1
    fi
    exec "$WEBCAM_LAUNCHER"
}

case "${1:-menu}" in
    train)
        train_model
        ;;
    webcam)
        launch_webcam
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
            2) launch_webcam ;;
            *) echo "Choose 1 or 2. Run "ui startup/isl-launcher.sh" to try again." >&2; exit 2 ;;
        esac
        ;;
    *)
        show_usage >&2
        exit 2
        ;;
esac
