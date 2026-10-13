#!/usr/bin/env bash
set -u

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"
CAMERA_INDEX="${1:-${ISL_CAMERA:-0}}"

pause_before_exit() {
    printf '\nPress Enter to close this window...'
    read -r _
}

if [[ ! -x "$PROJECT_DIR/.venv/bin/isl-webcam" ]]; then
    echo "The project environment is not set up."
    echo "Follow the Setup instructions in README.md, then double-click this launcher again."
    pause_before_exit
    exit 1
fi

if [[ ! "$CAMERA_INDEX" =~ ^[0-9]+$ ]]; then
    echo "Camera index must be a non-negative integer."
    pause_before_exit
    exit 2
fi

missing_models=()
for model in static_model.pt; do
    if [[ ! -f "$PROJECT_DIR/artifacts/$model" ]]; then
        missing_models+=("$model")
    fi
done

if ((${#missing_models[@]})); then
    echo "The webcam app needs trained model files in $PROJECT_DIR/artifacts:"
    printf '  - %s\n' "${missing_models[@]}"
    echo
    echo "Prepare and review the alphabet image dataset, then run isl-train-static"
    echo "as described in README.md."
    pause_before_exit
    exit 1
fi

if "$PROJECT_DIR/.venv/bin/isl-webcam" --camera "$CAMERA_INDEX"; then
    exit 0
else
    status=$?
    echo
    echo "The webcam app exited with an error (status $status)."
    echo "Check the Troubleshooting section in README.md."
    pause_before_exit
    exit "$status"
fi
