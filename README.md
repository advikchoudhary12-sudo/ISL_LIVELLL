# ISL Alphabet Recognition

A desktop app for recognizing static fingerspelling signs for **letters A–Z and numbers 0–9** in Indian Sign Language (ISL). It does not translate words or sentences.

## Recognition pipeline

```text
Image folders A–Z and 0–9 ─ validate ─ duplicate-aware split ─ MediaPipe landmarks ─ 36-class landmark MLP ─ native webcam UI ─ text buffer
```

The webcam app uses a native Tkinter window. It retains the camera preview, sign/confidence display, text buffer, manual character/space/backspace/clear controls, auto-type, pause, and fullscreen controls. Auto-type adds a stable sign after 500 ms; the same character will not repeat until the hand leaves or a different sign is confirmed. There is no motion mode.

## Dataset

The project uses the [Indian Sign Language (ISLRTC referred) dataset](https://www.kaggle.com/datasets/atharvadumbre/indian-sign-language-islrtc-referred), attributed to Atharva Dumbre. The Kaggle listing states CC BY-NC-SA 4.0. Its images are organized in letter folders A–Z and digit folders 0–9; all 36 folders are used for static training, including J and K. The source does not document signer/session identifiers, so the evaluation is not signer-independent.

The contributor's provenance and rights have not been independently audited. Respect the dataset's noncommercial, attribution, and ShareAlike terms. Do not redistribute dataset images or derived weights without checking the license requirements.

To download the pinned dataset version, configure your own Kaggle API credentials locally and run:

```bash
isl-download-static
```

It extracts under `data/raw/islrtc-referred/` and refuses to overwrite a non-empty destination. The dataset is already present in this workspace.

## Setup on Linux

Ubuntu with Python 3.12:

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install -r requirements-torch-cpu.txt
uv pip install --no-deps -e .
```

## Setup on Windows

Windows 10/11 with 64-bit Python 3.12 and `uv` installed:

```powershell
Set-Location C:\path\to\isl-linux
.\windows\setup.ps1
```

The setup script creates `.venv`, installs the application and CPU PyTorch, and makes the project commands available in that environment. To launch the interactive Windows menu, double-click `Run ISL on Windows.bat`. In PowerShell, run `powershell -ExecutionPolicy Bypass -File .\windows\isl-launcher.ps1 -Action webcam -CameraIndex 0` to select a camera directly. Use `-Action train` to prepare the dataset split and train a fresh model. Windows uses the same native Tkinter UI and OpenCV camera capture; no X11 or Linux subsystem is required.

## Prepare, train, and evaluate

```bash
cd /path/to/isl-linux
source .venv/bin/activate
isl-validate-static
isl-prepare-static
isl-train-static --epochs 25 --batch-size 32
isl-evaluate-static
```

Validation checks all 26 letter and 10 digit folders, verifies readable images, and reports duplicates or conflicting labels. The image preview shows one sample for each letter A–Z. The split groups exact and visually near-duplicate images before making train/validation/test partitions. Source signer/session IDs are unavailable, so the grouping cannot guarantee signer independence.

Training first extracts up to two hands of MediaPipe 21-point landmarks from each labeled image, normalizes each hand's 3D joint coordinates relative to its wrist and hand span, and caches the features at `artifacts/static_landmark_features.npz`. It then trains a compact multilayer perceptron (MLP) on the training split, monitors validation loss for early stopping, and writes the deployment model to `artifacts/static_model.pt`. The test partition remains held out for evaluation. You can tune `--epochs`, `--batch-size`, and `--seed`.

The live app uses those same landmark features directly. MediaPipe estimates the points for up to two hands; the MLP maps the 134-feature vector to scores for A–Z and 0–9. The checkpoint also stores the training split's per-class hand-count totals. Inference uses the detected hand count to suppress classes that had fewer than 25% of training examples with that count. This reduces one-hand/two-hand mix-ups when MediaPipe detects only one hand in a two-hand sign. If MediaPipe cannot locate a hand, the app does not guess a character.

The count rule is a dataset-supported eligibility filter, not a second hand detector. If the tracker misses a hand, the frame is treated as one-hand input and classes rarely represented with one detected hand are suppressed. The checkpoint must include `training_hand_count_counts`; older checkpoints without that metadata keep the original unrestricted behavior. The latest held-out evaluation with the count filter reports 76.38% accuracy among detected hands, 72.11% detection coverage, and 55.07% coverage-adjusted accuracy.

The dataset does not provide a verified signer/session split. Treat test accuracy as an in-dataset prototype metric, not a measure of performance on new signers or real-world conditions.

## Run the desktop app

```bash
isl-webcam --camera 0
```
 
You can also double-click `Launch ISL Webcam.desktop`. To choose a camera in the startup menu, select **Open webcam** and enter its camera index; press Enter for index 0. From a terminal, run `./ui startup/isl-launcher.sh webcam 1` to start camera index 1, or use `ISL_CAMERA=1 ./ui startup/isl-launcher.sh webcam`. The app uses `artifacts/static_model.pt` and works with the trained 36-class A–Z and 0–9 landmark MLP. Existing 26-class checkpoints must be retrained before number recognition is available; the UI reports when retraining is needed. Keep the camera view empty during initial background calibration. Keys: `A` add the confirmed character, `Space` insert a space, `Backspace` delete, `C` clear, `T` auto-type, `P` pause/resume, `F` fullscreen, `Q` quit. Auto-type starts off. The foreground-difference gate can be affected by lighting or movement; confidence is not calibrated probability.

On Windows, double-click `Run ISL on Windows.bat` and choose webcam, or run `powershell -ExecutionPolicy Bypass -File .\windows\isl-launcher.ps1 -Action webcam -CameraIndex 0`. Change `0` to another camera index if needed.

For a single image:

```bash
isl-infer-image --image /path/to/sign.jpg
```

## Paths and troubleshooting

Override default paths with `ISL_DATA_DIR`, `ISL_STATIC_DATA_DIR`, `ISL_ARTIFACT_DIR`, and `ISL_CONFIG`. Generated reports and weights are under `reports/generated/` and `artifacts/`.

- **Camera unavailable:** check camera permissions and index, and run from a desktop session. The app window stays open and displays the unavailable camera state.
- **No confident letter:** verify the hands are fully visible, check camera lighting and framing, and lower `--confidence` or `--margin` cautiously.
- **Missing model:** prepare the image manifest and run `isl-train-static`.
