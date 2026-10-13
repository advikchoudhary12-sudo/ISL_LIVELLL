import argparse
from collections import Counter, deque
from pathlib import Path
import time

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import torch

from isl_recognition.config import ARTIFACT_ROOT, load_config
from isl_recognition.inference.background import BackgroundGate
from isl_recognition.inference.landmarks import hand_count_compatible_scores
from isl_recognition.preprocessing import preprocess_image
from isl_recognition.text_buffer import TextBuffer


def _frame_tensor(frame: np.ndarray, size: int, device: torch.device) -> torch.Tensor:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pixels = preprocess_image(Image.fromarray(rgb), size=size)
    return torch.from_numpy(pixels).unsqueeze(0).to(device)


def _prediction(model, model_input, checkpoint, device, confidence, margin, hand_count=None):
    with torch.inference_mode():
        probabilities = torch.softmax(model(model_input.to(device)), dim=1)[0].cpu().numpy()
    if hand_count is not None:
        probabilities = hand_count_compatible_scores(probabilities, checkpoint, hand_count)
    probabilities = torch.as_tensor(probabilities)
    scores, indices = probabilities.topk(min(2, probabilities.numel()))
    top_index = int(indices[0])
    top_score = float(scores[0])
    score_margin = top_score - float(scores[1]) if len(scores) > 1 else 1.0
    label = checkpoint["labels"][top_index]
    if top_score < confidence or score_margin < margin:
        return None, label, top_score
    return label, label, top_score


def _draw_hand_grid(frame: np.ndarray, results, hands_module) -> None:
    for hand_landmarks in results.multi_hand_landmarks or []:
        mp.solutions.drawing_utils.draw_landmarks(
            frame,
            hand_landmarks,
            hands_module.HAND_CONNECTIONS,
            landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                color=(0, 220, 255), thickness=2, circle_radius=3
            ),
            connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                color=(80, 220, 80), thickness=2
            ),
        )


def _wrap_sidebar_line(line: str, max_width: int, font, font_scale: float) -> list[str]:
    words = line.split()
    if not words:
        return [""]
    wrapped = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        measured_width = cv2.getTextSize(candidate, font, font_scale, 1)[0][0]
        if measured_width <= max_width:
            current = candidate
            continue
        if current:
            wrapped.append(current)
        current = ""
        for character in word:
            candidate = current + character
            measured_width = cv2.getTextSize(candidate, font, font_scale, 1)[0][0]
            if measured_width > max_width and current:
                wrapped.append(current)
                current = character
            else:
                current = candidate
        if current:
            wrapped.append(current)
            current = ""
    if current:
        wrapped.append(current)
    return wrapped


class _DwellAutoTyper:
    def __init__(self, dwell_seconds: float = 0.5) -> None:
        self.dwell_seconds = dwell_seconds
        self._candidate: str | None = None
        self._candidate_since: float | None = None
        self._last_typed: str | None = None

    def reset(self) -> None:
        self._candidate = None
        self._candidate_since = None
        self._last_typed = None

    def update(self, candidate: str | None, foreground_present: bool, now: float) -> str | None:
        if not foreground_present:
            self._candidate = None
            self._candidate_since = None
            self._last_typed = None
            return None
        if candidate is None:
            self._candidate = None
            self._candidate_since = None
            return None
        if candidate != self._candidate:
            self._candidate = candidate
            self._candidate_since = now
        if candidate == self._last_typed or self._candidate_since is None:
            return None
        if now - self._candidate_since >= self.dwell_seconds:
            self._last_typed = candidate
            return candidate
        return None


def _compose_display(
    frame: np.ndarray,
    mode: str,
    state: str,
    hand_count: int,
    candidate: str,
    confidence: float,
    text_value: str,
    auto_type_enabled: bool,
    control_lines: list[str],
    hand_results,
    hands_module,
    canvas_size: tuple[int, int] | None = None,
) -> np.ndarray:
    camera = frame.copy()
    if hand_results is not None:
        _draw_hand_grid(camera, hand_results, hands_module)

    frame_height, frame_width = camera.shape[:2]
    height, canvas_width = canvas_size or (frame_height, frame_width + max(320, min(420, frame_width // 2)))
    sidebar_width = min(max(280, round(canvas_width * 0.29)), max(280, canvas_width - 320))
    camera_width = canvas_width - sidebar_width
    sidebar_color = (28, 32, 42)
    display = np.full((height, canvas_width, 3), sidebar_color, dtype=np.uint8)
    scale = min(camera_width / frame_width, height / frame_height)
    resized_width = max(1, round(frame_width * scale))
    resized_height = max(1, round(frame_height * scale))
    resized_camera = cv2.resize(
        camera,
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
    )
    camera_x = (camera_width - resized_width) // 2
    camera_y = (height - resized_height) // 2
    display[
        camera_y:camera_y + resized_height,
        camera_x:camera_x + resized_width,
    ] = resized_camera
    cv2.line(display, (camera_width, 0), (camera_width, height), (70, 78, 92), 1)

    margin = max(16, round(sidebar_width * 0.055))
    x = camera_width + margin
    text_width = sidebar_width - margin * 2
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale_factor = min(height / 720, sidebar_width / 340)
    font_scale = max(0.42, min(0.75, 0.58 * scale_factor))
    thickness = 1
    line_height = max(18, round(29 * font_scale / 0.58))
    title_scale = font_scale * 0.88
    title_y = max(24, round(height * 0.055))
    cv2.putText(
        display, "ISL SIGN RECOGNITION", (x, title_y), font, title_scale, (235, 240, 248), 1,
        cv2.LINE_AA,
    )
    divider_color = (70, 78, 92)
    cv2.line(display, (x, title_y + 12), (canvas_width - margin, title_y + 12), divider_color, 1)

    section_scale = font_scale * 0.78
    section_color = (100, 210, 255)
    current_sign_y = title_y + line_height * 2
    cv2.putText(
        display, "CURRENT SIGN", (x, current_sign_y), font, section_scale, section_color, 1,
        cv2.LINE_AA,
    )
    sign_text = candidate if candidate else "..."
    sign_scale = min(2.4 * scale_factor, max(0.9, (text_width * 0.8) / max(
        1, cv2.getTextSize(sign_text, font, 2.4 * scale_factor, 2)[0][0]
    ) * 2.4 * scale_factor))
    sign_color = (70, 230, 130) if candidate not in ("", "uncertain", "...") else (150, 160, 175)
    sign_y = current_sign_y + max(42, round(height * 0.115))
    cv2.putText(display, sign_text, (x, sign_y), font, sign_scale, sign_color, 2, cv2.LINE_AA)
    mode_state = f"{mode.upper()}  |  {state}  |  hands: {hand_count}"
    for index, line in enumerate(_wrap_sidebar_line(mode_state, text_width, font, font_scale)):
        cv2.putText(
            display, line, (x, sign_y + line_height * (index + 1)), font, font_scale,
            (205, 216, 230), thickness, cv2.LINE_AA,
        )

    confidence_y = sign_y + line_height * 3
    cv2.putText(
        display, f"Confidence  {confidence:.0%}", (x, confidence_y), font, font_scale,
        (205, 216, 230), thickness, cv2.LINE_AA,
    )
    bar_top = confidence_y + max(8, round(line_height * 0.4))
    bar_bottom = bar_top + max(7, round(9 * scale_factor))
    cv2.rectangle(display, (x, bar_top), (canvas_width - margin, bar_bottom), (55, 62, 74), -1)
    fill_width = round(text_width * min(1.0, max(0.0, confidence)))
    if fill_width:
        cv2.rectangle(
            display, (x, bar_top), (x + fill_width, bar_bottom), (70, 210, 130), -1
        )

    text_header_y = bar_bottom + line_height * 2
    cv2.line(display, (x, text_header_y - line_height), (canvas_width - margin, text_header_y - line_height), divider_color, 1)
    cv2.putText(
        display, "RECOGNIZED TEXT", (x, text_header_y), font, section_scale, section_color, 1,
        cv2.LINE_AA,
    )
    text_top = text_header_y + max(18, round(line_height * 0.7))
    text_bottom = min(height - line_height * (len(control_lines) + 3), text_top + round(height * 0.13))
    cv2.rectangle(display, (x, text_top), (canvas_width - margin, max(text_top + 1, text_bottom)), (18, 22, 30), -1)
    wrapped_text = []
    for line in text_value.splitlines() or [""]:
        wrapped_text.extend(_wrap_sidebar_line(line, text_width - 16, font, font_scale))
    max_text_rows = max(1, (text_bottom - text_top - 10) // line_height)
    for index, line in enumerate(wrapped_text[-max_text_rows:]):
        y = text_top + line_height * (index + 1)
        cv2.putText(display, line, (x + 8, y), font, font_scale, (235, 240, 248), thickness, cv2.LINE_AA)

    auto_status_y = text_bottom + line_height
    auto_label = "AUTO-TYPE ON  |  hold 500 ms" if auto_type_enabled else "AUTO-TYPE OFF"
    cv2.putText(
        display, auto_label, (x, auto_status_y), font, font_scale * 0.86,
        (70, 230, 130) if auto_type_enabled else (150, 160, 175), thickness, cv2.LINE_AA,
    )
    control_header_y = min(height - line_height * (len(control_lines) + 1), auto_status_y + line_height * 2)
    cv2.line(display, (x, control_header_y - line_height), (canvas_width - margin, control_header_y - line_height), divider_color, 1)
    cv2.putText(
        display, "CONTROLS", (x, control_header_y), font, section_scale, section_color, 1,
        cv2.LINE_AA,
    )
    control_font_scale = max(0.38, font_scale * 0.86)
    control_line_height = max(16, round(line_height * 0.88))
    for index, line in enumerate(control_lines):
        y = control_header_y + control_line_height * (index + 1)
        if y >= height - 4:
            break
        cv2.putText(
            display, line, (x, y), font, control_font_scale, (225, 230, 238),
            thickness, cv2.LINE_AA,
        )
    return display


def main() -> None:
    defaults = load_config()
    parser = argparse.ArgumentParser(description="Run the trained ISL alphabet webcam application.")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--static-model", type=Path, default=ARTIFACT_ROOT / "static_model.pt")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--confidence", type=float, default=defaults["confidence_threshold"])
    parser.add_argument("--margin", type=float, default=defaults["margin_threshold"])
    parser.add_argument("--confirmation-frames", type=int, default=defaults["confirmation_frames"])
    parser.add_argument("--cooldown-frames", type=int, default=defaults["cooldown_frames"])
    parser.add_argument("--foreground-min", type=float, default=0.012)
    parser.add_argument("--foreground-max", type=float, default=0.65)
    parser.add_argument("--calibration-frames", type=int, default=30)
    args = parser.parse_args()
    if args.confirmation_frames < 1 or args.cooldown_frames < 0:
        parser.error("confirmation/cooldown frame values are invalid")
    if not 0 <= args.foreground_min < args.foreground_max <= 1:
        parser.error("foreground thresholds must satisfy 0 <= min < max <= 1")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available")
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto" else torch.device(args.device)
    )
    if args.confirmation_frames < 2:
        raise ValueError("At least two frames are required for prediction smoothing")

    from isl_recognition.app.native_ui import run as run_native_ui

    run_native_ui(
        args,
        device,
    )


if __name__ == "__main__":
    main()
