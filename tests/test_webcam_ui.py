import cv2
from types import SimpleNamespace
import numpy as np
import mediapipe as mp

from isl_recognition.app.webcam import (
    _DwellAutoTyper,
    _compose_display,
    _draw_hand_grid,
    _wrap_sidebar_line,
)


def test_sidebar_is_opaque_and_keeps_camera_feed_intact():
    frame = np.full((480, 640, 3), (4, 5, 6), dtype=np.uint8)

    display = _compose_display(
        frame,
        "STATIC",
        "Uncertain",
        0,
        "uncertain",
        0.0,
        "",
        False,
        ["F Fullscreen", "Q / Esc Quit"],
        None,
        None,
    )

    assert display.shape == (480, 960, 3)
    np.testing.assert_array_equal(display[400, 320], (4, 5, 6))
    np.testing.assert_array_equal(display[400, 950], (28, 32, 42))


def test_sidebar_wraps_long_unbroken_text_to_fit():
    lines = _wrap_sidebar_line(
        f"Text: {'A' * 120}", 250, cv2.FONT_HERSHEY_SIMPLEX, 0.47
    )

    assert "".join(lines).replace(" ", "") == f"Text:{'A' * 120}"
    assert all(
        cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.47, 1)[0][0] <= 250
        for line in lines
    )


def test_fullscreen_canvas_uses_display_dimensions_without_stretching_camera():
    frame = np.full((480, 640, 3), (4, 5, 6), dtype=np.uint8)

    display = _compose_display(
        frame,
        "STATIC",
        "Confirmed",
        1,
        "A",
        0.9,
        "A",
        True,
        ["F Fullscreen"],
        None,
        None,
        canvas_size=(1080, 1920),
    )

    assert display.shape == (1080, 1920, 3)
    np.testing.assert_array_equal(display[1050, 1918], (28, 32, 42))
    np.testing.assert_array_equal(display[540, 600], (4, 5, 6))


def test_auto_typer_emits_once_after_one_second_dwell():
    typer = _DwellAutoTyper()

    assert typer.update("A", True, 10.0) is None
    assert typer.update("A", True, 10.99) is None
    assert typer.update("A", True, 11.0) == "A"
    assert typer.update("A", True, 12.0) is None


def test_auto_typer_waits_for_new_sign_and_resets_when_hand_leaves():
    typer = _DwellAutoTyper()
    typer.update("A", True, 0.0)
    assert typer.update("A", True, 1.0) == "A"

    assert typer.update("B", True, 2.0) is None
    assert typer.update("B", True, 3.0) == "B"
    assert typer.update("A", True, 4.0) is None
    assert typer.update(None, False, 4.1) is None
    assert typer.update("A", True, 5.0) is None
    assert typer.update("A", True, 6.0) == "A"


def test_hand_grid_draws_both_detected_hands(monkeypatch):
    drawn = []
    monkeypatch.setattr(
        mp.solutions.drawing_utils,
        "draw_landmarks",
        lambda frame, landmarks, connections, **kwargs: drawn.append(landmarks),
    )
    hands = SimpleNamespace(HAND_CONNECTIONS=())
    results = SimpleNamespace(multi_hand_landmarks=["left", "right"])

    _draw_hand_grid(np.zeros((20, 20, 3), dtype=np.uint8), results, hands)

    assert drawn == ["left", "right"]
