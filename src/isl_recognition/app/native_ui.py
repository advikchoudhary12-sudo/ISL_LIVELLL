"""Native desktop UI and live inference loop for the webcam recognizer."""

from collections import Counter, deque
import base64
import time
import tkinter as tk
from tkinter import ttk

import cv2
import mediapipe as mp
import numpy as np
import torch

from isl_recognition.app.webcam import _DwellAutoTyper, _draw_hand_grid, _prediction
from isl_recognition.inference.background import BackgroundGate
from isl_recognition.inference.landmarks import LANDMARK_FEATURES, landmarks_to_features
from isl_recognition.text_buffer import TextBuffer


def run(args, device: torch.device) -> None:
    root = tk.Tk()
    # Establish Tk's X11 connection on the main thread before MediaPipe starts
    # its background inference threads; this avoids an xcb sequence assertion.
    root.update_idletasks()
    root.title("ISL Sign Recognition")
    root.geometry("1440x900")
    root.minsize(1000, 650)
    root.configure(bg="#111827")
    root.columnconfigure(0, weight=1)
    root.rowconfigure(1, weight=1)

    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure("Title.TLabel", background="#0f3460", foreground="white",
                    font=("DejaVu Sans", 17, "bold"))
    style.configure("Subtitle.TLabel", background="#0f3460", foreground="#cbd5e1",
                    font=("DejaVu Sans", 10))
    style.configure("Section.TLabel", background="#1f2937", foreground="#38bdf8",
                    font=("DejaVu Sans", 10, "bold"))
    style.configure("Panel.TLabel", background="#1f2937", foreground="#f3f4f6")
    style.configure("Muted.TLabel", background="#1f2937", foreground="#9ca3af")
    style.configure("Sign.TLabel", background="#1f2937", foreground="#9ca3af",
                    font=("DejaVu Sans", 48, "bold"))
    style.configure("Dark.TLabel", background="#0b1220", foreground="#f3f4f6")
    style.configure("Footer.TLabel", background="#0b1220", foreground="#9ca3af")
    style.configure("Warning.TLabel", background="#1f2937", foreground="#fbbf24")
    style.configure("Camera.TLabel", background="#05080e", foreground="#9ca3af",
                    font=("DejaVu Sans", 12))

    palette = {"bg": "#111827", "panel": "#1f2937", "dark": "#0b1220",
               "text": "#f3f4f6", "muted": "#9ca3af", "accent": "#38bdf8",
               "green": "#34d399", "yellow": "#fbbf24", "red": "#f87171"}
    title = tk.Frame(root, bg="#0f3460", padx=20, pady=13)
    title.grid(row=0, column=0, sticky="ew")
    ttk.Label(title, text="ISL SIGN RECOGNITION", style="Title.TLabel").pack(anchor="w")
    ttk.Label(title, text="Live fingerspelling and number recognition", style="Subtitle.TLabel").pack(
        anchor="w", pady=(2, 0))

    body = tk.Frame(root, bg=palette["bg"], padx=14, pady=14)
    body.grid(row=1, column=0, sticky="nsew")
    body.columnconfigure(0, weight=1)
    body.rowconfigure(0, weight=1)
    camera_panel = tk.Frame(body, bg="#05080e", highlightthickness=1,
                            highlightbackground="#334155")
    camera_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 14))
    camera_panel.rowconfigure(0, weight=1)
    camera_panel.columnconfigure(0, weight=1)
    camera_label = ttk.Label(camera_panel, text="Opening camera…", style="Camera.TLabel")
    camera_label.grid(row=0, column=0, sticky="nsew")

    sidebar = tk.Frame(body, bg=palette["panel"], padx=18, pady=16, width=340)
    sidebar.grid(row=0, column=1, sticky="nsew")
    sidebar.grid_propagate(False)
    sidebar.columnconfigure(0, weight=1)

    def section(label: str, row: int) -> None:
        ttk.Label(sidebar, text=label, style="Section.TLabel", anchor="w").grid(
                     row=row, column=0, sticky="ew", pady=(8, 4))

    section("CURRENT SIGN", 0)
    sign_var = tk.StringVar(value="…")
    ttk.Label(sidebar, textvariable=sign_var, style="Sign.TLabel", anchor="center").grid(
                 row=1, column=0, sticky="ew")
    status_var = tk.StringVar(value="Keep camera view empty for background calibration.")
    mode_var = tk.StringVar(value="STATIC A–Z · 0–9  |  Calibrating")
    ttk.Label(sidebar, textvariable=mode_var, style="Muted.TLabel",
             font=("DejaVu Sans", 9), wraplength=300, justify="center").grid(
                 row=2, column=0, sticky="ew", pady=(0, 8))
    confidence_var = tk.StringVar(value="Confidence  0%")
    ttk.Label(sidebar, textvariable=confidence_var, style="Warning.TLabel",
              anchor="w").grid(row=3, column=0, sticky="ew")
    progress_var = tk.DoubleVar(value=0)
    ttk.Progressbar(sidebar, variable=progress_var, maximum=100).grid(
        row=4, column=0, sticky="ew", pady=(4, 10))
    hands_var = tk.StringVar(value="Hands detected: 0")
    ttk.Label(sidebar, textvariable=hands_var, style="Muted.TLabel",
              anchor="w").grid(row=5, column=0, sticky="ew", pady=(0, 6))

    section("RECOGNIZED TEXT", 6)
    text_var = tk.StringVar(value="")
    text_panel = tk.Frame(sidebar, bg=palette["dark"], height=112)
    text_panel.grid(row=7, column=0, sticky="ew", pady=(0, 6))
    text_panel.grid_propagate(False)
    text_panel.columnconfigure(0, weight=1)
    text_panel.rowconfigure(0, weight=1)
    ttk.Label(text_panel, textvariable=text_var, style="Dark.TLabel", anchor="nw",
              justify="left", wraplength=280, font=("DejaVu Sans", 14),
              padding=10).grid(row=0, column=0, sticky="nsew")
    auto_var = tk.StringVar(value="AUTO-TYPE OFF")
    ttk.Label(sidebar, textvariable=auto_var, style="Muted.TLabel",
              anchor="w").grid(row=8, column=0, sticky="ew", pady=(0, 8))
    section("CONTROLS", 9)
    style.configure("App.TButton", font=("DejaVu Sans", 9, "bold"),
                    foreground="white", background="#0f3460", padding=7,
                    borderwidth=0)
    style.map("App.TButton", background=[("active", "#164e7a")])
    style.configure("App.Success.TButton", font=("DejaVu Sans", 9, "bold"),
                    foreground="white", background="#1b5e20", padding=7,
                    borderwidth=0)
    style.map("App.Success.TButton", background=[("active", "#2e7d32")])

    state = {"paused": False, "auto": False,
             "candidate": None, "confirmed": None, "score": 0.0,
             "frame_number": 0, "last_appended": -10**9,
             "stable_hand_count": None, "fullscreen": False}
    text = TextBuffer()
    recent = deque(maxlen=args.confirmation_frames)
    hand_count_history = deque(maxlen=5)
    auto_typer = _DwellAutoTyper()

    def update_text() -> None:
        text_var.set(text.text)

    def add_letter() -> None:
        letter = state["confirmed"]
        if letter is None:
            status_var.set("No stable, above-threshold sign to add.")
            return
        if state["frame_number"] - state["last_appended"] < args.cooldown_frames:
            status_var.set("Letter append ignored during cooldown.")
            return
        text.append_character(letter)
        state["last_appended"] = state["frame_number"]
        update_text()

    def toggle_auto() -> None:
        state["auto"] = not state["auto"]
        auto_var.set("AUTO-TYPE ON · hold sign 500 ms" if state["auto"] else "AUTO-TYPE OFF")
        auto_typer.reset()

    def toggle_pause() -> None:
        state["paused"] = not state["paused"]
        recent.clear()
        state["candidate"] = state["confirmed"] = None
        state["score"] = 0.0
        auto_typer.reset()
        status_var.set("PAUSED" if state["paused"] else "Recognition resumed.")

    def toggle_fullscreen() -> None:
        state["fullscreen"] = not state["fullscreen"]
        root.attributes("-fullscreen", state["fullscreen"])

    button_grid = tk.Frame(sidebar, bg=palette["panel"])
    button_grid.grid(row=10, column=0, sticky="ew")
    button_grid.columnconfigure((0, 1), weight=1)
    controls = [
        ("✓  ADD CHARACTER", add_letter, 0, 0, "#1b5e20"),
        ("SPACE", lambda: (text.append_space(), update_text()), 0, 1, "#0f3460"),
        ("⌫  BACKSPACE", lambda: (text.backspace(), update_text()), 1, 0, "#0f3460"),
        ("✕  CLEAR", lambda: (text.clear(), update_text()), 1, 1, "#0f3460"),
        ("T  AUTO-TYPE", toggle_auto, 2, 0, "#1b5e20"),
        ("A–Z · 0–9", None, 2, 1, "#0f3460"),
        ("P  PAUSE / RESUME", toggle_pause, 3, 0, "#0f3460"),
        ("F  FULLSCREEN", toggle_fullscreen, 3, 1, "#0f3460"),
    ]
    for label, command, row, column, color in controls:
        button_style = "App.Success.TButton" if color == "#1b5e20" else "App.TButton"
        button = ttk.Button(button_grid, text=label, command=command, style=button_style)
        if command is None:
            button.state(["disabled"])
        button.grid(row=row, column=column, sticky="ew", padx=3, pady=3)
    ttk.Label(sidebar, text="Keys: A add · Space insert · Backspace undo · C clear\n"
             "T auto-type · P pause · F fullscreen · Q quit",
             style="Muted.TLabel", font=("DejaVu Sans", 8), justify="left",
             wraplength=300).grid(row=11, column=0, sticky="ew", pady=(14, 4))
    ttk.Label(root, textvariable=status_var, style="Footer.TLabel",
              anchor="w", padding=(14, 6)).grid(row=2, column=0, sticky="ew")

    def on_key(event):
        key = event.keysym.lower()
        if key in ("q", "escape"):
            root.destroy()
        elif key == "a":
            add_letter()
        elif key == "space":
            text.append_space(); update_text()
        elif key == "backspace":
            text.backspace(); update_text()
        elif key == "c":
            text.clear(); update_text()
        elif key == "t":
            toggle_auto()
        elif key == "p":
            toggle_pause()
        elif key == "f":
            toggle_fullscreen()
        return "break"

    root.bind_all("<KeyPress>", on_key)
    # Load PyTorch only after Tk has created and flushed its X11 widgets. This
    # prevents its worker threads racing Xlib while fonts and cursors initialize.
    from isl_recognition.evaluation.common import load_model

    static_model, static_checkpoint, _ = load_model(
        args.static_model.expanduser().resolve(), "static", device
    )
    if static_checkpoint.get("architecture") != "static_landmark_mlp":
        raise ValueError("The model is not a MediaPipe landmark model. Retrain with isl-train-static.")
    from isl_recognition.labels import STATIC_LABELS
    checkpoint_labels = tuple(static_checkpoint.get("labels", ()))
    mode_prefix = ("STATIC A–Z · 0–9" if checkpoint_labels == STATIC_LABELS else
                   f"RETRAIN MODEL · {len(checkpoint_labels)}/36 classes")
    if checkpoint_labels != STATIC_LABELS:
        status_var.set("This checkpoint has no digit classes. Run isl-prepare-static, then isl-train-static.")

    hands_module = mp.solutions.hands
    tracker = hands_module.Hands(static_image_mode=False, max_num_hands=2,
                                 model_complexity=0, min_detection_confidence=0.35,
                                 min_tracking_confidence=0.5)
    capture = cv2.VideoCapture(args.camera)
    camera_available = capture.isOpened()
    gate = BackgroundGate(args.calibration_frames)
    text_buffer = text

    def update_camera() -> None:
        if not root.winfo_exists():
            return
        if not camera_available:
            status_var.set(f"Camera {args.camera} unavailable; connect a camera and restart.")
            mode_var.set(f"{mode_prefix}  |  Camera unavailable")
            sign_var.set("—")
            height, width = 480, 640
            blank = np.zeros((height, width, 3), dtype=np.uint8)
            cv2.putText(blank, "Camera unavailable", (150, 225),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (148, 163, 184), 2, cv2.LINE_AA)
            cv2.putText(blank, f"Requested camera index: {args.camera}", (150, 265),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 116, 139), 1, cv2.LINE_AA)
            encoded, png = cv2.imencode(".png", blank)
            if not encoded:
                raise RuntimeError("Could not encode the camera placeholder frame")
            photo = tk.PhotoImage(master=root, data=base64.b64encode(png).decode("ascii"),
                                  format="PNG")
            camera_label.configure(image=photo, text="")
            camera_label.image = photo
            root.after(500, update_camera)
            return
        ok, frame = capture.read()
        if not ok:
            status_var.set(f"Camera {args.camera} stopped returning frames.")
            root.after(500, update_camera)
            return
        state["frame_number"] += 1
        # Keep the classifier input in the same left/right orientation as the
        # training photos. Mirror only the final preview for a natural camera view.
        results = None
        foreground = False
        if not gate.calibrated:
            gate.update_background(frame)
            state["candidate"] = state["confirmed"] = None
            state["score"] = 0.0
            display_state = (f"Background calibration: keep the view empty "
                             f"({gate.calibration_samples}/{args.calibration_frames})")
        else:
            results = tracker.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            hands = results.multi_hand_landmarks or []
            hand_count = len(hands)
            hand_count_history.append(hand_count)
            stable_hand_count = Counter(hand_count_history).most_common(1)[0][0]
            if stable_hand_count != state["stable_hand_count"]:
                recent.clear()
                state["stable_hand_count"] = stable_hand_count
            landmark_features = landmarks_to_features(hands, frame.shape)
            foreground = landmark_features is not None and gate.has_foreground(
                frame, args.foreground_min, args.foreground_max
            )
            now = time.monotonic()
            if not state["paused"] and foreground:
                model_input = torch.from_numpy(landmark_features).reshape(1, LANDMARK_FEATURES).to(device)
                candidate, _, score = _prediction(static_model, model_input,
                    static_checkpoint, device, args.confidence, args.margin,
                    hand_count=hand_count)
                # UI-only rule: with one stable hand, show X as I. The short
                # hand-count vote filters brief detector blips; rewrite old votes
                # too so a prior X frame cannot flash or confirm as X.
                if stable_hand_count == 1:
                    if candidate == "X":
                        candidate = "I"
                    if "X" in recent:
                        previous_votes = list(recent)
                        recent.clear()
                        recent.extend("I" if vote == "X" else vote for vote in previous_votes)
                state["candidate"], state["score"] = candidate, score
                recent.append(candidate)
            elif not foreground or state["paused"]:
                recent.clear()
                state["candidate"], state["score"] = None, 0.0
            votes = Counter(label for label in recent if label is not None)
            confirmed = None
            if votes:
                label, count = votes.most_common(1)[0]
                if count >= max(2, int(np.ceil(args.confirmation_frames * 0.75))):
                    confirmed = label
            state["confirmed"] = confirmed
            buffered_candidate = votes.most_common(1)[0][0] if votes else None
            if state["auto"]:
                letter = auto_typer.update(confirmed, foreground and not state["paused"], now)
                if letter:
                    text_buffer.append_character(letter)
                    update_text()
            if state["paused"]:
                display_state = "PAUSED"
            elif not foreground:
                display_state = "Show your hand"
            elif confirmed:
                display_state = f"Confirmed: {confirmed}"
            elif state["candidate"]:
                display_state = "Stabilizing…"
            else:
                display_state = "Uncertain"
            hands_var.set(f"Hands detected: {hand_count}")
            status_var.set(display_state)
            sign_var.set(buffered_candidate or "…")
            mode_var.set(f"{mode_prefix}  |  {display_state}")
            confidence_var.set(f"Confidence  {state['score']:.0%}")
            progress_var.set(100 if confirmed else (min(100, len(recent) * 100 /
                                max(1, args.confirmation_frames))))
        if results is not None:
            _draw_hand_grid(frame, results, hands_module)
        display_frame = cv2.flip(frame, 1)
        max_width = max(1, camera_label.winfo_width())
        max_height = max(1, camera_label.winfo_height())
        frame_height, frame_width = display_frame.shape[:2]
        scale = min(max_width / frame_width, max_height / frame_height)
        if abs(scale - 1.0) > 1e-3:
            interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
            display_frame = cv2.resize(
                display_frame,
                (max(1, round(frame_width * scale)), max(1, round(frame_height * scale))),
                interpolation=interpolation,
            )
        encoded, png = cv2.imencode(".png", display_frame)
        if not encoded:
            raise RuntimeError("Could not encode the camera frame")
        photo = tk.PhotoImage(master=root, data=base64.b64encode(png).decode("ascii"),
                              format="PNG")
        camera_label.configure(image=photo, text="")
        camera_label.image = photo
        root.after(15, update_camera)

    root.protocol("WM_DELETE_WINDOW", root.destroy)
    root.after(0, update_camera)
    try:
        root.mainloop()
    finally:
        capture.release()
        tracker.close()
