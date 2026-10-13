import cv2
import numpy as np


class BackgroundGate:
    """Lightweight, user-calibrated foreground gate; not a hand detector."""

    def __init__(self, calibration_frames: int = 30, difference_threshold: int = 25):
        if calibration_frames < 1:
            raise ValueError("calibration_frames must be positive")
        self.calibration_frames = calibration_frames
        self.difference_threshold = difference_threshold
        self._background: np.ndarray | None = None
        self._samples = 0

    @property
    def calibrated(self) -> bool:
        return self._samples >= self.calibration_frames

    @property
    def calibration_samples(self) -> int:
        return self._samples

    def update_background(self, frame: np.ndarray) -> bool:
        sample = frame.astype(np.float32)
        if self._background is None:
            self._background = sample
        else:
            cv2.accumulateWeighted(sample, self._background, 1.0 / (self._samples + 1))
        self._samples += 1
        return self.calibrated

    def foreground_fraction(self, frame: np.ndarray) -> float:
        if not self.calibrated or self._background is None:
            return 0.0
        background = cv2.convertScaleAbs(self._background)
        difference = cv2.absdiff(frame, background)
        gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, self.difference_threshold, 255, cv2.THRESH_BINARY)
        return cv2.countNonZero(mask) / mask.size

    def has_foreground(self, frame: np.ndarray, minimum: float, maximum: float) -> bool:
        fraction = self.foreground_fraction(frame)
        return minimum <= fraction <= maximum
