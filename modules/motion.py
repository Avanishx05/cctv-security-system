"""
modules/motion.py — Motion Detection Module
============================================
Frame differencing motion detector.
Gates all downstream inference — YOLOv8 and FaceNet only run when motion
is detected OR when the idle inference interval is hit.

Usage:
    from modules.motion import MotionDetector
    from config import MOTION

    detector = MotionDetector(MOTION)

    motion, mask = detector.detect(frame)
    if motion:
        # run YOLOv8 + FaceNet
"""

import cv2
import time
import numpy as np
from config import MOTION


class MotionDetector:
    """
    Background subtraction motion detector using frame differencing.

    How it works:
    1. Convert frame to grayscale + blur (removes noise)
    2. Compute absolute difference vs stored background frame
    3. Threshold the diff to get a binary motion mask
    4. Find contours in mask — contours above area threshold = motion
    5. Slowly update background to adapt to lighting changes

    The background update rate controls how fast the detector adapts:
    - High rate (0.2+) = adapts quickly, may miss slow-moving objects
    - Low rate (0.02)  = adapts slowly, more robust but slower to recover
      from sudden lighting changes (e.g. lights turning on)
    """

    def __init__(self, config: dict = MOTION):
        """
        Args:
            config: MOTION dict from config.py
        """
        self.threshold       = config["threshold"]
        self.blur_kernel     = config["blur_kernel"]
        self.bg_update_rate  = config["bg_update_rate"]
        self.cooldown_s      = config["cooldown_s"]
        self.idle_every_n    = config["idle_every_n"]

        # Internal state
        self._bg_frame         = None    # Running background model
        self._last_trigger_t   = 0.0    # Timestamp of last motion trigger
        self._frame_count      = 0      # Total frames processed

    def detect(self, frame: np.ndarray) -> tuple[bool, np.ndarray]:
        """
        Detects motion in the given frame.

        Args:
            frame: BGR numpy array from cv2.VideoCapture

        Returns:
            Tuple of:
                should_infer (bool) — True if downstream models should run
                motion_mask (np.ndarray) — binary mask showing motion regions
                                           (useful for debug visualisation)
        """
        self._frame_count += 1

        # ── Preprocess frame ──────────────────────────────────────────────────
        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (self.blur_kernel, self.blur_kernel), 0)

        # ── Initialise background on first frame ──────────────────────────────
        if self._bg_frame is None:
            self._bg_frame = blurred.copy()
            return False, np.zeros_like(gray)

        # ── Frame difference ──────────────────────────────────────────────────
        diff      = cv2.absdiff(self._bg_frame, blurred)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

        # Dilate to fill small gaps in motion regions
        thresh = cv2.dilate(thresh, None, iterations=2)

        # ── Find motion contours ──────────────────────────────────────────────
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Motion = at least one contour larger than threshold area
        raw_motion = any(
            cv2.contourArea(c) > self.threshold for c in contours
        )

        # ── Update background model ───────────────────────────────────────────
        # Weighted average — slowly blends current frame into background
        # This lets the detector adapt to gradual lighting changes
        self._bg_frame = cv2.addWeighted(
            self._bg_frame, 1.0 - self.bg_update_rate,
            blurred,         self.bg_update_rate,
            0
        )

        # ── Decide whether to trigger inference ───────────────────────────────
        now          = time.time()
        idle_trigger = (self._frame_count % self.idle_every_n == 0)

        # Motion trigger: raw motion detected AND cooldown has elapsed
        motion_trigger = (
            raw_motion and
            (now - self._last_trigger_t) >= self.cooldown_s
        )

        should_infer = motion_trigger or idle_trigger

        if should_infer:
            self._last_trigger_t = now

        return should_infer, thresh

    def reset(self):
        """
        Resets the background model.
        Call this if the camera is moved or scene changes significantly.
        """
        self._bg_frame       = None
        self._last_trigger_t = 0.0
        self._frame_count    = 0

    @property
    def frame_count(self) -> int:
        """Total frames processed since init or last reset."""
        return self._frame_count

    @property
    def is_initialised(self) -> bool:
        """True once the background model has been established."""
        return self._bg_frame is not None


# ══════════════════════════════════════════════════════════════════════════════
# QUICK TEST — run this file directly to test motion detection on webcam
# python modules/motion.py
# Press Q to quit, R to reset background model
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    sys.path.append("..")   # Allow import from project root when run directly

    from config import CAMERA

    detector = MotionDetector()
    cap      = cv2.VideoCapture(CAMERA["index"])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA["height"])

    print("Motion detector test — press Q to quit, R to reset background")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera read failed.")
            break

        should_infer, mask = detector.detect(frame)

        # Draw motion status on frame
        status = "MOTION" if should_infer else "idle"
        color  = (0, 255, 0) if should_infer else (100, 100, 100)
        cv2.putText(frame, status, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        cv2.putText(frame, f"Frames: {detector.frame_count}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

        # Show frame + motion mask side by side
        mask_bgr  = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        combined  = np.hstack([frame, mask_bgr])
        cv2.imshow("Motion Detector Test (left=feed, right=mask)", combined)

        key = cv2.waitKey(1)
        if key == ord("q"):
            break
        elif key == ord("r"):
            detector.reset()
            print("Background model reset.")

    cap.release()
    cv2.destroyAllWindows()