"""
security_app.py — Main Pipeline
================================
Wires all modules into a single real-time security monitoring pipeline.

Pipeline per frame:
    Webcam → MotionDetector → (gate) → DangerDetector + FaceRecognizer (threaded)
    → Alerter → SecurityLogger → OpenCV overlay → display

Usage:
    python security_app.py

Controls:
    Q — quit
    R — reset motion detector background model
    M — toggle motion mask debug view
    L — toggle LLM / placeholder alert mode
    F — register a new face (pauses main loop)
"""

import cv2
import sys
import time
import logging
import threading
import numpy as np
from pathlib import Path
from datetime import datetime

# ── Project imports ───────────────────────────────────────────────────────────
from config import (
    PATHS, CAMERA, MOTION, YOLO, FACENET, LLM, ALERTS, UI,
    init_dirs, validate, warmup_chromadb
)
from modules.motion     import MotionDetector
from modules.detector   import DangerDetector
from modules.recognizer import FaceRecognizer
from modules.logger     import SecurityLogger
from modules.alerter    import Alerter

# ── Create directories and pre-warm ChromaDB before anything else ─────────────
init_dirs()
warmup_chromadb()

# ══════════════════════════════════════════════════════════════════════════════
# LOGGING SETUP
# ══════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(PATHS["app_log"]))
    ]
)
log = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# OVERLAY RENDERER
# ══════════════════════════════════════════════════════════════════════════════

SEVERITY_BGR = {
    "CRITICAL": (0,   34,  255),
    "HIGH":     (0,   102, 255),
    "MEDIUM":   (0,   170, 255),
    "LOW":      (0,   187, 68),
    "NONE":     (128, 128, 128),
}


def draw_overlay(frame: np.ndarray,
                 yolo_output: dict,
                 facenet_output: dict,
                 alert: dict,
                 motion_detected: bool,
                 show_motion_mask: bool = False,
                 motion_mask: np.ndarray = None) -> np.ndarray:
    """
    Draws all detection overlays onto the frame.

    Elements:
    - YOLOv8 bounding boxes + tier labels (colour coded by danger tier)
    - FaceNet bounding boxes + identity labels (green=known, red=unknown)
    - Alert panel at bottom (severity + summary + action)
    - Status bar at top (motion indicator + mode + timestamp)
    - Optional motion mask debug panel (top-right corner)
    """
    vis = frame.copy()
    h, w = vis.shape[:2]

    # ── YOLOv8 boxes ─────────────────────────────────────────────────────────
    tier_display = YOLO["tier_display"]
    for obj in yolo_output["objects"]:
        x1, y1, x2, y2 = obj["bbox"]
        tier    = obj["danger_tier"]
        display = tier_display.get(tier, tier_display[-1])
        color   = display["color"]
        label   = f"{obj['class']} | {display['label']} | {obj['confidence']:.2f}"

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(vis, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(vis, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    # ── FaceNet boxes ─────────────────────────────────────────────────────────
    for face in facenet_output["faces"]:
        x1, y1, x2, y2 = face["bbox"]
        color = (0, 200, 0) if face["is_known"] else (0, 0, 255)
        label = f"{face['name']} ({face['confidence']:.2f})"
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(vis, label, (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # ── Alert panel ───────────────────────────────────────────────────────────
    if not alert["suppressed"] and alert["severity"] not in ("NONE", "suppressed"):
        panel_h     = 90
        alert_color = SEVERITY_BGR.get(alert["severity"], (128, 128, 128))

        overlay = vis.copy()
        cv2.rectangle(overlay, (0, h - panel_h), (w, h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, vis, 0.25, 0, vis)

        cv2.putText(vis, f"[ {alert['severity']} ]",
                    (10, h - panel_h + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, alert_color, 2)

        lines = [l for l in alert["alert_text"].split("\n") if l.strip()]
        if len(lines) > 1:
            cv2.putText(vis, lines[1][:90], (10, h - panel_h + 58),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (210, 210, 210), 1)
        if len(lines) > 2:
            cv2.putText(vis, lines[2][:90], (10, h - panel_h + 78),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)

        source = alert.get("source", "?").upper()
        cv2.putText(vis, f"[{source}]", (w - 110, h - panel_h + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1)

    # ── Status bar ────────────────────────────────────────────────────────────
    cv2.rectangle(vis, (0, 0), (w, 34), (25, 25, 40), -1)

    motion_text  = "● MOTION" if motion_detected else "○ idle"
    motion_color = (0, 220, 0) if motion_detected else (100, 100, 100)
    cv2.putText(vis, motion_text, (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, motion_color, 1)

    mode_text  = "LLM" if LLM["use_llm_alerts"] else "RULE"
    mode_color = (0, 180, 255) if LLM["use_llm_alerts"] else (180, 180, 180)
    cv2.putText(vis, f"MODE:{mode_text}", (w // 2 - 50, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, mode_color, 1)

    ts = datetime.now().strftime("%H:%M:%S")
    cv2.putText(vis, ts, (w - 80, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1)

    # ── Motion mask debug panel ───────────────────────────────────────────────
    if show_motion_mask and motion_mask is not None:
        mask_bgr = cv2.cvtColor(motion_mask, cv2.COLOR_GRAY2BGR)
        mask_bgr = cv2.resize(mask_bgr, (w // 3, h // 3))
        vis[34:34 + mask_bgr.shape[0], w - mask_bgr.shape[1]:w] = mask_bgr
        cv2.putText(vis, "motion mask", (w - mask_bgr.shape[1] + 4, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    return vis


# ══════════════════════════════════════════════════════════════════════════════
# FACE REGISTRATION
# ══════════════════════════════════════════════════════════════════════════════

def register_face_interactive(recognizer: FaceRecognizer):
    """
    Pauses the main loop and registers a new face via the webcam.
    Prompts for a name in the terminal, captures shots, rebuilds embeddings.
    """
    print("\n" + "=" * 50)
    print("FACE REGISTRATION")
    print("=" * 50)
    name = input("Enter name for new person (or Enter to cancel): ").strip()

    if not name:
        print("Registration cancelled.")
        return

    print(f"Registering '{name}' — get in position.")
    recognizer.register_person(name=name, n_shots=4, delay_between=1.5)
    print(f"Registration complete. '{name}' added to known faces.")
    print("=" * 50 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def main():
    log.info("=" * 60)
    log.info("Offline AI Security Application — Starting")
    log.info("=" * 60)

    # ── Startup ───────────────────────────────────────────────────────────────
    validate()

    # ── Load all modules ──────────────────────────────────────────────────────
    log.info("Loading modules...")

    motion_detector = MotionDetector(MOTION)
    log.info("MotionDetector ready.")

    danger_detector = DangerDetector(YOLO)
    log.info("DangerDetector ready.")

    face_recognizer = FaceRecognizer(FACENET, PATHS)
    log.info(f"FaceRecognizer ready. Known: {face_recognizer.known_people}")

    security_logger = SecurityLogger(PATHS, LLM)
    security_logger.sync_known_faces(PATHS["known_faces_dir"])
    log.info("SecurityLogger ready.")

    alerter = Alerter(LLM, ALERTS, security_logger)
    log.info("Alerter ready.")

    # ── Open webcam ───────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(CAMERA["index"])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA["height"])

    if not cap.isOpened():
        log.error(f"Could not open camera {CAMERA['index']}.")
        sys.exit(1)

    log.info(f"Camera opened: {CAMERA['width']}x{CAMERA['height']}")

    # ── State ─────────────────────────────────────────────────────────────────
    # Last known outputs — persist across non-inference frames so overlay
    # doesn't flicker between detections
    last_yolo    = {"objects": [], "alert_level": 0}
    last_facenet = {"faces": [], "known_count": 0, "unknown_count": 0}
    last_alert   = {
        "alert_text": "", "severity": "NONE",
        "suppressed": False, "source": "none", "timestamp": ""
    }

    show_mask   = False
    frame_count = 0
    fps_timer   = time.time()
    fps         = 0.0

    # ── FaceNet background thread ─────────────────────────────────────────────
    # FaceNet is too slow (~500ms) to run on the main loop thread.
    # We run it in a background thread so it never blocks the video feed.
    # Main loop reads last_facenet which updates whenever the thread finishes.
    facenet_lock    = threading.Lock()
    facenet_running = threading.Event()  # set = thread is busy, clear = free

    def run_facenet_async(frame_copy: np.ndarray):
        """Runs FaceNet in background. Updates last_facenet when done."""
        nonlocal last_facenet
        try:
            t0 = time.time()
            result = face_recognizer.recognize(frame_copy)
            t1 = time.time()
            print(f"FaceNet (background): {(t1-t0)*1000:.1f}ms")
            with facenet_lock:
                last_facenet = result
        except Exception as e:
            log.debug(f"FaceNet thread error: {e}")
        finally:
            facenet_running.clear()  # mark thread as free

    log.info("Main loop started.")
    print("\nControls: Q=quit | R=reset bg | M=mask | L=toggle LLM | F=add face\n")

    # ══════════════════════════════════════════════════════════════════════════
    # MAIN LOOP
    # ══════════════════════════════════════════════════════════════════════════

    while True:
        ret, frame = cap.read()
        if not ret:
            log.error("Camera read failed.")
            break

        frame_count += 1

        # Rolling FPS calculation — updated every 30 frames
        if frame_count % 30 == 0:
            elapsed   = time.time() - fps_timer
            fps       = 30 / elapsed if elapsed > 0 else 0
            fps_timer = time.time()

        # ── Motion detection (runs every frame — very cheap) ──────────────────
        should_infer, motion_mask = motion_detector.detect(frame)

        # ── Inference gate ────────────────────────────────────────────────────
        # Only runs when motion detected OR every idle_every_n frames
        if should_infer:

            # YOLO runs on main thread
            t0 = time.time()
            last_yolo = danger_detector.detect(frame)
            t1 = time.time()

            # FaceNet runs in background thread — non-blocking
            # Only spawns a new thread if the previous one has finished
            if not facenet_running.is_set():
                facenet_running.set()
                frame_copy = frame.copy()  # thread needs its own copy
                t = threading.Thread(
                    target=run_facenet_async,
                    args=(frame_copy,),
                    daemon=True  # thread dies automatically with main process
                )
                t.start()
            t2 = time.time()

            # Alert uses last known facenet result
            # May be 1-2 frames old — acceptable for security use case
            last_alert = alerter.generate(last_yolo, last_facenet)
            t3 = time.time()
            print(f"YOLO: {(t1-t0)*1000:.1f}ms | FaceNet thread spawn: {(t2-t1)*1000:.1f}ms | Alert: {(t3-t2)*1000:.1f}ms | FPS: {fps:.1f}")

            # Log to JSONL + ChromaDB
            security_logger.log_event(last_yolo, last_facenet, last_alert)

            # Console output for significant alerts
            if (not last_alert["suppressed"] and
                    last_alert["severity"] not in ("NONE", "suppressed")):
                log.info(
                    f"ALERT [{last_alert['severity']}] "
                    f"objects={len(last_yolo['objects'])} "
                    f"unknown_faces={last_facenet['unknown_count']} "
                    f"source={last_alert['source']}"
                )

            # ── Draw overlay ──────────────────────────────────────────────────────
            display = danger_detector.draw(frame, last_yolo)
            display = face_recognizer.draw(display, last_facenet)
        else:
            display = frame.copy()
            
        display = draw_overlay(
            display, last_yolo, last_facenet, last_alert,
            should_infer, show_mask, motion_mask
        )

        # FPS counter
        cv2.putText(display, f"{fps:.1f}fps",
                    (display.shape[1] - 75, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1)

        cv2.imshow("Offline AI Security Monitor", display)

        # ── Keyboard controls ─────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            log.info("Quit.")
            break

        elif key == ord("r"):
            motion_detector.reset()
            log.info("Background model reset.")

        elif key == ord("m"):
            show_mask = not show_mask

        elif key == ord("l"):
            new_mode = not alerter.use_llm
            alerter.set_mode(new_mode)
            LLM["use_llm_alerts"] = new_mode

        elif key == ord("f"):
            register_face_interactive(face_recognizer)
            security_logger.sync_known_faces(PATHS["known_faces_dir"])

    # ── Cleanup ───────────────────────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()
    log.info(f"Stopped. Frames: {frame_count} | Events: {security_logger.event_count}")


if __name__ == "__main__":
    main()