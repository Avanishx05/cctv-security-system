"""
modules/detector.py — YOLOv8 Weapon / Danger Detection Module
==============================================================
Wraps YOLOv8 inference and maps detected classes to danger tiers.

Usage:
    from modules.detector import DangerDetector
    from config import YOLO

    detector = DangerDetector(YOLO)
    result   = detector.detect(frame)

    # result = {
    #     'objects':     [{'class', 'danger_tier', 'confidence', 'bbox'}],
    #     'alert_level': int   (0=safe, 1=risky, 2=dangerous)
    # }
"""

import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from config import YOLO as YOLO_CONFIG


class DangerDetector:
    """
    YOLOv8-based weapon and dangerous object detector.

    Loads fine-tuned weights from config and runs inference on frames.
    Maps predicted class names to danger tiers (risky=1, dangerous=2).
    Returns structured output compatible with the cross-model schema.
    """

    def __init__(self, config: dict = YOLO_CONFIG):
        """
        Args:
            config: YOLO dict from config.py
        """
        self.config     = config
        self.model_path = Path(config["model_path"])
        self.conf       = config["conf"]
        self.imgsz      = config["imgsz"]
        self.device     = config["device"]
        self.tier_map   = config["tier_map"]
        self.tier_display = config["tier_display"]

        # Load model
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"YOLOv8 weights not found: {self.model_path}\n"
                f"Copy best.pt to models/danger_detection_best.pt"
            )

        print(f"[DangerDetector] Loading model from {self.model_path}...")
        self._model = YOLO(str(self.model_path))
        print(f"[DangerDetector] Model loaded. Device: {self.device}")

    def detect(self, frame: np.ndarray) -> dict:
        """
        Runs YOLOv8 inference on a single frame.

        Args:
            frame: BGR numpy array (H, W, 3)

        Returns:
            {
                'objects': [
                    {
                        'class':       str,   # 'risky' or 'dangerous'
                        'danger_tier': int,   # 1 or 2
                        'confidence':  float,
                        'bbox':        [x1, y1, x2, y2]  # pixel coords
                    }
                ],
                'alert_level': int  # highest tier detected (0 if nothing)
            }
        """
        results = self._model.predict(
            frame,
            conf=self.conf,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False       # suppress per-frame console output
        )

        objects = []

        for result in results:
            for box in result.boxes:
                cls_name = self._model.names[int(box.cls[0])]
                conf     = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Map class name to danger tier
                tier = self.tier_map.get(cls_name, 0)

                objects.append({
                    "class":       cls_name,
                    "danger_tier": tier,
                    "confidence":  round(conf, 3),
                    "bbox":        [x1, y1, x2, y2]
                })

        # Overall alert level = highest tier detected in this frame
        alert_level = max((o["danger_tier"] for o in objects), default=0)

        return {
            "objects":     objects,
            "alert_level": alert_level
        }

    def draw(self, frame: np.ndarray, result: dict) -> np.ndarray:
        """
        Draws detection bounding boxes and tier labels onto a frame.
        Returns a copy of the frame with overlays drawn.

        Args:
            frame:  BGR numpy array
            result: output dict from detect()

        Returns:
            BGR numpy array with detections drawn
        """
        vis = frame.copy()

        for obj in result["objects"]:
            x1, y1, x2, y2 = obj["bbox"]
            tier    = obj["danger_tier"]
            display = self.tier_display.get(tier, self.tier_display[-1])
            color   = display["color"]
            label   = f"{obj['class']} | {display['label']} | {obj['confidence']:.2f}"

            # Bounding box
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

            # Label background for readability
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
            )
            cv2.rectangle(vis, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(vis, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        return vis

    @property
    def class_names(self) -> list:
        """Returns the class names from the loaded model."""
        return list(self._model.names.values())


# ══════════════════════════════════════════════════════════════════════════════
# QUICK TEST — run directly to test detector on webcam
# python modules/detector.py
# Press Q to quit
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    sys.path.append("..")

    from config import CAMERA

    detector = DangerDetector()
    cap      = cv2.VideoCapture(CAMERA["index"])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA["height"])

    print("Danger detector test — press Q to quit")
    print(f"Model classes: {detector.class_names}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = detector.detect(frame)
        vis    = detector.draw(frame, result)

        # Print detections to console
        if result["objects"]:
            for obj in result["objects"]:
                print(f"  {obj['class']} | tier={obj['danger_tier']} | conf={obj['confidence']:.2f}")

        cv2.imshow("Danger Detector Test", vis)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()