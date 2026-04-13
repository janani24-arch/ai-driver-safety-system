"""
phone_detector.py — YOLO Phone Distraction Detector
=====================================================
Uses YOLOv8 (ultralytics) to detect mobile phones in the frame.
Automatically downloads the YOLOv8n model on first run (~6 MB).

Install:
    pip install ultralytics

COCO class ID for cell phone = 67
"""

import numpy as np
import cv2
from config import Config


# COCO dataset class ID for "cell phone"
PHONE_CLASS_ID = 67


class PhoneDetector:
    """
    Detects mobile phones using YOLOv8n.
    Lightweight nano model — fast enough for real-time use.
    """

    def __init__(self):
        self._model  = None
        self._loaded = False
        self._load_model()

    def _load_model(self):
        try:
            from ultralytics import YOLO
            print("[PhoneDetector] Loading YOLOv8n model...")
            self._model  = YOLO("yolov8n.pt")   # auto-downloads if not present
            self._loaded = True
            print("[PhoneDetector] YOLOv8n loaded successfully.")
        except ImportError:
            print("[PhoneDetector] 'ultralytics' not installed.")
            print("                Run:  pip install ultralytics")
            print("                Phone detection will be DISABLED.")
        except Exception as e:
            print(f"[PhoneDetector] Failed to load model: {e}")
            print("                Phone detection will be DISABLED.")

    @property
    def is_available(self) -> bool:
        return self._loaded

    def detect(self, frame: np.ndarray):
        """
        Run YOLO on frame and return list of phone bounding boxes.

        Args:
            frame: BGR image from OpenCV.

        Returns:
            List of (x1, y1, x2, y2, confidence) tuples for each phone detected.
            Empty list if no phone found or model not loaded.
        """
        if not self._loaded:
            return []

        results = self._model(
            frame,
            verbose=False,
            conf=Config.PHONE_CONFIDENCE,
            classes=[PHONE_CLASS_ID],   # only detect phones
        )

        phones = []
        for r in results:
            for box in r.boxes:
                cls  = int(box.cls[0])
                conf = float(box.conf[0])
                if cls == PHONE_CLASS_ID and conf >= Config.PHONE_CONFIDENCE:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    phones.append((x1, y1, x2, y2, conf))

        return phones

    def draw_detections(self, frame: np.ndarray, detections: list) -> np.ndarray:
        """
        Draw bounding boxes around detected phones.

        Args:
            frame      : BGR image.
            detections : List of (x1, y1, x2, y2, conf) tuples.

        Returns:
            Annotated frame.
        """
        for (x1, y1, x2, y2, conf) in detections:
            cv2.rectangle(frame, (x1, y1), (x2, y2), Config.PHONE_COLOR, 2)
            label = f"Phone {conf:.0%}"
            label_y = y1 - 10 if y1 - 10 > 10 else y1 + 20
            cv2.putText(frame, label, (x1, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, Config.PHONE_COLOR, 2)
        return frame