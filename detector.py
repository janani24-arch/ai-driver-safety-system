"""
detector.py — Core Detection Engine
=====================================
Integrates:
  1. MediaPipe FaceLandmarker  → EAR (drowsiness) + MAR (yawning)
  2. YOLOv8n                   → Phone distraction detection
  3. SoundAlert                → Beep alerts for each event
  4. AlertManager              → On-screen banners and HUD

Compatible with mediapipe 0.10.30+ (Tasks API).
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import FaceLandmarkerOptions
import numpy as np
import urllib.request
import os

from config import Config
from landmarks import LEFT_EYE, RIGHT_EYE, MOUTH, eye_aspect_ratio, mouth_aspect_ratio, extract_coords
from alert import AlertManager
from sound import SoundAlert
from phone_detector import PhoneDetector


# ─── MediaPipe Model Download ─────────────────────────────────────────────────

MODEL_PATH = "face_landmarker.task"
MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/"
              "face_landmarker/face_landmarker/float16/1/face_landmarker.task")

def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("[INFO] Downloading MediaPipe face landmark model (~30 MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("[INFO] Model downloaded successfully.")


class DriverSafetyDetector:
    """
    Full driver safety detection system.

    Detects:
      - Drowsiness  via EAR (Eye Aspect Ratio)
      - Yawning     via MAR (Mouth Aspect Ratio)
      - Phone use   via YOLOv8n object detection

    Alerts:
      - On-screen red/orange banners
      - Audible beep (different pitch per alert type)
      - Live HUD with all metrics
    """

    def __init__(self):
        # ── MediaPipe FaceLandmarker ──────────────────────────
        ensure_model()
        base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
        options = FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1,
            min_face_detection_confidence=Config.DETECTION_CONFIDENCE,
            min_face_presence_confidence=Config.DETECTION_CONFIDENCE,
            min_tracking_confidence=Config.TRACKING_CONFIDENCE,
        )
        self._face_landmarker = mp_vision.FaceLandmarker.create_from_options(options)

        # ── Phone Detector (YOLO) ─────────────────────────────
        self._phone_detector = PhoneDetector()

        # ── Alert & Sound ─────────────────────────────────────
        self._alert_mgr = AlertManager()
        self._sound     = SoundAlert()

        # ── State Counters ────────────────────────────────────
        self.blink_counter = 0
        self.yawn_counter  = 0
        self.phone_counter = 0

        self.total_blinks  = 0
        self.total_yawns   = 0
        self.total_phones  = 0

        self._ear = 0.0
        self._mar = 0.0

    # ─── Per-Frame Processing ─────────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Analyse one video frame for all driver safety events.

        Returns:
            Annotated frame with all overlays.
        """
        h, w = frame.shape[:2]
        drowsy  = False
        yawning = False
        phone   = False

        # ═══════════════════════════════════════════════════════
        #  1. FACE LANDMARK DETECTION  (EAR + MAR)
        # ═══════════════════════════════════════════════════════
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        face_result = self._face_landmarker.detect(mp_image)

        if face_result.face_landmarks:
            face_lms = face_result.face_landmarks[0]

            left_eye  = extract_coords(face_lms, LEFT_EYE,  w, h)
            right_eye = extract_coords(face_lms, RIGHT_EYE, w, h)
            mouth_pts = extract_coords(face_lms, MOUTH,     w, h)

            # ── EAR ───────────────────────────────────────────
            self._ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0

            # ── MAR ───────────────────────────────────────────
            self._mar = mouth_aspect_ratio(mouth_pts)

            # ── Drowsiness logic ──────────────────────────────
            if self._ear < Config.EAR_THRESHOLD:
                self.blink_counter += 1
                if self.blink_counter >= Config.BLINK_CONSEC_FRAMES:
                    drowsy = True
                    self._sound.alert_drowsy()
            else:
                if self.blink_counter >= Config.BLINK_CONSEC_FRAMES:
                    self.total_blinks += 1
                self.blink_counter = 0

            # ── Yawn logic ────────────────────────────────────
            if self._mar > Config.MAR_THRESHOLD:
                self.yawn_counter += 1
                if self.yawn_counter >= Config.YAWN_CONSEC_FRAMES:
                    yawning = True
                    self._sound.alert_yawn()
            else:
                if self.yawn_counter >= Config.YAWN_CONSEC_FRAMES:
                    self.total_yawns += 1
                self.yawn_counter = 0

            # ── Draw eye contours ─────────────────────────────
            self._draw_contour(frame, left_eye,  drowsy)
            self._draw_contour(frame, right_eye, drowsy)

        # ═══════════════════════════════════════════════════════
        #  2. PHONE DETECTION  (YOLO)
        # ═══════════════════════════════════════════════════════
        if self._phone_detector.is_available:
            phone_detections = self._phone_detector.detect(frame)
            self._phone_detector.draw_detections(frame, phone_detections)

            if phone_detections:
                self.phone_counter += 1
                if self.phone_counter >= Config.PHONE_CONSEC_FRAMES:
                    phone = True
                    self._sound.alert_phone()
            else:
                if self.phone_counter >= Config.PHONE_CONSEC_FRAMES:
                    self.total_phones += 1
                self.phone_counter = 0

        # ═══════════════════════════════════════════════════════
        #  3. OVERLAYS
        # ═══════════════════════════════════════════════════════
        if drowsy:
            AlertManager.draw_drowsiness_alert(frame)
        if yawning:
            AlertManager.draw_yawn_alert(frame)
        if phone:
            AlertManager.draw_phone_alert(frame)

        AlertManager.draw_hud(
            frame,
            self._ear, self._mar,
            self.total_blinks, self.total_yawns, self.total_phones,
            self.blink_counter, self.yawn_counter, self.phone_counter,
        )
        AlertManager.draw_status(frame, drowsy, yawning, phone)

        return frame

    # ─── Helper ───────────────────────────────────────────────────────────────

    @staticmethod
    def _draw_contour(frame, eye_coords, alert):
        pts   = eye_coords.astype(np.int32).reshape((-1, 1, 2))
        color = Config.ALERT_COLOR if alert else Config.NORMAL_COLOR
        cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=1)

    # ─── Main Camera Loop ─────────────────────────────────────────────────────

    def run(self):
        """Start webcam feed and run detection. Press Q to quit."""
        cap = cv2.VideoCapture(Config.CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  Config.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.FRAME_HEIGHT)

        if not cap.isOpened():
            print("[ERROR] Cannot open camera. Check CAMERA_INDEX in config.py")
            return

        print("[INFO] Camera opened. All systems active.")
        print("[INFO] Monitoring: Drowsiness | Yawning | Phone Distraction")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to read frame from camera.")
                break

            frame = cv2.flip(frame, 1)          # mirror view
            frame = self.process_frame(frame)

            cv2.imshow("AI Driver Safety System — Press Q to Quit", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # ── Cleanup ──────────────────────────────────────────
        cap.release()
        cv2.destroyAllWindows()
        self._face_landmarker.close()

        print("\n╔══════════════════════════════╗")
        print("║       SESSION SUMMARY        ║")
        print("╠══════════════════════════════╣")
        print(f"║  Blinks detected  : {self.total_blinks:<9}║")
        print(f"║  Yawns detected   : {self.total_yawns:<9}║")
        print(f"║  Phone alerts     : {self.total_phones:<9}║")
        print("╚══════════════════════════════╝")