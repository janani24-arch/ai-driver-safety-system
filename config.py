"""
config.py — Configuration & Tunable Parameters
================================================
All system thresholds and settings are defined here.
Adjust these values to tune sensitivity.
"""


class Config:
    # ─── Camera ───────────────────────────────────────────────
    CAMERA_INDEX = 0
    FRAME_WIDTH  = 640
    FRAME_HEIGHT = 480

    # ─── Eye Aspect Ratio (EAR) — Drowsiness ──────────────────
    # EAR drops below threshold when eyes are closed
    EAR_THRESHOLD       = 0.22   # safe value — won't false-trigger on normal blinks
    BLINK_CONSEC_FRAMES = 25    # frames eyes must stay closed → drowsy alert

    # ─── Mouth Aspect Ratio (MAR) — Yawning ───────────────────
    MAR_THRESHOLD       = 0.55   # balanced — catches real yawns not talking
    YAWN_CONSEC_FRAMES  = 15    # frames mouth must stay open → yawn alert

    # ─── Phone Distraction Detection (YOLO) ───────────────────
    PHONE_CONSEC_FRAMES = 5     # fewer frames needed → faster phone alert
    PHONE_CONFIDENCE    = 0.30  # lowered — detects phone more easily

    # ─── MediaPipe Face Mesh ──────────────────────────────────
    DETECTION_CONFIDENCE = 0.7
    TRACKING_CONFIDENCE  = 0.7

    # ─── Display Colors (BGR format) ──────────────────────────
    ALERT_COLOR  = (0, 0, 255)      # Red
    NORMAL_COLOR = (0, 255, 0)      # Green
    INFO_COLOR   = (255, 255, 0)    # Cyan
    PHONE_COLOR  = (0, 165, 255)    # Orange
    TEXT_THICKNESS = 2
    FONT_SCALE     = 0.7

    # ─── Sound Alerts (Windows beep — no extra files needed) ──
    ENABLE_SOUND     = True
    DROWSY_BEEP_FREQ = 2500   # Hz — high pitch for drowsiness
    YAWN_BEEP_FREQ   = 1800    # Hz — medium pitch for yawn
    PHONE_BEEP_FREQ  = 1200    # Hz — low pitch for phone distraction
    BEEP_DURATION_MS = 1500    # milliseconds