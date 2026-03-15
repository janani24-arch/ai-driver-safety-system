"""
landmarks.py — Facial Landmark Indices & Ratio Calculations
============================================================
Defines MediaPipe landmark indices for eyes and mouth,
and provides functions to compute EAR and MAR.
"""

import numpy as np


# ─── Landmark Index Groups (MediaPipe Face Mesh 468 points) ──────────────────

LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]
MOUTH     = [13,  14,  78,  308, 82,  312]


# ─── Euclidean Distance ───────────────────────────────────────────────────────

def _distance(p1: np.ndarray, p2: np.ndarray) -> float:
    return np.linalg.norm(p1 - p2)


# ─── Eye Aspect Ratio (EAR) ───────────────────────────────────────────────────

def eye_aspect_ratio(eye_landmarks: np.ndarray) -> float:
    """
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    Open eye  → ~0.25–0.30
    Closed eye → ~0.0
    """
    A = _distance(eye_landmarks[1], eye_landmarks[5])
    B = _distance(eye_landmarks[2], eye_landmarks[4])
    C = _distance(eye_landmarks[0], eye_landmarks[3])
    return (A + B) / (2.0 * C)


# ─── Mouth Aspect Ratio (MAR) ─────────────────────────────────────────────────

def mouth_aspect_ratio(mouth_landmarks: np.ndarray) -> float:
    """
    MAR = vertical_opening / horizontal_width
    Closed mouth → low value
    Yawning      → high value
    """
    vertical   = _distance(mouth_landmarks[0], mouth_landmarks[1])
    horizontal = _distance(mouth_landmarks[2], mouth_landmarks[3])
    return vertical / (horizontal + 1e-6)


# ─── Extract Landmark Coordinates ────────────────────────────────────────────

def extract_coords(face_landmarks, indices: list, frame_w: int, frame_h: int) -> np.ndarray:
    """
    Extract pixel (x, y) coords for given landmark indices.
    Works with both mediapipe Tasks API (plain list) and legacy NormalizedLandmarkList.
    """
    coords = []
    for idx in indices:
        # Tasks API returns a plain list of landmark objects
        lm = face_landmarks[idx]
        coords.append([int(lm.x * frame_w), int(lm.y * frame_h)])
    return np.array(coords, dtype=np.float32)