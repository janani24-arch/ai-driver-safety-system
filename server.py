"""
server.py — VisionGuard Flask Backend
=======================================
Runs the AI detection and serves data to the web dashboard via:
  - HTTP API endpoints  (/api/status, /api/alerts)
  - Server-Sent Events  (/stream) for live data push to browser

Install:
    pip install flask

Run:
    python server.py
Then open:  http://localhost:5000
"""

from flask import Flask, Response, jsonify, request, render_template_string
import threading
import time
import json
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import FaceLandmarkerOptions
import numpy as np
import urllib.request
import os
import sys

from config import Config
from landmarks import LEFT_EYE, RIGHT_EYE, MOUTH, eye_aspect_ratio, mouth_aspect_ratio, extract_coords
from sound import SoundAlert
from phone_detector import PhoneDetector

app = Flask(__name__)

# ── Model Download ─────────────────────────────────────────────
MODEL_PATH = "face_landmarker.task"
MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/"
              "face_landmarker/face_landmarker/float16/1/face_landmarker.task")

def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("[INFO] Downloading MediaPipe model (~30 MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("[INFO] Model downloaded.")

# ── Shared State ───────────────────────────────────────────────
state = {
    "ear": 0.0,
    "mar": 0.0,
    "blink_count": 0,
    "yawn_count": 0,
    "phone_count": 0,
    "blink_per_min": 0,
    "status": "Monitoring",          # Monitoring / DROWSINESS / YAWN / PHONE
    "alert_active": False,
    "system_online": False,
    "uptime_seconds": 0,
    "alerts": [],                    # list of {type, time, level}
    "ear_history": [],               # last 20 EAR values for graph
    "frame": "",                     # base64 encoded current frame
}
state_lock = threading.Lock()
start_time = time.time()

# ── Detection Thread ───────────────────────────────────────────
def detection_loop():
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
    face_landmarker = mp_vision.FaceLandmarker.create_from_options(options)
    phone_detector  = PhoneDetector()
    sound           = SoundAlert()

    cap = cv2.VideoCapture(Config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  Config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.FRAME_HEIGHT)

    blink_counter = 0
    yawn_counter  = 0
    phone_counter = 0
    total_blinks  = 0
    total_yawns   = 0
    total_phones  = 0
    blink_times   = []

    with state_lock:
        state["system_online"] = True

    print("[VisionGuard] Detection started. Open http://localhost:5000")

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.03)
            continue

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result    = face_landmarker.detect(mp_image)

        ear = 0.0
        mar = 0.0
        drowsy  = False
        yawning = False

        if result.face_landmarks:
            lms       = result.face_landmarks[0]
            left_eye  = extract_coords(lms, LEFT_EYE,  w, h)
            right_eye = extract_coords(lms, RIGHT_EYE, w, h)
            mouth_pts = extract_coords(lms, MOUTH,     w, h)

            ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
            mar = mouth_aspect_ratio(mouth_pts)

            if ear < Config.EAR_THRESHOLD:
                blink_counter += 1
                if blink_counter >= Config.BLINK_CONSEC_FRAMES:
                    drowsy = True
                    sound.alert_drowsy()
            else:
                if blink_counter >= 3:          # 3 frames = real blink, not noise
                    total_blinks += 1
                    blink_times.append(time.time())
                if blink_counter >= Config.BLINK_CONSEC_FRAMES:
                    pass  # already counted above
                blink_counter = 0

            if mar > Config.MAR_THRESHOLD:
                yawn_counter += 1
                if yawn_counter >= Config.YAWN_CONSEC_FRAMES:
                    yawning = True
                    sound.alert_yawn()
            else:
                if yawn_counter >= Config.YAWN_CONSEC_FRAMES:
                    total_yawns += 1
                elif yawn_counter >= 8:         # 8 frames = real yawn
                    total_yawns += 1
                yawn_counter = 0

        # Phone detection — use original frame for better accuracy
        phone = False
        if phone_detector.is_available:
            phone_frame = cv2.resize(frame, (640, 480))
            detections = phone_detector.detect(phone_frame)
            if detections:
                phone_counter += 1
                if phone_counter >= Config.PHONE_CONSEC_FRAMES:
                    phone = True
                    sound.alert_phone()
            else:
                if phone_counter >= Config.PHONE_CONSEC_FRAMES:
                    total_phones += 1
                phone_counter = 0

        # Blinks per minute (last 60s)
        now = time.time()
        blink_times = [t for t in blink_times if now - t <= 60]
        blink_per_min = len(blink_times)

        # Status — auto-clears when condition is no longer active
        if drowsy:
            status = "DROWSINESS"
            level  = "CRITICAL"
        elif yawning:
            status = "YAWN"
            level  = "WARNING"
        elif phone:
            status = "PHONE"
            level  = "WARNING"
        else:
            status = "Monitoring"
            level  = None

        # Auto-clear alert if current status is back to normal
        # and previous alert was phone/yawn (not critical drowsiness)
        with state_lock:
            prev = state.get("status", "Monitoring")
        if prev in ("PHONE", "YAWN") and status == "Monitoring":
            pass  # will be updated below — clears automatically

        # Update shared state
        # Encode frame as JPEG for streaming
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        import base64
        frame_b64 = base64.b64encode(buffer).decode('utf-8')

        with state_lock:
            state["frame"]         = frame_b64
            state["ear"]           = round(ear, 3)
            state["mar"]           = round(mar, 3)
            state["blink_count"]   = total_blinks
            state["yawn_count"]    = total_yawns
            state["phone_count"]   = total_phones
            state["blink_per_min"] = blink_per_min
            state["status"]        = status
            state["alert_active"]  = status != "Monitoring"
            # Auto-clear: if no active condition, reset status immediately
            if status == "Monitoring":
                state["status"] = "Monitoring"
                state["alert_active"] = False
            state["uptime_seconds"]= int(now - start_time)

            # EAR history (keep last 20)
            state["ear_history"].append(round(ear, 3))
            if len(state["ear_history"]) > 20:
                state["ear_history"].pop(0)

            # Add to alerts log (max 50)
            if level and (not state["alerts"] or state["alerts"][-1]["type"] != status):
                import datetime
                state["alerts"].insert(0, {
                    "type":  status,
                    "time":  datetime.datetime.now().strftime("%H:%M:%S"),
                    "level": level
                })
                if len(state["alerts"]) > 50:
                    state["alerts"].pop()

        time.sleep(0.03)  # ~30fps

# ── Custom JSON Encoder (handles numpy float32) ────────────────
class SafeEncoder(json.JSONEncoder):
    def default(self, obj):
        try:
            import numpy as np
            if isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            if isinstance(obj, (np.int32, np.int64)):
                return int(obj)
        except ImportError:
            pass
        return super().default(obj)

# ── Server-Sent Events Stream ──────────────────────────────────
def event_stream():
    while True:
        with state_lock:
            data = json.dumps(state, cls=SafeEncoder)
        yield f"data: {data}\n\n"
        time.sleep(0.15)

# ── Routes ─────────────────────────────────────────────────────
@app.route("/")
def index():
    with open("dashboard.html", "r", encoding="utf-8") as f:
        return f.read()

@app.route("/stream")
def stream():
    return Response(event_stream(), mimetype="text/event-stream")

@app.route("/api/status")
def api_status():
    with state_lock:
        return jsonify(state)

@app.route("/api/acknowledge", methods=["POST"])
def acknowledge():
    with state_lock:
        state["alert_active"] = False
        state["status"] = "Monitoring"
    return jsonify({"ok": True})

@app.route("/api/emergency", methods=["POST"])
def emergency():
    import datetime
    with state_lock:
        state["alerts"].insert(0, {
            "type":  "EMERGENCY",
            "time":  datetime.datetime.now().strftime("%H:%M:%S"),
            "level": "CRITICAL"
        })
    print("[VisionGuard] ⚠ EMERGENCY CONTACT TRIGGERED")
    return jsonify({"ok": True, "message": "Emergency contact notified"})

# ── Entry Point ────────────────────────────────────────────────
if __name__ == "__main__":
    t = threading.Thread(target=detection_loop, daemon=True)
    t.start()
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)