# AI-Based Driver Safety & Accident Prevention System

Real-time driver drowsiness and yawn detection using Python, OpenCV, and MediaPipe.

---

## Project Structure

```
driver_safety_system/
│
├── main.py           ← Entry point — run this file
├── config.py         ← All thresholds & settings (tune here)
├── detector.py       ← Core detection engine & webcam loop
├── landmarks.py      ← EAR / MAR maths & landmark indices
├── alert.py          ← Alert display & optional sound
└── requirements.txt  ← Python dependencies
```

---

## How It Works

1. **Webcam** captures live video of the driver.
2. **MediaPipe Face Mesh** detects 468 facial landmarks per frame.
3. **EAR (Eye Aspect Ratio)** is computed from eye landmarks.  
   → If EAR stays below threshold for N frames → **Drowsiness Alert**
4. **MAR (Mouth Aspect Ratio)** is computed from mouth landmarks.  
   → If MAR stays above threshold for N frames → **Yawn Alert**
5. Alerts are shown as **on-screen banners** (and optionally as sound).

---

## Setup & Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the system
python main.py

# Press Q to quit
```

---

## Tuning Sensitivity

Edit `config.py`:

| Parameter            | Default | Effect                                      |
|----------------------|---------|---------------------------------------------|
| `EAR_THRESHOLD`      | 0.25    | Lower → less sensitive to blinking          |
| `BLINK_CONSEC_FRAMES`| 20      | Higher → more frames needed before alert    |
| `MAR_THRESHOLD`      | 0.65    | Higher → less sensitive to yawning          |
| `YAWN_CONSEC_FRAMES` | 15      | Higher → more frames needed before alert    |
| `ENABLE_SOUND`       | False   | Set True + provide `alert.wav` for sound    |

---

## Technologies Used

| Tool          | Purpose                              |
|---------------|--------------------------------------|
| Python 3.8+   | Programming language                 |
| OpenCV        | Video capture & image processing     |
| MediaPipe     | Pre-trained facial landmark detection|
| NumPy         | EAR / MAR mathematical calculations  |
| pygame        | Alert sound playback (optional)      |

---

## Future Improvements

- Mobile phone detection using YOLO
- Head pose / distraction detection
- Real-time sound alarm system
- Data logging & session reports
- Embedded deployment in vehicle camera systems
