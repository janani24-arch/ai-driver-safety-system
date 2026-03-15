from detector import DriverSafetyDetector
from config import Config


def main():
    print("=" * 55)
    print("  AI-Based Driver Safety & Accident Prevention System")
    print("=" * 55)
    print(f"  EAR Threshold      : {Config.EAR_THRESHOLD}")
    print(f"  MAR Threshold      : {Config.MAR_THRESHOLD}")
    print(f"  Blink Alert Count  : {Config.BLINK_CONSEC_FRAMES} frames")
    print(f"  Yawn Alert Count   : {Config.YAWN_CONSEC_FRAMES} frames")
    print("=" * 55)
    print("  Press 'Q' to quit.")
    print("=" * 55)

    detector = DriverSafetyDetector()
    detector.run()


if __name__ == "__main__":
    main()
