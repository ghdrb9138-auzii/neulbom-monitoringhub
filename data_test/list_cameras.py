"""Probe camera indices 0-4 and print which ones return frames.

Useful for finding the index of an external webcam (e.g., Logitech) vs the
built-in FaceTime camera.
"""

import cv2

MAX_INDEX = 5


def main() -> None:
    print("Probing camera indices 0 through {}...".format(MAX_INDEX - 1))
    available = []
    for i in range(MAX_INDEX):
        cap = cv2.VideoCapture(i)
        if not cap.isOpened():
            print(f"  index {i}: not opened")
            continue
        ok, frame = cap.read()
        if ok and frame is not None:
            h, w = frame.shape[:2]
            print(f"  index {i}: OK  resolution={w}x{h}")
            available.append((i, w, h))
        else:
            print(f"  index {i}: opened but no frame")
        cap.release()

    print()
    if not available:
        print("No cameras detected. Check camera permission for your terminal.")
        return
    print("Available cameras:")
    for i, w, h in available:
        print(f"  --camera {i}   ({w}x{h})")
    print()
    print("Run the detector with the camera you want, for example:")
    print("  .venv/bin/python drowsiness_detector.py --camera 1")


if __name__ == "__main__":
    main()
