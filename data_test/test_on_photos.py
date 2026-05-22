"""Batch EAR test on a folder of face images.

Usage:
    .venv/bin/python test_on_photos.py path/to/folder
    .venv/bin/python test_on_photos.py path/to/folder --threshold 0.21

Supports JPEG/PNG natively; HEIC is converted via macOS `sips` if available.
Prints EAR for each image and predicted state (open / closed).
"""

import argparse
import subprocess
from pathlib import Path

import cv2
import mediapipe as mp

from src.config import EAR_THRESHOLD
from src.ear import compute_both_ear

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".heic"}


def load_image(path: Path):
    if path.suffix.lower() == ".heic":
        tmp = path.with_suffix(".tmp.jpg")
        try:
            subprocess.run(
                ["sips", "-s", "format", "jpeg", str(path), "--out", str(tmp)],
                check=True,
                capture_output=True,
            )
            img = cv2.imread(str(tmp))
        finally:
            tmp.unlink(missing_ok=True)
        return img
    return cv2.imread(str(path))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="Folder containing face images")
    parser.add_argument(
        "--threshold",
        type=float,
        default=EAR_THRESHOLD,
        help=f"EAR threshold (default: {EAR_THRESHOLD})",
    )
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.is_dir():
        raise SystemExit(f"Not a directory: {folder}")

    images = sorted(p for p in folder.iterdir() if p.suffix.lower() in SUPPORTED_EXTS)
    if not images:
        raise SystemExit(f"No supported images in {folder}")

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.3,
    )

    print(f"{'file':<30} {'L-EAR':>7} {'R-EAR':>7} {'avg':>7} {'pred'}")
    print("-" * 60)

    for path in images:
        img = load_image(path)
        if img is None:
            print(f"{path.name:<30} load-fail")
            continue
        h, w = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)
        if not result.multi_face_landmarks:
            print(f"{path.name:<30} no-face")
            continue
        landmarks = result.multi_face_landmarks[0].landmark
        l, r, avg = compute_both_ear(landmarks, w, h)
        pred = "closed" if avg < args.threshold else "open"
        print(f"{path.name:<30} {l:>7.3f} {r:>7.3f} {avg:>7.3f} {pred}")

    face_mesh.close()


if __name__ == "__main__":
    main()
