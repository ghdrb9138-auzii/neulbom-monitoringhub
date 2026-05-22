"""Eye Aspect Ratio (EAR) calculation from MediaPipe face mesh landmarks.

Reference: Soukupová & Čech, "Real-Time Eye Blink Detection using Facial Landmarks" (2016).

EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)

where p1..p6 are the 6 eye landmarks ordered as:
    p2  p3
  p1      p4
    p6  p5
"""

import numpy as np

from src.config import LEFT_EYE_IDX, RIGHT_EYE_IDX

__all__ = [
    "LEFT_EYE_IDX",
    "RIGHT_EYE_IDX",
    "compute_ear",
    "compute_both_ear",
    "eye_polygon",
]


def _to_pixel(landmark, image_w: int, image_h: int) -> np.ndarray:
    return np.array([landmark.x * image_w, landmark.y * image_h], dtype=np.float32)


def compute_ear(landmarks, eye_idx: list[int], image_w: int, image_h: int) -> float:
    pts = [_to_pixel(landmarks[i], image_w, image_h) for i in eye_idx]
    vertical_1 = np.linalg.norm(pts[1] - pts[5])
    vertical_2 = np.linalg.norm(pts[2] - pts[4])
    horizontal = np.linalg.norm(pts[0] - pts[3])
    if horizontal < 1e-6:
        return 0.0
    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def compute_both_ear(landmarks, image_w: int, image_h: int) -> tuple[float, float, float]:
    left = compute_ear(landmarks, LEFT_EYE_IDX, image_w, image_h)
    right = compute_ear(landmarks, RIGHT_EYE_IDX, image_w, image_h)
    return left, right, (left + right) / 2.0


def eye_polygon(landmarks, eye_idx: list[int], image_w: int, image_h: int) -> np.ndarray:
    return np.array(
        [[int(landmarks[i].x * image_w), int(landmarks[i].y * image_h)] for i in eye_idx],
        dtype=np.int32,
    )
