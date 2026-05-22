"""Visualization helpers: draw eye landmarks, head pose axes, status, and alarms."""

import cv2
import numpy as np

GREEN = (0, 255, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
ORANGE = (0, 128, 255)
WHITE = (255, 255, 255)
BLUE = (255, 0, 0)


def draw_eyes(frame: np.ndarray, left_poly: np.ndarray, right_poly: np.ndarray, is_closed: bool) -> None:
    color = RED if is_closed else GREEN
    cv2.polylines(frame, [left_poly], isClosed=True, color=color, thickness=1)
    cv2.polylines(frame, [right_poly], isClosed=True, color=color, thickness=1)


def draw_head_axes(frame: np.ndarray, axes_2d: np.ndarray) -> None:
    """Draw the projected X/Y/Z axes anchored at the nose tip."""
    origin = tuple(axes_2d[0])
    cv2.line(frame, origin, tuple(axes_2d[1]), RED, 3)    # X (right)
    cv2.line(frame, origin, tuple(axes_2d[2]), GREEN, 3)  # Y (up)
    cv2.line(frame, origin, tuple(axes_2d[3]), BLUE, 3)   # Z (forward)


def draw_status(
    frame: np.ndarray,
    ear: float,
    state: str,
    fps: float,
    closed_frames: int,
    head_bent_frames: int,
    pitch: float,
    yaw: float,
    roll: float,
) -> None:
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 100), (0, 0, 0), thickness=-1)
    cv2.putText(frame, f"EAR: {ear:.3f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2)
    cv2.putText(frame, f"State: {state}", (200, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (480, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1)

    cv2.putText(frame, f"Pitch: {pitch:>6.1f}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
    cv2.putText(frame, f"Yaw:   {yaw:>6.1f}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
    cv2.putText(frame, f"Roll:  {roll:>6.1f}", (180, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)

    cv2.putText(
        frame, f"Closed: {closed_frames}", (180, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1
    )
    cv2.putText(
        frame, f"HeadBent: {head_bent_frames}", (340, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1
    )


def _apply_color_mask(frame: np.ndarray, color: tuple, intensity: float, text: str) -> None:
    layer = np.zeros_like(frame)
    layer[:] = color
    cv2.addWeighted(layer, intensity, frame, 1.0 - intensity, 0, dst=frame)
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, thickness=8)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.8, 4)
    cv2.putText(
        frame,
        text,
        ((w - tw) // 2, (h + th) // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.8,
        WHITE,
        4,
    )


def apply_drowsy_mask(frame: np.ndarray) -> None:
    _apply_color_mask(frame, RED, 0.35, "!! DROWSY !!")


def apply_head_bent_mask(frame: np.ndarray) -> None:
    _apply_color_mask(frame, ORANGE, 0.30, "!! HEAD BENT !!")


def apply_danger_mask(frame: np.ndarray) -> None:
    _apply_color_mask(frame, RED, 0.45, "!! DANGER !!")
