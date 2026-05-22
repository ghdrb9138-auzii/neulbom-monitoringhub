"""Head pose estimation from MediaPipe Face Mesh landmarks via cv2.solvePnP.

Pose angles (degrees):
    pitch  nod up/down  (positive = head down, chin toward chest)
    yaw    look left/right
    roll   tilt left/right

The 3D model points below are a canonical adult face in millimeters. They are
accurate enough for relative pose estimation; absolute distance estimation
would require camera calibration.
"""

import math
from typing import Optional, Tuple

import cv2
import numpy as np

from src.config import HEAD_POSE_IDX, MODEL_POINTS_3D

LANDMARK_INDICES = HEAD_POSE_IDX


def _camera_matrix(image_w: int, image_h: int) -> np.ndarray:
    focal_length = float(image_w)
    return np.array(
        [
            [focal_length, 0.0, image_w / 2.0],
            [0.0, focal_length, image_h / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _rotation_to_euler(rotation_matrix: np.ndarray) -> Tuple[float, float, float]:
    """Decompose a rotation matrix into (pitch, yaw, roll) in degrees."""
    sy = math.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        pitch = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        yaw = math.atan2(-rotation_matrix[2, 0], sy)
        roll = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        pitch = math.atan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        yaw = math.atan2(-rotation_matrix[2, 0], sy)
        roll = 0.0
    return math.degrees(pitch), math.degrees(yaw), math.degrees(roll)


def estimate_head_pose(
    landmarks, image_w: int, image_h: int
) -> Optional[Tuple[float, float, float, np.ndarray, np.ndarray]]:
    """Return (pitch, yaw, roll, rotation_vector, translation_vector) in degrees.

    Returns None if pose could not be solved.
    """
    image_points = np.array(
        [[landmarks[i].x * image_w, landmarks[i].y * image_h] for i in LANDMARK_INDICES],
        dtype=np.float64,
    )
    camera_matrix = _camera_matrix(image_w, image_h)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    success, rvec, tvec = cv2.solvePnP(
        MODEL_POINTS_3D,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return None

    rotation_matrix, _ = cv2.Rodrigues(rvec)
    pitch, yaw, roll = _rotation_to_euler(rotation_matrix)

    # Normalize pitch so that "head forward / chin to chest" reads as positive.
    # solvePnP with this model gives pitch near 180/-180 for level head; remap to ~0.
    if pitch > 90:
        pitch = pitch - 180
    elif pitch < -90:
        pitch = pitch + 180

    return pitch, yaw, roll, rvec, tvec


def project_axes(
    rvec: np.ndarray, tvec: np.ndarray, image_w: int, image_h: int, length_mm: float = 60.0
) -> np.ndarray:
    """Project a 3D coordinate axis (X right, Y up, Z forward) to 2D."""
    axis_3d = np.float64(
        [[0, 0, 0], [length_mm, 0, 0], [0, length_mm, 0], [0, 0, length_mm]]
    )
    camera_matrix = _camera_matrix(image_w, image_h)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)
    projected, _ = cv2.projectPoints(axis_3d, rvec, tvec, camera_matrix, dist_coeffs)
    return projected.reshape(-1, 2).astype(int)
