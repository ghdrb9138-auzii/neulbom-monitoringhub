"""Single source of truth loader for landmark indices, 3D face model, and thresholds.

Reads JSON files from the repo-root ``shared/`` directory so that the Python PoC
(this package) and the Web PWA (``web_pwa/``) stay in sync. Edit the JSON files
to change values — do not hardcode them in the modules that import from here.
"""

import json
from pathlib import Path

import numpy as np

_SHARED_DIR = Path(__file__).resolve().parents[2] / "shared"


def _load(name: str) -> dict:
    with (_SHARED_DIR / name).open() as f:
        return json.load(f)


_LANDMARKS = _load("landmarks.json")
_FACE_MODEL = _load("face_model_3d.json")
_THRESHOLDS = _load("thresholds.json")

LEFT_EYE_IDX: list[int] = list(_LANDMARKS["eye"]["left"])
RIGHT_EYE_IDX: list[int] = list(_LANDMARKS["eye"]["right"])

HEAD_POSE_IDX: list[int] = list(_LANDMARKS["head_pose"]["indices"])
MODEL_POINTS_3D: np.ndarray = np.array(_FACE_MODEL["points"], dtype=np.float64)

EAR_THRESHOLD: float = float(_THRESHOLDS["ear"]["threshold"])
DROWSY_FRAMES: int = int(_THRESHOLDS["ear"]["drowsy_frames"])
HEAD_PITCH_THRESHOLD: float = float(_THRESHOLDS["head_pose"]["pitch_threshold_deg"])
HEAD_ROLL_THRESHOLD: float = float(_THRESHOLDS["head_pose"]["roll_threshold_deg"])
HEAD_BENT_FRAMES: int = int(_THRESHOLDS["head_pose"]["bent_frames"])
