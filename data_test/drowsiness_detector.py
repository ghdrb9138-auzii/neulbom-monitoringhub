"""Real-time drowsiness + head-pose detector using MediaPipe Face Mesh.

Detects two failure modes:
    1. Eyes closed for an extended period  (drowsiness)
    2. Head tilted out of upright posture   (positional-asphyxia risk in car seats / strollers)
       - forward  (pitch > threshold)         e.g. chin to chest
       - sideways (|roll|  > threshold)       e.g. head slumped to shoulder
       - diagonal (both axes exceed)          combined slump

Combined "DANGER" alarm fires when both eyes-closed and head-bent conditions hold simultaneously.

Usage:
    .venv/bin/python drowsiness_detector.py                # built-in camera
    .venv/bin/python drowsiness_detector.py --camera 1     # external camera

Keys:
    q   quit
    [   lower EAR threshold         ]  raise EAR threshold
    ,   lower pitch threshold       .  raise pitch threshold   (forward tilt)
    ;   lower roll threshold        '  raise roll threshold    (sideways tilt)
"""

import argparse
import time

import cv2
import mediapipe as mp

from src.config import (
    DROWSY_FRAMES,
    EAR_THRESHOLD,
    HEAD_BENT_FRAMES,
    HEAD_PITCH_THRESHOLD,
    HEAD_ROLL_THRESHOLD,
)
from src.ear import LEFT_EYE_IDX, RIGHT_EYE_IDX, compute_both_ear, eye_polygon
from src.head_pose import estimate_head_pose, project_axes
from src.overlay import (
    apply_danger_mask,
    apply_drowsy_mask,
    apply_head_bent_mask,
    draw_eyes,
    draw_head_axes,
    draw_status,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera index {args.camera}. "
            "Run list_cameras.py to see available cameras, "
            "and check camera permission for your terminal."
        )

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    ear_threshold = EAR_THRESHOLD
    pitch_threshold = HEAD_PITCH_THRESHOLD
    roll_threshold = HEAD_ROLL_THRESHOLD
    closed_frames = 0
    head_bent_frames = 0
    prev_time = time.time()
    fps = 0.0

    print(f"Starting detector on camera {args.camera}. Press 'q' to quit.")
    print(f"EAR threshold: {ear_threshold:.2f}, drowsy frames: {DROWSY_FRAMES}")
    print(
        f"Pitch threshold: {pitch_threshold:.1f} deg, "
        f"roll threshold: {roll_threshold:.1f} deg, "
        f"head-bent frames: {HEAD_BENT_FRAMES}"
    )

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(rgb)

            ear = 0.0
            pitch = yaw = roll = 0.0
            state = "NO FACE"
            is_closed = False
            is_head_bent = False
            bent_dir = ""

            if result.multi_face_landmarks:
                landmarks = result.multi_face_landmarks[0].landmark
                _, _, ear = compute_both_ear(landmarks, w, h)
                is_closed = ear < ear_threshold

                is_pitch_bent = False
                is_roll_bent = False
                pose = estimate_head_pose(landmarks, w, h)
                if pose is not None:
                    pitch, yaw, roll, rvec, tvec = pose
                    is_pitch_bent = pitch > pitch_threshold
                    is_roll_bent = abs(roll) > roll_threshold
                    is_head_bent = is_pitch_bent or is_roll_bent
                    axes_2d = project_axes(rvec, tvec, w, h)
                    draw_head_axes(frame, axes_2d)

                if is_pitch_bent and is_roll_bent:
                    bent_dir = "DIAG"
                elif is_pitch_bent:
                    bent_dir = "FWD"
                elif is_roll_bent:
                    bent_dir = "SIDE"

                closed_frames = closed_frames + 1 if is_closed else 0
                head_bent_frames = head_bent_frames + 1 if is_head_bent else 0

                is_drowsy = closed_frames >= DROWSY_FRAMES
                is_head_bent_sustained = head_bent_frames >= HEAD_BENT_FRAMES

                if is_drowsy and is_head_bent_sustained:
                    state = f"DANGER ({bent_dir})"
                elif is_head_bent_sustained:
                    state = f"HEAD BENT ({bent_dir})"
                elif is_drowsy:
                    state = "DROWSY"
                elif is_closed and is_head_bent:
                    state = f"SLEEPING? ({bent_dir})"
                elif is_closed:
                    state = "EYES CLOSED"
                elif is_head_bent:
                    state = f"HEAD {bent_dir}"
                else:
                    state = "AWAKE"

                left_poly = eye_polygon(landmarks, LEFT_EYE_IDX, w, h)
                right_poly = eye_polygon(landmarks, RIGHT_EYE_IDX, w, h)
                draw_eyes(frame, left_poly, right_poly, is_closed)
            else:
                closed_frames = 0
                head_bent_frames = 0

            now = time.time()
            dt = now - prev_time
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)
            prev_time = now

            if state.startswith("DANGER"):
                apply_danger_mask(frame)
            elif state.startswith("HEAD BENT"):
                apply_head_bent_mask(frame)
            elif state == "DROWSY":
                apply_drowsy_mask(frame)

            draw_status(frame, ear, state, fps, closed_frames, head_bent_frames, pitch, yaw, roll)
            cv2.imshow("Drowsiness Detector", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("["):
                ear_threshold = max(0.10, ear_threshold - 0.01)
                print(f"EAR threshold: {ear_threshold:.2f}")
            elif key == ord("]"):
                ear_threshold = min(0.40, ear_threshold + 0.01)
                print(f"EAR threshold: {ear_threshold:.2f}")
            elif key == ord(","):
                pitch_threshold = max(5.0, pitch_threshold - 2.0)
                print(f"Pitch threshold: {pitch_threshold:.1f}")
            elif key == ord("."):
                pitch_threshold = min(60.0, pitch_threshold + 2.0)
                print(f"Pitch threshold: {pitch_threshold:.1f}")
            elif key == ord(";"):
                roll_threshold = max(5.0, roll_threshold - 2.0)
                print(f"Roll threshold: {roll_threshold:.1f}")
            elif key == ord("'"):
                roll_threshold = min(60.0, roll_threshold + 2.0)
                print(f"Roll threshold: {roll_threshold:.1f}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        face_mesh.close()


if __name__ == "__main__":
    main()
