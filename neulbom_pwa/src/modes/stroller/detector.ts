import {
  FilesetResolver,
  PoseLandmarker,
  type NormalizedLandmark,
  type PoseLandmarkerResult,
} from "@mediapipe/tasks-vision";

import hazardConfig from "@shared/hazard.json";

const WASM_BASE =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm";

const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task";

// MediaPipe Pose landmark indices used for the head+shoulders bbox.
// 0=nose, 1-6=eye inner/center/outer (L,R), 7-8=ears (L,R), 9-10=mouth corners, 11-12=shoulders.
const FACE_IDX = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
const LEFT_SHOULDER = 11;
const RIGHT_SHOULDER = 12;
const HEAD_SHOULDERS_IDX = [...FACE_IDX, LEFT_SHOULDER, RIGHT_SHOULDER];

export interface HazardDetection {
  x: number;
  y: number;
  width: number;
  height: number;
  score: number;
  category: string;
}

export async function createHazardDetector(): Promise<PoseLandmarker> {
  const fileset = await FilesetResolver.forVisionTasks(WASM_BASE);
  return PoseLandmarker.createFromOptions(fileset, {
    baseOptions: {
      modelAssetPath: MODEL_URL,
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numPoses: hazardConfig.pose.num_poses,
    minPoseDetectionConfidence: hazardConfig.pose.min_pose_detection_confidence,
    minPosePresenceConfidence: hazardConfig.pose.min_pose_presence_confidence,
    minTrackingConfidence: hazardConfig.pose.min_tracking_confidence,
  });
}

function isVisible(lm: NormalizedLandmark | undefined, minVis: number): boolean {
  if (!lm) return false;
  return (lm.visibility ?? 1.0) >= minVis;
}

export function toHazardDetections(
  result: PoseLandmarkerResult,
  frameW: number,
  frameH: number,
): HazardDetection[] {
  const minVis = hazardConfig.pose.min_landmark_visibility;
  const padPct = hazardConfig.pose.bbox_padding_pct;
  const out: HazardDetection[] = [];

  for (const landmarks of result.landmarks ?? []) {
    const hasShoulder =
      isVisible(landmarks[LEFT_SHOULDER], minVis) ||
      isVisible(landmarks[RIGHT_SHOULDER], minVis);
    const hasFace = FACE_IDX.some((i) => isVisible(landmarks[i], minVis));
    if (!hasShoulder || !hasFace) continue;

    let minX = Infinity;
    let minY = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    let minObservedVis = 1.0;

    for (const idx of HEAD_SHOULDERS_IDX) {
      const lm = landmarks[idx];
      if (!isVisible(lm, minVis)) continue;
      const px = lm.x * frameW;
      const py = lm.y * frameH;
      if (px < minX) minX = px;
      if (px > maxX) maxX = px;
      if (py < minY) minY = py;
      if (py > maxY) maxY = py;
      const v = lm.visibility ?? 1.0;
      if (v < minObservedVis) minObservedVis = v;
    }

    if (!isFinite(minX) || !isFinite(minY)) continue;
    const rawW = maxX - minX;
    const rawH = maxY - minY;
    if (rawW <= 0 || rawH <= 0) continue;

    const padX = rawW * padPct;
    const padY = rawH * padPct;
    const x = Math.max(0, minX - padX);
    const y = Math.max(0, minY - padY);
    const w = Math.min(frameW - x, rawW + 2 * padX);
    const h = Math.min(frameH - y, rawH + 2 * padY);

    out.push({
      x,
      y,
      width: w,
      height: h,
      score: minObservedVis,
      category: "person",
    });
  }
  return out;
}
