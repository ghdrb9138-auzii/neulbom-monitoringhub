import landmarksConfig from "@shared/landmarks.json";

export const LEFT_EYE_IDX: readonly number[] = landmarksConfig.eye.left;
export const RIGHT_EYE_IDX: readonly number[] = landmarksConfig.eye.right;

export interface Point2D {
  x: number;
  y: number;
}

function dist(a: Point2D, b: Point2D, imageW: number, imageH: number): number {
  const dx = (a.x - b.x) * imageW;
  const dy = (a.y - b.y) * imageH;
  return Math.hypot(dx, dy);
}

export function computeEAR(
  landmarks: readonly Point2D[],
  eyeIdx: readonly number[],
  imageW: number,
  imageH: number,
): number {
  const p = eyeIdx.map((i) => landmarks[i]);
  const vertical1 = dist(p[1], p[5], imageW, imageH);
  const vertical2 = dist(p[2], p[4], imageW, imageH);
  const horizontal = dist(p[0], p[3], imageW, imageH);
  if (horizontal < 1e-6) return 0;
  return (vertical1 + vertical2) / (2 * horizontal);
}

export function computeBothEAR(
  landmarks: readonly Point2D[],
  imageW: number,
  imageH: number,
): { left: number; right: number; avg: number } {
  const left = computeEAR(landmarks, LEFT_EYE_IDX, imageW, imageH);
  const right = computeEAR(landmarks, RIGHT_EYE_IDX, imageW, imageH);
  return { left, right, avg: (left + right) / 2 };
}
