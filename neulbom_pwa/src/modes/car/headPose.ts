/**
 * Decompose MediaPipe Tasks Vision facialTransformationMatrix into pitch/yaw/roll (deg).
 *
 * The matrix transforms canonical face-mesh coordinates into camera/image space.
 * Layout per the typings: 4x4, row-major, length-16 array (Matrix.data).
 *
 * Euler decomposition uses ZYX order (yaw=Y, pitch=X, roll=Z) to match the
 * Python solvePnP-based path in data_test/src/head_pose.py.
 */

const RAD2DEG = 180 / Math.PI;

export interface Pose {
  pitch: number;
  yaw: number;
  roll: number;
}

export interface AxesScreen {
  origin: { x: number; y: number };
  xEnd: { x: number; y: number };
  yEnd: { x: number; y: number };
  zEnd: { x: number; y: number };
}

/**
 * Project the 3 face-model axes (X right, Y up, Z forward) onto the screen
 * using only the rotation part of MediaPipe's facialTransformationMatrix.
 *
 * Anchored at the nose tip; rotated unit vectors are used as 2D screen
 * offsets. MediaPipe's transformation matrix maps canonical face mesh
 * coordinates into screen space (Y-down), so the rotation rows can be
 * used as screen-space directions directly without sign flipping.
 */
export function projectAxes(
  matrix: ArrayLike<number>,
  origin: { x: number; y: number },
  lengthPx: number,
): AxesScreen {
  // Row-major 3x3 rotation rows (upper-left of the 4x4 matrix).
  const r00 = matrix[0], r01 = matrix[1], r02 = matrix[2];
  const r10 = matrix[4], r11 = matrix[5], r12 = matrix[6];

  return {
    origin,
    xEnd: { x: origin.x + r00 * lengthPx, y: origin.y + r10 * lengthPx },
    yEnd: { x: origin.x + r01 * lengthPx, y: origin.y + r11 * lengthPx },
    zEnd: { x: origin.x + r02 * lengthPx, y: origin.y + r12 * lengthPx },
  };
}

/**
 * Extract rotation 3x3 from a 4x4 row-major matrix as a flat 9-array
 * [r00, r01, r02, r10, r11, r12, r20, r21, r22].
 */
function extractRotation(data: ArrayLike<number>): number[] {
  return [
    data[0], data[1], data[2],
    data[4], data[5], data[6],
    data[8], data[9], data[10],
  ];
}

export function decomposeEuler(matrix: ArrayLike<number>): Pose {
  const [r00, , r02, r10, r11, r12, r20, r21, r22] = extractRotation(matrix);

  const sy = Math.sqrt(r00 * r00 + r10 * r10);
  const singular = sy < 1e-6;

  let pitch: number;
  let yaw: number;
  let roll: number;

  if (!singular) {
    pitch = Math.atan2(r21, r22);
    yaw = Math.atan2(-r20, sy);
    roll = Math.atan2(r10, r00);
  } else {
    pitch = Math.atan2(-r12, r11);
    yaw = Math.atan2(-r20, sy);
    roll = 0;
    void r02;
  }

  pitch *= RAD2DEG;
  yaw *= RAD2DEG;
  roll *= RAD2DEG;

  if (pitch > 90) pitch -= 180;
  else if (pitch < -90) pitch += 180;

  // MediaPipe's facialTransformationMatrix maps face mesh -> screen space,
  // which inverts pitch relative to OpenCV solvePnP. Flip so that
  // "chin to chest" reads as positive (matches data_test/ Python convention).
  pitch = -pitch;

  return { pitch, yaw, roll };
}
