import { LEFT_EYE_IDX, RIGHT_EYE_IDX, type Point2D } from "./ear";
import type { AlarmLevel } from "./stateMachine";

const COLOR = {
  green: "#00ff00",
  red: "#ff3030",
  orange: "#ff8c00",
  white: "#ffffff",
  panel: "rgba(0, 0, 0, 0.65)",
} as const;

function drawEyePolygon(
  ctx: CanvasRenderingContext2D,
  landmarks: readonly Point2D[],
  eyeIdx: readonly number[],
  w: number,
  h: number,
  isClosed: boolean,
): void {
  ctx.strokeStyle = isClosed ? COLOR.red : COLOR.green;
  ctx.lineWidth = 2;
  ctx.beginPath();
  eyeIdx.forEach((i, k) => {
    const p = landmarks[i];
    const x = p.x * w;
    const y = p.y * h;
    if (k === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.closePath();
  ctx.stroke();
}

export function drawEyes(
  ctx: CanvasRenderingContext2D,
  landmarks: readonly Point2D[],
  w: number,
  h: number,
  isClosed: boolean,
): void {
  drawEyePolygon(ctx, landmarks, LEFT_EYE_IDX, w, h, isClosed);
  drawEyePolygon(ctx, landmarks, RIGHT_EYE_IDX, w, h, isClosed);
}

export interface StatusBar {
  ear: number;
  state: string;
  fps: number;
  closedFrames: number;
  headBentFrames: number;
  pitch: number;
  yaw: number;
  roll: number;
}

export function drawStatus(
  ctx: CanvasRenderingContext2D,
  w: number,
  s: StatusBar,
): void {
  ctx.fillStyle = COLOR.panel;
  ctx.fillRect(0, 0, w, 100);
  ctx.fillStyle = COLOR.white;
  ctx.font = "600 16px -apple-system, system-ui, sans-serif";
  ctx.fillText(`EAR: ${s.ear.toFixed(3)}`, 10, 25);
  ctx.fillText(`State: ${s.state}`, 200, 25);
  ctx.fillText(`FPS: ${s.fps.toFixed(1)}`, Math.max(500, w - 130), 25);

  ctx.font = "13px -apple-system, system-ui, sans-serif";
  ctx.fillText(`Pitch: ${s.pitch.toFixed(1)}`, 10, 55);
  ctx.fillText(`Yaw:   ${s.yaw.toFixed(1)}`, 10, 80);
  ctx.fillText(`Roll:  ${s.roll.toFixed(1)}`, 180, 80);
  ctx.fillText(`Closed: ${s.closedFrames}`, 180, 55);
  ctx.fillText(`HeadBent: ${s.headBentFrames}`, 340, 55);
}

interface MaskConfig {
  fillRgb: [number, number, number];
  baseAlpha: number;
  border: string;
  text: string;
  pulseHz: number;
}

const MASK: Record<Exclude<AlarmLevel, "none">, MaskConfig> = {
  drowsy: {
    fillRgb: [255, 48, 48],
    baseAlpha: 0.35,
    border: COLOR.red,
    text: "!! DROWSY !!",
    pulseHz: 1.2,
  },
  head_bent: {
    fillRgb: [255, 140, 0],
    baseAlpha: 0.30,
    border: COLOR.orange,
    text: "!! HEAD BENT !!",
    pulseHz: 1.8,
  },
  danger: {
    fillRgb: [255, 0, 0],
    baseAlpha: 0.45,
    border: COLOR.red,
    text: "!! DANGER !!",
    pulseHz: 3.0,
  },
};

export function applyAlarmMask(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  alarm: AlarmLevel,
  timeMs: number,
): void {
  if (alarm === "none") return;
  const cfg = MASK[alarm];

  // Pulse: 0..1, scaled so alarm never disappears entirely (range 0.55–1.0).
  const pulse = 0.55 + 0.45 * (0.5 + 0.5 * Math.sin((timeMs / 1000) * cfg.pulseHz * Math.PI * 2));
  const [r, g, b] = cfg.fillRgb;

  ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${(cfg.baseAlpha * pulse).toFixed(3)})`;
  ctx.fillRect(0, 0, w, h);

  ctx.strokeStyle = cfg.border;
  ctx.lineWidth = 6 + 6 * pulse;
  ctx.strokeRect(4, 4, w - 8, h - 8);

  ctx.fillStyle = `rgba(255, 255, 255, ${(0.7 + 0.3 * pulse).toFixed(3)})`;
  ctx.font = `bold ${48 + Math.round(12 * pulse)}px -apple-system, system-ui, sans-serif`;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(cfg.text, w / 2, h / 2);
  ctx.textAlign = "start";
  ctx.textBaseline = "alphabetic";
}
