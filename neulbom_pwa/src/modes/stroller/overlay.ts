import type { HazardLevel, TrackHazard } from "./stateMachine";

const LEVEL_COLOR: Record<HazardLevel, string> = {
  safe: "#5ec8ff",
  warn: "#ffb43c",
  danger: "#ff5050",
};

const LEVEL_WIDTH: Record<HazardLevel, number> = {
  safe: 3,
  warn: 4,
  danger: 5,
};

interface MaskConfig {
  fillRgb: [number, number, number];
  baseAlpha: number;
  border: string;
  text: string;
  pulseHz: number;
}

const MASK: Partial<Record<HazardLevel, MaskConfig>> = {
  // WARN doesn't get a full-screen mask — bbox highlight + TTS is enough.
  // Reserving the heavy visual occlusion for DANGER avoids alarm fatigue and
  // keeps the camera view clear so the parent can react.
  danger: {
    fillRgb: [255, 0, 0],
    baseAlpha: 0.4,
    border: "#ff3030",
    text: "위험\n사람이 빠르게 접근합니다",
    pulseHz: 3.0,
  },
};

export function drawTracks(
  ctx: CanvasRenderingContext2D,
  trackHazards: readonly TrackHazard[],
): void {
  ctx.font = "14px -apple-system, BlinkMacSystemFont, sans-serif";
  ctx.textBaseline = "top";

  for (const th of trackHazards) {
    const color = LEVEL_COLOR[th.level];
    ctx.lineWidth = LEVEL_WIDTH[th.level];
    ctx.strokeStyle = color;
    const { x, y, width, height } = th.track.bbox;
    ctx.strokeRect(x, y, width, height);

    const growth = th.metrics.growthPctPerSec;
    const sign = growth >= 0 ? "+" : "";
    const label =
      `#${th.track.id} ${th.level.toUpperCase()}` +
      `  ${(th.metrics.areaRatio * 100).toFixed(1)}%` +
      `  ${sign}${growth.toFixed(0)}%/s`;

    const padX = 6;
    const padY = 3;
    const m = ctx.measureText(label);
    const labelW = m.width + padX * 2;
    const labelH = 20;
    const labelY = y - labelH;
    ctx.globalAlpha = 0.9;
    ctx.fillStyle = color;
    ctx.fillRect(x, labelY, labelW, labelH);
    ctx.globalAlpha = 1.0;
    ctx.fillStyle = "#001020";
    ctx.fillText(label, x + padX, labelY + padY);
  }
}

export function applyAlarmMask(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  level: HazardLevel,
  timeMs: number,
): void {
  const cfg = MASK[level];
  if (!cfg) return;

  // Pulse never fades to zero (0.55–1.0) so the alarm stays visible.
  const pulse =
    0.55 + 0.45 * (0.5 + 0.5 * Math.sin((timeMs / 1000) * cfg.pulseHz * Math.PI * 2));
  const [r, g, b] = cfg.fillRgb;

  ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${(cfg.baseAlpha * pulse).toFixed(3)})`;
  ctx.fillRect(0, 0, w, h);

  ctx.strokeStyle = cfg.border;
  ctx.lineWidth = 6 + 6 * pulse;
  ctx.strokeRect(4, 4, w - 8, h - 8);

  ctx.fillStyle = `rgba(255, 255, 255, ${(0.7 + 0.3 * pulse).toFixed(3)})`;
  const lines = cfg.text.split("\n");
  const maxWidth = w * 0.86;
  let fontSize = 48 + Math.round(12 * pulse);
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  for (;;) {
    ctx.font = `bold ${fontSize}px -apple-system, system-ui, sans-serif`;
    const widest = Math.max(...lines.map((l) => ctx.measureText(l).width));
    if (widest <= maxWidth || fontSize <= 18) break;
    fontSize -= 2;
  }
  const lineHeight = fontSize * 1.15;
  const startY = h / 2 - (lineHeight * (lines.length - 1)) / 2;
  lines.forEach((line, i) => {
    ctx.fillText(line, w / 2, startY + i * lineHeight);
  });
  ctx.textAlign = "start";
  ctx.textBaseline = "alphabetic";
}
