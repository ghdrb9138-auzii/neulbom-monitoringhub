import "./style.css";
import { registerSW } from "virtual:pwa-register";
import { env, pipeline, RawImage } from "@huggingface/transformers";

type SegState = "idle" | "loading" | "ready" | "error";
type RiskState = "unknown" | "safe" | "warn" | "danger";
type MaskKind = "road" | "sidewalk";

type SemanticMask = {
  width: number;
  height: number;
  data: Uint8Array;
};

type Zone = {
  xMin: number;
  xMax: number;
  yMin: number;
  yMax: number;
};

type RawMaskLike = {
  width?: number;
  height?: number;
  size?: [number, number];
  channels?: number;
  numChannels?: number;
  data?: ArrayLike<number>;
};

type RawSegmentLike = {
  label?: string;
  mask?: unknown;
};

const ROAD_SEG_LOCAL_MODEL_DIR = "/models/segformer-cityscapes";
const ROAD_SEG_CONFIG_URL = "/models/segformer-cityscapes/config.json";
const ROAD_SEG_PREPROCESSOR_URL = "/models/segformer-cityscapes/preprocessor_config.json";
const ROAD_SEG_ONNX_URL = "/models/segformer-cityscapes/onnx/model_quantized.onnx";
const ROAD_SEG_INPUT_W = 160;
const ROAD_SEG_INPUT_H = 90;
const ROAD_SEG_INTERVAL_MS = 900;
const ROAD_SEG_SLOW_INTERVAL_MS = 1500;

const NEAR_ZONE: Zone = { xMin: 0.35, xMax: 0.65, yMin: 0.75, yMax: 1.0 };
const LOOKAHEAD_ZONE: Zone = { xMin: 0.32, xMax: 0.68, yMin: 0.52, yMax: 0.78 };
const FAR_ZONE: Zone = { xMin: 0.30, xMax: 0.70, yMin: 0.35, yMax: 0.55 };

const WARN_TTS_INTERVAL_MS = 4000;
const DANGER_TTS_INTERVAL_MS = 1500;
const DANGER_HOLD_MS = 1500;
const WARN_HOLD_MS = 900;
const LOW_POWER_ENTER_FPS = 12;
const LOW_POWER_EXIT_FPS = 16;
const NORMAL_INFER_EVERY_N_FRAMES = 2;
const LOW_POWER_INFER_EVERY_N_FRAMES = 3;

const originalFetch = globalThis.fetch.bind(globalThis);
globalThis.fetch = async (input: RequestInfo | URL, init?: RequestInit) => {
  const url =
    typeof input === "string"
      ? input
      : input instanceof URL
        ? input.toString()
        : input.url;

  const response = await originalFetch(input, init);

  if (url.includes("/models/") || url.includes("segformer")) {
    const contentType = response.headers.get("content-type") ?? "";
    console.info("[model fetch]", response.status, contentType, url);

    if (contentType.includes("text/html")) {
      console.error("[model fetch] HTML returned for model request:", url);
    }
  }

  return response;
};

const app = document.getElementById("app");
if (!app) {
  throw new Error("#app element not found");
}

app.innerHTML = `
  <main class="shell">
    <section class="stage-shell">
      <div class="stage-header">
        <div>
          <div class="eyebrow">ENVIRONMENT PWA</div>
          <h1>인도 / 도로 감지 PWA</h1>
          <p class="lede">
            SegFormer로 인도와 도로를 구분하고, 전방 도로 접근 시 경고합니다.
          </p>
        </div>
        <div class="status-stack">
          <span class="pill pill-neutral" id="app-state">대기 중</span>
          <span class="pill pill-neutral" id="cam-state">카메라 미실행</span>
          <span class="pill pill-neutral" id="seg-state">지면 분석 대기</span>
          <span class="pill pill-neutral" id="risk-state">지면 인식 불안정</span>
          <span class="pill pill-neutral" id="power-state">normal</span>
          <span class="pill pill-neutral" id="tts-state">TTS 비활성</span>
        </div>
      </div>

      <div class="video-frame">
        <video id="video" autoplay muted playsinline webkit-playsinline></video>
        <canvas id="overlay"></canvas>
        <div class="overlay-badge" id="overlay-badge">지면 인식 불안정</div>
      </div>

      <div id="seg-error" class="error-banner hidden" role="status" aria-live="polite"></div>
    </section>

    <section class="controls">
      <button id="start" type="button">Start</button>
      <button id="stop" type="button" class="secondary">Stop</button>
      <button id="announce" type="button" class="secondary">TTS 테스트</button>
    </section>

    <section class="info-grid">
      <article class="card">
        <div class="card-label">카메라</div>
        <div class="card-value" id="camera-label">대기</div>
        <div class="card-note">권장 해상도 960x540, 최대 frameRate 30</div>
      </article>

      <article class="card">
        <div class="card-label">지면</div>
        <div class="card-value" id="seg-label">대기</div>
        <div class="card-note" id="seg-note">Camera + SegFormer road/sidewalk segmentation + Risk Warning + TTS</div>
      </article>

      <article class="card">
        <div class="card-label">Risk</div>
        <div class="card-value" id="risk-label">지면 인식 불안정</div>
        <div class="card-note" id="risk-note">지면 인식 불안정</div>
      </article>

      <article class="card">
        <div class="card-label">Performance</div>
        <div class="card-value" id="fps-label">FPS --</div>
        <div class="card-note" id="power-note">정상 모드</div>
      </article>

      <article class="card">
        <div class="card-label">TTS</div>
        <div class="card-value" id="tts-label">준비 필요</div>
        <div class="card-note">경고 상태별 cooldown 적용</div>
      </article>
    </section>
  </main>
`;

const video = document.getElementById("video") as HTMLVideoElement;
const overlay = document.getElementById("overlay") as HTMLCanvasElement;
const overlayBadge = document.getElementById("overlay-badge") as HTMLDivElement;
const startButton = document.getElementById("start") as HTMLButtonElement;
const stopButton = document.getElementById("stop") as HTMLButtonElement;
const announceButton = document.getElementById("announce") as HTMLButtonElement;

const appState = document.getElementById("app-state") as HTMLSpanElement;
const camState = document.getElementById("cam-state") as HTMLSpanElement;
const segStateEl = document.getElementById("seg-state") as HTMLSpanElement;
const riskStateEl = document.getElementById("risk-state") as HTMLSpanElement;
const powerStateEl = document.getElementById("power-state") as HTMLSpanElement;
const ttsState = document.getElementById("tts-state") as HTMLSpanElement;

const cameraLabel = document.getElementById("camera-label") as HTMLDivElement;
const segLabel = document.getElementById("seg-label") as HTMLDivElement;
const segNote = document.getElementById("seg-note") as HTMLDivElement;
const riskLabel = document.getElementById("risk-label") as HTMLDivElement;
const riskNote = document.getElementById("risk-note") as HTMLDivElement;
const fpsLabel = document.getElementById("fps-label") as HTMLDivElement;
const powerNote = document.getElementById("power-note") as HTMLDivElement;
const ttsLabel = document.getElementById("tts-label") as HTMLDivElement;
const segError = document.getElementById("seg-error") as HTMLDivElement;

const overlayCtx = overlay.getContext("2d");
const roadSegInputCanvas = document.createElement("canvas");
roadSegInputCanvas.width = ROAD_SEG_INPUT_W;
roadSegInputCanvas.height = ROAD_SEG_INPUT_H;
const roadSegInputCtx = roadSegInputCanvas.getContext("2d", { willReadFrequently: true });

async function clearDevServiceWorkersAndCaches(): Promise<void> {
  if (!import.meta.env.DEV) return;

  try {
    if ("serviceWorker" in navigator) {
      const registrations = await navigator.serviceWorker.getRegistrations();
      await Promise.all(registrations.map((registration) => registration.unregister()));
    }

    if ("caches" in window) {
      const keys = await caches.keys();
      await Promise.all(keys.map((key) => caches.delete(key)));
    }

    console.info("[dev] service workers and caches cleared");

    if (!sessionStorage.getItem("dev-sw-cleared")) {
      sessionStorage.setItem("dev-sw-cleared", "1");
      window.location.reload();
    }
  } catch (error) {
    console.warn("[dev] failed to clear service workers/caches", error);
  }
}

void clearDevServiceWorkersAndCaches();

env.localModelPath = "/models/";
env.allowLocalModels = true;
env.allowRemoteModels = false;
env.useBrowserCache = false;

if (import.meta.env.PROD) {
  registerSW({ immediate: true });
}

type RoadSegmenter = (image: RawImage) => Promise<unknown>;

let stream: MediaStream | null = null;
let running = false;
let rafId = 0;
let resizeObserver: ResizeObserver | null = null;
let frameCount = 0;
let lastFrameAt = 0;
let fps = 0;
let lowPowerMode = false;
let hasPrimedTts = false;

let roadSegState: SegState = "idle";
let roadSegErrorMessage = "";
let roadSegBusy = false;
let roadSegLoadPromise: Promise<void> | null = null;
let segmenter: RoadSegmenter | null = null;

let roadMask: SemanticMask | null = null;
let sidewalkMask: SemanticMask | null = null;
let roadMaskCanvas: HTMLCanvasElement | null = null;
let sidewalkMaskCanvas: HTMLCanvasElement | null = null;

let nearSidewalkCoverage = 0;
let nearRoadCoverage = 0;
let lookaheadSidewalkCoverage = 0;
let lookaheadRoadCoverage = 0;
let farRoadCoverage = 0;
let nearSidewalkCoverageKnown = false;
let nearRoadCoverageKnown = false;
let lookaheadSidewalkCoverageKnown = false;
let lookaheadRoadCoverageKnown = false;
let farRoadCoverageKnown = false;

let rawRiskState: RiskState = "unknown";
let currentRiskState: RiskState = "unknown";
let lastRawRiskState: RiskState = "unknown";
let rawRiskStreak = 0;
let riskHoldUntil = 0;
let lastWarnTtsAt = 0;
let lastDangerTtsAt = 0;
let lastSegAt = 0;

function setPillText(
  el: HTMLSpanElement,
  text: string,
  variant: "success" | "neutral" | "warn" | "bad",
): void {
  el.textContent = text;
  el.classList.remove("pill-success", "pill-neutral", "pill-warn", "pill-bad");
  el.classList.add(
    variant === "success"
      ? "pill-success"
      : variant === "warn"
        ? "pill-warn"
        : variant === "bad"
          ? "pill-bad"
          : "pill-neutral",
  );
}

function showSegError(message: string): void {
  segError.textContent = message;
  segError.classList.remove("hidden");
}

function hideSegError(): void {
  segError.textContent = "";
  segError.classList.add("hidden");
}

function getRiskBadgeText(state: RiskState): string {
  if (state === "safe") return "인도 보행 중";
  if (state === "warn") return "전방 도로 접근 주의";
  if (state === "danger") return "도로 진입 위험";
  return "지면 인식 불안정";
}

function getCoverageText(value: number | null): string {
  return value === null ? "unknown" : `${(value * 100).toFixed(1)}%`;
}

function updateSegUi(): void {
  if (roadSegState === "idle") {
    setPillText(segStateEl, "지면 분석 대기", "neutral");
    segLabel.textContent = "대기";
    segNote.textContent = "Camera + SegFormer road/sidewalk segmentation + Risk Warning + TTS";
    hideSegError();
    return;
  }

  if (roadSegState === "loading") {
    setPillText(segStateEl, "지면 분석 중", "warn");
    segLabel.textContent = "분석 중";
    segNote.textContent = "Local SegFormer 확인 중";
    hideSegError();
    return;
  }

  if (roadSegState === "ready") {
    setPillText(segStateEl, "지면 분석 완료", "success");
    segLabel.textContent = "Seg 준비 완료";
    segNote.textContent = "Local SegFormer 사용 중: /models/segformer-cityscapes";
    hideSegError();
    return;
  }

  setPillText(segStateEl, "지면 분석 오류", "bad");
  segLabel.textContent = "Seg 오류";
  segNote.textContent = "인도 / 도로 분석 실패";
  showSegError(roadSegErrorMessage || "지면 분석에 실패했습니다.");
}

function updateRiskUi(): void {
  const riskText = getRiskBadgeText(currentRiskState);
  riskStateEl.dataset.rawState = rawRiskState;
  riskLabel.textContent = riskText;
  riskNote.textContent = [
    `near sidewalk ${getCoverageText(nearSidewalkCoverageKnown ? nearSidewalkCoverage : null)}`,
    `near road ${getCoverageText(nearRoadCoverageKnown ? nearRoadCoverage : null)}`,
    `lookahead road ${getCoverageText(lookaheadRoadCoverageKnown ? lookaheadRoadCoverage : null)}`,
    `far road ${getCoverageText(farRoadCoverageKnown ? farRoadCoverage : null)}`,
  ].join(" · ");

  setPillText(
    riskStateEl,
    riskText,
    currentRiskState === "safe" || currentRiskState === "unknown"
      ? "neutral"
      : currentRiskState === "warn"
        ? "warn"
        : "bad",
  );
}

function updatePowerUi(): void {
  setPillText(powerStateEl, lowPowerMode ? "low power" : "normal", lowPowerMode ? "warn" : "success");
  powerNote.textContent = lowPowerMode ? "FPS 12 이하, 3프레임마다 추론" : "정상 모드";
}

function updateCommonUi(): void {
  setPillText(appState, running ? "실행 중" : "대기 중", running ? "success" : "neutral");
  setPillText(camState, stream ? "카메라 실행" : "카메라 미실행", stream ? "success" : "neutral");
  setPillText(ttsState, hasPrimedTts ? "TTS 활성" : "TTS 비활성", hasPrimedTts ? "success" : "neutral");

  cameraLabel.textContent = stream ? "실행 중" : "대기";
  fpsLabel.textContent = `FPS ${fps > 0 ? fps.toFixed(1) : "--"}`;
  overlayBadge.textContent = running ? getRiskBadgeText(currentRiskState) : "지면 인식 불안정";
}

function setStoppedState(): void {
  cameraLabel.textContent = "대기";
  frameCount = 0;
  lastFrameAt = 0;
  fps = 0;
  rawRiskState = "unknown";
  currentRiskState = "unknown";
  lastRawRiskState = "unknown";
  rawRiskStreak = 0;
  riskHoldUntil = 0;
  lastWarnTtsAt = 0;
  lastDangerTtsAt = 0;
  roadMask = null;
  sidewalkMask = null;
  roadMaskCanvas = null;
  sidewalkMaskCanvas = null;
  nearSidewalkCoverageKnown = false;
  nearRoadCoverageKnown = false;
  lookaheadSidewalkCoverageKnown = false;
  lookaheadRoadCoverageKnown = false;
  farRoadCoverageKnown = false;
  nearSidewalkCoverage = 0;
  nearRoadCoverage = 0;
  lookaheadSidewalkCoverage = 0;
  lookaheadRoadCoverage = 0;
  farRoadCoverage = 0;
  updateSegUi();
  updateRiskUi();
  updatePowerUi();
  updateCommonUi();
  ttsLabel.textContent = "준비 필요";
}

function syncCanvasSize(): void {
  const rect = video.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const width = Math.max(1, Math.round(rect.width * dpr));
  const height = Math.max(1, Math.round(rect.height * dpr));

  if (overlay.width !== width) overlay.width = width;
  if (overlay.height !== height) overlay.height = height;
}

function clearOverlay(): void {
  if (!overlayCtx) return;
  overlayCtx.setTransform(1, 0, 0, 1, 0, 0);
  overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
}

function fitContain(
  sourceWidth: number,
  sourceHeight: number,
  targetWidth: number,
  targetHeight: number,
): { x: number; y: number; width: number; height: number } {
  if (sourceWidth <= 0 || sourceHeight <= 0 || targetWidth <= 0 || targetHeight <= 0) {
    return { x: 0, y: 0, width: targetWidth, height: targetHeight };
  }

  const scale = Math.min(targetWidth / sourceWidth, targetHeight / sourceHeight);
  const width = sourceWidth * scale;
  const height = sourceHeight * scale;
  return {
    x: (targetWidth - width) / 2,
    y: (targetHeight - height) / 2,
    width,
    height,
  };
}

function drawSemanticMask(
  sourceCanvas: HTMLCanvasElement | null,
  sourceWidth: number,
  sourceHeight: number,
  kind: MaskKind,
): void {
  if (!overlayCtx || !sourceCanvas) return;
  const rect = fitContain(sourceWidth, sourceHeight, overlay.width, overlay.height);
  overlayCtx.save();
  overlayCtx.globalAlpha = kind === "road" ? 0.82 : 0.76;
  overlayCtx.drawImage(sourceCanvas, rect.x, rect.y, rect.width, rect.height);
  overlayCtx.restore();
}

function drawZoneOverlay(sourceWidth: number, sourceHeight: number): void {
  if (!overlayCtx) return;

  const rect = fitContain(sourceWidth, sourceHeight, overlay.width, overlay.height);
  const zones = [
    { zone: NEAR_ZONE, label: "NEAR", desc: "현재 위치", stroke: "rgba(255,255,255,0.95)", fill: "rgba(255,255,255,0.05)" },
    { zone: LOOKAHEAD_ZONE, label: "LOOKAHEAD", desc: "진행 방향", stroke: "rgba(255,193,7,0.95)", fill: "rgba(255,193,7,0.07)" },
    { zone: FAR_ZONE, label: "FAR", desc: "전방", stroke: "rgba(255,94,94,0.95)", fill: "rgba(255,94,94,0.07)" },
  ] as const;

  overlayCtx.save();
  overlayCtx.font = `600 ${Math.max(10, Math.round(rect.width * 0.03))}px Inter, system-ui, sans-serif`;
  overlayCtx.textBaseline = "top";

  for (const entry of zones) {
    const x = rect.x + rect.width * entry.zone.xMin;
    const y = rect.y + rect.height * entry.zone.yMin;
    const width = rect.width * (entry.zone.xMax - entry.zone.xMin);
    const height = rect.height * (entry.zone.yMax - entry.zone.yMin);
    const labelPadX = 6;
    const labelPadY = 4;
    const labelHeight = Math.max(20, Math.round(rect.height * 0.04));

    overlayCtx.fillStyle = entry.fill;
    overlayCtx.fillRect(x, y, width, height);
    overlayCtx.strokeStyle = entry.stroke;
    overlayCtx.lineWidth = Math.max(2, Math.round(rect.width * 0.004));
    overlayCtx.strokeRect(x, y, width, height);

    const labelText = `${entry.label} · ${entry.desc}`;
    const labelWidth = overlayCtx.measureText(labelText).width + labelPadX * 2;
    overlayCtx.fillStyle = "rgba(15, 23, 42, 0.75)";
    overlayCtx.fillRect(x + 6, y + 6, labelWidth, labelHeight);
    overlayCtx.strokeStyle = entry.stroke;
    overlayCtx.lineWidth = 1;
    overlayCtx.strokeRect(x + 6, y + 6, labelWidth, labelHeight);
    overlayCtx.fillStyle = "rgba(255,255,255,0.95)";
    overlayCtx.fillText(labelText, x + 6 + labelPadX, y + 6 + labelPadY);
  }

  overlayCtx.restore();
}

function maskToSemanticMask(mask: unknown): SemanticMask | null {
  const raw = mask as RawMaskLike | null | undefined;
  if (!raw) return null;

  const width = Number(raw.width ?? raw.size?.[0] ?? 0);
  const height = Number(raw.height ?? raw.size?.[1] ?? 0);
  const sourceData = raw.data;
  if (!width || !height || !sourceData) return null;

  const channels = Math.max(1, Number(raw.channels ?? raw.numChannels ?? 1));
  const data = new Uint8Array(width * height);

  for (let i = 0; i < data.length; i += 1) {
    const value = Number(sourceData[i * channels] ?? 0);
    data[i] = value > 0 ? 1 : 0;
  }

  return { width, height, data };
}

function buildSemanticMaskCanvas(mask: SemanticMask, kind: MaskKind): HTMLCanvasElement | null {
  const canvas = document.createElement("canvas");
  canvas.width = mask.width;
  canvas.height = mask.height;
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;

  const imageData = ctx.createImageData(mask.width, mask.height);
  const color = kind === "road" ? [255, 107, 53] : [46, 204, 113];
  const alpha = kind === "road" ? 92 : 108;

  for (let i = 0; i < mask.data.length; i += 1) {
    const base = i * 4;
    if (mask.data[i] === 1) {
      imageData.data[base] = color[0];
      imageData.data[base + 1] = color[1];
      imageData.data[base + 2] = color[2];
      imageData.data[base + 3] = alpha;
    } else {
      imageData.data[base + 3] = 0;
    }
  }

  ctx.putImageData(imageData, 0, 0);
  return canvas;
}

function updateSemanticMasks(segments: RawSegmentLike[]): void {
  const roadSegment = segments.find((segment) => String(segment?.label ?? "").toLowerCase() === "road");
  const sidewalkSegment = segments.find((segment) => String(segment?.label ?? "").toLowerCase() === "sidewalk");

  roadMask = maskToSemanticMask(roadSegment?.mask);
  sidewalkMask = maskToSemanticMask(sidewalkSegment?.mask);
  roadMaskCanvas = roadMask ? buildSemanticMaskCanvas(roadMask, "road") : null;
  sidewalkMaskCanvas = sidewalkMask ? buildSemanticMaskCanvas(sidewalkMask, "sidewalk") : null;
}

function calculateMaskCoverage(mask: SemanticMask | null, zone: Zone): number | null {
  if (!mask) return null;

  const xMin = Math.max(0, Math.floor(mask.width * zone.xMin));
  const xMax = Math.min(mask.width, Math.ceil(mask.width * zone.xMax));
  const yMin = Math.max(0, Math.floor(mask.height * zone.yMin));
  const yMax = Math.min(mask.height, Math.ceil(mask.height * zone.yMax));

  let total = 0;
  let active = 0;

  for (let y = yMin; y < yMax; y += 1) {
    for (let x = xMin; x < xMax; x += 1) {
      total += 1;
      if (mask.data[y * mask.width + x] === 1) {
        active += 1;
      }
    }
  }

  if (total === 0) return null;
  return active / total;
}

function updateCoverageMetrics(): void {
  const nearSidewalk = calculateMaskCoverage(sidewalkMask, NEAR_ZONE);
  const nearRoad = calculateMaskCoverage(roadMask, NEAR_ZONE);
  const lookaheadSidewalk = calculateMaskCoverage(sidewalkMask, LOOKAHEAD_ZONE);
  const lookaheadRoad = calculateMaskCoverage(roadMask, LOOKAHEAD_ZONE);
  const farRoad = calculateMaskCoverage(roadMask, FAR_ZONE);

  nearSidewalkCoverageKnown = nearSidewalk !== null;
  nearRoadCoverageKnown = nearRoad !== null;
  lookaheadSidewalkCoverageKnown = lookaheadSidewalk !== null;
  lookaheadRoadCoverageKnown = lookaheadRoad !== null;
  farRoadCoverageKnown = farRoad !== null;

  nearSidewalkCoverage = nearSidewalk ?? 0;
  nearRoadCoverage = nearRoad ?? 0;
  lookaheadSidewalkCoverage = lookaheadSidewalk ?? 0;
  lookaheadRoadCoverage = lookaheadRoad ?? 0;
  farRoadCoverage = farRoad ?? 0;
}

function deriveRawRiskState(): RiskState {
  if (!roadMask && !sidewalkMask) return "unknown";

  const knownCount = [
    nearSidewalkCoverageKnown,
    nearRoadCoverageKnown,
    lookaheadSidewalkCoverageKnown,
    lookaheadRoadCoverageKnown,
    farRoadCoverageKnown,
  ].filter(Boolean).length;

  if (knownCount < 3) return "unknown";

  const nearSidewalk = nearSidewalkCoverageKnown ? nearSidewalkCoverage : null;
  const nearRoad = nearRoadCoverageKnown ? nearRoadCoverage : null;
  const lookaheadSidewalk = lookaheadSidewalkCoverageKnown ? lookaheadSidewalkCoverage : null;
  const lookaheadRoad = lookaheadRoadCoverageKnown ? lookaheadRoadCoverage : null;
  const farRoad = farRoadCoverageKnown ? farRoadCoverage : null;

  if ((nearRoad !== null && nearRoad >= 0.35) || (lookaheadRoad !== null && lookaheadRoad >= 0.55)) {
    return "danger";
  }

  const onSidewalk =
    nearSidewalk !== null &&
    nearRoad !== null &&
    nearSidewalk >= 0.5 &&
    nearRoad < 0.35;

  if (!onSidewalk) return "unknown";

  if (
    (lookaheadRoad !== null && lookaheadRoad >= 0.35) ||
    (lookaheadSidewalk !== null && lookaheadSidewalk <= 0.4) ||
    (farRoad !== null && farRoad >= 0.45)
  ) {
    return "warn";
  }

  if (
    lookaheadRoad !== null &&
    lookaheadRoad < 0.25 &&
    lookaheadSidewalk !== null &&
    lookaheadSidewalk >= 0.45
  ) {
    return "safe";
  }

  return "unknown";
}

function syncRiskState(now: number): void {
  const nextRaw = deriveRawRiskState();
  rawRiskState = nextRaw;

  if (nextRaw === lastRawRiskState) {
    rawRiskStreak += 1;
  } else {
    lastRawRiskState = nextRaw;
    rawRiskStreak = 1;
  }

  if (rawRiskStreak < 3) return;

  if (currentRiskState === "danger" && nextRaw !== "danger" && now < riskHoldUntil) return;
  if (currentRiskState === "warn" && nextRaw !== "warn" && now < riskHoldUntil) return;

  if (currentRiskState !== nextRaw) {
    currentRiskState = nextRaw;
    if (nextRaw === "danger") {
      riskHoldUntil = now + DANGER_HOLD_MS;
    } else if (nextRaw === "warn") {
      riskHoldUntil = now + WARN_HOLD_MS;
    }
  }
}

function speak(text: string): void {
  if (!("speechSynthesis" in window)) return;

  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.lang = "ko-KR";
  utterance.rate = 1;
  utterance.pitch = 1;
  utterance.volume = 1;
  window.speechSynthesis.speak(utterance);
  ttsLabel.textContent = text;
  hasPrimedTts = true;
}

function primeTts(): void {
  if (!("speechSynthesis" in window)) {
    hasPrimedTts = true;
    return;
  }

  window.speechSynthesis.cancel();
  hasPrimedTts = true;
}

function maybeSpeakRisk(now: number): void {
  if (currentRiskState === "warn" && now - lastWarnTtsAt >= WARN_TTS_INTERVAL_MS) {
    speak("전방에 도로가 가까워지고 있습니다. 인도 안쪽으로 이동하세요.");
    lastWarnTtsAt = now;
  }

  if (currentRiskState === "danger" && now - lastDangerTtsAt >= DANGER_TTS_INTERVAL_MS) {
    speak("위험, 도로 진입이 감지되었습니다. 즉시 멈추세요.");
    lastDangerTtsAt = now;
  }
}

async function assertJsonFile(url: string): Promise<void> {
  const response = await fetch(url, {
    method: "GET",
    cache: "no-store",
  });

  if (!response.ok) {
    throw new Error(`${url} returned HTTP ${response.status}`);
  }

  const text = await response.text();
  const trimmed = text.trimStart();

  if (trimmed.startsWith("<!doctype") || trimmed.startsWith("<html")) {
    throw new Error(`${url} returned HTML instead of JSON`);
  }

  try {
    JSON.parse(text);
  } catch {
    throw new Error(`${url} is not valid JSON`);
  }
}

async function assertBinaryFile(url: string, minBytes: number): Promise<void> {
  const response = await fetch(url, {
    method: "GET",
    cache: "no-store",
  });

  if (!response.ok) {
    throw new Error(`${url} returned HTTP ${response.status}`);
  }

  const buffer = await response.arrayBuffer();
  if (buffer.byteLength < minBytes) {
    throw new Error(`${url} is too small: ${buffer.byteLength} bytes`);
  }

  const firstText = new TextDecoder().decode(buffer.slice(0, 80)).trimStart().toLowerCase();
  if (firstText.startsWith("<!doctype") || firstText.startsWith("<html")) {
    throw new Error(`${url} returned HTML instead of ONNX binary`);
  }
}

async function assertNoHtmlFallback(url: string): Promise<void> {
  const response = await fetch(url, {
    method: "GET",
    cache: "no-store",
  });

  const text = await response.text();
  const trimmed = text.trimStart().toLowerCase();

  if (trimmed.startsWith("<!doctype") || trimmed.startsWith("<html")) {
    throw new Error(`${url} returned HTML fallback. Vite or service worker is serving index.html for missing model file.`);
  }
}

async function validateLocalSegformerFiles(): Promise<void> {
  await assertJsonFile(ROAD_SEG_CONFIG_URL);
  await assertJsonFile(ROAD_SEG_PREPROCESSOR_URL);
  await assertBinaryFile(ROAD_SEG_ONNX_URL, 1_000_000);

  await assertNoHtmlFallback("/models/segformer-cityscapes/feature_extractor_config.json").catch((error) => {
    console.warn(error);
  });

  await assertNoHtmlFallback("/models/segformer-cityscapes/processor_config.json").catch((error) => {
    console.warn(error);
  });

  await assertNoHtmlFallback("/models/segformer-cityscapes/onnx/model.onnx").catch((error) => {
    console.warn(error);
  });
}

async function ensureRoadSegLoad(): Promise<void> {
  if (roadSegLoadPromise || roadSegState === "ready") {
    return roadSegLoadPromise ?? Promise.resolve();
  }

  roadSegState = "loading";
  roadSegErrorMessage = "";
  updateSegUi();

  roadSegLoadPromise = (async () => {
    try {
      await validateLocalSegformerFiles();

      env.localModelPath = "/models/";
      env.allowLocalModels = true;
      env.allowRemoteModels = false;
      env.useBrowserCache = false;

      segmenter = (await pipeline(
        "image-segmentation",
        ROAD_SEG_LOCAL_MODEL_DIR,
        {
          model_file_name: "model",
          local_files_only: true,
        },
      )) as RoadSegmenter;

      roadSegState = "ready";
      roadSegErrorMessage = "";
      segNote.textContent = "Local SegFormer 사용 중: /models/segformer-cityscapes";
      updateSegUi();
    } catch (error) {
      segmenter = null;
      roadSegState = "error";
      roadSegErrorMessage =
        error instanceof Error
          ? `SegFormer 로딩 실패: ${error.message}`
          : `SegFormer 로딩 실패: ${String(error)}`;
      console.error("SegFormer local load failed", error);
      updateSegUi();
    } finally {
      roadSegLoadPromise = null;
    }
  })();

  return roadSegLoadPromise;
}

async function runRoadSegmentation(): Promise<void> {
  if (!segmenter || roadSegBusy || !running) return;
  if (video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) return;

  const now = performance.now();
  if (now - lastSegAt < (lowPowerMode ? ROAD_SEG_SLOW_INTERVAL_MS : ROAD_SEG_INTERVAL_MS)) return;

  lastSegAt = now;
  roadSegBusy = true;

  try {
    if (!roadSegInputCtx) throw new Error("road segmentation canvas context not available");
    roadSegInputCtx.drawImage(video, 0, 0, ROAD_SEG_INPUT_W, ROAD_SEG_INPUT_H);
    const rawImage = RawImage.fromCanvas(roadSegInputCanvas);
    const result = await segmenter(rawImage);
    const segments = Array.isArray(result) ? (result as RawSegmentLike[]) : [result as RawSegmentLike];
    updateSemanticMasks(segments);
  } catch (error) {
    roadSegState = "error";
    roadSegErrorMessage =
      error instanceof Error ? `SegFormer 추론 실패: ${error.message}` : `SegFormer 추론 실패: ${String(error)}`;
    updateSegUi();
    console.error("road segmentation inference failed", error);
  } finally {
    roadSegBusy = false;
  }
}

function updateFpsAndPower(now: number): void {
  if (lastFrameAt > 0) {
    const instant = 1000 / Math.max(1, now - lastFrameAt);
    fps = fps === 0 ? instant : fps * 0.85 + instant * 0.15;
  }

  lastFrameAt = now;

  const shouldEnterLow = fps > 0 && fps <= LOW_POWER_ENTER_FPS;
  const shouldExitLow = fps >= LOW_POWER_EXIT_FPS;

  if (!lowPowerMode && shouldEnterLow) {
    lowPowerMode = true;
    updatePowerUi();
  } else if (lowPowerMode && shouldExitLow) {
    lowPowerMode = false;
    updatePowerUi();
  }
}

function renderOverlayFrame(sourceWidth: number, sourceHeight: number): void {
  if (!overlayCtx) return;
  clearOverlay();
  if (sourceWidth <= 0 || sourceHeight <= 0) return;

  if (roadMaskCanvas) {
    drawSemanticMask(roadMaskCanvas, roadMaskCanvas.width, roadMaskCanvas.height, "road");
  }

  if (sidewalkMaskCanvas) {
    drawSemanticMask(sidewalkMaskCanvas, sidewalkMaskCanvas.width, sidewalkMaskCanvas.height, "sidewalk");
  }

  drawZoneOverlay(sourceWidth, sourceHeight);
}

async function start(): Promise<void> {
  if (running) return;

  startButton.disabled = true;
  primeTts();

  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: { ideal: "environment" },
        width: { ideal: 960 },
        height: { ideal: 540 },
      },
      audio: false,
    });

    video.srcObject = stream;
    await video.play();

    running = true;
    frameCount = 0;
    lastFrameAt = 0;
    fps = 0;
    lowPowerMode = false;
    roadSegState = "idle";
    roadSegErrorMessage = "";
    roadSegBusy = false;
    segmenter = null;
    roadMask = null;
    sidewalkMask = null;
    roadMaskCanvas = null;
    sidewalkMaskCanvas = null;
    nearSidewalkCoverageKnown = false;
    nearRoadCoverageKnown = false;
    lookaheadSidewalkCoverageKnown = false;
    lookaheadRoadCoverageKnown = false;
    farRoadCoverageKnown = false;
    nearSidewalkCoverage = 0;
    nearRoadCoverage = 0;
    lookaheadSidewalkCoverage = 0;
    lookaheadRoadCoverage = 0;
    farRoadCoverage = 0;
    rawRiskState = "unknown";
    currentRiskState = "unknown";
    lastRawRiskState = "unknown";
    rawRiskStreak = 0;
    riskHoldUntil = 0;
    lastWarnTtsAt = 0;
    lastDangerTtsAt = 0;

    updateSegUi();
    updateRiskUi();
    updatePowerUi();
    updateCommonUi();

    syncCanvasSize();
    if (typeof ResizeObserver !== "undefined") {
      resizeObserver = new ResizeObserver(syncCanvasSize);
      resizeObserver.observe(video);
    }

    void ensureRoadSegLoad();
    rafId = window.requestAnimationFrame(renderLoop);
  } catch (error) {
    console.error("camera start failed", error);
    stream?.getTracks().forEach((track) => track.stop());
    stream = null;
    running = false;
    updateCommonUi();
  } finally {
    startButton.disabled = false;
  }
}

function stop(): void {
  running = false;
  startButton.disabled = false;
  cancelAnimationFrame(rafId);
  resizeObserver?.disconnect();
  resizeObserver = null;

  if ("speechSynthesis" in window) {
    window.speechSynthesis.cancel();
  }

  const currentStream = stream;
  stream = null;
  currentStream?.getTracks().forEach((track) => track.stop());
  video.srcObject = null;
  clearOverlay();
  lowPowerMode = false;
  roadSegBusy = false;
  roadMask = null;
  sidewalkMask = null;
  roadMaskCanvas = null;
  sidewalkMaskCanvas = null;
  setStoppedState();
}

function renderLoop(now: number): void {
  if (!running) return;

  updateFpsAndPower(now);

  frameCount += 1;
  if (frameCount % (lowPowerMode ? LOW_POWER_INFER_EVERY_N_FRAMES : NORMAL_INFER_EVERY_N_FRAMES) === 0) {
    void runRoadSegmentation();
  }

  if (roadSegState === "idle") {
    void ensureRoadSegLoad();
  }

  updateCoverageMetrics();
  syncRiskState(now);
  renderOverlayFrame(video.videoWidth || ROAD_SEG_INPUT_W, video.videoHeight || ROAD_SEG_INPUT_H);
  updateRiskUi();
  updateCommonUi();
  maybeSpeakRisk(now);

  rafId = window.requestAnimationFrame(renderLoop);
}

startButton.addEventListener("click", () => {
  void start();
});

stopButton.addEventListener("click", () => {
  stop();
});

announceButton.addEventListener("click", () => {
  primeTts();
  speak("TTS 테스트입니다.");
});

window.addEventListener("resize", syncCanvasSize);
window.addEventListener("beforeunload", stop);

updateSegUi();
updateRiskUi();
updatePowerUi();
updateCommonUi();
setStoppedState();
