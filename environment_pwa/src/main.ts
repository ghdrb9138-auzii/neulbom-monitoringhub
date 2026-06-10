import "./style.css";
import { registerSW } from "virtual:pwa-register";
import type { WorkerMessage, WorkerRequest, WorkerSegResult, WorkerMask } from "./seg-types";

type SegState = "idle" | "loading" | "ready" | "error";
type RiskState = "unknown" | "safe" | "crosswalk" | "warn" | "danger";
type AlertProfile = {
  state: RiskState;
  visualClass: string;
  label: string;
  beepPattern: number[];
  beepFrequency: number;
  vibrationPattern: number[];
  ttsText: string | null;
  cooldownMs: number;
};
type MaskKind = "road" | "roadWarn" | "sidewalk" | "crosswalk" | "curb";

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

const SEG_INPUT_SIZE_NORMAL = 512;
const SEG_INPUT_SIZE_FAST = 320;
const USE_FAST_CAMERA_SEGMENTATION = true;
const SHOW_IMAGE_TEST_MASK_OVERLAY = true;
const SHOW_LIVE_MASK_OVERLAY = true;
const ROAD_SEG_INPUT_W = SEG_INPUT_SIZE_NORMAL;
const ROAD_SEG_INPUT_H = SEG_INPUT_SIZE_NORMAL;
const ROAD_SEG_INTERVAL_MS = 1300;
const ROAD_SEG_SLOW_INTERVAL_MS = 2000;
const NEAR_ZONE: Zone = {
  xMin: 0.25,
  xMax: 0.75,
  yMin: 0.72,
  yMax: 1.0,
};

const LOOKAHEAD_ZONE: Zone = {
  xMin: 0.22,
  xMax: 0.78,
  yMin: 0.48,
  yMax: 0.76,
};

const FAR_ZONE: Zone = {
  xMin: 0.25,
  xMax: 0.75,
  yMin: 0.28,
  yMax: 0.50,
};

const DANGER_HOLD_MS = 2500;
const WARN_HOLD_MS = 1800;
const CROSSWALK_HOLD_MS = 1800;
const LOW_POWER_ENTER_FPS = 12;
const LOW_POWER_EXIT_FPS = 16;
const PREVIEW_FPS = 15;
const PREVIEW_INTERVAL_MS = 1000 / PREVIEW_FPS;
const COVERAGE_EMA_ALPHA = 0.35;
const DEBUG_MODEL_FETCH = false;
const COMMON_UI_INTERVAL_MS = 500;
const DANGER_HEARTBEAT_MS = 3000;
const WARN_HEARTBEAT_MS = 7000;
const SPEECH_MIN_INTERVAL_MS = 2500;
const SEG_INPUT_SIZE_LOW_POWER = 320;
const WARN_CONFIRM_STREAK = 2;
const CROSSWALK_CONFIRM_STREAK = 2;
const DANGER_CONFIRM_STREAK = 2;
const HARD_NEAR_ROAD_DANGER = 0.65;
const HARD_LOOKAHEAD_ROAD_DANGER = 0.70;
const ON_SIDEWALK_MIN = 0.35;
const LOOKAHEAD_ROAD_DANGER = 0.50;
const LOOKAHEAD_CURB_WARN = 0.06;
const NEAR_CURB_WARN = 0.10;
const WARN_ENTER_LOOKAHEAD_ROAD = 0.30;
const WARN_EXIT_LOOKAHEAD_ROAD = 0.20;
const DANGER_ENTER_NEAR_ROAD = 0.48;
const DANGER_EXIT_NEAR_ROAD = 0.34;
const NEAR_ROAD_WARN_MIN = 0.26;
const LOOKAHEAD_ROAD_WARN_MIN = 0.18;
const CROSSWALK_ENTER = 0.22;
const CROSSWALK_EXIT = 0.14;
const GROUND_CONFIDENCE_MIN = 0.14;
const SIDEWALK_DROP_WARN = 0.25;
const RISK_VOTE_WINDOW = 4;

// If the local SegFormer model fails with 384x384 on a device, set SEG_INPUT_SIZE_FAST back to 512
// and revert public/models/segformer-sidewalk/preprocessor_config.json to 512x512.
function getActiveSegInputSize(): number {
  if (imageTestMode) {
    return SEG_INPUT_SIZE_NORMAL;
  }

  if (lowPowerMode) {
    return SEG_INPUT_SIZE_LOW_POWER;
  }

  return USE_FAST_CAMERA_SEGMENTATION ? SEG_INPUT_SIZE_FAST : SEG_INPUT_SIZE_NORMAL;
}

const ALERT_PROFILES: Record<RiskState, AlertProfile> = {
  unknown: {
    state: "unknown",
    visualClass: "pill-neutral",
    label: "지면 인식 불안정",
    beepPattern: [],
    beepFrequency: 0,
    vibrationPattern: [],
    ttsText: null,
    cooldownMs: 0,
  },
  safe: {
    state: "safe",
    visualClass: "pill-neutral",
    label: "안전 보행 중",
    beepPattern: [],
    beepFrequency: 0,
    vibrationPattern: [],
    ttsText: null,
    cooldownMs: 0,
  },
  crosswalk: {
    state: "crosswalk",
    visualClass: "pill-neutral",
    label: "횡단보도 보행 구간",
    beepPattern: [100],
    beepFrequency: 880,
    vibrationPattern: [60],
    ttsText: "횡단보도 구간입니다. 좌우를 확인하세요.",
    cooldownMs: 8000,
  },
  warn: {
    state: "warn",
    visualClass: "pill-warn",
    label: "앞으로 경계 주의",
    beepPattern: [120, 120, 120],
    beepFrequency: 660,
    vibrationPattern: [90, 80, 90],
    ttsText: "앞쪽에 경계가 있습니다. 주의하세요.",
    cooldownMs: 6000,
  },
  danger: {
    state: "danger",
    visualClass: "pill-bad",
    label: "앞으로 진입 위험",
    beepPattern: [160, 90, 160, 90, 160],
    beepFrequency: 1046,
    vibrationPattern: [180, 90, 180, 90, 180],
    ttsText: "위험, 앞으로 진입했습니다. 즉시 멈추세요.",
    cooldownMs: 2500,
  },
};

type AlertToastProfile = {
  title: string;
  message: string;
  className: string;
};

const ALERT_TOASTS: Record<RiskState, AlertToastProfile> = {
  unknown: {
    title: "지면 인식 불안정",
    message: "카메라 인식을 확인하세요.",
    className: "toast-unknown",
  },
  safe: {
    title: "안전",
    message: "안정적으로 보행 중입니다.",
    className: "toast-safe",
  },
  crosswalk: {
    title: "횡단보도",
    message: "횡단보도 구간입니다.",
    className: "toast-crosswalk",
  },
  warn: {
    title: "주의",
    message: "앞쪽 경계가 감지되었습니다.",
    className: "toast-warn",
  },
  danger: {
    title: "앞으로 진입 위험",
    message: "즉시 멈추세요.",
    className: "toast-danger",
  },
};

const originalFetch = globalThis.fetch.bind(globalThis);
globalThis.fetch = async (input: RequestInfo | URL, init?: RequestInit) => {
  const url =
    typeof input === "string"
      ? input
      : input instanceof URL
        ? input.toString()
        : input.url;

  const response = await originalFetch(input, init);

  if (DEBUG_MODEL_FETCH && (url.includes("/models/") || url.includes("segformer"))) {
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
            SegFormer로 인도와 도로를 분할하고, 위험과 보행 구간을 화면과 음성으로 안내합니다.
          </p>
        </div>
        <div class="status-stack">
          <span class="pill pill-neutral" id="app-state">준비 필요</span>
          <span class="pill pill-neutral" id="cam-state">카메라 미시작</span>
          <span class="pill pill-neutral" id="seg-state">Worker 준비 중</span>
          <span class="pill pill-neutral" id="risk-state">지면 인식 불안정</span>
          <span class="pill pill-neutral" id="power-state">normal</span>
          <span class="pill pill-neutral" id="tts-state">TTS 비활성</span>
        </div>
      </div>

      <div class="video-frame">
        <video id="video" class="hidden" autoplay muted playsinline webkit-playsinline></video>
        <canvas id="camera-preview-canvas" class="hidden"></canvas>
        <canvas id="image-test-canvas" class="hidden"></canvas>
        <canvas id="overlay"></canvas>
        <div class="overlay-badge" id="overlay-badge">지면 인식 불안정</div>
      </div>

      <div id="alert-toast" class="alert-toast hidden">
        <strong id="alert-toast-title">도로 진입 위험</strong>
        <span id="alert-toast-message">즉시 멈추세요.</span>
      </div>

      <div id="seg-error" class="error-banner hidden" role="status" aria-live="polite"></div>
    </section>

    <section class="controls">
      <input id="image-test-input" type="file" accept="image/*" />
      <button id="image-test-run" type="button" class="secondary">이미지 분석</button>
      <button id="camera-mode" type="button" class="secondary">카메라 모드</button>
      <button id="enable-alerts" type="button" class="secondary">알림 활성화</button>
      <button id="test-danger-alert" type="button" class="secondary">위험 알림 테스트</button>
      <button id="start" type="button">Start</button>
      <button id="stop" type="button" class="secondary">Stop</button>
      <button id="announce" type="button" class="secondary">TTS 테스트</button>
    </section>

    <section class="info-grid">
      <article class="card">
        <div class="card-label">카메라</div>
        <div class="card-value" id="camera-label">대기</div>
        <div class="card-note">권장 해상도 640x640, 목표 15fps</div>
      </article>

      <article class="card">
        <div class="card-label">세그멘테이션</div>
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
        <div class="perf-metrics">
          <div class="perf-metric"><span>Worker inference</span><strong id="worker-inference-label">--</strong></div>
          <div class="perf-metric"><span>Worker total</span><strong id="worker-total-label">--</strong></div>
        </div>
        <div class="card-note" id="power-note">대기 중</div>
        <div class="card-note" id="perf-note">fps=--</div>
      </article>

      <article class="card">
        <div class="card-label">TTS</div>
        <div class="card-value" id="tts-label">대기 필요</div>
        <div class="card-note">경고 상태별 cooldown 적용</div>
      </article>
    </section>
  </main>
`;

const video = document.getElementById("video") as HTMLVideoElement;
const imageTestInput = document.querySelector<HTMLInputElement>("#image-test-input");
const imageTestRun = document.querySelector<HTMLButtonElement>("#image-test-run");
const cameraModeButton = document.querySelector<HTMLButtonElement>("#camera-mode");
const imageTestCanvas = document.querySelector<HTMLCanvasElement>("#image-test-canvas");
const imageTestCtx = imageTestCanvas?.getContext("2d") ?? null;
const cameraPreviewCanvas = document.querySelector<HTMLCanvasElement>("#camera-preview-canvas");
const cameraPreviewCtx = cameraPreviewCanvas?.getContext("2d", { willReadFrequently: true }) ?? null;
const overlay = document.getElementById("overlay") as HTMLCanvasElement;
const overlayBadge = document.getElementById("overlay-badge") as HTMLDivElement;
const alertToast = document.getElementById("alert-toast") as HTMLDivElement;
const alertToastTitle = document.getElementById("alert-toast-title") as HTMLElement;
const alertToastMessage = document.getElementById("alert-toast-message") as HTMLSpanElement;
const stageShell = document.querySelector<HTMLElement>(".stage-shell");
const videoFrame = document.querySelector<HTMLElement>(".video-frame");
const startButton = document.getElementById("start") as HTMLButtonElement;
const stopButton = document.getElementById("stop") as HTMLButtonElement;
const announceButton = document.getElementById("announce") as HTMLButtonElement;
const enableAlertsButton = document.querySelector<HTMLButtonElement>("#enable-alerts");
const testDangerAlertButton = document.querySelector<HTMLButtonElement>("#test-danger-alert");

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
const workerInferenceLabel = document.getElementById("worker-inference-label") as HTMLDivElement;
const workerTotalLabel = document.getElementById("worker-total-label") as HTMLDivElement;
const powerNote = document.getElementById("power-note") as HTMLDivElement;
const perfNote = document.getElementById("perf-note") as HTMLDivElement;
const ttsLabel = document.getElementById("tts-label") as HTMLDivElement;
const segError = document.getElementById("seg-error") as HTMLDivElement;

const overlayCtx = overlay.getContext("2d");
if (
  !imageTestInput ||
  !imageTestRun ||
  !cameraModeButton ||
  !imageTestCanvas ||
  !imageTestCtx ||
  !cameraPreviewCanvas ||
  !cameraPreviewCtx
) {
  throw new Error("이미지 테스트 컨트롤을 찾을 수 없습니다.");
}

const imageTestInputEl: HTMLInputElement = imageTestInput;
const cameraModeButtonEl: HTMLButtonElement = cameraModeButton;
const imageTestCanvasEl: HTMLCanvasElement = imageTestCanvas;
const cameraPreviewCanvasEl: HTMLCanvasElement = cameraPreviewCanvas;

cameraPreviewCanvasEl.width = ROAD_SEG_INPUT_W;
cameraPreviewCanvasEl.height = ROAD_SEG_INPUT_H;
imageTestCanvasEl.width = ROAD_SEG_INPUT_W;
imageTestCanvasEl.height = ROAD_SEG_INPUT_H;

async function clearDevServiceWorkersAndCaches(): Promise<void> {
  if (!import.meta.env.DEV) return;

  try {
    if ("serviceWorker" in navigator) {
      const registrations =
        await navigator.serviceWorker.getRegistrations();

      await Promise.all(
        registrations.map((registration) =>
          registration.unregister(),
        ),
      );
    }

    if ("caches" in window) {
      const keys = await caches.keys();

      await Promise.all(
        keys.map((key) =>
          caches.delete(key),
        ),
      );
    }

    console.info(
      "[dev] service workers and caches cleared",
    );
  } catch (error) {
    console.warn("[dev] failed to clear service workers/caches", error);
  }
}

void clearDevServiceWorkersAndCaches();


if (import.meta.env.PROD) {
  registerSW({ immediate: true });
}

let stream: MediaStream | null = null;
let running = false;
let rafId = 0;
let resizeObserver: ResizeObserver | null = null;
let lastFrameAt = 0;
let fps = 0;
let lowPowerMode = false;
let hasPrimedTts = false;

let roadSegState: SegState = "idle";
let roadSegErrorMessage = "";
let roadSegBusy = false;
let roadSegInFlight = false;
let roadSegLoadPromise: Promise<void> | null = null;
let segWorker: Worker | null = null;
let segWorkerReady = false;
let segWorkerWarmupResolve: (() => void) | null = null;
let segWorkerWarmupReject: ((reason?: unknown) => void) | null = null;
let segWorkerWarmupPromise: Promise<void> | null = null;
let segRequestSeq = 0;
let latestSegRequestId = 0;
let segResultResolve: ((result: WorkerSegResult) => void) | null = null;
let segResultReject: ((reason?: unknown) => void) | null = null;
let segResultPromise: Promise<WorkerSegResult> | null = null;
let imageTestMode = false;
let imageTestBitmap: ImageBitmap | null = null;
let workerInferenceMs = 0;
let workerTotalMs = 0;

let roadMask: SemanticMask | null = null;
let roadWarnMask: SemanticMask | null = null;
let sidewalkMask: SemanticMask | null = null;
let crosswalkMask: SemanticMask | null = null;
let curbMask: SemanticMask | null = null;
let roadMaskCanvas: HTMLCanvasElement | null = null;
let roadWarnMaskCanvas: HTMLCanvasElement | null = null;
let sidewalkMaskCanvas: HTMLCanvasElement | null = null;
let crosswalkMaskCanvas: HTMLCanvasElement | null = null;
let curbMaskCanvas: HTMLCanvasElement | null = null;
let detectedLabels: string[] = [];

let nearSidewalkCoverage = 0;
let nearRoadCoverage = 0;
let nearRoadWarnCoverage = 0;
let nearCrosswalkCoverage = 0;
let lookaheadSidewalkCoverage = 0;
let lookaheadRoadCoverage = 0;
let lookaheadRoadWarnCoverage = 0;
let lookaheadCrosswalkCoverage = 0;
let nearCurbCoverage = 0;
let lookaheadCurbCoverage = 0;
let farRoadCoverage = 0;
let farRoadWarnCoverage = 0;
let nearSidewalkCoverageKnown = false;
let nearRoadCoverageKnown = false;
let nearRoadWarnCoverageKnown = false;
let nearCrosswalkCoverageKnown = false;
let lookaheadSidewalkCoverageKnown = false;
let lookaheadRoadCoverageKnown = false;
let lookaheadRoadWarnCoverageKnown = false;
let lookaheadCrosswalkCoverageKnown = false;
let nearCurbCoverageKnown = false;
let lookaheadCurbCoverageKnown = false;
let farRoadWarnCoverageKnown = false;

let rawRiskState: RiskState = "unknown";
let votedRiskState: RiskState = "unknown";
let currentRiskState: RiskState = "unknown";
let lastRawRiskState: RiskState = "unknown";
let rawRiskStreak = 0;
let riskVoteHistory: RiskState[] = [];
let riskHoldUntil = 0;
let segmentationRevision = 0;
let lastRiskProcessedRevision = -1;
let lastOverlayDrawRevision = -1;
let lastAlertProcessedState: RiskState = "unknown";
let audioContext: AudioContext | null = null;
let alertsEnabled = false;
let lastAlertAt = 0;
let lastHeartbeatAlertAt = 0;
let lastInferenceMs = 0;
let lastSegAt = 0;
let lastCameraSegAttemptAt = 0;
let lastPreviewDrawAt = 0;
let lastCommonUiAt = 0;
let lastSpokenText = "";
let lastSpeechAt = 0;

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
  if (state === "safe") return "안전 보행 중";
  if (state === "crosswalk") return "횡단보도 보행 구간";
  if (state === "warn") return "앞으로 경계 주의";
  if (state === "danger") return "앞으로 진입 위험";
  return "지면 인식 불안정";
}

function getCoverageText(value: number | null): string {
  return value === null ? "unknown" : `${(value * 100).toFixed(1)}%`;
}

function getRiskSeverity(state: RiskState): number {
  switch (state) {
    case "danger":
      return 3;
    case "warn":
      return 2;
    case "crosswalk":
      return 1;
    default:
      return 0;
  }
}


function pushRiskVote(nextRiskState: RiskState): RiskState {
  riskVoteHistory.push(nextRiskState);
  if (riskVoteHistory.length > RISK_VOTE_WINDOW) {
    riskVoteHistory.shift();
  }

  const voteCounts = new Map<RiskState, number>();
  for (const vote of riskVoteHistory) {
    voteCounts.set(vote, (voteCounts.get(vote) ?? 0) + 1);
  }

  let majorityState: RiskState = nextRiskState;
  let majorityCount = 0;

  for (const [state, count] of voteCounts) {
    if (count > majorityCount) {
      majorityState = state;
      majorityCount = count;
    }
  }

  const majorityThreshold = Math.floor(riskVoteHistory.length / 2) + 1;
  if (majorityCount >= majorityThreshold) {
    return majorityState;
  }

  return currentRiskState;
}

function smoothCoverage(prev: number, next: number): number {
  return prev * (1 - COVERAGE_EMA_ALPHA) + next * COVERAGE_EMA_ALPHA;
}
function updateSegUi(): void {
  if (roadSegState === "idle") {
    setPillText(segStateEl, "지면 인식 대기", "neutral");
    segLabel.textContent = "대기";
    segNote.textContent = "카메라 + SegFormer 도로/보도 분할 + 위험 경고 + TTS";
    hideSegError();
    return;
  }

  if (roadSegState === "loading") {
    setPillText(segStateEl, "지면 인식 로딩 중", "warn");
    segLabel.textContent = "분석 중";
    segNote.textContent = "로컬 SegFormer 확인 중";
    hideSegError();
    return;
  }

  if (roadSegState === "ready") {
    setPillText(segStateEl, "지면 인식 완료", "success");
    segLabel.textContent = "분석 준비 완료";
    segNote.textContent = "로컬 SegFormer 사용 중 /models/segformer-sidewalk";
    hideSegError();
    return;
  }

  setPillText(segStateEl, "지면 인식 오류", "bad");
  segLabel.textContent = "Seg 오류";
  segNote.textContent = "SegFormer 로딩 실패";
  showSegError(roadSegErrorMessage || "SegFormer 로딩 실패.");
}
function updateRiskUi(): void {
  const riskText = getRiskBadgeText(currentRiskState);
  const labelsText = detectedLabels.length > 0 ? detectedLabels.join(", ") : "none";
  const activeSegSize = getActiveSegInputSize();
  const activeSegInterval = imageTestMode
    ? "manual"
    : `${lowPowerMode ? ROAD_SEG_SLOW_INTERVAL_MS : ROAD_SEG_INTERVAL_MS}ms`;
  const nearSidewalkText = `${getCoverageText(nearSidewalkCoverageKnown ? nearSidewalkCoverage : null)}`;
  const nearRoadText = `${getCoverageText(nearRoadCoverageKnown ? nearRoadCoverage : null)}`;
  const nearRoadWarnText = `${getCoverageText(nearRoadWarnCoverageKnown ? nearRoadWarnCoverage : null)}`;
  const nearCrosswalkText = `${getCoverageText(nearCrosswalkCoverageKnown ? nearCrosswalkCoverage : null)}`;
  const lookaheadSidewalkText = `${getCoverageText(lookaheadSidewalkCoverageKnown ? lookaheadSidewalkCoverage : null)}`;
  const lookaheadRoadText = `${getCoverageText(lookaheadRoadCoverageKnown ? lookaheadRoadCoverage : null)}`;
  const lookaheadRoadWarnText = `${getCoverageText(
    lookaheadRoadWarnCoverageKnown ? lookaheadRoadWarnCoverage : null,
  )}`;
  const lookaheadCrosswalkText = `${getCoverageText(lookaheadCrosswalkCoverageKnown ? lookaheadCrosswalkCoverage : null)}`;
  const nearCurbText = `${getCoverageText(nearCurbCoverageKnown ? nearCurbCoverage : null)}`;
  const lookaheadCurbText = `${getCoverageText(lookaheadCurbCoverageKnown ? lookaheadCurbCoverage : null)}`;

  riskStateEl.dataset.rawState = rawRiskState;
  riskLabel.textContent = riskText;
  riskNote.textContent = [
    `mode=${imageTestMode ? "image" : "camera"}`,
    `input=${activeSegSize}x${activeSegSize}`,
    `seg=${activeSegInterval}`,
    `inference=${lastInferenceMs > 0 ? `${Math.round(lastInferenceMs)}ms` : "--"}`,
    `raw=${rawRiskState}`,
    `voted=${votedRiskState}`,
    `current=${currentRiskState}`,
    `near sidewalk ${nearSidewalkText}`,
    `near road ${nearRoadText}`,
    `near roadWarn ${nearRoadWarnText}`,
    `near crosswalk ${nearCrosswalkText}`,
    `lookahead sidewalk ${lookaheadSidewalkText}`,
    `lookahead road ${lookaheadRoadText}`,
    `lookahead roadWarn ${lookaheadRoadWarnText}`,
    `lookahead crosswalk ${lookaheadCrosswalkText}`,
    `near curb ${nearCurbText}`,
    `lookahead curb ${lookaheadCurbText}`,
    `labels=${labelsText}`,
    `summary=danger road: flat-road, flat-railtrack | warn road: flat-cyclinglane, flat-parkingdriveway | boundary: flat-curb, construction-stairs`,
  ].join(" ");

  setPillText(
    riskStateEl,
    riskText,
    currentRiskState === "safe" || currentRiskState === "crosswalk" || currentRiskState === "unknown"
      ? "neutral"
      : currentRiskState === "warn"
        ? "warn"
        : "bad",
  );

  applyVisualAlert(currentRiskState);
}


function updateCommonUiThrottled(now: number): void {
  if (now - lastCommonUiAt < COMMON_UI_INTERVAL_MS) return;

  lastCommonUiAt = now;
  updateCommonUi();
}

function applyVisualAlert(state: RiskState): void {
  const riskClass = `risk-${state}`;
  stageShell?.classList.remove("risk-unknown", "risk-safe", "risk-crosswalk", "risk-warn", "risk-danger");
  videoFrame?.classList.remove("risk-unknown", "risk-safe", "risk-crosswalk", "risk-warn", "risk-danger");
  stageShell?.classList.add(riskClass);
  videoFrame?.classList.add(riskClass);
  updateAlertToast(state);
}

function updateAlertToast(state: RiskState): void {
  const profile = ALERT_TOASTS[state];
  if (!profile) return;

  alertToastTitle.textContent = profile.title;
  alertToastMessage.textContent = profile.message;

  alertToast.classList.remove("toast-unknown", "toast-safe", "toast-crosswalk", "toast-warn", "toast-danger");
  alertToast.classList.add(profile.className);

  if (state === "safe") {
    alertToast.classList.add("hidden");
    return;
  }

  if (state === "unknown") {
    alertToast.classList.add("hidden");
    return;
  }

  alertToast.classList.remove("hidden");
}

function updatePowerUi(): void {
  setPillText(
    powerStateEl,
    lowPowerMode ? "절전" : "일반",
    lowPowerMode ? "warn" : "success",
  );
  powerNote.textContent = lowPowerMode ? "FPS 12 미만, 캡처 속도 낮춤" : "일반 모드";
}

function formatPerfMs(value: number): string {
  return value > 0 ? `${Math.round(value)}ms` : "--";
}

function updateCommonUi(): void {
  setPillText(appState, running ? "실행 중" : "준비 필요", running ? "success" : "neutral");
  setPillText(camState, stream ? "카메라 켜짐" : "카메라 꺼짐", stream ? "success" : "neutral");
  setPillText(ttsState, hasPrimedTts ? "TTS 활성" : "TTS 비활성", hasPrimedTts ? "success" : "neutral");

  cameraLabel.textContent = stream ? "실행 중" : "대기";
  fpsLabel.textContent = `FPS ${fps > 0 ? fps.toFixed(1) : "--"}`;
  workerInferenceLabel.textContent = formatPerfMs(workerInferenceMs);
  workerTotalLabel.textContent = formatPerfMs(workerTotalMs);
  perfNote.textContent = [
    `fps=${fps > 0 ? fps.toFixed(1) : "--"}`,
    `input=${getActiveSegInputSize()}x${getActiveSegInputSize()}`,
    `worker inference=${formatPerfMs(workerInferenceMs)}`,
    `worker total=${formatPerfMs(workerTotalMs)}`,
    `worker ready=${segWorkerReady}`,
    `interval=${imageTestMode ? "manual" : `${lowPowerMode ? ROAD_SEG_SLOW_INTERVAL_MS : ROAD_SEG_INTERVAL_MS}ms`}`,
    `inFlight=${roadSegInFlight}`,
  ].join(" ");
  overlayBadge.textContent = running ? getRiskBadgeText(currentRiskState) : "대기";
}
function setStoppedState(): void {
  cameraLabel.textContent = "대기";
  lastFrameAt = 0;
  fps = 0;
  rawRiskState = "unknown";
  votedRiskState = "unknown";
  currentRiskState = "unknown";
  lastRawRiskState = "unknown";
  rawRiskStreak = 0;
  riskHoldUntil = 0;
  lastInferenceMs = 0;
  roadMask = null;
  roadWarnMask = null;
  sidewalkMask = null;
  crosswalkMask = null;
  curbMask = null;
  roadMaskCanvas = null;
  roadWarnMaskCanvas = null;
  sidewalkMaskCanvas = null;
  crosswalkMaskCanvas = null;
  curbMaskCanvas = null;
  detectedLabels = [];
  nearSidewalkCoverageKnown = false;
  nearRoadCoverageKnown = false;
  nearRoadWarnCoverageKnown = false;
  nearCrosswalkCoverageKnown = false;
  lookaheadSidewalkCoverageKnown = false;
  lookaheadRoadCoverageKnown = false;
  lookaheadRoadWarnCoverageKnown = false;
  lookaheadCrosswalkCoverageKnown = false;
  nearCurbCoverageKnown = false;
  lookaheadCurbCoverageKnown = false;
  farRoadWarnCoverageKnown = false;
  nearSidewalkCoverage = 0;
  nearRoadCoverage = 0;
  nearRoadWarnCoverage = 0;
  lookaheadSidewalkCoverage = 0;
  lookaheadRoadCoverage = 0;
  lookaheadRoadWarnCoverage = 0;
  nearCurbCoverage = 0;
  lookaheadCurbCoverage = 0;
  farRoadCoverage = 0;
  farRoadWarnCoverage = 0;
  updateSegUi();
  updateRiskUi();
  updatePowerUi();
  updateCommonUi();
  ttsLabel.textContent = "대기 필요";
}
function syncCanvasSize(): void {
  const activeCanvas = imageTestMode ? imageTestCanvasEl : cameraPreviewCanvasEl;
  const width = activeCanvas.width || ROAD_SEG_INPUT_W;
  const height = activeCanvas.height || ROAD_SEG_INPUT_H;

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
    {
      zone: NEAR_ZONE,
      label: "NEAR",
      desc: "현재 위치",
      stroke: "rgba(255,255,255,0.95)",
      fill: "rgba(255,255,255,0.05)",
    },
    {
      zone: LOOKAHEAD_ZONE,
      label: "LOOKAHEAD",
      desc: "진행 방향",
      stroke: "rgba(255,193,7,0.95)",
      fill: "rgba(255,193,7,0.07)",
    },
    {
      zone: FAR_ZONE,
      label: "FAR",
      desc: "전방",
      stroke: "rgba(255,94,94,0.95)",
      fill: "rgba(255,94,94,0.07)",
    },
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

    const labelText = `${entry.label}  ${entry.desc}`;
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

function drawCoverBottom(
  ctx: CanvasRenderingContext2D,
  source: CanvasImageSource,
  sourceWidth: number,
  sourceHeight: number,
  targetWidth: number,
  targetHeight: number,
): void {
  const scale = Math.max(targetWidth / sourceWidth, targetHeight / sourceHeight);
  const drawWidth = sourceWidth * scale;
  const drawHeight = sourceHeight * scale;
  const dx = (targetWidth - drawWidth) / 2;
  const dy = targetHeight - drawHeight;

  ctx.clearRect(0, 0, targetWidth, targetHeight);
  ctx.drawImage(source, dx, dy, drawWidth, drawHeight);
}

function buildSemanticMaskCanvas(mask: SemanticMask, kind: MaskKind): HTMLCanvasElement | null {
  const canvas = document.createElement("canvas");
  canvas.width = mask.width;
  canvas.height = mask.height;
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;

  const imageData = ctx.createImageData(mask.width, mask.height);
  const color =
    kind === "road"
      ? [255, 64, 64]
      : kind === "roadWarn"
        ? [255, 180, 0]
        : kind === "sidewalk"
          ? [46, 204, 113]
          : kind === "crosswalk"
            ? [66, 165, 245]
            : [255, 215, 0];
  const alpha =
    kind === "road"
      ? 115
      : kind === "roadWarn"
        ? 100
        : kind === "sidewalk"
          ? 80
          : kind === "crosswalk"
            ? 105
            : 125;

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
  const nearRoadWarn = calculateMaskCoverage(roadWarnMask, NEAR_ZONE);
  const nearCrosswalk = calculateMaskCoverage(crosswalkMask, NEAR_ZONE);
  const nearCurb = calculateMaskCoverage(curbMask, NEAR_ZONE);
  const lookaheadSidewalk = calculateMaskCoverage(sidewalkMask, LOOKAHEAD_ZONE);
  const lookaheadRoad = calculateMaskCoverage(roadMask, LOOKAHEAD_ZONE);
  const lookaheadRoadWarn = calculateMaskCoverage(roadWarnMask, LOOKAHEAD_ZONE);
  const lookaheadCrosswalk = calculateMaskCoverage(crosswalkMask, LOOKAHEAD_ZONE);
  const lookaheadCurb = calculateMaskCoverage(curbMask, LOOKAHEAD_ZONE);
  const farRoad = calculateMaskCoverage(roadMask, FAR_ZONE);
  const farRoadWarn = calculateMaskCoverage(roadWarnMask, FAR_ZONE);

  nearSidewalkCoverageKnown = nearSidewalk !== null;
  nearRoadCoverageKnown = nearRoad !== null;
  nearRoadWarnCoverageKnown = nearRoadWarn !== null;
  nearCrosswalkCoverageKnown = nearCrosswalk !== null;
  nearCurbCoverageKnown = nearCurb !== null;
  lookaheadSidewalkCoverageKnown = lookaheadSidewalk !== null;
  lookaheadRoadCoverageKnown = lookaheadRoad !== null;
  lookaheadRoadWarnCoverageKnown = lookaheadRoadWarn !== null;
  lookaheadCrosswalkCoverageKnown = lookaheadCrosswalk !== null;
  lookaheadCurbCoverageKnown = lookaheadCurb !== null;
  farRoadWarnCoverageKnown = farRoadWarn !== null;

  if (imageTestMode) {
    nearSidewalkCoverage = nearSidewalk ?? 0;
    nearRoadCoverage = nearRoad ?? 0;
    nearRoadWarnCoverage = nearRoadWarn ?? 0;
    nearCrosswalkCoverage = nearCrosswalk ?? 0;
    nearCurbCoverage = nearCurb ?? 0;
    lookaheadSidewalkCoverage = lookaheadSidewalk ?? 0;
    lookaheadRoadCoverage = lookaheadRoad ?? 0;
    lookaheadRoadWarnCoverage = lookaheadRoadWarn ?? 0;
    lookaheadCrosswalkCoverage = lookaheadCrosswalk ?? 0;
    lookaheadCurbCoverage = lookaheadCurb ?? 0;
    farRoadCoverage = farRoad ?? 0;
    farRoadWarnCoverage = farRoadWarn ?? 0;
    return;
  }

  nearSidewalkCoverage = nearSidewalk === null ? 0 : smoothCoverage(nearSidewalkCoverage, nearSidewalk);
  nearRoadCoverage = nearRoad === null ? 0 : smoothCoverage(nearRoadCoverage, nearRoad);
  nearRoadWarnCoverage = nearRoadWarn === null ? 0 : smoothCoverage(nearRoadWarnCoverage, nearRoadWarn);
  nearCrosswalkCoverage = nearCrosswalk === null ? 0 : smoothCoverage(nearCrosswalkCoverage, nearCrosswalk);
  nearCurbCoverage = nearCurb === null ? 0 : smoothCoverage(nearCurbCoverage, nearCurb);
  lookaheadSidewalkCoverage =
    lookaheadSidewalk === null ? 0 : smoothCoverage(lookaheadSidewalkCoverage, lookaheadSidewalk);
  lookaheadRoadCoverage = lookaheadRoad === null ? 0 : smoothCoverage(lookaheadRoadCoverage, lookaheadRoad);
  lookaheadRoadWarnCoverage =
    lookaheadRoadWarn === null ? 0 : smoothCoverage(lookaheadRoadWarnCoverage, lookaheadRoadWarn);
  lookaheadCrosswalkCoverage =
    lookaheadCrosswalk === null ? 0 : smoothCoverage(lookaheadCrosswalkCoverage, lookaheadCrosswalk);
  lookaheadCurbCoverage = lookaheadCurb === null ? 0 : smoothCoverage(lookaheadCurbCoverage, lookaheadCurb);
  farRoadCoverage = farRoad === null ? 0 : smoothCoverage(farRoadCoverage, farRoad);
  farRoadWarnCoverage = farRoadWarn === null ? 0 : smoothCoverage(farRoadWarnCoverage, farRoadWarn);
}

function deriveRawRiskState(): RiskState {
  const nearSidewalk = nearSidewalkCoverageKnown ? nearSidewalkCoverage : null;
  const nearRoad = nearRoadCoverageKnown ? nearRoadCoverage : null;
  const nearRoadWarn = nearRoadWarnCoverageKnown ? nearRoadWarnCoverage : null;
  const lookaheadSidewalk = lookaheadSidewalkCoverageKnown ? lookaheadSidewalkCoverage : null;
  const lookaheadRoad = lookaheadRoadCoverageKnown ? lookaheadRoadCoverage : null;
  const lookaheadRoadWarn = lookaheadRoadWarnCoverageKnown ? lookaheadRoadWarnCoverage : null;
  const nearCrosswalk = nearCrosswalkCoverageKnown ? nearCrosswalkCoverage : null;
  const lookaheadCrosswalk = lookaheadCrosswalkCoverageKnown ? lookaheadCrosswalkCoverage : null;
  const nearCurb = nearCurbCoverageKnown ? nearCurbCoverage : null;
  const lookaheadCurb = lookaheadCurbCoverageKnown ? lookaheadCurbCoverage : null;

  if (
    nearSidewalk === null &&
    nearRoad === null &&
    lookaheadRoad === null &&
    nearCrosswalk === null &&
    lookaheadCrosswalk === null
  ) {
    return "unknown";
  }

  if (currentRiskState === "danger") {
    if (isHardDanger()) {
      return "danger";
    }

    if (
      (nearRoad !== null && nearRoad >= DANGER_EXIT_NEAR_ROAD) ||
      (lookaheadRoad !== null && lookaheadRoad >= LOOKAHEAD_ROAD_DANGER)
    ) {
      return "danger";
    }
  }

  if (currentRiskState === "crosswalk") {
    if (
      (nearCrosswalk !== null && nearCrosswalk >= CROSSWALK_EXIT) ||
      (lookaheadCrosswalk !== null && lookaheadCrosswalk >= CROSSWALK_EXIT)
    ) {
      return "crosswalk";
    }
  }

  if (currentRiskState === "warn") {
    if (
      (lookaheadRoad !== null && lookaheadRoad >= WARN_EXIT_LOOKAHEAD_ROAD) ||
      (nearCurb !== null && nearCurb >= NEAR_CURB_WARN) ||
      (lookaheadCurb !== null && lookaheadCurb >= LOOKAHEAD_CURB_WARN)
    ) {
      return "warn";
    }
  }

  if (
    nearRoad !== null &&
    nearRoad >= DANGER_ENTER_NEAR_ROAD &&
    (nearCrosswalk === null || nearCrosswalk < CROSSWALK_ENTER)
  ) {
    return "danger";
  }

  if (
    lookaheadRoad !== null &&
    lookaheadRoad >= LOOKAHEAD_ROAD_DANGER &&
    (lookaheadCrosswalk === null || lookaheadCrosswalk < CROSSWALK_ENTER)
  ) {
    return "danger";
  }

  if (nearCrosswalk !== null && nearCrosswalk >= CROSSWALK_ENTER) {
    return "crosswalk";
  }

  if (
    lookaheadCrosswalk !== null &&
    lookaheadCrosswalk >= CROSSWALK_ENTER &&
    (lookaheadRoad === null || lookaheadRoad < LOOKAHEAD_ROAD_DANGER)
  ) {
    return "crosswalk";
  }

  if (
    lookaheadRoad !== null &&
    lookaheadRoad >= WARN_ENTER_LOOKAHEAD_ROAD
  ) {
    return "warn";
  }

  if (
    nearRoadWarn !== null &&
    nearRoadWarn >= NEAR_ROAD_WARN_MIN
  ) {
    return "warn";
  }

  if (
    lookaheadRoadWarn !== null &&
    lookaheadRoadWarn >= LOOKAHEAD_ROAD_WARN_MIN
  ) {
    return "warn";
  }

  if (
    nearCurb !== null &&
    nearCurb >= NEAR_CURB_WARN
  ) {
    return "warn";
  }

  if (
    lookaheadCurb !== null &&
    lookaheadCurb >= LOOKAHEAD_CURB_WARN
  ) {
    return "warn";
  }

  const sidewalkDrop =
    nearSidewalk !== null && lookaheadSidewalk !== null
      ? nearSidewalk - lookaheadSidewalk
      : 0;
  const hasRoadOrCurbAhead =
    (lookaheadRoad !== null && lookaheadRoad >= WARN_ENTER_LOOKAHEAD_ROAD) ||
    (lookaheadCurb !== null && lookaheadCurb >= LOOKAHEAD_CURB_WARN);

  if (
    nearSidewalk !== null &&
    nearSidewalk >= ON_SIDEWALK_MIN &&
    sidewalkDrop >= SIDEWALK_DROP_WARN &&
    hasRoadOrCurbAhead
  ) {
    return "warn";
  }

  if (
    nearSidewalk !== null &&
    nearSidewalk >= ON_SIDEWALK_MIN &&
    nearRoad !== null &&
    nearRoad < DANGER_ENTER_NEAR_ROAD &&
    lookaheadRoad !== null &&
    lookaheadRoad < WARN_EXIT_LOOKAHEAD_ROAD &&
    (nearCrosswalk === null || nearCrosswalk < CROSSWALK_EXIT) &&
    (lookaheadCrosswalk === null || lookaheadCrosswalk < CROSSWALK_EXIT)
  ) {
    return "safe";
  }

  return currentRiskState;
}

function isHardDanger(): boolean {
  return (
    (nearRoadCoverageKnown && nearRoadCoverage >= HARD_NEAR_ROAD_DANGER) ||
    (lookaheadRoadCoverageKnown && lookaheadRoadCoverage >= HARD_LOOKAHEAD_ROAD_DANGER)
  );
}

function getGroundCoverageTotal(): number {
  const nearTotal =
    (nearSidewalkCoverageKnown ? nearSidewalkCoverage : 0) +
    (nearRoadCoverageKnown ? nearRoadCoverage : 0) +
    (nearRoadWarnCoverageKnown ? nearRoadWarnCoverage : 0) +
    (nearCrosswalkCoverageKnown ? nearCrosswalkCoverage : 0) +
    (nearCurbCoverageKnown ? nearCurbCoverage : 0);

  const lookaheadTotal =
    (lookaheadSidewalkCoverageKnown ? lookaheadSidewalkCoverage : 0) +
    (lookaheadRoadCoverageKnown ? lookaheadRoadCoverage : 0) +
    (lookaheadRoadWarnCoverageKnown ? lookaheadRoadWarnCoverage : 0) +
    (lookaheadCrosswalkCoverageKnown ? lookaheadCrosswalkCoverage : 0) +
    (lookaheadCurbCoverageKnown ? lookaheadCurbCoverage : 0);

  const farTotal = farRoadWarnCoverageKnown ? farRoadWarnCoverage : 0;

  return Math.max(nearTotal, lookaheadTotal, farTotal);
}

function syncRiskState(now: number): void {
  if (!imageTestMode && getGroundCoverageTotal() < GROUND_CONFIDENCE_MIN) {
    rawRiskState = "unknown";
    votedRiskState = currentRiskState;
    return;
  }

  const nextRaw = deriveRawRiskState();
  const votedRaw = imageTestMode ? nextRaw : pushRiskVote(nextRaw);
  rawRiskState = nextRaw;
  votedRiskState = votedRaw;

  if (imageTestMode) {
    currentRiskState = votedRaw;
    lastRawRiskState = votedRaw;
    rawRiskStreak = 1;
    riskHoldUntil = 0;
    return;
  }

  if (votedRaw === lastRawRiskState) {
    rawRiskStreak += 1;
  } else {
    lastRawRiskState = votedRaw;
    rawRiskStreak = 1;
  }

  if (votedRaw === "danger") {
    if (!isHardDanger() && rawRiskStreak < DANGER_CONFIRM_STREAK) {
      return;
    }

    currentRiskState = "danger";
    riskHoldUntil = now + DANGER_HOLD_MS;
    return;
  }

  if (votedRaw === "crosswalk" && rawRiskStreak < CROSSWALK_CONFIRM_STREAK) {
    return;
  }

  if (votedRaw === "crosswalk") {
    currentRiskState = "crosswalk";
    riskHoldUntil = now + CROSSWALK_HOLD_MS;
    return;
  }

  if (votedRaw === "warn" && rawRiskStreak < WARN_CONFIRM_STREAK) {
    return;
  }

  if (votedRaw === "safe") {
    if (currentRiskState === "safe") {
      riskHoldUntil = 0;
      return;
    }

    if (rawRiskStreak < 3) return;

    currentRiskState = "safe";
    riskHoldUntil = 0;
    return;
  }

  if (currentRiskState === "danger" && now < riskHoldUntil) return;
  if (rawRiskStreak < 2) return;

  if (currentRiskState !== votedRaw) {
    currentRiskState = votedRaw;
    if (votedRaw === "warn") {
      riskHoldUntil = now + WARN_HOLD_MS;
    } else {
      riskHoldUntil = 0;
    }
  }
}

function speak(text: string): void {
  if (!("speechSynthesis" in window)) return;
  const now = performance.now();

  if (text === lastSpokenText && now - lastSpeechAt < SPEECH_MIN_INTERVAL_MS) {
    return;
  }

  lastSpokenText = text;
  lastSpeechAt = now;
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

type WindowWithWebkitAudioContext = Window & {
  webkitAudioContext?: typeof AudioContext;
};

async function ensureAudioContext(): Promise<AudioContext | null> {
  if (audioContext && audioContext.state !== "closed") {
    if (audioContext.state === "suspended") {
      try {
        await audioContext.resume();
      } catch {
        return audioContext;
      }
    }

    return audioContext;
  }

  const AudioContextCtor =
    window.AudioContext ?? (window as WindowWithWebkitAudioContext).webkitAudioContext;
  if (!AudioContextCtor) return null;

  audioContext = new AudioContextCtor();

  if (audioContext.state === "suspended") {
    try {
      await audioContext.resume();
    } catch {
      return audioContext;
    }
  }

  return audioContext;
}

async function playBeepPattern(pattern: number[], frequency: number): Promise<void> {
  if (!pattern.length || frequency <= 0) return;

  const context = await ensureAudioContext();
  if (!context || context.state === "closed") return;

  let cursor = context.currentTime + 0.02;

  for (let i = 0; i < pattern.length; i += 1) {
    const durationMs = Math.max(0, pattern[i]);
    if (durationMs === 0) continue;

    if (i % 2 === 0) {
      const durationSec = durationMs / 1000;
      const attack = Math.min(0.01, durationSec / 4);
      const release = Math.min(0.02, durationSec / 4);
      const oscillator = context.createOscillator();
      const gainNode = context.createGain();

      oscillator.type = "sine";
      oscillator.frequency.setValueAtTime(frequency, cursor);
      gainNode.gain.setValueAtTime(0.0001, cursor);
      gainNode.gain.exponentialRampToValueAtTime(0.12, cursor + attack);
      gainNode.gain.setValueAtTime(
        0.12,
        Math.max(cursor + attack, cursor + durationSec - release),
      );
      gainNode.gain.exponentialRampToValueAtTime(0.0001, cursor + durationSec);

      oscillator.connect(gainNode);
      gainNode.connect(context.destination);
      oscillator.start(cursor);
      oscillator.stop(cursor + durationSec + 0.03);
    }

    cursor += durationMs / 1000;
  }
}

function vibratePattern(pattern: number[]): void {
  if (!pattern.length || !("vibrate" in navigator)) return;

  navigator.vibrate(pattern);
}

function triggerAlertNow(profile: AlertProfile, now: number): void {
  lastAlertProcessedState = profile.state;
  lastAlertAt = now;
  lastHeartbeatAlertAt = now;

  void playBeepPattern(profile.beepPattern, profile.beepFrequency);
  vibratePattern(profile.vibrationPattern);

  if (profile.ttsText) {
    speak(profile.ttsText);
  }
}

function maybeTriggerAlertHeartbeat(now: number): void {
  if (!alertsEnabled) return;

  const profile = ALERT_PROFILES[currentRiskState];
  const severity = getRiskSeverity(currentRiskState);
  if (severity === 0 || profile.cooldownMs <= 0) return;

  const previousSeverity = getRiskSeverity(lastAlertProcessedState);
  const severityRaised = severity > previousSeverity;
  const heartbeatInterval =
    currentRiskState === "danger"
      ? DANGER_HEARTBEAT_MS
      : currentRiskState === "warn"
        ? WARN_HEARTBEAT_MS
        : profile.cooldownMs;
  const heartbeatDue = now - lastHeartbeatAlertAt >= heartbeatInterval;
  const profileCooldownDue = now - lastAlertAt >= profile.cooldownMs;

  if (severityRaised || heartbeatDue || profileCooldownDue) {
    triggerAlertNow(profile, now);
  }
}

function maybeTriggerAlert(now: number): void {
  maybeTriggerAlertHeartbeat(now);
}

function handleSegWorkerMessage(event: MessageEvent<WorkerMessage>): void {
  const message = event.data;

  if (message.type === "ready") {
    segWorkerReady = true;
    roadSegState = "ready";
    roadSegErrorMessage = "";
    segWorkerWarmupResolve?.();
    segWorkerWarmupResolve = null;
    segWorkerWarmupReject = null;
    segWorkerWarmupPromise = null;
    updateSegUi();
    return;
  }

  if (message.type === "error") {
    if (typeof message.requestId === "number" && message.requestId !== latestSegRequestId) {
      return;
    }

    roadSegErrorMessage = message.message;

    if (typeof message.requestId === "number") {
      segResultReject?.(new Error(message.message));
      segResultResolve = null;
      segResultReject = null;
      segResultPromise = null;
      roadSegBusy = false;
      roadSegInFlight = false;
      roadSegState = segWorkerReady ? "ready" : "error";
      updateSegUi();
      updateCommonUi();
      return;
    } else {
      segWorkerReady = false;
      roadSegState = "error";
      segWorkerWarmupReject?.(new Error(message.message));
      segWorkerWarmupResolve = null;
      segWorkerWarmupReject = null;
      segWorkerWarmupPromise = null;
      segWorker?.terminate();
      segWorker = null;
    }

    updateSegUi();
    updateCommonUi();
    return;
  }

  if (message.type === "seg-result") {
    handleWorkerSegResult(message);
  }
}

function handleSegWorkerRuntimeError(event: ErrorEvent): void {
  segWorkerReady = false;
  roadSegState = "error";
  roadSegErrorMessage = event.message || "SegFormer worker error";
  segWorkerWarmupReject?.(new Error(roadSegErrorMessage));
  segResultReject?.(new Error(roadSegErrorMessage));
  segWorkerWarmupResolve = null;
  segWorkerWarmupReject = null;
  segWorkerWarmupPromise = null;
  segResultResolve = null;
  segResultReject = null;
  segResultPromise = null;
  roadSegBusy = false;
  roadSegInFlight = false;
  segWorker?.terminate();
  segWorker = null;
  updateSegUi();
}

function restoreWorkerMask(mask: WorkerMask | null): SemanticMask | null {
  if (!mask) return null;

  return {
    width: mask.width,
    height: mask.height,
    data: new Uint8Array(mask.data),
  };
}

function handleWorkerSegResult(result: WorkerSegResult): void {
  if (result.requestId !== latestSegRequestId) {
    return;
  }

  segWorkerReady = true;
  roadSegState = "ready";
  roadSegErrorMessage = "";
  lastInferenceMs = result.inferenceMs;
  workerInferenceMs = result.inferenceMs;
  workerTotalMs = result.totalMs;
  detectedLabels = result.labels;
  roadMask = restoreWorkerMask(result.roadMask);
  roadWarnMask = restoreWorkerMask(result.roadWarnMask);
  sidewalkMask = restoreWorkerMask(result.sidewalkMask);
  crosswalkMask = restoreWorkerMask(result.crosswalkMask);
  curbMask = restoreWorkerMask(result.curbMask);
  roadMaskCanvas = roadMask ? buildSemanticMaskCanvas(roadMask, "road") : null;
  roadWarnMaskCanvas = roadWarnMask ? buildSemanticMaskCanvas(roadWarnMask, "roadWarn") : null;
  sidewalkMaskCanvas = sidewalkMask ? buildSemanticMaskCanvas(sidewalkMask, "sidewalk") : null;
  crosswalkMaskCanvas = crosswalkMask ? buildSemanticMaskCanvas(crosswalkMask, "crosswalk") : null;
  curbMaskCanvas = curbMask ? buildSemanticMaskCanvas(curbMask, "curb") : null;
  segmentationRevision += 1;
  updateCoverageMetrics();
  syncRiskState(performance.now());
  renderOverlayFrame();
  updateRiskUi();
  updateCommonUi();
  lastRiskProcessedRevision = segmentationRevision;
  lastOverlayDrawRevision = segmentationRevision;
  roadSegBusy = false;
  roadSegInFlight = false;
  segResultResolve?.(result);
  segResultResolve = null;
  segResultReject = null;
  segResultPromise = null;
  updateSegUi();
}

function ensureSegWorker(): Promise<void> {
  if (segWorkerReady) {
    return Promise.resolve();
  }

  if (segWorkerWarmupPromise) {
    return segWorkerWarmupPromise;
  }

  if (!segWorker) {
    segWorker = new Worker(new URL("./seg-worker.ts", import.meta.url), { type: "module" });
    segWorker.addEventListener("message", handleSegWorkerMessage);
    segWorker.addEventListener("error", handleSegWorkerRuntimeError);
  }

  segWorkerWarmupPromise = new Promise<void>((resolve, reject) => {
    segWorkerWarmupResolve = resolve;
    segWorkerWarmupReject = reject;
  });

  const request: WorkerRequest = { type: "warmup" };
  segWorker.postMessage(request);

  return segWorkerWarmupPromise;
}

async function ensureRoadSegLoad(): Promise<void> {
  if (roadSegState === "ready") {
    return Promise.resolve();
  }

  if (roadSegLoadPromise) {
    return roadSegLoadPromise;
  }

  roadSegState = "loading";
  roadSegErrorMessage = "";
  updateSegUi();

  roadSegLoadPromise = (async () => {
    try {
      await ensureSegWorker();
      segWorkerReady = true;
      roadSegState = "ready";
      roadSegErrorMessage = "";
      updateSegUi();
    } catch (error) {
      segWorkerReady = false;
      roadSegState = "error";
      roadSegErrorMessage =
        error instanceof Error
          ? `SegFormer worker warmup failed: ${error.message}`
          : `SegFormer worker warmup failed: ${String(error)}`;
      updateSegUi();
    } finally {
      roadSegLoadPromise = null;
    }
  })();

  return roadSegLoadPromise;
}

function resetSegWorker(): void {
  segWorker?.terminate();
  segWorker = null;
  segWorkerReady = false;
  roadSegState = "idle";
  roadSegErrorMessage = "";
  roadSegLoadPromise = null;
  segWorkerWarmupResolve = null;
  segWorkerWarmupReject = null;
  segWorkerWarmupPromise = null;
  segResultResolve = null;
  segResultReject = null;
  segResultPromise = null;
  roadSegBusy = false;
  roadSegInFlight = false;
  workerInferenceMs = 0;
  workerTotalMs = 0;
}

async function runRoadSegmentation(): Promise<void> {
  if (imageTestMode) {
    if (!imageTestCanvasEl.width || !imageTestCanvasEl.height) return;
    await runRoadSegmentationFromSource(imageTestCanvasEl);
    return;
  }

  if (!cameraPreviewCanvasEl.width || !cameraPreviewCanvasEl.height) return;
  await runRoadSegmentationFromSource(cameraPreviewCanvasEl);
}

function maybeRunCameraSegmentation(now: number): void {
  if (imageTestMode) return;
  if (!running) return;
  if (roadSegState !== "ready") return;
  if (roadSegInFlight || roadSegBusy) return;
  if (!cameraPreviewCanvasEl.width || !cameraPreviewCanvasEl.height) return;

  const interval = lowPowerMode ? ROAD_SEG_SLOW_INTERVAL_MS : ROAD_SEG_INTERVAL_MS;

  if (now - lastCameraSegAttemptAt < interval) return;

  lastCameraSegAttemptAt = now;
  void runRoadSegmentation();
}

async function runRoadSegmentationFromSource(
  source: HTMLCanvasElement | OffscreenCanvas,
): Promise<void> {
  const now = performance.now();
  const activeSize = getActiveSegInputSize();

  if (!imageTestMode && now - lastSegAt < (lowPowerMode ? ROAD_SEG_SLOW_INTERVAL_MS : ROAD_SEG_INTERVAL_MS)) {
    return;
  }

  if (source.width !== activeSize || source.height !== activeSize) {
    return;
  }

  if (roadSegInFlight || roadSegBusy) return;
  if (!segWorker) {
    await ensureSegWorker();
  }
  if (!segWorker || roadSegState !== "ready") return;

  roadSegInFlight = true;
  roadSegBusy = true;

  try {
    lastSegAt = now;
    const imageBitmap = await createImageBitmap(source);
    const requestId = ++segRequestSeq;
    latestSegRequestId = requestId;

    segResultPromise = new Promise<WorkerSegResult>((resolve, reject) => {
      segResultResolve = resolve;
      segResultReject = reject;
    });

    const request: WorkerRequest = {
      type: "segment",
      requestId,
      imageBitmap,
    };

    segWorker.postMessage(request, [imageBitmap]);
    await segResultPromise;
  } catch (error) {
    if (error instanceof Error && error.message === "Segmentation stopped") {
      return;
    }
    roadSegState = "error";
    roadSegErrorMessage =
      error instanceof Error ? `SegFormer inference failed: ${error.message}` : `SegFormer inference failed: ${String(error)}`;
    segResultReject?.(error);
    segResultResolve = null;
    segResultReject = null;
    segResultPromise = null;
    updateSegUi();
    console.error("SegFormer inference failed", error);
  } finally {
    roadSegInFlight = false;
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

function renderOverlayFrame(): void {
  if (!overlayCtx) return;
  clearOverlay();
  const activeSize = getActiveSegInputSize();
  if (activeSize <= 0) return;

  const showMaskOverlay = imageTestMode ? SHOW_IMAGE_TEST_MASK_OVERLAY : SHOW_LIVE_MASK_OVERLAY;

  if (showMaskOverlay && sidewalkMaskCanvas) {
    drawSemanticMask(sidewalkMaskCanvas, sidewalkMaskCanvas.width, sidewalkMaskCanvas.height, "sidewalk");
  }

  if (showMaskOverlay && crosswalkMaskCanvas) {
    drawSemanticMask(crosswalkMaskCanvas, crosswalkMaskCanvas.width, crosswalkMaskCanvas.height, "crosswalk");
  }

  if (showMaskOverlay && roadWarnMaskCanvas) {
    drawSemanticMask(roadWarnMaskCanvas, roadWarnMaskCanvas.width, roadWarnMaskCanvas.height, "roadWarn");
  }

  if (showMaskOverlay && curbMaskCanvas) {
    drawSemanticMask(curbMaskCanvas, curbMaskCanvas.width, curbMaskCanvas.height, "curb");
  }

  if (showMaskOverlay && roadMaskCanvas) {
    drawSemanticMask(roadMaskCanvas, roadMaskCanvas.width, roadMaskCanvas.height, "road");
  }

  drawZoneOverlay(activeSize, activeSize);
}

function stopCameraOnlyButKeepUi(): void {
  if (stream) {
    stream.getTracks().forEach((track) => track.stop());
    stream = null;
  }

  video.pause();
  video.srcObject = null;
}

function drawCameraPreviewSource(): void {
  if (!cameraPreviewCtx) return;
  if (!video.videoWidth || !video.videoHeight) return;

  const activeSize = getActiveSegInputSize();

  if (cameraPreviewCanvasEl.width !== activeSize) {
    cameraPreviewCanvasEl.width = activeSize;
  }
  if (cameraPreviewCanvasEl.height !== activeSize) {
    cameraPreviewCanvasEl.height = activeSize;
  }

  drawCoverBottom(
    cameraPreviewCtx,
    video,
    video.videoWidth,
    video.videoHeight,
    activeSize,
    activeSize,
  );
}

function drawImageTestSource(): void {
  if (!imageTestBitmap || !imageTestCtx) return;

  if (imageTestCanvasEl.width !== ROAD_SEG_INPUT_W) {
    imageTestCanvasEl.width = ROAD_SEG_INPUT_W;
  }
  if (imageTestCanvasEl.height !== ROAD_SEG_INPUT_H) {
    imageTestCanvasEl.height = ROAD_SEG_INPUT_H;
  }

  drawCoverBottom(
    imageTestCtx,
    imageTestBitmap,
    imageTestBitmap.width,
    imageTestBitmap.height,
    SEG_INPUT_SIZE_NORMAL,
    SEG_INPUT_SIZE_NORMAL,
  );

  overlay.width = SEG_INPUT_SIZE_NORMAL;
  overlay.height = SEG_INPUT_SIZE_NORMAL;

  video.classList.add("hidden");
  cameraPreviewCanvasEl.classList.add("hidden");
  imageTestCanvasEl.classList.remove("hidden");
}

async function loadImageTestFile(file: File): Promise<void> {
  imageTestBitmap?.close?.();
  imageTestBitmap = await createImageBitmap(file);
  imageTestMode = true;

  stopCameraOnlyButKeepUi();
  drawImageTestSource();

  setPillText(camState, "이미지 모드", "warn");
  cameraLabel.textContent = "이미지 모드";
}

async function runImageTestAnalysis(): Promise<void> {
  if (!imageTestMode || !imageTestBitmap || !imageTestRun) return;

  imageTestRun.disabled = true;
  imageTestRun.textContent = "분석 중...";

  try {
    await ensureRoadSegLoad();
    drawImageTestSource();
    clearOverlay();

    lastSegAt = 0;

    await runRoadSegmentation();

    setPillText(camState, "이미지 모드", "warn");
    cameraLabel.textContent = "이미지 모드";
  } catch (error) {
    console.error("이미지 분석 실패", error);
    setPillText(camState, "이미지 분석 실패", "bad");
  } finally {
    imageTestRun.disabled = false;
    imageTestRun.textContent = "이미지 분석";
  }
}

function setCameraMode(): void {
  imageTestMode = false;
  imageTestCanvasEl.classList.add("hidden");
  cameraPreviewCanvasEl.classList.remove("hidden");
  clearOverlay();
  updateCommonUi();
}

async function start(): Promise<void> {
  if (running) return;

  startButton.disabled = true;
  primeTts();
  setCameraMode();

  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: { ideal: "environment" },
        width: { ideal: 640 },
        height: { ideal: 640 },
        frameRate: { ideal: 15, max: 24 },
      },
      audio: false,
    });

    video.srcObject = stream;
    await video.play();

    running = true;
    imageTestMode = false;
    lastFrameAt = 0;
    fps = 0;
    lowPowerMode = false;
    roadSegErrorMessage = "";
    roadSegBusy = false;
    roadSegInFlight = false;
    roadMask = null;
    sidewalkMask = null;
    crosswalkMask = null;
    curbMask = null;
    roadMaskCanvas = null;
    sidewalkMaskCanvas = null;
    crosswalkMaskCanvas = null;
    curbMaskCanvas = null;
    nearCurbCoverageKnown = false;
    lookaheadCurbCoverageKnown = false;
    nearCurbCoverage = 0;
    lookaheadCurbCoverage = 0;
    nearSidewalkCoverageKnown = false;
    nearRoadCoverageKnown = false;
    lookaheadSidewalkCoverageKnown = false;
    lookaheadRoadCoverageKnown = false;
    nearSidewalkCoverage = 0;
    nearRoadCoverage = 0;
    lookaheadSidewalkCoverage = 0;
    lookaheadRoadCoverage = 0;
    farRoadCoverage = 0;
    rawRiskState = "unknown";
    votedRiskState = "unknown";
    currentRiskState = "unknown";
    lastRawRiskState = "unknown";
    rawRiskStreak = 0;
    riskVoteHistory = [];
    riskHoldUntil = 0;
    segmentationRevision = 0;
    lastRiskProcessedRevision = -1;
    lastOverlayDrawRevision = -1;
    lastAlertProcessedState = "unknown";
    lastAlertAt = 0;
    lastPreviewDrawAt = 0;
    lastCommonUiAt = 0;
    lastInferenceMs = 0;
    lastCameraSegAttemptAt = 0;

    const activeSize = getActiveSegInputSize();
    cameraPreviewCanvasEl.width = activeSize;
    cameraPreviewCanvasEl.height = activeSize;
    cameraPreviewCanvasEl.classList.remove("hidden");
    imageTestCanvasEl.classList.add("hidden");

    updateSegUi();
    updateRiskUi();
    updatePowerUi();
    updateCommonUi();
    document.body.classList.add("live-mode");

    syncCanvasSize();
    if (typeof ResizeObserver !== "undefined") {
      resizeObserver = new ResizeObserver(syncCanvasSize);
      resizeObserver.observe(video);
    }

    if (!segWorker || roadSegState !== "ready") {
      roadSegState = "idle";
      void ensureRoadSegLoad();
    }
    rafId = window.requestAnimationFrame(renderLoop);
  } catch (error) {
    console.error("카메라 시작 실패", error);
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

  document.body.classList.remove("live-mode");

  lastAlertAt = 0;
  lastAlertProcessedState = "unknown";
  lastHeartbeatAlertAt = 0;
  lastInferenceMs = 0;
  workerInferenceMs = 0;
  workerTotalMs = 0;
  votedRiskState = "unknown";
  lastCameraSegAttemptAt = 0;
  lastSpokenText = "";
  lastSpeechAt = 0;
  segmentationRevision = 0;
  lastRiskProcessedRevision = -1;
  lastOverlayDrawRevision = -1;
  lastPreviewDrawAt = 0;
  lastCommonUiAt = 0;
  riskVoteHistory = [];
  latestSegRequestId = 0;
  segRequestSeq = 0;
  segResultReject?.(new Error("Segmentation stopped"));
  segResultResolve = null;
  segResultReject = null;
  segResultPromise = null;

  const currentStream = stream;
  stream = null;
  currentStream?.getTracks().forEach((track) => track.stop());
  video.srcObject = null;
  clearOverlay();
  lowPowerMode = false;
  cameraPreviewCanvasEl.classList.add("hidden");
  roadSegBusy = false;
  roadSegInFlight = false;
  setStoppedState();
}

async function enableAlerts(): Promise<void> {
  alertsEnabled = true;
  await ensureAudioContext();
  void playBeepPattern([90], 880);
  vibratePattern([40]);
  primeTts();
  if (enableAlertsButton) {
    enableAlertsButton.textContent = "알림 활성화됨";
  }
}

async function testDangerAlert(): Promise<void> {
  await enableAlerts();
  const previousState = currentRiskState;
  currentRiskState = "danger";
  lastAlertProcessedState = "unknown";
  lastHeartbeatAlertAt = 0;
  maybeTriggerAlert(performance.now());
  applyVisualAlert("danger");
  updateRiskUi();

  window.setTimeout(() => {
    currentRiskState = previousState;
    applyVisualAlert(previousState);
    updateRiskUi();
  }, 1800);
}

function renderLoop(now: number): void {
  if (!running) return;

  updateFpsAndPower(now);

  if (!imageTestMode && now - lastPreviewDrawAt >= PREVIEW_INTERVAL_MS) {
    lastPreviewDrawAt = now;
    drawCameraPreviewSource();
  }

  maybeRunCameraSegmentation(now);

  const hasNewSegmentation = segmentationRevision !== lastRiskProcessedRevision;

  if (hasNewSegmentation) {
    updateCoverageMetrics();
    syncRiskState(now);

    lastRiskProcessedRevision = segmentationRevision;

    if (lastOverlayDrawRevision !== segmentationRevision) {
      renderOverlayFrame();
      lastOverlayDrawRevision = segmentationRevision;
    }

    updateRiskUi();
  }

  maybeTriggerAlertHeartbeat(now);
  updateCommonUiThrottled(now);

  rafId = window.requestAnimationFrame(renderLoop);
}

startButton.addEventListener("click", () => {
  void start();
});

imageTestInputEl.addEventListener("change", async () => {
  const file = imageTestInputEl.files?.[0];
  if (!file) return;

  try {
    await loadImageTestFile(file);
  } catch (error) {
    console.error("이미지 파일 로드 실패", error);
    showSegError(error instanceof Error ? error.message : String(error));
  } finally {
    imageTestInputEl.value = "";
  }
});

imageTestRun?.addEventListener("click", () => {
  void runImageTestAnalysis();
});

cameraModeButtonEl.addEventListener("click", () => {
  setCameraMode();
});

enableAlertsButton?.addEventListener("click", () => {
  void enableAlerts();
});

testDangerAlertButton?.addEventListener("click", () => {
  void testDangerAlert();
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

// Debug only: reset worker state without auto-running it.
(window as Window & { resetSegWorker?: () => void }).resetSegWorker = resetSegWorker;

updateSegUi();
updateRiskUi();
updatePowerUi();
updateCommonUi();
setStoppedState();
