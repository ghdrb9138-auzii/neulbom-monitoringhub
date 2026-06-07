import "./style.css";
import { registerSW } from "virtual:pwa-register";
import * as ort from "onnxruntime-web";
import { env, pipeline, RawImage } from "@huggingface/transformers";

type Detection = {
  classId: number;
  label: string;
  confidence: number;
  x: number;
  y: number;
  width: number;
  height: number;
};

type PreprocessMeta = {
  scale: number;
  padX: number;
  padY: number;
  sourceWidth: number;
  sourceHeight: number;
};

type ModelState = "idle" | "loading" | "ready" | "disabled" | "error";
type SegState = "idle" | "loading" | "ready" | "error";
type DepthState = "idle" | "loading" | "ready" | "error";
type RiskState = "safe" | "warn" | "danger";
type RoadMask = {
  width: number;
  height: number;
  data: Uint8Array;
};
type DepthMap = {
  width: number;
  height: number;
  data: Float32Array;
  source: "model" | "placeholder";
};
type StepCandidate = {
  x: number;
  y: number;
  width: number;
  height: number;
  score: number;
  source: "model" | "placeholder";
};

const MODEL_URL = "/models/road_obstacle_yolov8n_320.onnx";
const MOCK_OVERLAY_BADGE = "mock curb_or_step · sidewalk demo";
const ROAD_SEG_LOCAL_MODEL_ID = "segformer-cityscapes";
const ROAD_SEG_REMOTE_MODEL_ID = "Xenova/segformer-b0-finetuned-cityscapes-640-1280";
const DEPTH_REMOTE_MODEL_ID = "onnx-community/depth-anything-v2-small";
const WALKABLE_CLASSES = ["sidewalk"] as const;
const WALKABLE_CLASS_OPTIONS = ["sidewalk", "road"] as const;

const INPUT_SIZE = 640;
const ROAD_SEG_INPUT_W = 160;
const ROAD_SEG_INPUT_H = 90;
const ROAD_SEG_INTERVAL_MS = 900;
const ROAD_SEG_SLOW_INTERVAL_MS = 1500;
const DEPTH_INTERVAL_MS = 1200;
const DEPTH_INPUT_W = 160;
const DEPTH_INPUT_H = 90;
const DEPTH_PLACEHOLDER_ENABLED = true;
const DEPTH_GRADIENT_THRESHOLD = 0.12;
const DEPTH_PLACEHOLDER_STEP_Y_RATIO = 0.72;
const DEPTH_PLACEHOLDER_STEP_THICKNESS = 0.08;
const STEP_CANDIDATE_BOTTOM_RATIO = 0.7;
const CONF_THRESHOLD = 0.45;
const NMS_THRESHOLD = 0.45;
const TARGET_CLASSES = [
  "curb_or_step",
  "speed_bump",
  "road_cone",
  "debris",
  "unknown_obstacle",
] as const;

const MOCK_DETECTIONS: Detection[] = [
  {
    classId: 0,
    label: "curb_or_step",
    confidence: 0.82,
    x: 317,
    y: 367,
    width: 326,
    height: 86,
  },
];

const WARN_TTS_INTERVAL_MS = 4000;
const DANGER_TTS_INTERVAL_MS = 1500;
const DANGER_HOLD_MS = 1500;
const WARN_HOLD_MS = 850;
const LOW_POWER_ENTER_FPS = 12;
const LOW_POWER_EXIT_FPS = 16;
const NORMAL_INFER_EVERY_N_FRAMES = 2;
const LOW_POWER_INFER_EVERY_N_FRAMES = 3;
const STROLLER_CORRIDOR_X_MIN = 0.3;
const STROLLER_CORRIDOR_X_MAX = 0.7;
const STROLLER_CORRIDOR_Y_MIN = 0.5;
const STROLLER_CORRIDOR_BOTTOM_RATIO = 0.7;
const STROLLER_CORRIDOR_SAFE_COVERAGE = 0.55;
const STROLLER_CORRIDOR_WARN_COVERAGE = 0.3;
const ROI_X_MIN = 0.2;
const ROI_X_MAX = 0.8;
const ROI_Y_MIN = 0.45;
const DANGER_BOTTOM_RATIO = 0.75;
const DANGER_AREA_RATIO = 0.12;
const MATCH_IOU_THRESHOLD = 0.45;
const TRACK_STABLE_FRAMES = 3;

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
          <h1>늘봄 유모차 보행로 감지 PWA</h1>
          <p class="lede">
            Start 버튼으로 카메라를 시작하고, SegFormer sidewalk segmentation과 mock curb_or_step 감지 결과, TTS 안내를 함께 확인합니다.
          </p>
        </div>
        <div class="status-stack">
          <span class="pill pill-neutral" id="app-state">대기 중</span>
          <span class="pill pill-neutral" id="cam-state">카메라 미실행</span>
          <span class="pill pill-neutral" id="model-state">YOLO 미실행</span>
          <span class="pill pill-neutral" id="seg-state">Seg 미실행</span>
          <span class="pill pill-neutral" id="depth-state">Depth 미실행</span>
          <span class="pill pill-neutral" id="risk-state">safe</span>
          <span class="pill pill-neutral" id="power-state">normal</span>
          <span class="pill pill-neutral" id="tts-state">TTS 비활성</span>
        </div>
      </div>

      <div class="video-frame">
        <video id="video" autoplay muted playsinline webkit-playsinline></video>
        <canvas id="overlay"></canvas>
        <div class="overlay-badge" id="overlay-badge">mock curb_or_step · sidewalk demo</div>
      </div>

      <div id="model-error" class="error-banner hidden" role="status" aria-live="polite"></div>
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
        <div class="card-label">YOLO 모델</div>
        <div class="card-value" id="model-label">대기</div>
        <div class="card-note" id="model-note">/public/models/road_obstacle_yolov8n_320.onnx</div>
      </article>

      <article class="card">
        <div class="card-label">SegFormer</div>
        <div class="card-value" id="seg-label">대기</div>
        <div class="card-note" id="seg-note">sidewalk segmentation overlay</div>
      </article>

      <article class="card">
        <div class="card-label">Depth</div>
        <div class="card-value" id="depth-label">대기</div>
        <div class="card-note" id="depth-note">depth-anything placeholder + optional model skeleton</div>
      </article>

      <article class="card">
        <div class="card-label">Risk</div>
        <div class="card-value" id="risk-label">safe</div>
        <div class="card-note" id="risk-note">보행 가속 영역 + 유모차 감속 영역 기준</div>
      </article>

      <article class="card">
        <div class="card-label">Performance</div>
        <div class="card-value" id="fps-label">FPS --</div>
        <div class="card-note" id="power-note">정상 모드</div>
      </article>

      <article class="card">
        <div class="card-label">Detection</div>
        <div class="card-value" id="detection-label">curb_or_step 0.82</div>
        <div class="card-note">mock curb_or_step는 2초 1회, 감지 상승 시 3초 1회로 유지</div>
      </article>

      <article class="card">
        <div class="card-label">TTS</div>
        <div class="card-value" id="tts-label">준비 필요</div>
        <div class="card-note">warn / danger 상태별 cooldown 적용</div>
      </article>
    </section>

    <section class="details">
      <div><span>패키지명</span><code>neulbom-environment-pwa</code></div>
      <div><span>개발 포트</span><code>5176</code></div>
      <div><span>구성</span><code>Camera + Canvas + TTS + mock curb_or_step + SegFormer(sidewalk)</code></div>
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
const modelStateEl = document.getElementById("model-state") as HTMLSpanElement;
const segStateEl = document.getElementById("seg-state") as HTMLSpanElement;
const depthStateEl = document.getElementById("depth-state") as HTMLSpanElement;
const riskStateEl = document.getElementById("risk-state") as HTMLSpanElement;
const powerStateEl = document.getElementById("power-state") as HTMLSpanElement;
const ttsState = document.getElementById("tts-state") as HTMLSpanElement;

const cameraLabel = document.getElementById("camera-label") as HTMLDivElement;
const modelLabel = document.getElementById("model-label") as HTMLDivElement;
const modelNote = document.getElementById("model-note") as HTMLDivElement;
const segLabel = document.getElementById("seg-label") as HTMLDivElement;
const segNote = document.getElementById("seg-note") as HTMLDivElement;
const depthLabel = document.getElementById("depth-label") as HTMLDivElement;
const depthNote = document.getElementById("depth-note") as HTMLDivElement;
const riskLabel = document.getElementById("risk-label") as HTMLDivElement;
const riskNote = document.getElementById("risk-note") as HTMLDivElement;
const fpsLabel = document.getElementById("fps-label") as HTMLDivElement;
const powerNote = document.getElementById("power-note") as HTMLDivElement;
const detectionLabel = document.getElementById("detection-label") as HTMLDivElement;
const ttsLabel = document.getElementById("tts-label") as HTMLDivElement;
const modelError = document.getElementById("model-error") as HTMLDivElement;
const segError = document.getElementById("seg-error") as HTMLDivElement;

const overlayCtx = overlay.getContext("2d");
const preprocessCanvas = document.createElement("canvas");
preprocessCanvas.width = INPUT_SIZE;
preprocessCanvas.height = INPUT_SIZE;
const preprocessCtx = preprocessCanvas.getContext("2d", { willReadFrequently: true });

const roadSegInputCanvas = document.createElement("canvas");
roadSegInputCanvas.width = ROAD_SEG_INPUT_W;
roadSegInputCanvas.height = ROAD_SEG_INPUT_H;
const roadSegInputCtx = roadSegInputCanvas.getContext("2d", { willReadFrequently: true });

const depthInputCanvas = document.createElement("canvas");
depthInputCanvas.width = DEPTH_INPUT_W;
depthInputCanvas.height = DEPTH_INPUT_H;
const depthInputCtx = depthInputCanvas.getContext("2d", { willReadFrequently: true });

env.localModelPath = "/models/";
env.allowLocalModels = true;
env.allowRemoteModels = true;

registerSW({ immediate: true });

ort.env.wasm.numThreads = 1;
ort.env.wasm.proxy = false;
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";

type RoadSegmenter = (image: RawImage) => Promise<any>;

let stream: MediaStream | null = null;
let running = false;
let rafId = 0;
let resizeObserver: ResizeObserver | null = null;
let frameCount = 0;
let lastFrameAt = 0;
let fps = 0;
let lowPowerMode = false;
let isInferBusy = false;
let hasPrimedTts = false;

let modelState: ModelState = "idle";
let modelErrorMessage = "";
let modelSession: ort.InferenceSession | null = null;
let modelLoadPromise: Promise<void> | null = null;
let modelAvailability: boolean | null = null;
let modelCheckPromise: Promise<boolean> | null = null;

let roadSegState: SegState = "idle";
let roadSegErrorMessage = "";
let roadSegBusy = false;
let roadSegLoadPromise: Promise<void> | null = null;
let segmenter: RoadSegmenter | null = null;
let depthState: DepthState = "idle";
let depthErrorMessage = "";
let depthBusy = false;
let depthLoadPromise: Promise<void> | null = null;
let depthEstimator: ((image: RawImage) => Promise<any>) | null = null;
let lastDepthMap: DepthMap | null = null;
let lastStepCandidate: StepCandidate | null = null;
let lastDepthAt = 0;
let lastRoadMask: RoadMask | null = null;
let lastRoadMaskCanvas: HTMLCanvasElement | null = null;
let lastRoadMaskCoverage = 0;
let lastCorridorCoverage = 0;
let lastSegAt = 0;

let lastDetections: Detection[] = [...MOCK_DETECTIONS];
let trackedDetection: Detection | null = null;
let stableFrames = 0;
let currentRiskState: RiskState = "safe";
let riskHoldUntil = 0;
let lastWarnTtsAt = 0;
let lastDangerTtsAt = 0;

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

function showModelError(message: string): void {
  modelError.textContent = message;
  modelError.classList.remove("hidden");
}

function hideModelError(): void {
  modelError.textContent = "";
  modelError.classList.add("hidden");
}

function showSegError(message: string): void {
  segError.textContent = message;
  segError.classList.remove("hidden");
}

function hideSegError(): void {
  segError.textContent = "";
  segError.classList.add("hidden");
}

function updateModelUi(): void {
  if (modelState === "idle") {
    setPillText(modelStateEl, "YOLO 미실행", "neutral");
    modelLabel.textContent = "대기";
    modelNote.textContent = "시작 시 모델 존재 여부를 확인합니다";
    hideModelError();
    return;
  }

  if (modelState === "disabled") {
    setPillText(modelStateEl, "YOLO 비활성 / mock mode", "neutral");
    modelLabel.textContent = "mock mode";
    modelNote.textContent = "모델 파일이 없어 mock curb_or_step를 사용합니다";
    hideModelError();
    return;
  }

  if (modelState === "loading") {
    setPillText(modelStateEl, "YOLO 로딩 중", "warn");
    modelLabel.textContent = "로딩 중";
    modelNote.textContent = MODEL_URL;
    hideModelError();
    return;
  }

  if (modelState === "ready") {
    setPillText(modelStateEl, "YOLO 준비 완료", "success");
    modelLabel.textContent = "준비 완료";
    modelNote.textContent = "YOLOv8 Nano ONNX 추론 사용 중";
    hideModelError();
    return;
  }

  setPillText(modelStateEl, "YOLO 오류", "bad");
  modelLabel.textContent = "오류";
  modelNote.textContent = modelErrorMessage || "YOLO 모델 로딩 또는 추론 실패";
  showModelError(modelErrorMessage || "YOLO 모델 로딩 실패");
}

function updateSegUi(): void {
  if (roadSegState === "idle") {
    setPillText(segStateEl, "Seg 미실행", "neutral");
    segLabel.textContent = "대기";
    segNote.textContent = "Road mask: fallback ROI only";
    hideSegError();
    return;
  }

  if (roadSegState === "loading") {
    setPillText(segStateEl, "Seg 로딩 중", "warn");
    segLabel.textContent = "로딩 중";
    segNote.textContent = `${ROAD_SEG_LOCAL_MODEL_ID} → ${ROAD_SEG_REMOTE_MODEL_ID}`;
    hideSegError();
    return;
  }

  if (roadSegState === "ready") {
    setPillText(segStateEl, "Seg 준비 완료", "success");
    segLabel.textContent = "준비 완료";
    segNote.textContent = lastRoadMask
      ? `Walkable mask: ready\nMask coverage: ${(lastRoadMaskCoverage * 100).toFixed(1)}%\nStroller corridor: ${(lastCorridorCoverage * 100).toFixed(1)}%`
      : "Road mask: ready\nRoad filter: fallback ROI only";
    hideSegError();
    return;
  }

  setPillText(segStateEl, "Seg 오류", "bad");
  segLabel.textContent = "오류";
  segNote.textContent = "Road filter: fallback ROI only";
  showSegError(roadSegErrorMessage || "SegFormer 로딩 또는 추론 실패");
}

function updateDepthUi(): void {
  if (depthState === "idle") {
    setPillText(depthStateEl, "Depth 미실행", "neutral");
    depthLabel.textContent = "대기";
    depthNote.textContent = "depth placeholder active";
    return;
  }

  if (depthState === "loading") {
    setPillText(depthStateEl, "Depth 로딩 중", "warn");
    depthLabel.textContent = "로딩 중";
    depthNote.textContent = DEPTH_REMOTE_MODEL_ID;
    return;
  }

  if (depthState === "ready") {
    setPillText(depthStateEl, "Depth 준비 완료", "success");
    depthLabel.textContent = depthEstimator ? "model" : "placeholder";
    depthNote.textContent = lastStepCandidate
      ? `step_candidate ${(lastStepCandidate.score * 100).toFixed(0)}%`
      : "depth placeholder / optional model";
    return;
  }

  setPillText(depthStateEl, "Depth 오류", "bad");
  depthLabel.textContent = "fallback";
  depthNote.textContent = depthErrorMessage || "depth module fallback active";
}

function updateRiskUi(): void {
  riskLabel.textContent = currentRiskState;
  setPillText(
    riskStateEl,
    currentRiskState,
    currentRiskState === "safe" ? "neutral" : currentRiskState === "warn" ? "warn" : "bad",
  );

  if (currentRiskState === "safe") {
    riskNote.textContent = `stroller corridor coverage ${(lastCorridorCoverage * 100).toFixed(1)}%`;
  } else if (currentRiskState === "warn") {
    riskNote.textContent = `stroller corridor coverage ${(lastCorridorCoverage * 100).toFixed(1)}%`;
  } else {
    riskNote.textContent = `corridor obstacle detected · coverage ${(lastCorridorCoverage * 100).toFixed(1)}%`;
  }
}

function updatePowerUi(): void {
  setPillText(powerStateEl, lowPowerMode ? "low power" : "normal", lowPowerMode ? "warn" : "success");
  powerNote.textContent = lowPowerMode ? "FPS 12 이하 대응, 3프레임 1회 추론" : "정상 모드";
}

function updateCommonUi(): void {
  setPillText(appState, running ? "실행 중" : "대기 중", running ? "success" : "neutral");
  setPillText(camState, stream ? "카메라 실행" : "카메라 미실행", stream ? "success" : "neutral");
  setPillText(ttsState, hasPrimedTts ? "TTS 준비 완료" : "TTS 비활성", hasPrimedTts ? "success" : "neutral");

  const primary = lastDetections[0] ?? null;
  detectionLabel.textContent = primary ? `${primary.label} ${primary.confidence.toFixed(2)}` : "no detection";
  fpsLabel.textContent = `FPS ${fps > 0 ? fps.toFixed(1) : "--"}`;

  if (roadSegState === "ready") {
    overlayBadge.textContent = primary
      ? `${primary.label} ${primary.confidence.toFixed(2)} · ${currentRiskState}`
      : `${currentRiskState}`;
  }

  if (modelState === "disabled" || !modelSession) {
    overlayBadge.textContent = MOCK_OVERLAY_BADGE;
  } else if (lastStepCandidate) {
    overlayBadge.textContent = `step_candidate ${lastStepCandidate.score.toFixed(2)} · ${currentRiskState}`;
  } else {
    overlayBadge.textContent = primary
      ? `${primary.label} ${primary.confidence.toFixed(2)} · ${currentRiskState}`
      : currentRiskState;
  }
}

function setStoppedState(): void {
  cameraLabel.textContent = "대기";
  ttsLabel.textContent = "준비 필요";
  lastDetections = [...MOCK_DETECTIONS];
  trackedDetection = null;
  stableFrames = 0;
  currentRiskState = "safe";
  riskHoldUntil = 0;
  updateModelUi();
  updateSegUi();
  updateDepthUi();
  updateRiskUi();
  updatePowerUi();
  updateCommonUi();
}

function syncCanvasSize(): void {
  const rect = video.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const width = Math.max(1, Math.round(rect.width * dpr));
  const height = Math.max(1, Math.round(rect.height * dpr));

  if (overlay.width !== width) {
    overlay.width = width;
  }

  if (overlay.height !== height) {
    overlay.height = height;
  }
}

function clearOverlay(): void {
  if (!overlayCtx) {
    return;
  }

  overlayCtx.setTransform(1, 0, 0, 1, 0, 0);
  overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
}

function fitContain(
  sourceWidth: number,
  sourceHeight: number,
  viewWidth: number,
  viewHeight: number,
): { scale: number; x: number; y: number; width: number; height: number } {
  const scale = Math.min(viewWidth / sourceWidth, viewHeight / sourceHeight);
  const width = sourceWidth * scale;
  const height = sourceHeight * scale;
  return {
    scale,
    width,
    height,
    x: (viewWidth - width) / 2,
    y: (viewHeight - height) / 2,
  };
}

function drawRoadMask(maskCanvas: HTMLCanvasElement, sourceWidth: number, sourceHeight: number): void {
  if (!overlayCtx) {
    return;
  }

  const rect = video.getBoundingClientRect();
  const frame = fitContain(sourceWidth, sourceHeight, rect.width, rect.height);
  overlayCtx.save();
  overlayCtx.globalAlpha = lowPowerMode ? 0.22 : 0.34;
  overlayCtx.drawImage(maskCanvas, frame.x, frame.y, frame.width, frame.height);
  overlayCtx.restore();
}

function drawStrollerCorridor(sourceWidth: number, sourceHeight: number): void {
  if (!overlayCtx) {
    return;
  }

  const rect = video.getBoundingClientRect();
  const frame = fitContain(sourceWidth, sourceHeight, rect.width, rect.height);
  const x = frame.x + frame.width * STROLLER_CORRIDOR_X_MIN;
  const y = frame.y + frame.height * STROLLER_CORRIDOR_Y_MIN;
  const width = frame.width * (STROLLER_CORRIDOR_X_MAX - STROLLER_CORRIDOR_X_MIN);
  const height = frame.height * (1 - STROLLER_CORRIDOR_Y_MIN);

  overlayCtx.save();
  overlayCtx.setLineDash([12, 8]);
  overlayCtx.lineWidth = 2.5;
  overlayCtx.strokeStyle = "rgba(74, 222, 128, 0.95)";
  overlayCtx.fillStyle = "rgba(74, 222, 128, 0.08)";
  overlayCtx.fillRect(x, y, width, height);
  overlayCtx.strokeRect(x, y, width, height);
  overlayCtx.restore();
}

function drawStepCandidate(candidate: StepCandidate | null, sourceWidth: number, sourceHeight: number): void {
  if (!overlayCtx || !candidate) {
    return;
  }

  const rect = video.getBoundingClientRect();
  const frame = fitContain(sourceWidth, sourceHeight, rect.width, rect.height);
  const x = frame.x + candidate.x * frame.scale;
  const y = frame.y + candidate.y * frame.scale;
  const width = candidate.width * frame.scale;
  const height = candidate.height * frame.scale;

  overlayCtx.save();
  overlayCtx.setLineDash([8, 6]);
  overlayCtx.lineWidth = 2.5;
  overlayCtx.strokeStyle = "rgba(251, 191, 36, 0.98)";
  overlayCtx.fillStyle = "rgba(251, 191, 36, 0.16)";
  overlayCtx.fillRect(x, y, width, height);
  overlayCtx.strokeRect(x, y, width, height);
  overlayCtx.restore();
}

function getRoadSegIntervalMs(): number {
  return lowPowerMode ? ROAD_SEG_SLOW_INTERVAL_MS : ROAD_SEG_INTERVAL_MS;
}

function drawDetections(detections: Detection[], sourceWidth: number, sourceHeight: number): void {
  if (!overlayCtx) {
    return;
  }

  const rect = video.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const frame = fitContain(sourceWidth, sourceHeight, rect.width, rect.height);
  overlayCtx.setTransform(dpr, 0, 0, dpr, 0, 0);

  const detectionsToDraw = lowPowerMode ? detections.slice(0, 1) : detections;
  const primary = detectionsToDraw[0] ?? null;

  detectionsToDraw.forEach((detection, index) => {
    const x = frame.x + detection.x * frame.scale;
    const y = frame.y + detection.y * frame.scale;
    const width = detection.width * frame.scale;
    const height = detection.height * frame.scale;
    const isPrimary = index === 0;
    const isRisk = isPrimary && currentRiskState !== "safe";

    const strokeColor = isRisk
      ? currentRiskState === "danger"
        ? "rgba(255, 90, 90, 0.98)"
        : "rgba(255, 180, 60, 0.96)"
      : "rgba(120, 180, 255, 0.92)";

    overlayCtx.lineWidth = lowPowerMode ? 2 : 3;
    overlayCtx.strokeStyle = strokeColor;
    overlayCtx.fillStyle = lowPowerMode
      ? "rgba(255, 255, 255, 0.03)"
      : isRisk
        ? currentRiskState === "danger"
          ? "rgba(255, 90, 90, 0.16)"
          : "rgba(255, 180, 60, 0.16)"
        : "rgba(120, 180, 255, 0.12)";
    overlayCtx.fillRect(x, y, width, height);
    overlayCtx.strokeRect(x, y, width, height);

    if (!lowPowerMode && isPrimary) {
      const label = `${detection.label} ${detection.confidence.toFixed(2)}`;
      overlayCtx.font = "600 14px system-ui, -apple-system, BlinkMacSystemFont, sans-serif";
      const textWidth = overlayCtx.measureText(label).width;
      const labelBoxWidth = Math.max(74, textWidth + 18);
      const labelBoxHeight = 26;
      const labelX = x;
      const labelY = Math.max(8, y - labelBoxHeight - 6);

      overlayCtx.fillStyle = "rgba(15, 23, 42, 0.92)";
      overlayCtx.fillRect(labelX, labelY, labelBoxWidth, labelBoxHeight);
      overlayCtx.strokeStyle = strokeColor;
      overlayCtx.lineWidth = 1.5;
      overlayCtx.strokeRect(labelX, labelY, labelBoxWidth, labelBoxHeight);
      overlayCtx.fillStyle = "#fff7d6";
      overlayCtx.textBaseline = "middle";
      overlayCtx.fillText(label, labelX + 9, labelY + labelBoxHeight / 2 + 0.5);
    }
  });

  overlayBadge.textContent = lowPowerMode
    ? `low power 쨌 ${currentRiskState}`
    : primary
      ? `${primary.label} ${primary.confidence.toFixed(2)} 쨌 ${currentRiskState}`
      : currentRiskState;
}

function renderOverlayFrame(sourceWidth: number, sourceHeight: number): void {
  if (!overlayCtx) {
    return;
  }

  const rect = video.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  syncCanvasSize();
  overlayCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
  overlayCtx.clearRect(0, 0, rect.width, rect.height);

  if (lastRoadMaskCanvas) {
    drawRoadMask(lastRoadMaskCanvas, sourceWidth, sourceHeight);
  }

  drawDetections(lastDetections, sourceWidth, sourceHeight);
  drawStepCandidate(lastStepCandidate, sourceWidth, sourceHeight);
  drawStrollerCorridor(sourceWidth, sourceHeight);
}

function speak(text: string): boolean {
  if (!("speechSynthesis" in window)) {
    ttsLabel.textContent = "브라우저 TTS 미지원";
    setPillText(ttsState, "TTS 미지원", "warn");
    return false;
  }

  window.speechSynthesis.cancel();

  const utterance = new SpeechSynthesisUtterance(text);
  utterance.lang = "ko-KR";
  utterance.rate = 1.0;
  utterance.pitch = 1.0;
  utterance.volume = 1.0;
  utterance.onstart = () => {
    ttsLabel.textContent = "말하기 중";
    setPillText(ttsState, "TTS ?쒖꽦", "success");
  };
  utterance.onend = () => {
    ttsLabel.textContent = "대기";
    setPillText(ttsState, "TTS ?쒖꽦", "success");
  };
  utterance.onerror = () => {
    ttsLabel.textContent = "TTS 오류";
    setPillText(ttsState, "TTS 오류", "bad");
  };

  window.speechSynthesis.speak(utterance);
  return true;
}

function primeTts(): void {
  if (!("speechSynthesis" in window)) {
    hasPrimedTts = false;
    updateCommonUi();
    return;
  }

  hasPrimedTts = true;
  window.speechSynthesis.cancel();

  const warmup = new SpeechSynthesisUtterance(" ");
  warmup.lang = "ko-KR";
  warmup.volume = 0;
  warmup.rate = 1;
  warmup.pitch = 1;
  window.speechSynthesis.speak(warmup);
  updateCommonUi();
}

async function openCamera(): Promise<MediaStream> {
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error("브라우저가 카메라를 지원하지 않습니다.");
  }

  const commonVideo = {
    width: { ideal: 960 },
    height: { ideal: 540 },
    frameRate: { ideal: 30, max: 30 },
  };

  const attempts: MediaStreamConstraints[] = [
    {
      video: {
        ...commonVideo,
        facingMode: { ideal: "environment" },
      },
      audio: false,
    },
    {
      video: {
        ...commonVideo,
        facingMode: { ideal: "user" },
      },
      audio: false,
    },
    {
      video: commonVideo,
      audio: false,
    },
  ];

  let lastError: unknown = null;

  for (const constraints of attempts) {
    try {
      return await navigator.mediaDevices.getUserMedia(constraints);
    } catch (error) {
      lastError = error;
    }
  }

  throw lastError instanceof Error ? lastError : new Error("移대찓?쇰? ?????놁뒿?덈떎.");
}

function preprocessFrame(): { tensor: ort.Tensor; meta: PreprocessMeta } {
  if (!preprocessCtx) {
    throw new Error("preprocess canvas context not available");
  }

  const sourceWidth = video.videoWidth;
  const sourceHeight = video.videoHeight;

  if (!sourceWidth || !sourceHeight) {
    throw new Error("video metadata not ready");
  }

  const scale = Math.min(INPUT_SIZE / sourceWidth, INPUT_SIZE / sourceHeight);
  const resizedWidth = Math.round(sourceWidth * scale);
  const resizedHeight = Math.round(sourceHeight * scale);
  const padX = Math.floor((INPUT_SIZE - resizedWidth) / 2);
  const padY = Math.floor((INPUT_SIZE - resizedHeight) / 2);

  preprocessCtx.fillStyle = "#000000";
  preprocessCtx.fillRect(0, 0, INPUT_SIZE, INPUT_SIZE);
  preprocessCtx.drawImage(
    video,
    0,
    0,
    sourceWidth,
    sourceHeight,
    padX,
    padY,
    resizedWidth,
    resizedHeight,
  );

  const imageData = preprocessCtx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE).data;
  const tensorData = new Float32Array(3 * INPUT_SIZE * INPUT_SIZE);
  const planeSize = INPUT_SIZE * INPUT_SIZE;

  for (let i = 0; i < planeSize; i += 1) {
    const base = i * 4;
    tensorData[i] = imageData[base] / 255;
    tensorData[planeSize + i] = imageData[base + 1] / 255;
    tensorData[planeSize * 2 + i] = imageData[base + 2] / 255;
  }

  return {
    tensor: new ort.Tensor("float32", tensorData, [1, 3, INPUT_SIZE, INPUT_SIZE]),
    meta: { scale, padX, padY, sourceWidth, sourceHeight },
  };
}

function sigmoid(value: number): number {
  return 1 / (1 + Math.exp(-value));
}

function decodeOutput(
  output: ort.Tensor,
  numClasses: number,
): Array<{ cx: number; cy: number; width: number; height: number; scores: number[] }> {
  const dims = output.dims.map((dim) => Number(dim));
  const values = output.data as Float32Array;

  if (dims.length !== 3) {
    return [];
  }

  const attrs = 4 + numClasses;
  const [, dim1, dim2] = dims;
  const candidates: Array<{ cx: number; cy: number; width: number; height: number; scores: number[] }> = [];

  if (dim1 === attrs) {
    const numPredictions = dim2;
    for (let i = 0; i < numPredictions; i += 1) {
      const cx = values[0 * numPredictions + i];
      const cy = values[1 * numPredictions + i];
      const width = values[2 * numPredictions + i];
      const height = values[3 * numPredictions + i];
      const scores = new Array<number>(numClasses);

      for (let c = 0; c < numClasses; c += 1) {
        scores[c] = values[(4 + c) * numPredictions + i];
      }

      candidates.push({ cx, cy, width, height, scores });
    }

    return candidates;
  }

  if (dim2 === attrs) {
    const numPredictions = dim1;
    for (let i = 0; i < numPredictions; i += 1) {
      const offset = i * attrs;
      const cx = values[offset];
      const cy = values[offset + 1];
      const width = values[offset + 2];
      const height = values[offset + 3];
      const scores = new Array<number>(numClasses);

      for (let c = 0; c < numClasses; c += 1) {
        scores[c] = values[offset + 4 + c];
      }

      candidates.push({ cx, cy, width, height, scores });
    }
  }

  return candidates;
}

function normalizeBox(box: { cx: number; cy: number; width: number; height: number }): {
  cx: number;
  cy: number;
  width: number;
  height: number;
} {
  const maxValue = Math.max(box.cx, box.cy, box.width, box.height);
  if (maxValue <= 2) {
    return {
      cx: box.cx * INPUT_SIZE,
      cy: box.cy * INPUT_SIZE,
      width: box.width * INPUT_SIZE,
      height: box.height * INPUT_SIZE,
    };
  }

  return box;
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function convertToSourceBox(
  box: { cx: number; cy: number; width: number; height: number },
  meta: PreprocessMeta,
): Omit<Detection, "label" | "confidence" | "classId"> | null {
  const normalized = normalizeBox(box);
  const left = (normalized.cx - normalized.width / 2 - meta.padX) / meta.scale;
  const top = (normalized.cy - normalized.height / 2 - meta.padY) / meta.scale;
  const width = normalized.width / meta.scale;
  const height = normalized.height / meta.scale;

  const x = clamp(left, 0, meta.sourceWidth - 1);
  const y = clamp(top, 0, meta.sourceHeight - 1);
  const right = clamp(left + width, 0, meta.sourceWidth);
  const bottom = clamp(top + height, 0, meta.sourceHeight);
  const clippedWidth = right - x;
  const clippedHeight = bottom - y;

  if (clippedWidth <= 1 || clippedHeight <= 1) {
    return null;
  }

  return { x, y, width: clippedWidth, height: clippedHeight };
}

function iou(a: Detection, b: Detection): number {
  const left = Math.max(a.x, b.x);
  const top = Math.max(a.y, b.y);
  const right = Math.min(a.x + a.width, b.x + b.width);
  const bottom = Math.min(a.y + a.height, b.y + b.height);
  const intersectionWidth = Math.max(0, right - left);
  const intersectionHeight = Math.max(0, bottom - top);
  const intersection = intersectionWidth * intersectionHeight;
  const union = a.width * a.height + b.width * b.height - intersection;
  return union <= 0 ? 0 : intersection / union;
}

function applyNms(detections: Detection[], threshold: number): Detection[] {
  const sorted = [...detections].sort((a, b) => b.confidence - a.confidence);
  const kept: Detection[] = [];

  for (const detection of sorted) {
    const hasOverlap = kept.some(
      (other) => other.label === detection.label && iou(other, detection) > threshold,
    );
    if (!hasOverlap) {
      kept.push(detection);
    }
  }

  return kept;
}

function postprocess(output: ort.Tensor, meta: PreprocessMeta): Detection[] {
  const decoded = decodeOutput(output, TARGET_CLASSES.length);
  const detections: Detection[] = [];

  for (const candidate of decoded) {
    let bestClassId = 0;
    let bestScore = -Infinity;

    for (let i = 0; i < candidate.scores.length; i += 1) {
      const score =
        candidate.scores[i] >= 0 && candidate.scores[i] <= 1
          ? candidate.scores[i]
          : sigmoid(candidate.scores[i]);
      if (score > bestScore) {
        bestScore = score;
        bestClassId = i;
      }
    }

    if (bestScore < CONF_THRESHOLD) {
      continue;
    }

    const sourceBox = convertToSourceBox(candidate, meta);
    if (!sourceBox) {
      continue;
    }

    detections.push({
      classId: bestClassId,
      label: TARGET_CLASSES[bestClassId] ?? "unknown_obstacle",
      confidence: bestScore,
      ...sourceBox,
    });
  }

  return applyNms(detections, NMS_THRESHOLD);
}

function getOutputTensor(outputs: Record<string, ort.Tensor>): ort.Tensor | null {
  return Object.values(outputs)[0] ?? null;
}

function updateDetectionsUi(detections: Detection[]): void {
  detectionLabel.textContent =
    detections.length > 0 ? `${detections[0].label} ${detections[0].confidence.toFixed(2)}` : "no detection";
}

function isWithinRoi(detection: Detection, width: number, height: number): boolean {
  const centerX = detection.x + detection.width / 2;
  const centerY = detection.y + detection.height / 2;
  return (
    centerX >= width * ROI_X_MIN &&
    centerX <= width * ROI_X_MAX &&
    centerY >= height * ROI_Y_MIN
  );
}

function detectionAreaRatio(detection: Detection, width: number, height: number): number {
  return (detection.width * detection.height) / (width * height);
}

function detectionIsDanger(detection: Detection, width: number, height: number): boolean {
  const bottom = detection.y + detection.height;
  return (
    bottom >= height * DANGER_BOTTOM_RATIO ||
    detectionAreaRatio(detection, width, height) >= DANGER_AREA_RATIO
  );
}

function matchDetection(a: Detection, b: Detection): boolean {
  return a.label === b.label && iou(a, b) >= MATCH_IOU_THRESHOLD;
}

function isDetectionInStrollerCorridor(detection: Detection, width: number, height: number): boolean {
  const centerX = detection.x + detection.width / 2;
  const centerY = detection.y + detection.height / 2;

  return (
    centerX >= width * STROLLER_CORRIDOR_X_MIN &&
    centerX <= width * STROLLER_CORRIDOR_X_MAX &&
    centerY >= height * STROLLER_CORRIDOR_Y_MIN
  );
}

function calculateStrollerCorridorCoverage(
  roadMask: RoadMask | null,
  sourceWidth: number,
  sourceHeight: number,
): number | null {
  if (!roadMask) {
    return null;
  }

  const corridorXMin = Math.floor(sourceWidth * STROLLER_CORRIDOR_X_MIN);
  const corridorXMax = Math.ceil(sourceWidth * STROLLER_CORRIDOR_X_MAX);
  const corridorYMin = Math.floor(sourceHeight * STROLLER_CORRIDOR_Y_MIN);
  const corridorYMax = sourceHeight;

  const xMin = Math.floor((corridorXMin / sourceWidth) * roadMask.width);
  const xMax = Math.ceil((corridorXMax / sourceWidth) * roadMask.width);
  const yMin = Math.floor((corridorYMin / sourceHeight) * roadMask.height);
  const yMax = Math.ceil((corridorYMax / sourceHeight) * roadMask.height);

  let walkable = 0;
  let total = 0;

  for (let y = yMin; y < yMax; y += 1) {
    if (y < 0 || y >= roadMask.height) {
      continue;
    }

    for (let x = xMin; x < xMax; x += 1) {
      if (x < 0 || x >= roadMask.width) {
        continue;
      }

      total += 1;
      if (roadMask.data[y * roadMask.width + x] === 1) {
        walkable += 1;
      }
    }
  }

  return total > 0 ? walkable / total : null;
}

function updateTrackingAndRisk(now: number, width: number, height: number): void {
  const primary = lastDetections[0] ?? null;
  if (!primary) {
    trackedDetection = null;
    stableFrames = 0;
  } else if (trackedDetection && matchDetection(trackedDetection, primary)) {
    trackedDetection = primary;
    stableFrames += 1;
  } else {
    trackedDetection = primary;
    stableFrames = 1;
  }

  const candidate = trackedDetection;
  const corridorCoverage = calculateStrollerCorridorCoverage(lastRoadMask, width, height);
  if (corridorCoverage !== null) {
    lastCorridorCoverage = corridorCoverage;
  }

  let candidateRisk: RiskState = "safe";

  if (candidate && candidate.confidence >= CONF_THRESHOLD) {
    const inCorridor = isDetectionInStrollerCorridor(candidate, width, height);
    const dangerInCorridor =
      inCorridor && (candidate.y + candidate.height >= height * STROLLER_CORRIDOR_BOTTOM_RATIO || detectionIsDanger(candidate, width, height));

    if (dangerInCorridor) {
      candidateRisk = "danger";
    } else if (corridorCoverage !== null) {
      if (corridorCoverage >= STROLLER_CORRIDOR_SAFE_COVERAGE) {
        candidateRisk = "safe";
      } else {
        candidateRisk = "warn";
      }
    } else if (isWithinRoi(candidate, width, height)) {
      if (detectionIsDanger(candidate, width, height)) {
        candidateRisk = "danger";
      } else if (stableFrames >= TRACK_STABLE_FRAMES) {
        candidateRisk = "warn";
      }
    }
  } else if (corridorCoverage !== null) {
    if (corridorCoverage >= STROLLER_CORRIDOR_SAFE_COVERAGE) {
      candidateRisk = "safe";
    } else {
      candidateRisk = "warn";
    }
  }

  if (corridorCoverage !== null && corridorCoverage < STROLLER_CORRIDOR_WARN_COVERAGE) {
    candidateRisk = candidateRisk === "danger" ? "danger" : "warn";
  }

  if (currentRiskState === "danger" && now < riskHoldUntil) {
    return;
  }

  if (candidateRisk === "danger") {
    currentRiskState = "danger";
    riskHoldUntil = now + DANGER_HOLD_MS;
    return;
  }

  if (currentRiskState === "danger" && now >= riskHoldUntil) {
    currentRiskState = candidateRisk === "warn" ? "warn" : "safe";
    riskHoldUntil = currentRiskState === "warn" ? now + WARN_HOLD_MS : 0;
    return;
  }

  if (candidateRisk === "warn") {
    currentRiskState = "warn";
    riskHoldUntil = now + WARN_HOLD_MS;
    return;
  }

  if (currentRiskState === "warn" && now < riskHoldUntil) {
    return;
  }

  currentRiskState = "safe";
  riskHoldUntil = 0;
}

function maybeSpeakRisk(now: number): void {
  if (currentRiskState === "warn") {
    if (now - lastWarnTtsAt >= WARN_TTS_INTERVAL_MS) {
      speak("전방 보행로에 주의가 필요합니다");
      lastWarnTtsAt = now;
    }
    return;
  }

  if (currentRiskState === "danger") {
    if (now - lastDangerTtsAt >= DANGER_TTS_INTERVAL_MS) {
      speak("위험, 전방 턱 또는 장애물을 주의하세요");
      lastDangerTtsAt = now;
    }
  }
}

function maskToRoadMask(mask: any): RoadMask | null {
  if (!mask) {
    return null;
  }

  const width = Number(mask.width ?? mask.size?.[0] ?? 0);
  const height = Number(mask.height ?? mask.size?.[1] ?? 0);
  const data = mask.data as ArrayLike<number> | undefined;
  if (!width || !height || !data) {
    return null;
  }

  const channels = Number(mask.channels ?? mask.numChannels ?? 1);
  const binary = new Uint8Array(width * height);

  for (let i = 0; i < binary.length; i += 1) {
    const raw = channels > 1 ? data[i * channels] ?? 0 : data[i] ?? 0;
    const normalized = raw > 1 ? raw / 255 : raw;
    binary[i] = normalized > 0.5 ? 1 : 0;
  }

  return { width, height, data: binary };
}

function maskToCanvas(roadMask: RoadMask): HTMLCanvasElement | null {
  const canvas = document.createElement("canvas");
  canvas.width = roadMask.width;
  canvas.height = roadMask.height;
  const ctx = canvas.getContext("2d");

  if (!ctx) {
    return null;
  }

  const imageData = ctx.createImageData(roadMask.width, roadMask.height);

  for (let i = 0; i < roadMask.data.length; i += 1) {
    const base = i * 4;
    imageData.data[base] = 74;
    imageData.data[base + 1] = 222;
    imageData.data[base + 2] = 128;
    imageData.data[base + 3] = roadMask.data[i] ? 84 : 0;
  }

  ctx.putImageData(imageData, 0, 0);
  return canvas;
}

function buildDepthPlaceholder(): DepthMap {
  const width = DEPTH_INPUT_W;
  const height = DEPTH_INPUT_H;
  const data = new Float32Array(width * height);
  const stepLine = Math.floor(height * DEPTH_PLACEHOLDER_STEP_Y_RATIO);
  const stepThickness = Math.max(1, Math.round(height * DEPTH_PLACEHOLDER_STEP_THICKNESS));

  for (let y = 0; y < height; y += 1) {
    const vertical = y / Math.max(1, height - 1);
    const stepBoost = y >= stepLine ? 0.35 : 0;

    for (let x = 0; x < width; x += 1) {
      const horizontal = (x / Math.max(1, width - 1) - 0.5) * 0.08;
      const inStepBand = y >= stepLine && y < stepLine + stepThickness;
      const bandBoost = inStepBand ? 0.18 : 0;
      data[y * width + x] = Math.max(0, Math.min(1, vertical + horizontal + stepBoost + bandBoost));
    }
  }

  return { width, height, data, source: "placeholder" };
}

function extractDepthMap(output: any): DepthMap | null {
  const candidates = Array.isArray(output) ? output : [output];

  for (const candidate of candidates) {
    const depthLike = candidate?.depth ?? candidate?.predicted_depth ?? candidate?.depth_map ?? candidate;
    const width = Number(depthLike?.width ?? depthLike?.size?.[0] ?? 0);
    const height = Number(depthLike?.height ?? depthLike?.size?.[1] ?? 0);
    const data = depthLike?.data as ArrayLike<number> | undefined;

    if (!width || !height || !data) {
      continue;
    }

    const depthData = new Float32Array(width * height);
    for (let i = 0; i < depthData.length; i += 1) {
      const value = Number(data[i] ?? 0);
      depthData[i] = value > 1 ? value / 255 : value;
    }

    return {
      width,
      height,
      data: depthData,
      source: "model",
    };
  }

  return null;
}

function detectStepCandidate(depthMap: DepthMap | null, sourceWidth: number, sourceHeight: number): StepCandidate | null {
  if (!depthMap) {
    return null;
  }

  const xMin = Math.floor(depthMap.width * STROLLER_CORRIDOR_X_MIN);
  const xMax = Math.ceil(depthMap.width * STROLLER_CORRIDOR_X_MAX);
  const yMin = Math.floor(depthMap.height * STROLLER_CORRIDOR_Y_MIN);
  const yMax = depthMap.height - 1;

  let bestRow = -1;
  let bestScore = 0;

  for (let y = yMin; y < yMax; y += 1) {
    let rowGradient = 0;
    let count = 0;

    for (let x = xMin; x < xMax; x += 1) {
      const current = depthMap.data[y * depthMap.width + x];
      const next = depthMap.data[(y + 1) * depthMap.width + x];
      rowGradient += Math.abs(next - current);
      count += 1;
    }

    const average = count > 0 ? rowGradient / count : 0;
    if (average > bestScore) {
      bestScore = average;
      bestRow = y;
    }
  }

  if (bestRow < 0 || bestScore < DEPTH_GRADIENT_THRESHOLD) {
    return null;
  }

  const frameX = sourceWidth * STROLLER_CORRIDOR_X_MIN;
  const frameWidth = sourceWidth * (STROLLER_CORRIDOR_X_MAX - STROLLER_CORRIDOR_X_MIN);
  const stepY = (bestRow / Math.max(1, depthMap.height)) * sourceHeight;
  if (stepY < sourceHeight * STEP_CANDIDATE_BOTTOM_RATIO) {
    return null;
  }
  const stepHeight = Math.max(16, sourceHeight * 0.07);

  return {
    x: frameX,
    y: Math.max(sourceHeight * STROLLER_CORRIDOR_Y_MIN, stepY - stepHeight / 2),
    width: frameWidth,
    height: stepHeight,
    score: bestScore,
    source: depthMap.source,
  };
}

function updateRoadSegMask(segments: any[]): void {
  const walkableSegment = segments.find((segment) => {
    const label = String(segment?.label ?? "").toLowerCase();
    return WALKABLE_CLASSES.includes(label as (typeof WALKABLE_CLASSES)[number]);
  });

  const roadSegment =
    walkableSegment ??
    segments.find((segment) => {
      const label = String(segment?.label ?? "").toLowerCase();
      return WALKABLE_CLASS_OPTIONS.includes(label as (typeof WALKABLE_CLASS_OPTIONS)[number]);
    });

  if (!roadSegment) {
    return;
  }

  const roadMask = maskToRoadMask(roadSegment.mask);
  if (!roadMask) {
    return;
  }

  const coverage = roadMask.data.length > 0 ? roadMask.data.reduce((sum, value) => sum + value, 0) / roadMask.data.length : 0;
  const canvas = maskToCanvas(roadMask);
  if (canvas) {
    lastRoadMask = roadMask;
    lastRoadMaskCanvas = canvas;
    lastRoadMaskCoverage = coverage;
  }
}

async function ensureModelLoad(): Promise<void> {
  if (modelLoadPromise || modelState === "ready" || modelState === "disabled") {
    return modelLoadPromise ?? Promise.resolve();
  }

  modelLoadPromise = (async () => {
    try {
      const available = await ensureModelAvailable();
      if (!available) {
        modelSession = null;
        modelState = "disabled";
        modelErrorMessage = "";
        updateModelUi();
        return;
      }

      modelState = "loading";
      modelErrorMessage = "";
      updateModelUi();

      modelSession = await ort.InferenceSession.create(MODEL_URL, {
        executionProviders: ["wasm"],
      });
      modelState = "ready";
      modelErrorMessage = "";
      updateModelUi();
    } catch (error) {
      modelSession = null;
      modelState = "error";
      modelErrorMessage =
        error instanceof Error ? `YOLO 모델 로딩 실패: ${error.message}` : `YOLO 모델 로딩 실패: ${String(error)}`;
      updateModelUi();
      console.error("model load failed", error);
    } finally {
      modelLoadPromise = null;
    }
  })();

  return modelLoadPromise;
}

async function ensureModelAvailable(): Promise<boolean> {
  if (modelAvailability !== null) {
    return modelAvailability;
  }

  if (modelCheckPromise) {
    return modelCheckPromise;
  }

  modelCheckPromise = (async () => {
    try {
      const response = await fetch(MODEL_URL, {
        method: "HEAD",
        cache: "no-store",
      });
      modelAvailability = response.ok;
      return modelAvailability;
    } catch (error) {
      console.warn("model existence check failed, switching to mock mode", error);
      modelAvailability = false;
      return false;
    } finally {
      modelCheckPromise = null;
    }
  })();

  return modelCheckPromise;
}

async function ensureRoadSegLoad(): Promise<void> {
  if (roadSegLoadPromise || roadSegState === "ready") {
    return roadSegLoadPromise ?? Promise.resolve();
  }

  roadSegState = "loading";
  roadSegErrorMessage = "";
  updateSegUi();

  roadSegLoadPromise = (async () => {
    const previousRemoteMode = env.allowRemoteModels;

    try {
      env.allowRemoteModels = false;
      segmenter = (await pipeline("image-segmentation", ROAD_SEG_LOCAL_MODEL_ID)) as RoadSegmenter;
      roadSegState = "ready";
      roadSegErrorMessage = "";
      updateSegUi();
      return;
    } catch (localError) {
      console.warn("road segmentation local model load failed", localError);
    } finally {
      env.allowRemoteModels = true;
    }

    try {
      segmenter = (await pipeline("image-segmentation", ROAD_SEG_REMOTE_MODEL_ID)) as RoadSegmenter;
      roadSegState = "ready";
      roadSegErrorMessage = "";
      updateSegUi();
    } catch (error) {
      segmenter = null;
      roadSegState = "error";
      roadSegErrorMessage =
        error instanceof Error
          ? `SegFormer 로딩 실패: ${error.message}`
          : `SegFormer 로딩 실패: ${String(error)}`;
      updateSegUi();
      console.error("road segmentation load failed", error);
    } finally {
      env.allowRemoteModels = previousRemoteMode;
      roadSegLoadPromise = null;
    }
  })();

  return roadSegLoadPromise;
}

async function ensureDepthLoad(): Promise<void> {
  if (depthLoadPromise || depthState === "ready") {
    return depthLoadPromise ?? Promise.resolve();
  }

  depthState = "loading";
  depthErrorMessage = "";
  updateDepthUi();

  depthLoadPromise = (async () => {
    try {
      depthEstimator = (await pipeline("depth-estimation", DEPTH_REMOTE_MODEL_ID)) as (image: RawImage) => Promise<any>;
      depthState = "ready";
      depthErrorMessage = "";
      updateDepthUi();
    } catch (error) {
      depthEstimator = null;
      depthErrorMessage =
        error instanceof Error ? `Depth 모델 로딩 실패: ${error.message}` : `Depth 모델 로딩 실패: ${String(error)}`;
      console.warn("depth module load failed, placeholder fallback enabled", error);
      depthState = "ready";
      updateDepthUi();
    } finally {
      depthLoadPromise = null;
    }
  })();

  return depthLoadPromise;
}

async function runInference(): Promise<void> {
  if (!modelSession || isInferBusy || !running) {
    return;
  }

  if (video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
    return;
  }

  isInferBusy = true;

  try {
    const { tensor, meta } = preprocessFrame();
    const inputName = modelSession.inputNames[0];
    const outputs = await modelSession.run({ [inputName]: tensor });
    const outputTensor = getOutputTensor(outputs);

    if (!outputTensor) {
      throw new Error("YOLO output tensor not found");
    }

    const result = postprocess(outputTensor, meta);
    lastDetections = result.length > 0 ? result : [...MOCK_DETECTIONS];
  } catch (error) {
    modelState = "error";
    modelErrorMessage =
      error instanceof Error ? `YOLO 추론 실패: ${error.message}` : `YOLO 추론 실패: ${String(error)}`;
    updateModelUi();
    console.error("inference failed", error);
  } finally {
    isInferBusy = false;
  }
}

async function runRoadSegmentation(): Promise<void> {
  if (!segmenter || roadSegBusy || !running) {
    return;
  }

  if (video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
    return;
  }

  const now = performance.now();
  if (now - lastSegAt < getRoadSegIntervalMs()) {
    return;
  }

  roadSegBusy = true;

  try {
    if (!roadSegInputCtx) {
      throw new Error("road segmentation canvas context not available");
    }

    roadSegInputCtx.drawImage(video, 0, 0, ROAD_SEG_INPUT_W, ROAD_SEG_INPUT_H);
    const rawImage = RawImage.fromCanvas(roadSegInputCanvas);
    const result = await segmenter(rawImage);
    const segments = Array.isArray(result) ? result : [result];
    updateRoadSegMask(segments);
    lastSegAt = now;
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

function getDepthIntervalMs(): number {
  return DEPTH_INTERVAL_MS;
}

async function runDepthEstimation(): Promise<void> {
  if (depthBusy || !running) {
    return;
  }

  if (video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
    return;
  }

  const now = performance.now();
  if (now - lastDepthAt < getDepthIntervalMs()) {
    return;
  }

  depthBusy = true;

  try {
    if (!depthInputCtx) {
      throw new Error("depth canvas context not available");
    }

    depthInputCtx.drawImage(video, 0, 0, DEPTH_INPUT_W, DEPTH_INPUT_H);

    let depthMap = buildDepthPlaceholder();

    if (depthEstimator) {
      try {
        const rawImage = RawImage.fromCanvas(depthInputCanvas);
        const output = await depthEstimator(rawImage);
        const extracted = extractDepthMap(output);
        if (extracted) {
          depthMap = extracted;
        }
      } catch (error) {
        depthErrorMessage =
          error instanceof Error ? `Depth 추론 실패: ${error.message}` : `Depth 추론 실패: ${String(error)}`;
        console.warn("depth inference failed, using placeholder", error);
      }
    }

    lastDepthMap = depthMap;
    lastStepCandidate = detectStepCandidate(depthMap, video.videoWidth || INPUT_SIZE, video.videoHeight || INPUT_SIZE);
    depthState = "ready";
    updateDepthUi();
    lastDepthAt = now;
  } catch (error) {
    depthErrorMessage =
      error instanceof Error ? `Depth placeholder 실패: ${error.message}` : `Depth placeholder 실패: ${String(error)}`;
    depthState = DEPTH_PLACEHOLDER_ENABLED ? "ready" : "error";
    lastDepthMap = DEPTH_PLACEHOLDER_ENABLED ? buildDepthPlaceholder() : null;
    lastStepCandidate = lastDepthMap
      ? detectStepCandidate(lastDepthMap, video.videoWidth || INPUT_SIZE, video.videoHeight || INPUT_SIZE)
      : null;
    updateDepthUi();
    console.error("depth module failed", error);
  } finally {
    depthBusy = false;
  }
}

function getInferIntervalFrames(): number {
  return lowPowerMode ? LOW_POWER_INFER_EVERY_N_FRAMES : NORMAL_INFER_EVERY_N_FRAMES;
}

function updateFpsAndPower(now: number): void {
  if (lastFrameAt > 0) {
    const instant = 1000 / Math.max(1, now - lastFrameAt);
    fps = fps === 0 ? instant : fps * 0.9 + instant * 0.1;
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

  fpsLabel.textContent = `FPS ${fps > 0 ? fps.toFixed(1) : "--"}`;
}

async function start(): Promise<void> {
  if (running) {
    return;
  }

  startButton.disabled = true;
  primeTts();
  setPillText(appState, "시작 중", "warn");
  setPillText(camState, "카메라 요청 중", "neutral");
  cameraLabel.textContent = "연결 중";
  ttsLabel.textContent = "준비 중";

  try {
    stream = await openCamera();
    video.srcObject = stream;
    await video.play();

    running = true;
    frameCount = 0;
    fps = 0;
    lastFrameAt = 0;
    lowPowerMode = false;
    lastDetections = [...MOCK_DETECTIONS];
    trackedDetection = null;
    stableFrames = 0;
    currentRiskState = "safe";
    riskHoldUntil = 0;
    lastWarnTtsAt = 0;
    lastDangerTtsAt = 0;
    lastDepthMap = null;
    lastStepCandidate = null;
    lastDepthAt = 0;
    lastRoadMask = null;
    lastSegAt = 0;

    updateModelUi();
    updateSegUi();
    updateDepthUi();
    updateRiskUi();
    updatePowerUi();
    updateCommonUi();
    cameraLabel.textContent = "실행 중";
    ttsLabel.textContent = hasPrimedTts ? "준비 완료" : "비활성";

    syncCanvasSize();
    if (typeof ResizeObserver !== "undefined") {
      resizeObserver = new ResizeObserver(() => {
        syncCanvasSize();
      });
      resizeObserver.observe(video);
    }

    void ensureModelLoad();
    void ensureRoadSegLoad();
    void ensureDepthLoad();
    rafId = window.requestAnimationFrame(renderLoop);
  } catch (error) {
    console.error("camera start failed", error);
    stop();
    const message = error instanceof Error ? error.message : String(error);
    cameraLabel.textContent = "실행 실패";
    setPillText(appState, "시작 실패", "bad");
    setPillText(camState, `오류: ${message}`, "bad");
  } finally {
    startButton.disabled = false;
  }
}

function stop(): void {
  running = false;
  cancelAnimationFrame(rafId);
  resizeObserver?.disconnect();
  resizeObserver = null;

  window.speechSynthesis?.cancel?.();

  const currentStream = stream;
  stream = null;
  currentStream?.getTracks().forEach((track) => track.stop());
  video.srcObject = null;
  clearOverlay();
  lowPowerMode = false;
  lastDepthMap = null;
  lastStepCandidate = null;
  lastDepthAt = 0;
  lastRoadMask = null;
  lastRoadMaskCanvas = null;
  overlayBadge.textContent = MOCK_OVERLAY_BADGE;
  lastDetections = [...MOCK_DETECTIONS];

  setStoppedState();
}

function renderLoop(now: number): void {
  if (!running) {
    return;
  }

  updateFpsAndPower(now);

  frameCount += 1;
  if (frameCount % getInferIntervalFrames() === 0) {
    void runInference();
  }

  if (roadSegState === "ready") {
    void runRoadSegmentation();
  } else if (roadSegState === "idle") {
    void ensureRoadSegLoad();
  }

  if (depthState === "ready") {
    void runDepthEstimation();
  } else if (depthState === "idle") {
    void ensureDepthLoad();
  }

  const sourceWidth = video.videoWidth || INPUT_SIZE;
  const sourceHeight = video.videoHeight || INPUT_SIZE;
  updateTrackingAndRisk(now, sourceWidth, sourceHeight);
  renderOverlayFrame(sourceWidth, sourceHeight);
  updateDetectionsUi(lastDetections);
  updateSegUi();
  updateDepthUi();
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
  speak("전방 보행로에 주의가 필요합니다");
});

window.addEventListener("resize", () => {
  syncCanvasSize();
});

window.addEventListener("beforeunload", () => {
  stop();
});

updateModelUi();
updateSegUi();
updateDepthUi();
updateRiskUi();
updatePowerUi();
setStoppedState();

