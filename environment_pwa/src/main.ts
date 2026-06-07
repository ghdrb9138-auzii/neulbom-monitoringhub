import "./style.css";
import { registerSW } from "virtual:pwa-register";
import * as ort from "onnxruntime-web";

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

type ModelState = "idle" | "loading" | "ready" | "error";
type RiskState = "safe" | "warn" | "danger";
type PowerMode = "normal" | "low";

const MODEL_URL = "/models/road_obstacle_yolov8n_320.onnx";
const INPUT_SIZE = 320;
const CONF_THRESHOLD = 0.45;
const NMS_THRESHOLD = 0.45;
const TARGET_CLASSES = [
  "pothole",
  "speed_bump",
  "road_cone",
  "debris",
  "unknown_obstacle",
] as const;

const MOCK_DETECTIONS: Detection[] = [
  {
    classId: 0,
    label: "pothole",
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
          <h1>늘봄 도로 위험 감지 PWA</h1>
          <p class="lede">
            Start 버튼을 누르면 후면 카메라를 우선 사용하고, PC에서는 일반 webcam으로 fallback 합니다.
            YOLOv8 Nano ONNX가 없더라도 앱은 유지되고, 상태와 에러는 UI에 표시됩니다.
          </p>
        </div>
        <div class="status-stack">
          <span class="pill pill-neutral" id="app-state">대기 중</span>
          <span class="pill pill-neutral" id="cam-state">카메라 미실행</span>
          <span class="pill pill-neutral" id="model-state">모델 미실행</span>
          <span class="pill pill-neutral" id="risk-state">safe</span>
          <span class="pill pill-neutral" id="power-state">normal</span>
          <span class="pill pill-neutral" id="tts-state">TTS 비활성</span>
        </div>
      </div>

      <div class="video-frame">
        <video id="video" autoplay muted playsinline webkit-playsinline></video>
        <canvas id="overlay"></canvas>
        <div class="overlay-badge" id="overlay-badge">mock detection</div>
      </div>

      <div id="model-error" class="error-banner hidden" role="status" aria-live="polite"></div>
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
        <div class="card-note">권장 해상도 960x540, frameRate 최대 30</div>
      </article>

      <article class="card">
        <div class="card-label">YOLO 모델</div>
        <div class="card-value" id="model-label">대기</div>
        <div class="card-note" id="model-note">/public/models/road_obstacle_yolov8n_320.onnx</div>
      </article>

      <article class="card">
        <div class="card-label">Risk</div>
        <div class="card-value" id="risk-label">safe</div>
        <div class="card-note" id="risk-note">주행 ROI와 유지 프레임 조건을 기준으로 판단</div>
      </article>

      <article class="card">
        <div class="card-label">Performance</div>
        <div class="card-value" id="fps-label">FPS --</div>
        <div class="card-note" id="power-note">저전력 모드 비활성</div>
      </article>

      <article class="card">
        <div class="card-label">Detection</div>
        <div class="card-value" id="detection-label">pothole 0.82</div>
        <div class="card-note">저전력 시 디버그 오버레이를 줄이고 3프레임에 1번 추론합니다</div>
      </article>

      <article class="card">
        <div class="card-label">TTS</div>
        <div class="card-value" id="tts-label">준비 필요</div>
        <div class="card-note">warn/danger 상태별 cooldown과 반복 간격을 적용합니다</div>
      </article>
    </section>

    <section class="details">
      <div><span>패키지명</span><code>neulbom-environment-pwa</code></div>
      <div><span>개발 포트</span><code>5176</code></div>
      <div><span>상태</span><code>카메라 + Canvas + TTS + YOLO ONNX</code></div>
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
const riskStateEl = document.getElementById("risk-state") as HTMLSpanElement;
const powerStateEl = document.getElementById("power-state") as HTMLSpanElement;
const ttsState = document.getElementById("tts-state") as HTMLSpanElement;
const cameraLabel = document.getElementById("camera-label") as HTMLDivElement;
const modelLabel = document.getElementById("model-label") as HTMLDivElement;
const modelNote = document.getElementById("model-note") as HTMLDivElement;
const riskLabel = document.getElementById("risk-label") as HTMLDivElement;
const riskNote = document.getElementById("risk-note") as HTMLDivElement;
const fpsLabel = document.getElementById("fps-label") as HTMLDivElement;
const powerNote = document.getElementById("power-note") as HTMLDivElement;
const detectionLabel = document.getElementById("detection-label") as HTMLDivElement;
const ttsLabel = document.getElementById("tts-label") as HTMLDivElement;
const modelError = document.getElementById("model-error") as HTMLDivElement;

const overlayCtx = overlay.getContext("2d");
const preprocessCanvas = document.createElement("canvas");
preprocessCanvas.width = INPUT_SIZE;
preprocessCanvas.height = INPUT_SIZE;
const preprocessCtx = preprocessCanvas.getContext("2d", { willReadFrequently: true });

registerSW({ immediate: true });

ort.env.wasm.numThreads = 1;
ort.env.wasm.proxy = false;
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";

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

function setStateChip(el: HTMLSpanElement, state: RiskState | PowerMode | ModelState | string): void {
  el.classList.remove("pill-success", "pill-neutral", "pill-warn", "pill-bad");
  if (state === "danger" || state === "error") {
    el.classList.add("pill-bad");
    return;
  }
  if (state === "warn" || state === "loading") {
    el.classList.add("pill-warn");
    return;
  }
  if (state === "safe" || state === "ready" || state === "normal") {
    el.classList.add("pill-success");
    return;
  }
  el.classList.add("pill-neutral");
}

function showModelError(message: string): void {
  modelErrorMessage = message;
  modelError.textContent = message;
  modelError.classList.remove("hidden");
}

function hideModelError(): void {
  modelErrorMessage = "";
  modelError.textContent = "";
  modelError.classList.add("hidden");
}

function updateModelUi(): void {
  if (modelState === "idle") {
    setPillText(modelStateEl, "모델 미실행", "neutral");
    modelLabel.textContent = "대기";
    modelNote.textContent = "모델 로딩 전";
    hideModelError();
    return;
  }

  if (modelState === "loading") {
    setPillText(modelStateEl, "모델 로딩 중", "warn");
    modelLabel.textContent = "로딩 중";
    modelNote.textContent = MODEL_URL;
    hideModelError();
    return;
  }

  if (modelState === "ready") {
    setPillText(modelStateEl, "모델 준비 완료", "success");
    modelLabel.textContent = "준비 완료";
    modelNote.textContent = "YOLOv8 Nano ONNX 추론 가능";
    hideModelError();
    return;
  }

  setPillText(modelStateEl, "모델 오류", "bad");
  modelLabel.textContent = "오류";
  modelNote.textContent = modelErrorMessage || "모델 로딩 또는 추론 실패";
  showModelError(modelErrorMessage || "모델 로딩 실패");
}

function updateRiskUi(): void {
  riskLabel.textContent = currentRiskState;
  setPillText(riskStateEl, currentRiskState, currentRiskState === "safe" ? "neutral" : currentRiskState === "warn" ? "warn" : "bad");
  setStateChip(riskStateEl, currentRiskState);

  if (currentRiskState === "safe") {
    riskNote.textContent = "주행 ROI 밖이거나 유지/거리 조건이 충족되지 않았습니다";
  } else if (currentRiskState === "warn") {
    riskNote.textContent = "같은 객체가 3프레임 이상 유지되었습니다";
  } else {
    riskNote.textContent = "하단 침투 또는 큰 면적 조건으로 위험 상태를 유지합니다";
  }
}

function updatePowerUi(): void {
  setPillText(powerStateEl, lowPowerMode ? "low power" : "normal", lowPowerMode ? "warn" : "success");
  setStateChip(powerStateEl, lowPowerMode ? "warn" : "normal");
  powerNote.textContent = lowPowerMode ? "FPS 저하 감지, 3프레임 1회 추론 / 디버그 축소" : "정상 모드";
}

function updateCommonUi(): void {
  setPillText(appState, running ? "실행 중" : "대기 중", running ? "success" : "neutral");
  setPillText(camState, stream ? "카메라 실행됨" : "카메라 미실행", stream ? "success" : "neutral");
  setPillText(ttsState, hasPrimedTts ? "TTS 준비 완료" : "TTS 비활성", hasPrimedTts ? "success" : "neutral");
  detectionLabel.textContent =
    lastDetections.length > 0
      ? `${lastDetections[0].label} ${lastDetections[0].confidence.toFixed(2)}`
      : "no detection";
  fpsLabel.textContent = `FPS ${fps > 0 ? fps.toFixed(1) : "--"}`;
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

function fitContain(sourceWidth: number, sourceHeight: number, viewWidth: number, viewHeight: number) {
  const scale = Math.min(viewWidth / sourceWidth, viewHeight / sourceHeight);
  const width = sourceWidth * scale;
  const height = sourceHeight * scale;
  return {
    scale,
    x: (viewWidth - width) / 2,
    y: (viewHeight - height) / 2,
  };
}

function drawDetections(detections: Detection[], sourceWidth: number, sourceHeight: number): void {
  if (!overlayCtx || sourceWidth <= 0 || sourceHeight <= 0) {
    return;
  }

  const rect = video.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  syncCanvasSize();

  const frame = fitContain(sourceWidth, sourceHeight, rect.width, rect.height);
  overlayCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
  overlayCtx.clearRect(0, 0, rect.width, rect.height);

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
    ? `low power · ${currentRiskState}`
    : primary
      ? `${primary.label} ${primary.confidence.toFixed(2)} · ${currentRiskState}`
      : currentRiskState;
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
    ttsLabel.textContent = "알림 재생 중";
    setPillText(ttsState, "TTS 활성", "success");
  };
  utterance.onend = () => {
    ttsLabel.textContent = "대기 중";
    setPillText(ttsState, "TTS 활성", "success");
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
    throw new Error("이 브라우저는 카메라를 지원하지 않습니다.");
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

  throw lastError instanceof Error ? lastError : new Error("카메라를 열 수 없습니다.");
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

function normalizeBox(
  box: { cx: number; cy: number; width: number; height: number },
): { cx: number; cy: number; width: number; height: number } {
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
      const score = candidate.scores[i] >= 0 && candidate.scores[i] <= 1 ? candidate.scores[i] : sigmoid(candidate.scores[i]);
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
    detections.length > 0
      ? `${detections[0].label} ${detections[0].confidence.toFixed(2)}`
      : "no detection";
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
  return bottom >= height * DANGER_BOTTOM_RATIO || detectionAreaRatio(detection, width, height) >= DANGER_AREA_RATIO;
}

function matchDetection(a: Detection, b: Detection): boolean {
  return a.label === b.label && iou(a, b) >= MATCH_IOU_THRESHOLD;
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

  let candidateRisk: RiskState = "safe";
  const candidate = trackedDetection;

  if (candidate && candidate.confidence >= CONF_THRESHOLD && isWithinRoi(candidate, width, height)) {
    if (detectionIsDanger(candidate, width, height)) {
      candidateRisk = "danger";
    } else if (stableFrames >= TRACK_STABLE_FRAMES) {
      candidateRisk = "warn";
    }
  }

  if (currentRiskState === "danger" && now < riskHoldUntil) {
    return;
  }

  if (candidateRisk === "danger") {
    if (currentRiskState !== "danger") {
      currentRiskState = "danger";
    }
    riskHoldUntil = now + DANGER_HOLD_MS;
    return;
  }

  if (currentRiskState === "danger" && now >= riskHoldUntil) {
    currentRiskState = candidateRisk === "warn" ? "warn" : "safe";
    riskHoldUntil = currentRiskState === "warn" ? now + WARN_HOLD_MS : 0;
    return;
  }

  if (candidateRisk === "warn") {
    if (currentRiskState !== "warn") {
      currentRiskState = "warn";
    }
    riskHoldUntil = now + WARN_HOLD_MS;
    return;
  }

  if (currentRiskState === "warn" && now < riskHoldUntil) {
    return;
  }

  if (currentRiskState !== "safe") {
    currentRiskState = "safe";
  }
  riskHoldUntil = 0;
}

function maybeSpeakRisk(now: number): void {
  if (currentRiskState === "warn") {
    if (now - lastWarnTtsAt >= WARN_TTS_INTERVAL_MS) {
      speak("전방 도로 장애물이 감지되었습니다");
      lastWarnTtsAt = now;
    }
    return;
  }

  if (currentRiskState === "danger") {
    if (now - lastDangerTtsAt >= DANGER_TTS_INTERVAL_MS) {
      speak("위험, 전방 장애물을 주의하세요");
      lastDangerTtsAt = now;
    }
  }
}

async function ensureModelLoad(): Promise<void> {
  if (modelLoadPromise || modelState === "ready") {
    return modelLoadPromise ?? Promise.resolve();
  }

  modelState = "loading";
  modelErrorMessage = "";
  updateModelUi();

  modelLoadPromise = (async () => {
    try {
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
        error instanceof Error ? `모델 로딩 실패: ${error.message}` : `모델 로딩 실패: ${String(error)}`;
      updateModelUi();
      console.error("model load failed", error);
    } finally {
      modelLoadPromise = null;
    }
  })();

  return modelLoadPromise;
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
      error instanceof Error ? `추론 실패: ${error.message}` : `추론 실패: ${String(error)}`;
    updateModelUi();
    console.error("inference failed", error);
  } finally {
    isInferBusy = false;
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
  ttsLabel.textContent = "활성화 준비";

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

    updateModelUi();
    updateRiskUi();
    updatePowerUi();
    updateCommonUi();
    cameraLabel.textContent = "카메라 실행됨";
    ttsLabel.textContent = hasPrimedTts ? "준비 완료" : "비활성";

    syncCanvasSize();
    if (typeof ResizeObserver !== "undefined") {
      resizeObserver = new ResizeObserver(() => {
        syncCanvasSize();
      });
      resizeObserver.observe(video);
    }

    void ensureModelLoad();
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
  overlayBadge.textContent = "mock detection";
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

  const sourceWidth = video.videoWidth || INPUT_SIZE;
  const sourceHeight = video.videoHeight || INPUT_SIZE;
  updateTrackingAndRisk(now, sourceWidth, sourceHeight);
  drawDetections(lastDetections, sourceWidth, sourceHeight);
  updateDetectionsUi(lastDetections);
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
  speak("전방 도로 장애물이 감지되었습니다");
});

window.addEventListener("resize", () => {
  syncCanvasSize();
});

window.addEventListener("beforeunload", () => {
  stop();
});

updateModelUi();
updateRiskUi();
updatePowerUi();
setStoppedState();
