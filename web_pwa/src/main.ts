import {
  FaceLandmarker,
  FilesetResolver,
  type FaceLandmarkerResult,
} from "@mediapipe/tasks-vision";
import { registerSW } from "virtual:pwa-register";

import { computeBothEAR, type Point2D } from "./ear";
import { decomposeEuler, projectAxes } from "./headPose";
import { applyAlarmMask, drawEyes, drawHeadAxes } from "./overlay";
import {
  DEFAULT_CONFIG,
  Detector,
  type AlarmLevel,
  type BentDir,
  type DetectorConfig,
} from "./stateMachine";
import { getRepeatMs, isMuted, setMuted, speakAlarm, unlock } from "./tts";

registerSW({ immediate: true });

const WASM_BASE =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm";
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";

const CONFIG_STORAGE_KEY = "neulbom.config.v1";
const VIS_STORAGE_KEY = "neulbom.debugVis.v1";
const NOSE_TIP_IDX = 1;
const AXIS_LEN_PX = 80;

// ───────────────────────── DOM ─────────────────────────

const video = document.getElementById("video") as HTMLVideoElement;
const canvas = document.getElementById("canvas") as HTMLCanvasElement;
const intro = document.getElementById("intro") as HTMLDivElement;
const startBtn = document.getElementById("start") as HTMLButtonElement;
const help = document.getElementById("help") as HTMLDivElement;
const ctx = canvas.getContext("2d")!;

const panel = document.getElementById("panel") as HTMLElement;
const tunePanel = document.getElementById("tune") as HTMLElement;
const muteBtn = document.getElementById("btn-mute") as HTMLButtonElement;
const visBtn = document.getElementById("btn-vis") as HTMLButtonElement;
const tuneBtn = document.getElementById("btn-tune") as HTMLButtonElement;
const stopBtn = document.getElementById("btn-stop") as HTMLButtonElement;
const resetBtn = document.getElementById("btn-reset") as HTMLButtonElement;

const mState = document.getElementById("m-state") as HTMLSpanElement;
const mFps = document.getElementById("m-fps") as HTMLSpanElement;
const mEar = document.getElementById("m-ear") as HTMLSpanElement;
const mPitch = document.getElementById("m-pitch") as HTMLSpanElement;
const mYaw = document.getElementById("m-yaw") as HTMLSpanElement;
const mRoll = document.getElementById("m-roll") as HTMLSpanElement;
const mClosed = document.getElementById("m-closed") as HTMLSpanElement;
const mBent = document.getElementById("m-bent") as HTMLSpanElement;
const mEye = document.getElementById("m-eye") as HTMLSpanElement;
const mHead = document.getElementById("m-head") as HTMLSpanElement;

interface ParamControl {
  slider: HTMLInputElement;
  value: HTMLSpanElement;
  format: (n: number) => string;
  apply: (n: number) => void;
}

const detector = new Detector(loadConfig());

const PARAM_CONTROLS: Record<keyof DetectorConfig, ParamControl> = {
  earThreshold: {
    slider: document.getElementById("s-ear") as HTMLInputElement,
    value: document.getElementById("v-ear") as HTMLSpanElement,
    format: (n) => n.toFixed(2),
    apply: (n) => {
      detector.config.earThreshold = n;
    },
  },
  drowsyFrames: {
    slider: document.getElementById("s-drowsy-frames") as HTMLInputElement,
    value: document.getElementById("v-drowsy-frames") as HTMLSpanElement,
    format: (n) => `${n}`,
    apply: (n) => {
      detector.config.drowsyFrames = n;
    },
  },
  pitchThresholdDeg: {
    slider: document.getElementById("s-pitch") as HTMLInputElement,
    value: document.getElementById("v-pitch") as HTMLSpanElement,
    format: (n) => `${n.toFixed(0)}°`,
    apply: (n) => {
      detector.config.pitchThresholdDeg = n;
    },
  },
  rollThresholdDeg: {
    slider: document.getElementById("s-roll") as HTMLInputElement,
    value: document.getElementById("v-roll") as HTMLSpanElement,
    format: (n) => `${n.toFixed(0)}°`,
    apply: (n) => {
      detector.config.rollThresholdDeg = n;
    },
  },
  bentFrames: {
    slider: document.getElementById("s-bent-frames") as HTMLInputElement,
    value: document.getElementById("v-bent-frames") as HTMLSpanElement,
    format: (n) => `${n}`,
    apply: (n) => {
      detector.config.bentFrames = n;
    },
  },
};

// ──────────────────────── State ────────────────────────

let faceLandmarker: FaceLandmarker | null = null;
let fps = 0;
let prevTime = performance.now();
let running = false;

let lastAlarmLevel: AlarmLevel = "none";
let lastAlarmAt = 0;
let debugVis = loadDebugVis();

// ───────────────────── Persistence ─────────────────────

function loadConfig(): DetectorConfig {
  try {
    const raw = localStorage.getItem(CONFIG_STORAGE_KEY);
    if (!raw) return { ...DEFAULT_CONFIG };
    const parsed = JSON.parse(raw) as Partial<DetectorConfig>;
    return { ...DEFAULT_CONFIG, ...parsed };
  } catch {
    return { ...DEFAULT_CONFIG };
  }
}

function saveConfig(): void {
  try {
    localStorage.setItem(CONFIG_STORAGE_KEY, JSON.stringify(detector.config));
  } catch {
    // ignore (private mode / quota)
  }
}

function clearStoredConfig(): void {
  try {
    localStorage.removeItem(CONFIG_STORAGE_KEY);
  } catch {
    // ignore
  }
}

function loadDebugVis(): boolean {
  try {
    const v = localStorage.getItem(VIS_STORAGE_KEY);
    if (v === null) return true; // default: ON
    return v === "1";
  } catch {
    return true;
  }
}

function saveDebugVis(): void {
  try {
    localStorage.setItem(VIS_STORAGE_KEY, debugVis ? "1" : "0");
  } catch {
    // ignore
  }
}

// ─────────────────────── Alarm ─────────────────────────

function maybeSpeakAlarm(level: AlarmLevel, now: number): void {
  if (level === "none") {
    lastAlarmLevel = "none";
    return;
  }
  const isNewLevel = level !== lastAlarmLevel;
  const elapsed = now - lastAlarmAt;
  if (isNewLevel || elapsed >= getRepeatMs(level)) {
    speakAlarm(level);
    lastAlarmAt = now;
    lastAlarmLevel = level;
  }
}

// ──────────────────── Panel rendering ──────────────────

function renderMuteButton(): void {
  const muted = isMuted();
  muteBtn.textContent = muted ? "🔇 음성 OFF" : "🔊 음성 ON";
  muteBtn.classList.toggle("muted", muted);
}

function renderTuneButton(): void {
  const open = tunePanel.classList.contains("visible");
  tuneBtn.textContent = open ? "⚙ 파라미터 닫기" : "⚙ 파라미터";
  tuneBtn.classList.toggle("tune-open", open);
}

function renderVisButton(): void {
  visBtn.textContent = debugVis ? "📐 시각화 ON" : "📐 시각화 OFF";
  visBtn.classList.toggle("vis-off", !debugVis);
}

function renderSlider(key: keyof DetectorConfig): void {
  const ctrl = PARAM_CONTROLS[key];
  const n = detector.config[key];
  ctrl.slider.value = String(n);
  ctrl.value.textContent = ctrl.format(n);
}

function renderAllSliders(): void {
  (Object.keys(PARAM_CONTROLS) as Array<keyof DetectorConfig>).forEach(renderSlider);
}

interface PanelData {
  state: string;
  fps: number;
  ear: number;
  pitch: number;
  yaw: number;
  roll: number;
  closedFrames: number;
  headBentFrames: number;
  eye: "open" | "closed" | "—";
  head: "normal" | BentDir | "—";
}

function setChip(el: HTMLElement, label: string, kind: "good" | "warn" | "bad" | "neutral"): void {
  el.textContent = label;
  el.classList.toggle("good", kind === "good");
  el.classList.toggle("warn", kind === "warn");
  el.classList.toggle("bad", kind === "bad");
}

function renderPanel(d: PanelData): void {
  mState.textContent = d.state;
  mFps.textContent = `${d.fps.toFixed(1)} FPS`;
  mEar.textContent = d.ear.toFixed(3);
  mPitch.textContent = `${d.pitch.toFixed(1)}°`;
  mYaw.textContent = `${d.yaw.toFixed(1)}°`;
  mRoll.textContent = `${d.roll.toFixed(1)}°`;
  mClosed.textContent = `${d.closedFrames}`;
  mBent.textContent = `${d.headBentFrames}`;

  if (d.eye === "open") setChip(mEye, "Eye: OPEN", "good");
  else if (d.eye === "closed") setChip(mEye, "Eye: CLOSED", "warn");
  else setChip(mEye, "Eye: —", "neutral");

  if (d.head === "normal") setChip(mHead, "Head: NORMAL", "good");
  else if (d.head === "FWD") setChip(mHead, "Head: FWD", "warn");
  else if (d.head === "SIDE") setChip(mHead, "Head: SIDE", "warn");
  else if (d.head === "DIAG") setChip(mHead, "Head: DIAG", "warn");
  else setChip(mHead, "Head: —", "neutral");
}

// ──────────────────── MediaPipe / camera ───────────────

async function createLandmarker(): Promise<FaceLandmarker> {
  const fileset = await FilesetResolver.forVisionTasks(WASM_BASE);
  return FaceLandmarker.createFromOptions(fileset, {
    baseOptions: {
      modelAssetPath: MODEL_URL,
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numFaces: 1,
    outputFaceBlendshapes: false,
    outputFacialTransformationMatrixes: true,
  });
}

async function startCamera(): Promise<MediaStream> {
  return navigator.mediaDevices.getUserMedia({
    video: {
      facingMode: "user",
      width: { ideal: 1280 },
      height: { ideal: 720 },
    },
    audio: false,
  });
}

function resizeCanvas(): void {
  canvas.width = video.videoWidth || 640;
  canvas.height = video.videoHeight || 480;
}

// ─────────────────────── Render loop ───────────────────

function renderFrame(timestampMs: number): void {
  if (!faceLandmarker || !running) return;

  const now = performance.now();
  const dt = (now - prevTime) / 1000;
  if (dt > 0) fps = 0.9 * fps + 0.1 * (1 / dt);
  prevTime = now;

  const result: FaceLandmarkerResult = faceLandmarker.detectForVideo(
    video,
    timestampMs,
  );

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  if (result.faceLandmarks.length > 0) {
    const landmarks = result.faceLandmarks[0] as readonly Point2D[];
    const { avg: ear } = computeBothEAR(landmarks, canvas.width, canvas.height);

    let pitch = 0;
    let yaw = 0;
    let roll = 0;
    const matrix = result.facialTransformationMatrixes?.[0];
    if (matrix?.data) {
      ({ pitch, yaw, roll } = decomposeEuler(matrix.data));
    }

    const out = detector.step({ ear, pitch, roll });
    maybeSpeakAlarm(out.alarm, now);
    if (debugVis) {
      drawEyes(ctx, landmarks, canvas.width, canvas.height, out.isClosed);
      if (matrix?.data) {
        const nose = landmarks[NOSE_TIP_IDX];
        const origin = {
          x: nose.x * canvas.width,
          y: nose.y * canvas.height,
        };
        drawHeadAxes(ctx, projectAxes(matrix.data, origin, AXIS_LEN_PX));
      }
    }
    applyAlarmMask(ctx, canvas.width, canvas.height, out.alarm, now);
    renderPanel({
      state: out.state,
      fps,
      ear,
      pitch,
      yaw,
      roll,
      closedFrames: out.closedFrames,
      headBentFrames: out.headBentFrames,
      eye: out.isClosed ? "closed" : "open",
      head: out.isHeadBent ? (out.bentDir || "FWD") : "normal",
    });
  } else {
    const out = detector.noFace();
    maybeSpeakAlarm(out.alarm, now);
    renderPanel({
      state: out.state,
      fps,
      ear: 0,
      pitch: 0,
      yaw: 0,
      roll: 0,
      closedFrames: 0,
      headBentFrames: 0,
      eye: "—",
      head: "—",
    });
  }

  scheduleNextFrame();
}

function scheduleNextFrame(): void {
  if (!running) return;
  const v = video as HTMLVideoElement & {
    requestVideoFrameCallback?: (
      cb: (now: number, meta: { mediaTime: number }) => void,
    ) => number;
  };
  if (typeof v.requestVideoFrameCallback === "function") {
    v.requestVideoFrameCallback((now) => renderFrame(now));
  } else {
    requestAnimationFrame((now) => renderFrame(now));
  }
}

// ─────────────────────── Startup ───────────────────────

function stopDetection(): void {
  running = false;
  window.speechSynthesis.cancel();
  detector.reset();
  lastAlarmLevel = "none";
  lastAlarmAt = 0;
  fps = 0;

  const stream = video.srcObject as MediaStream | null;
  stream?.getTracks().forEach((t) => t.stop());
  video.srcObject = null;

  faceLandmarker?.close();
  faceLandmarker = null;

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  panel.classList.remove("visible");
  tunePanel.classList.remove("visible");
  renderTuneButton();

  intro.classList.remove("hidden");
  startBtn.disabled = false;
  startBtn.textContent = "Start";
  help.textContent = "카메라 권한 요청 팝업이 뜨면 '허용'해주세요!";
}

async function start(): Promise<void> {
  startBtn.disabled = true;
  startBtn.textContent = "Loading…";
  unlock();
  try {
    const stream = await startCamera();
    video.srcObject = stream;
    await new Promise<void>((resolve, reject) => {
      video.onloadedmetadata = () => resolve();
      video.onerror = () => reject(new Error("video element error"));
    });
    await video.play();
    resizeCanvas();
    faceLandmarker = await createLandmarker();
    running = true;
    intro.classList.add("hidden");
    panel.classList.add("visible");
    scheduleNextFrame();
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    console.error("start() failed", e);
    startBtn.disabled = false;
    startBtn.textContent = "Retry";
    help.textContent = `오류: ${msg}`;
  }
}

// ─────────────────────── Listeners ─────────────────────

startBtn.addEventListener("click", () => {
  void start();
});

muteBtn.addEventListener("click", () => {
  setMuted(!isMuted());
  renderMuteButton();
});

visBtn.addEventListener("click", () => {
  debugVis = !debugVis;
  saveDebugVis();
  renderVisButton();
});

tuneBtn.addEventListener("click", () => {
  tunePanel.classList.toggle("visible");
  renderTuneButton();
});

(Object.keys(PARAM_CONTROLS) as Array<keyof DetectorConfig>).forEach((key) => {
  const ctrl = PARAM_CONTROLS[key];
  ctrl.slider.addEventListener("input", () => {
    const n = parseFloat(ctrl.slider.value);
    ctrl.apply(n);
    ctrl.value.textContent = ctrl.format(n);
    saveConfig();
  });
});

resetBtn.addEventListener("click", () => {
  detector.config = { ...DEFAULT_CONFIG };
  clearStoredConfig();
  renderAllSliders();
});

stopBtn.addEventListener("click", () => {
  stopDetection();
});

window.addEventListener("resize", resizeCanvas);

window.addEventListener("beforeunload", () => {
  running = false;
  const stream = video.srcObject as MediaStream | null;
  stream?.getTracks().forEach((t) => t.stop());
  faceLandmarker?.close();
});

// ─────────────────────── Init ──────────────────────────

renderMuteButton();
renderVisButton();
renderTuneButton();
renderAllSliders();
