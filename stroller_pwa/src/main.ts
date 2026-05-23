import { type PoseLandmarker } from "@mediapipe/tasks-vision";

import { ApproachAnalyzer } from "./approach";
import { createHazardDetector, toHazardDetections } from "./detector";
import { applyAlarmMask, drawTracks } from "./overlay";
import {
  DEFAULT_HAZARD_CONFIG,
  HazardJudge,
  type HazardConfig,
  type HazardLevel,
  type TrackHazard,
} from "./stateMachine";
import { IouTracker } from "./tracker";
import { getRepeatMs, isMuted, setMuted, speakAlarm, unlock } from "./tts";

const CONFIG_STORAGE_KEY = "neulbom.stroller.config.v1";
const VIS_STORAGE_KEY = "neulbom.stroller.debugVis.v1";

const video = document.getElementById("video") as HTMLVideoElement;
const canvas = document.getElementById("canvas") as HTMLCanvasElement;
const intro = document.getElementById("intro") as HTMLDivElement;
const startBtn = document.getElementById("start") as HTMLButtonElement;
const help = document.getElementById("help") as HTMLDivElement;
const panel = document.getElementById("panel") as HTMLElement;
const mState = document.getElementById("m-state") as HTMLSpanElement;
const mFps = document.getElementById("m-fps") as HTMLSpanElement;
const mCount = document.getElementById("m-count") as HTMLSpanElement;
const mArea = document.getElementById("m-area") as HTMLSpanElement;
const mGrowth = document.getElementById("m-growth") as HTMLSpanElement;
const muteBtn = document.getElementById("btn-mute") as HTMLButtonElement;
const visBtn = document.getElementById("btn-vis") as HTMLButtonElement;
const tuneBtn = document.getElementById("btn-tune") as HTMLButtonElement;
const tunePanel = document.getElementById("tune") as HTMLElement;
const resetBtn = document.getElementById("btn-reset") as HTMLButtonElement;
const stopBtn = document.getElementById("btn-stop") as HTMLButtonElement;
const ctx = canvas.getContext("2d")!;

let detector: PoseLandmarker | null = null;
let running = false;
let fps = 0;
let prevTime = performance.now();
let lastAlarmLevel: HazardLevel = "safe";
let lastAlarmAt = 0;
let debugVis = loadDebugVis();
const tracker = new IouTracker();
const approach = new ApproachAnalyzer();
const judge = new HazardJudge(loadConfig());

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

function renderVisButton(): void {
  visBtn.textContent = debugVis ? "📐 시각화 ON" : "📐 시각화 OFF";
  visBtn.classList.toggle("vis-off", !debugVis);
}

function loadConfig(): HazardConfig {
  try {
    const raw = localStorage.getItem(CONFIG_STORAGE_KEY);
    if (!raw) return { ...DEFAULT_HAZARD_CONFIG };
    const parsed = JSON.parse(raw) as Partial<HazardConfig>;
    return { ...DEFAULT_HAZARD_CONFIG, ...parsed };
  } catch {
    return { ...DEFAULT_HAZARD_CONFIG };
  }
}

function saveConfig(): void {
  try {
    localStorage.setItem(CONFIG_STORAGE_KEY, JSON.stringify(judge.config));
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

interface ParamControl {
  slider: HTMLInputElement;
  value: HTMLSpanElement;
  format: (n: number) => string;
  apply: (n: number) => void;
}

const PARAM_CONTROLS: Record<keyof HazardConfig, ParamControl> = {
  areaWarnRatio: {
    slider: document.getElementById("s-area-warn") as HTMLInputElement,
    value: document.getElementById("v-area-warn") as HTMLSpanElement,
    format: (n) => `${(n * 100).toFixed(1)}%`,
    apply: (n) => {
      judge.config.areaWarnRatio = n;
    },
  },
  areaDangerRatio: {
    slider: document.getElementById("s-area-danger") as HTMLInputElement,
    value: document.getElementById("v-area-danger") as HTMLSpanElement,
    format: (n) => `${(n * 100).toFixed(1)}%`,
    apply: (n) => {
      judge.config.areaDangerRatio = n;
    },
  },
  growthWarnPctPerSec: {
    slider: document.getElementById("s-growth-warn") as HTMLInputElement,
    value: document.getElementById("v-growth-warn") as HTMLSpanElement,
    format: (n) => `${n.toFixed(0)}%/s`,
    apply: (n) => {
      judge.config.growthWarnPctPerSec = n;
    },
  },
  growthDangerPctPerSec: {
    slider: document.getElementById("s-growth-danger") as HTMLInputElement,
    value: document.getElementById("v-growth-danger") as HTMLSpanElement,
    format: (n) => `${n.toFixed(0)}%/s`,
    apply: (n) => {
      judge.config.growthDangerPctPerSec = n;
    },
  },
  warnFrames: {
    slider: document.getElementById("s-warn-frames") as HTMLInputElement,
    value: document.getElementById("v-warn-frames") as HTMLSpanElement,
    format: (n) => `${n}`,
    apply: (n) => {
      judge.config.warnFrames = n;
    },
  },
  dangerFrames: {
    slider: document.getElementById("s-danger-frames") as HTMLInputElement,
    value: document.getElementById("v-danger-frames") as HTMLSpanElement,
    format: (n) => `${n}`,
    apply: (n) => {
      judge.config.dangerFrames = n;
    },
  },
};

function renderSlider(key: keyof HazardConfig): void {
  const ctrl = PARAM_CONTROLS[key];
  const n = judge.config[key];
  ctrl.slider.value = String(n);
  ctrl.value.textContent = ctrl.format(n);
}

function renderAllSliders(): void {
  (Object.keys(PARAM_CONTROLS) as Array<keyof HazardConfig>).forEach(renderSlider);
}

function renderTuneButton(): void {
  const open = tunePanel.classList.contains("visible");
  tuneBtn.textContent = open ? "⚙ 파라미터 닫기" : "⚙ 파라미터";
  tuneBtn.classList.toggle("tune-open", open);
}

const LEVEL_LABEL: Record<HazardLevel, string> = {
  safe: "SAFE",
  warn: "WARN",
  danger: "DANGER",
};

function maybeSpeakAlarm(level: HazardLevel, nowMs: number): void {
  if (level === "safe") {
    lastAlarmLevel = "safe";
    return;
  }
  const isNewLevel = level !== lastAlarmLevel;
  const elapsed = nowMs - lastAlarmAt;
  if (isNewLevel || elapsed >= getRepeatMs(level)) {
    speakAlarm(level);
    lastAlarmAt = nowMs;
    lastAlarmLevel = level;
  }
}

function renderMuteButton(): void {
  const muted = isMuted();
  muteBtn.textContent = muted ? "🔇 음성 OFF" : "🔊 음성 ON";
  muteBtn.classList.toggle("muted", muted);
}

async function startCamera(): Promise<MediaStream> {
  return navigator.mediaDevices.getUserMedia({
    video: {
      // Stroller deployment uses the rear camera; on a laptop this falls back to the only webcam.
      facingMode: { ideal: "environment" },
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

function maxAreaRatio(ths: readonly TrackHazard[]): number {
  let max = 0;
  for (const t of ths) if (t.metrics.areaRatio > max) max = t.metrics.areaRatio;
  return max;
}

function maxGrowth(ths: readonly TrackHazard[]): number {
  let max = 0;
  let found = false;
  for (const t of ths) {
    if (!found || t.metrics.growthPctPerSec > max) {
      max = t.metrics.growthPctPerSec;
      found = true;
    }
  }
  return found ? max : 0;
}

function renderFrame(timestampMs: number): void {
  if (!detector || !running) return;

  const now = performance.now();
  const dt = (now - prevTime) / 1000;
  if (dt > 0) fps = 0.9 * fps + 0.1 * (1 / dt);
  prevTime = now;

  const result = detector.detectForVideo(video, timestampMs);
  const dets = toHazardDetections(result, canvas.width, canvas.height);
  const tracks = tracker.update(dets, now);
  const metricsById = approach.update(tracks, canvas.width, canvas.height, now);
  const { trackHazards, globalLevel } = judge.step(tracks, metricsById);

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  if (debugVis) drawTracks(ctx, trackHazards);
  applyAlarmMask(ctx, canvas.width, canvas.height, globalLevel, now);
  maybeSpeakAlarm(globalLevel, now);

  mState.textContent = LEVEL_LABEL[globalLevel];
  mState.classList.toggle("safe", globalLevel === "safe");
  mState.classList.toggle("warn", globalLevel === "warn");
  mState.classList.toggle("danger", globalLevel === "danger");
  mFps.textContent = `${fps.toFixed(1)} FPS`;
  mCount.textContent = `${tracks.length}`;
  mArea.textContent =
    trackHazards.length > 0
      ? `${(maxAreaRatio(trackHazards) * 100).toFixed(1)}%`
      : "—";
  const g = maxGrowth(trackHazards);
  mGrowth.textContent =
    trackHazards.length > 0
      ? `${g >= 0 ? "+" : ""}${g.toFixed(0)}%/s`
      : "—";

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

function stopDetection(): void {
  running = false;
  window.speechSynthesis.cancel();
  const stream = video.srcObject as MediaStream | null;
  stream?.getTracks().forEach((t) => t.stop());
  video.srcObject = null;
  detector?.close();
  detector = null;
  tracker.reset();
  approach.reset();
  judge.reset();
  lastAlarmLevel = "safe";
  lastAlarmAt = 0;
  mState.classList.remove("safe", "warn", "danger");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  panel.classList.remove("visible");
  tunePanel.classList.remove("visible");
  renderTuneButton();
  intro.classList.remove("hidden");
  startBtn.disabled = false;
  startBtn.textContent = "Start";
  help.textContent = "카메라 권한 요청 팝업이 뜨면 '허용'해주세요!";
  fps = 0;
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
    detector = await createHazardDetector();
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

(Object.keys(PARAM_CONTROLS) as Array<keyof HazardConfig>).forEach((key) => {
  const ctrl = PARAM_CONTROLS[key];
  ctrl.slider.addEventListener("input", () => {
    const n = parseFloat(ctrl.slider.value);
    ctrl.apply(n);
    ctrl.value.textContent = ctrl.format(n);
    saveConfig();
  });
});

resetBtn.addEventListener("click", () => {
  judge.config = { ...DEFAULT_HAZARD_CONFIG };
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
  detector?.close();
});

renderMuteButton();
renderVisButton();
renderTuneButton();
renderAllSliders();
