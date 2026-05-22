import {
  FaceLandmarker,
  FilesetResolver,
  type FaceLandmarkerResult,
} from "@mediapipe/tasks-vision";
import { registerSW } from "virtual:pwa-register";

import { computeBothEAR, type Point2D } from "./ear";
import { decomposeEuler } from "./headPose";
import { applyAlarmMask, drawEyes, drawStatus } from "./overlay";
import { Detector } from "./stateMachine";

registerSW({ immediate: true });

const WASM_BASE =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm";
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";

const video = document.getElementById("video") as HTMLVideoElement;
const canvas = document.getElementById("canvas") as HTMLCanvasElement;
const startBtn = document.getElementById("start") as HTMLButtonElement;
const help = document.getElementById("help") as HTMLDivElement;
const ctx = canvas.getContext("2d")!;
const detector = new Detector();

let faceLandmarker: FaceLandmarker | null = null;
let fps = 0;
let prevTime = performance.now();
let running = false;

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
    drawEyes(ctx, landmarks, canvas.width, canvas.height, out.isClosed);
    applyAlarmMask(ctx, canvas.width, canvas.height, out.alarm);
    drawStatus(ctx, canvas.width, {
      ear,
      state: out.state,
      fps,
      closedFrames: out.closedFrames,
      headBentFrames: out.headBentFrames,
      pitch,
      yaw,
      roll,
    });
  } else {
    const out = detector.noFace();
    drawStatus(ctx, canvas.width, {
      ear: 0,
      state: out.state,
      fps,
      closedFrames: 0,
      headBentFrames: 0,
      pitch: 0,
      yaw: 0,
      roll: 0,
    });
  }

  scheduleNextFrame();
}

function scheduleNextFrame(): void {
  if (!running) return;
  const v = video as HTMLVideoElement & {
    requestVideoFrameCallback?: (cb: (now: number, meta: { mediaTime: number }) => void) => number;
  };
  if (typeof v.requestVideoFrameCallback === "function") {
    v.requestVideoFrameCallback((now) => renderFrame(now));
  } else {
    requestAnimationFrame((now) => renderFrame(now));
  }
}

async function start(): Promise<void> {
  startBtn.disabled = true;
  startBtn.textContent = "Loading…";
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
    startBtn.classList.add("hidden");
    help.style.display = "none";
    scheduleNextFrame();
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    console.error("start() failed", e);
    startBtn.disabled = false;
    startBtn.textContent = "Retry";
    help.textContent = `Error: ${msg}`;
    help.style.opacity = "1";
  }
}

startBtn.addEventListener("click", () => {
  void start();
});

window.addEventListener("resize", resizeCanvas);

window.addEventListener("beforeunload", () => {
  running = false;
  const stream = video.srcObject as MediaStream | null;
  stream?.getTracks().forEach((t) => t.stop());
  faceLandmarker?.close();
});
