/// <reference lib="webworker" />

import { env, pipeline, RawImage } from "@huggingface/transformers";
import type {
  WorkerErrorMessage,
  WorkerMask,
  WorkerReadyMessage,
  WorkerRequest,
  WorkerSegResult,
  WorkerSegmentRequest,
} from "./seg-types";

const ROAD_SEG_MODEL_DIR = `${import.meta.env.BASE_URL}models/segformer-sidewalk`;
const ROAD_SEG_CONFIG_URL = `${import.meta.env.BASE_URL}models/segformer-sidewalk/config.json`;
const ROAD_SEG_PREPROCESSOR_URL = `${import.meta.env.BASE_URL}models/segformer-sidewalk/preprocessor_config.json`;
const ROAD_SEG_ONNX_URL = `${import.meta.env.BASE_URL}models/segformer-sidewalk/onnx/model.onnx`;

const ROAD_DANGER_LABELS = new Set(["flat-road", "flat-railtrack"]);
const ROAD_WARN_LABELS = new Set(["flat-cyclinglane", "flat-parkingdriveway"]);
const SIDEWALK_LABELS = new Set(["flat-sidewalk"]);
const CROSSWALK_LABELS = new Set(["flat-crosswalk"]);
const CURB_LABELS = new Set(["flat-curb", "construction-stairs"]);

type RoadSegmenter = (image: RawImage) => Promise<unknown>;

type BinaryMask = {
  width: number;
  height: number;
  data: Uint8Array;
};

type RawMaskLike = {
  width?: number;
  height?: number;
  size?: [number, number];
  channels?: number;
  numChannels?: number;
  data?: ArrayLike<number>;
};

env.localModelPath = `${import.meta.env.BASE_URL}models/`;
env.allowLocalModels = true;
env.allowRemoteModels = false;
env.useBrowserCache = false;

let segmenter: RoadSegmenter | null = null;
let segmenterPromise: Promise<RoadSegmenter> | null = null;

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

async function validateLocalSegformerFiles(): Promise<void> {
  await assertJsonFile(ROAD_SEG_CONFIG_URL);
  await assertJsonFile(ROAD_SEG_PREPROCESSOR_URL);
  await assertBinaryFile(ROAD_SEG_ONNX_URL, 1_000_000);
}

async function ensureSegmenter(): Promise<RoadSegmenter> {
  if (segmenter) {
    return segmenter;
  }

  if (!segmenterPromise) {
    segmenterPromise = (async () => {
      await validateLocalSegformerFiles();

      const loaded = (await pipeline("image-segmentation", ROAD_SEG_MODEL_DIR, {
        model_file_name: "model",
        local_files_only: true,
        dtype: "fp32",
      })) as RoadSegmenter;

      segmenter = loaded;
      return loaded;
    })();
  }

  return segmenterPromise;
}

function cloneMask(mask: BinaryMask): BinaryMask {
  return {
    width: mask.width,
    height: mask.height,
    data: new Uint8Array(mask.data),
  };
}

function maskFromRawMaskLike(mask: RawMaskLike): BinaryMask | null {
  const width = Number(mask.width ?? mask.size?.[0] ?? 0);
  const height = Number(mask.height ?? mask.size?.[1] ?? 0);
  const sourceData = mask.data;
  if (!width || !height || !sourceData) return null;

  const pixelCount = width * height;
  const channels = Math.max(
    1,
    Number(
      mask.channels ??
        mask.numChannels ??
        (pixelCount > 0 && sourceData.length % pixelCount === 0
          ? Math.max(1, Math.floor(sourceData.length / pixelCount))
          : 1),
    ),
  );
  const useAlpha = channels === 4 || sourceData.length === pixelCount * 4;
  const foregroundIndex = useAlpha ? 3 : 0;
  const data = new Uint8Array(pixelCount);

  for (let i = 0; i < pixelCount; i += 1) {
    const value = Number(sourceData[i * channels + foregroundIndex] ?? 0);
    data[i] = value > 0 ? 1 : 0;
  }

  return { width, height, data };
}

function inspectMask(mask: unknown): BinaryMask | null {
  if (!mask) return null;

  if (typeof ImageData !== "undefined" && mask instanceof ImageData) {
    const data = new Uint8Array(mask.width * mask.height);
    for (let i = 0; i < data.length; i += 1) {
      data[i] = (mask.data[i * 4 + 3] ?? 0) > 0 ? 1 : 0;
    }
    return { width: mask.width, height: mask.height, data };
  }

  if (typeof OffscreenCanvas !== "undefined" && mask instanceof OffscreenCanvas) {
    const ctx = mask.getContext("2d");
    if (!ctx) return null;
    return inspectMask(ctx.getImageData(0, 0, mask.width, mask.height));
  }

  if (typeof ImageBitmap !== "undefined" && mask instanceof ImageBitmap) {
    const canvas = new OffscreenCanvas(mask.width, mask.height);
    const ctx = canvas.getContext("2d");
    if (!ctx) return null;
    ctx.drawImage(mask, 0, 0);
    return inspectMask(ctx.getImageData(0, 0, canvas.width, canvas.height));
  }

  return maskFromRawMaskLike(mask as RawMaskLike);
}

function imageBitmapToRawImage(imageBitmap: ImageBitmap): RawImage {
  const canvas = new OffscreenCanvas(imageBitmap.width, imageBitmap.height);
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    throw new Error("Failed to create worker canvas context");
  }

  ctx.drawImage(imageBitmap, 0, 0);

  try {
    return RawImage.fromCanvas(canvas as unknown as HTMLCanvasElement);
  } catch {
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    return new RawImage(imageData.data, canvas.width, canvas.height, 4);
  }
}

function extractMaskFromSegment(segment: unknown): BinaryMask | null {
  if (!segment || typeof segment !== "object") return null;
  return inspectMask((segment as { mask?: unknown }).mask);
}

function mergeMask(existing: BinaryMask | null, next: BinaryMask): BinaryMask {
  if (!existing) {
    return cloneMask(next);
  }

  if (existing.width !== next.width || existing.height !== next.height) {
    return cloneMask(next);
  }

  const data = new Uint8Array(existing.data.length);
  for (let i = 0; i < data.length; i += 1) {
    data[i] = existing.data[i] || next.data[i] ? 1 : 0;
  }

  return {
    width: existing.width,
    height: existing.height,
    data,
  };
}

function toWorkerMask(mask: BinaryMask | null): WorkerMask | null {
  if (!mask) return null;
  return {
    width: mask.width,
    height: mask.height,
    data: mask.data.slice().buffer,
  };
}

function buildMasks(segments: Array<{ label?: string; mask?: unknown }>): {
  labels: string[];
  roadMask: BinaryMask | null;
  roadWarnMask: BinaryMask | null;
  sidewalkMask: BinaryMask | null;
  crosswalkMask: BinaryMask | null;
  curbMask: BinaryMask | null;
} {
  let roadMask: BinaryMask | null = null;
  let roadWarnMask: BinaryMask | null = null;
  let sidewalkMask: BinaryMask | null = null;
  let crosswalkMask: BinaryMask | null = null;
  let curbMask: BinaryMask | null = null;
  const labels = new Set<string>();

  for (const segment of segments) {
    const label = String(segment?.label ?? "").toLowerCase();
    if (!label) continue;
    labels.add(label);

    const mask = extractMaskFromSegment(segment);
    if (!mask) continue;

    if (ROAD_DANGER_LABELS.has(label)) {
      roadMask = mergeMask(roadMask, mask);
      continue;
    }

    if (ROAD_WARN_LABELS.has(label)) {
      roadWarnMask = mergeMask(roadWarnMask, mask);
      continue;
    }

    if (SIDEWALK_LABELS.has(label)) {
      sidewalkMask = mergeMask(sidewalkMask, mask);
      continue;
    }

    if (CROSSWALK_LABELS.has(label)) {
      crosswalkMask = mergeMask(crosswalkMask, mask);
      continue;
    }

    if (CURB_LABELS.has(label)) {
      curbMask = mergeMask(curbMask, mask);
    }
  }

  return {
    labels: Array.from(labels),
    roadMask,
    roadWarnMask,
    sidewalkMask,
    crosswalkMask,
    curbMask,
  };
}

async function postReady(): Promise<void> {
  await ensureSegmenter();
  const message: WorkerReadyMessage = { type: "ready" };
  self.postMessage(message);
}

function postError(messageText: string, requestId?: number): void {
  const message: WorkerErrorMessage = {
    type: "error",
    message: messageText,
    requestId,
  };
  self.postMessage(message);
}

async function handleSegmentRequest(request: WorkerSegmentRequest): Promise<void> {
  const requestStart = performance.now();

  try {
    const segmenterInstance = await ensureSegmenter();
    const rawImage = imageBitmapToRawImage(request.imageBitmap);
    const inferenceStart = performance.now();
    const result = await segmenterInstance(rawImage);
    const inferenceMs = performance.now() - inferenceStart;
    const segments = Array.isArray(result) ? result : [result];
    const masks = buildMasks(segments as Array<{ label?: string; mask?: unknown }>);

    const roadMask = toWorkerMask(masks.roadMask);
    const roadWarnMask = toWorkerMask(masks.roadWarnMask);
    const sidewalkMask = toWorkerMask(masks.sidewalkMask);
    const crosswalkMask = toWorkerMask(masks.crosswalkMask);
    const curbMask = toWorkerMask(masks.curbMask);

    const transferList: ArrayBuffer[] = [
      roadMask?.data,
      roadWarnMask?.data,
      sidewalkMask?.data,
      crosswalkMask?.data,
      curbMask?.data,
    ].filter((buffer): buffer is ArrayBuffer => Boolean(buffer));

    const message: WorkerSegResult = {
      type: "seg-result",
      requestId: request.requestId,
      inferenceMs,
      totalMs: performance.now() - requestStart,
      labels: masks.labels,
      roadMask,
      roadWarnMask,
      sidewalkMask,
      crosswalkMask,
      curbMask,
    };

    self.postMessage(message, transferList);
  } catch (error) {
    postError(error instanceof Error ? error.message : String(error), request.requestId);
  } finally {
    request.imageBitmap.close();
  }
}

self.addEventListener("message", (event: MessageEvent<WorkerRequest>) => {
  const request = event.data;

  if (request.type === "warmup") {
    void postReady().catch((error) => {
      postError(error instanceof Error ? error.message : String(error));
    });
    return;
  }

  if (request.type === "segment") {
    void handleSegmentRequest(request);
  }
});
