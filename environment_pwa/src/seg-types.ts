export type WorkerReadyMessage = {
  type: "ready";
};

export type WorkerErrorMessage = {
  type: "error";
  requestId?: number;
  message: string;
};

export type WorkerMask = {
  width: number;
  height: number;
  data: ArrayBuffer;
};

export type WorkerSegResult = {
  type: "seg-result";
  requestId: number;
  inferenceMs: number;
  totalMs: number;
  labels: string[];
  roadMask: WorkerMask | null;
  roadWarnMask: WorkerMask | null;
  sidewalkMask: WorkerMask | null;
  crosswalkMask: WorkerMask | null;
  curbMask: WorkerMask | null;
};

export type WorkerSegmentRequest = {
  type: "segment";
  requestId: number;
  imageBitmap: ImageBitmap;
};

export type WorkerMessage = WorkerReadyMessage | WorkerErrorMessage | WorkerSegResult;

export type WorkerRequest = {
  type: "warmup";
} | WorkerSegmentRequest;
