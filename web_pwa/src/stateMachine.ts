import thresholdsConfig from "@shared/thresholds.json";

export type BentDir = "" | "FWD" | "SIDE" | "DIAG";
export type AlarmLevel = "none" | "drowsy" | "head_bent" | "danger";

export interface DetectorConfig {
  earThreshold: number;
  drowsyFrames: number;
  pitchThresholdDeg: number;
  rollThresholdDeg: number;
  bentFrames: number;
}

export const DEFAULT_CONFIG: DetectorConfig = {
  earThreshold: thresholdsConfig.ear.threshold,
  drowsyFrames: thresholdsConfig.ear.drowsy_frames,
  pitchThresholdDeg: thresholdsConfig.head_pose.pitch_threshold_deg,
  rollThresholdDeg: thresholdsConfig.head_pose.roll_threshold_deg,
  bentFrames: thresholdsConfig.head_pose.bent_frames,
};

export interface StepInput {
  ear: number;
  pitch: number;
  roll: number;
}

export interface StepOutput {
  state: string;
  alarm: AlarmLevel;
  isClosed: boolean;
  isHeadBent: boolean;
  bentDir: BentDir;
  closedFrames: number;
  headBentFrames: number;
}

export class Detector {
  private closedFrames = 0;
  private headBentFrames = 0;

  constructor(public config: DetectorConfig = { ...DEFAULT_CONFIG }) {}

  reset(): void {
    this.closedFrames = 0;
    this.headBentFrames = 0;
  }

  /** Called when no face is detected — clears the sustained counters. */
  noFace(): StepOutput {
    this.reset();
    return {
      state: "NO FACE",
      alarm: "none",
      isClosed: false,
      isHeadBent: false,
      bentDir: "",
      closedFrames: 0,
      headBentFrames: 0,
    };
  }

  step({ ear, pitch, roll }: StepInput): StepOutput {
    const isClosed = ear < this.config.earThreshold;
    const isPitchBent = pitch > this.config.pitchThresholdDeg;
    const isRollBent = Math.abs(roll) > this.config.rollThresholdDeg;
    const isHeadBent = isPitchBent || isRollBent;

    let bentDir: BentDir = "";
    if (isPitchBent && isRollBent) bentDir = "DIAG";
    else if (isPitchBent) bentDir = "FWD";
    else if (isRollBent) bentDir = "SIDE";

    this.closedFrames = isClosed ? this.closedFrames + 1 : 0;
    this.headBentFrames = isHeadBent ? this.headBentFrames + 1 : 0;

    const isDrowsy = this.closedFrames >= this.config.drowsyFrames;
    const isHeadBentSustained = this.headBentFrames >= this.config.bentFrames;

    let state: string;
    let alarm: AlarmLevel = "none";

    if (isDrowsy && isHeadBentSustained) {
      state = `DANGER (${bentDir || "FWD"})`;
      alarm = "danger";
    } else if (isHeadBentSustained) {
      state = `HEAD BENT (${bentDir || "FWD"})`;
      alarm = "head_bent";
    } else if (isDrowsy) {
      state = "DROWSY";
      alarm = "drowsy";
    } else if (isClosed && isHeadBent) {
      state = `SLEEPING? (${bentDir || "FWD"})`;
    } else if (isClosed) {
      state = "EYES CLOSED";
    } else if (isHeadBent) {
      state = `HEAD ${bentDir || "FWD"}`;
    } else {
      state = "AWAKE";
    }

    return {
      state,
      alarm,
      isClosed,
      isHeadBent,
      bentDir,
      closedFrames: this.closedFrames,
      headBentFrames: this.headBentFrames,
    };
  }
}
