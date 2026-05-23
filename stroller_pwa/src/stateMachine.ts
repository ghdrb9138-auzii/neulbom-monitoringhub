import hazardConfig from "@shared/hazard.json";

import type { TrackMetrics } from "./approach";
import type { Track } from "./tracker";

export type HazardLevel = "safe" | "warn" | "danger";

export interface HazardConfig {
  areaWarnRatio: number;
  areaDangerRatio: number;
  growthWarnPctPerSec: number;
  growthDangerPctPerSec: number;
  warnFrames: number;
  dangerFrames: number;
}

export const DEFAULT_HAZARD_CONFIG: HazardConfig = {
  areaWarnRatio: hazardConfig.approach.area_warn_ratio,
  areaDangerRatio: hazardConfig.approach.area_danger_ratio,
  growthWarnPctPerSec: hazardConfig.approach.growth_warn_pct_per_sec,
  growthDangerPctPerSec: hazardConfig.approach.growth_danger_pct_per_sec,
  warnFrames: hazardConfig.approach.warn_frames,
  dangerFrames: hazardConfig.approach.danger_frames,
};

interface PerTrackState {
  warnFrames: number;
  dangerFrames: number;
  level: HazardLevel;
}

export interface TrackHazard {
  track: Track;
  metrics: TrackMetrics;
  level: HazardLevel;
}

export interface FrameJudgment {
  trackHazards: TrackHazard[];
  globalLevel: HazardLevel;
}

export class HazardJudge {
  config: HazardConfig;
  private states = new Map<number, PerTrackState>();

  constructor(config: HazardConfig = DEFAULT_HAZARD_CONFIG) {
    this.config = { ...config };
  }

  private instantLevel(m: TrackMetrics): HazardLevel {
    const c = this.config;
    if (m.areaRatio >= c.areaDangerRatio && m.growthPctPerSec >= c.growthDangerPctPerSec)
      return "danger";
    if (m.areaRatio >= c.areaWarnRatio || m.growthPctPerSec >= c.growthWarnPctPerSec)
      return "warn";
    return "safe";
  }

  step(
    tracks: readonly Track[],
    metricsById: Map<number, TrackMetrics>,
  ): FrameJudgment {
    const out: TrackHazard[] = [];
    const seen = new Set<number>();
    let global: HazardLevel = "safe";
    const c = this.config;

    for (const t of tracks) {
      seen.add(t.id);
      const m = metricsById.get(t.id) ?? { areaRatio: 0, growthPctPerSec: 0 };
      const inst = this.instantLevel(m);

      let s = this.states.get(t.id);
      if (!s) {
        s = { warnFrames: 0, dangerFrames: 0, level: "safe" };
        this.states.set(t.id, s);
      }

      if (inst === "danger") {
        s.dangerFrames++;
        s.warnFrames++;
      } else if (inst === "warn") {
        s.warnFrames++;
        s.dangerFrames = 0;
      } else {
        s.warnFrames = 0;
        s.dangerFrames = 0;
      }

      if (s.dangerFrames >= c.dangerFrames) s.level = "danger";
      else if (s.warnFrames >= c.warnFrames) s.level = "warn";
      else s.level = "safe";

      out.push({ track: t, metrics: m, level: s.level });
      if (s.level === "danger") global = "danger";
      else if (s.level === "warn" && global === "safe") global = "warn";
    }

    for (const id of Array.from(this.states.keys())) {
      if (!seen.has(id)) this.states.delete(id);
    }
    return { trackHazards: out, globalLevel: global };
  }

  reset(): void {
    this.states.clear();
  }
}
