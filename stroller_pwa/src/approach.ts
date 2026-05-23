import hazardConfig from "@shared/hazard.json";

import type { Track } from "./tracker";

const WINDOW_MS = hazardConfig.approach.history_window_ms;
const MIN_DELTA_MS = 100;

interface AreaSample {
  t: number;
  area: number;
}

export interface TrackMetrics {
  areaRatio: number;
  growthPctPerSec: number;
}

export class ApproachAnalyzer {
  private history = new Map<number, AreaSample[]>();

  update(
    tracks: readonly Track[],
    frameW: number,
    frameH: number,
    nowMs: number,
  ): Map<number, TrackMetrics> {
    const frameArea = Math.max(1, frameW * frameH);
    const result = new Map<number, TrackMetrics>();
    const seen = new Set<number>();

    for (const t of tracks) {
      seen.add(t.id);
      const area = Math.max(0, t.bbox.width * t.bbox.height);

      let buf = this.history.get(t.id);
      if (!buf) {
        buf = [];
        this.history.set(t.id, buf);
      }
      buf.push({ t: nowMs, area });
      while (buf.length > 0 && nowMs - buf[0].t > WINDOW_MS) buf.shift();

      let growthPctPerSec = 0;
      if (buf.length >= 2) {
        const oldest = buf[0];
        const dtMs = nowMs - oldest.t;
        if (dtMs >= MIN_DELTA_MS && oldest.area > 0) {
          const ratio = area / oldest.area;
          growthPctPerSec = ((ratio - 1) * 100) / (dtMs / 1000);
        }
      }

      result.set(t.id, {
        areaRatio: area / frameArea,
        growthPctPerSec,
      });
    }

    for (const id of Array.from(this.history.keys())) {
      if (!seen.has(id)) this.history.delete(id);
    }
    return result;
  }

  reset(): void {
    this.history.clear();
  }
}
