import type { HazardDetection } from "./detector";

export interface BBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface Track {
  id: number;
  bbox: BBox;
  score: number;
  category: string;
  firstSeenMs: number;
  lastSeenMs: number;
  missedFrames: number;
}

const IOU_MATCH_THRESHOLD = 0.3;
const MAX_MISSED_FRAMES = 5;

function iou(a: BBox, b: BBox): number {
  const ax2 = a.x + a.width;
  const ay2 = a.y + a.height;
  const bx2 = b.x + b.width;
  const by2 = b.y + b.height;
  const interW = Math.max(0, Math.min(ax2, bx2) - Math.max(a.x, b.x));
  const interH = Math.max(0, Math.min(ay2, by2) - Math.max(a.y, b.y));
  const interArea = interW * interH;
  if (interArea <= 0) return 0;
  const unionArea = a.width * a.height + b.width * b.height - interArea;
  return unionArea > 0 ? interArea / unionArea : 0;
}

export class IouTracker {
  private tracks: Track[] = [];
  private nextId = 1;

  update(detections: HazardDetection[], nowMs: number): Track[] {
    const existingCount = this.tracks.length;

    const pairs: Array<{ ti: number; di: number; v: number }> = [];
    for (let ti = 0; ti < existingCount; ti++) {
      const tBox = this.tracks[ti].bbox;
      for (let di = 0; di < detections.length; di++) {
        const d = detections[di];
        const v = iou(tBox, {
          x: d.x,
          y: d.y,
          width: d.width,
          height: d.height,
        });
        if (v >= IOU_MATCH_THRESHOLD) pairs.push({ ti, di, v });
      }
    }
    pairs.sort((a, b) => b.v - a.v);

    const matchedTracks = new Set<number>();
    const matchedDets = new Set<number>();
    for (const p of pairs) {
      if (matchedTracks.has(p.ti) || matchedDets.has(p.di)) continue;
      const track = this.tracks[p.ti];
      const d = detections[p.di];
      track.bbox = { x: d.x, y: d.y, width: d.width, height: d.height };
      track.score = d.score;
      track.category = d.category;
      track.lastSeenMs = nowMs;
      track.missedFrames = 0;
      matchedTracks.add(p.ti);
      matchedDets.add(p.di);
    }

    for (let ti = 0; ti < existingCount; ti++) {
      if (!matchedTracks.has(ti)) this.tracks[ti].missedFrames++;
    }

    for (let di = 0; di < detections.length; di++) {
      if (matchedDets.has(di)) continue;
      const d = detections[di];
      this.tracks.push({
        id: this.nextId++,
        bbox: { x: d.x, y: d.y, width: d.width, height: d.height },
        score: d.score,
        category: d.category,
        firstSeenMs: nowMs,
        lastSeenMs: nowMs,
        missedFrames: 0,
      });
    }

    this.tracks = this.tracks.filter((t) => t.missedFrames <= MAX_MISSED_FRAMES);
    return this.tracks.filter((t) => t.missedFrames === 0);
  }

  reset(): void {
    this.tracks = [];
    this.nextId = 1;
  }
}

export function trackColor(id: number): string {
  // Golden-ratio hue stepping keeps adjacent IDs visually distinct.
  const hue = (id * 137.508) % 360;
  return `hsl(${hue}, 80%, 60%)`;
}
