/**
 * Hardware flashlight (torch) control via the active camera MediaStreamTrack.
 *
 * Only the rear / environment-facing camera exposes a torch, and only on
 * iOS 17.5+ Safari/WebKit or Android Chrome. The front camera (car mode) and
 * older iOS report no torch capability — in that case `available` is false and
 * every method here is a no-op, so the caller's on-screen alarm mask stays as
 * the visual fallback.
 *
 * Shared by both modes: it contains no mode-specific logic, only generic
 * "blink the LED while a danger condition holds" behaviour.
 */

// `torch` is part of the MediaStream Image Capture spec but not yet in the
// TS DOM lib types, so we widen the relevant shapes locally.
type TorchCapabilities = MediaTrackCapabilities & { torch?: boolean };
type TorchConstraints = { advanced: Array<{ torch: boolean }> };

export interface TorchController {
  /** True only when a controllable torch was detected on this stream. */
  readonly available: boolean;
  /** Drive the blink loop: pass `true` while the danger condition holds. */
  setActive(active: boolean): void;
  /** Force the LED off and stop blinking. Call on stop / mode switch. */
  dispose(): void;
}

export function createTorch(stream: MediaStream, blinkMs = 500): TorchController {
  const track = stream.getVideoTracks()[0] as MediaStreamTrack | undefined;
  const caps = (track?.getCapabilities?.() ?? {}) as TorchCapabilities;
  const available = !!track && caps.torch === true;

  let active = false;
  let on = false;
  let timer: ReturnType<typeof setTimeout> | null = null;

  function applyTorch(value: boolean): void {
    if (!track || on === value) return;
    on = value;
    // applyConstraints is async; the blink loop intentionally fires and forgets.
    // The LED can transiently reject (e.g. mid teardown) — ignore those.
    void track
      .applyConstraints({ advanced: [{ torch: value }] } as unknown as TorchConstraints as MediaTrackConstraints)
      .catch(() => {});
  }

  function tick(): void {
    if (!active) return;
    applyTorch(!on);
    timer = setTimeout(tick, blinkMs);
  }

  return {
    available,
    setActive(next: boolean): void {
      if (!available || next === active) return;
      active = next;
      if (active) {
        applyTorch(true);
        timer = setTimeout(tick, blinkMs);
      } else {
        if (timer !== null) {
          clearTimeout(timer);
          timer = null;
        }
        applyTorch(false);
      }
    },
    dispose(): void {
      active = false;
      if (timer !== null) {
        clearTimeout(timer);
        timer = null;
      }
      applyTorch(false);
    },
  };
}
