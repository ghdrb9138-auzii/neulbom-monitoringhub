import alarmsConfig from "@shared/alarms.json";

import type { AlarmLevel } from "./stateMachine";

const MUTE_STORAGE_KEY = "neulbom.alarm.muted";

interface AlarmSpec {
  message_ko: string;
  level: number;
  repeat_ms: number;
}

const SPECS: Record<Exclude<AlarmLevel, "none">, AlarmSpec> = {
  drowsy: alarmsConfig.alarms.drowsy,
  head_bent: alarmsConfig.alarms.head_bent,
  danger: alarmsConfig.alarms.danger,
};

export function getRepeatMs(level: Exclude<AlarmLevel, "none">): number {
  return SPECS[level].repeat_ms;
}

let muted = (() => {
  try {
    return localStorage.getItem(MUTE_STORAGE_KEY) === "1";
  } catch {
    return false;
  }
})();

export function isMuted(): boolean {
  return muted;
}

export function setMuted(value: boolean): void {
  muted = value;
  try {
    localStorage.setItem(MUTE_STORAGE_KEY, value ? "1" : "0");
  } catch {
    // ignore quota / privacy mode failures
  }
  if (value) {
    window.speechSynthesis.cancel();
  }
}

/**
 * Trigger an empty utterance inside a user gesture handler so subsequent
 * automatic .speak() calls are not blocked by iOS Safari autoplay policy.
 */
export function unlock(): void {
  if (!("speechSynthesis" in window)) return;
  const u = new SpeechSynthesisUtterance("");
  u.volume = 0;
  window.speechSynthesis.speak(u);
}

export function speakAlarm(level: Exclude<AlarmLevel, "none">): void {
  if (muted) return;
  if (!("speechSynthesis" in window)) return;
  const spec = SPECS[level];
  const u = new SpeechSynthesisUtterance(spec.message_ko);
  u.lang = "ko-KR";
  u.rate = level === "danger" ? 1.2 : 1.0;
  u.pitch = 1.0;
  // Cancel any in-flight utterance so a more severe alarm preempts a milder one.
  window.speechSynthesis.cancel();
  window.speechSynthesis.speak(u);
}
