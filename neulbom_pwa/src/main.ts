import "./style.css";
import { registerSW } from "virtual:pwa-register";

registerSW({ immediate: true });

export type ModeId = "car" | "stroller";

export interface ModeController {
  stop(): void;
}

export interface ModeBootOptions {
  stage: HTMLElement;
  onExit: () => void;
  onSwitch: (target: ModeId) => void;
}

const chooser = document.getElementById("chooser") as HTMLDivElement;
const stage = document.getElementById("stage") as HTMLDivElement;
const carCard = document.getElementById("mode-car") as HTMLButtonElement;
const strollerCard = document.getElementById("mode-stroller") as HTMLButtonElement;
const errorOverlay = document.getElementById("error") as HTMLDivElement;
const errorMsg = document.getElementById("error-msg") as HTMLParagraphElement;
const errorBackBtn = document.getElementById("error-back") as HTMLButtonElement;

let active: ModeController | null = null;

function showChooser(): void {
  active?.stop();
  active = null;
  stage.innerHTML = "";
  stage.classList.remove("active");
  errorOverlay.classList.add("hidden");
  chooser.classList.remove("hidden");
}

function showError(e: unknown): void {
  const msg = e instanceof Error ? e.message : String(e);
  console.error("mode boot failed", e);
  errorMsg.textContent = `카메라/모델 초기화에 실패했습니다.\n${msg}`;
  stage.innerHTML = "";
  stage.classList.remove("active");
  chooser.classList.add("hidden");
  errorOverlay.classList.remove("hidden");
}

async function bootMode(id: ModeId): Promise<void> {
  chooser.classList.add("hidden");
  stage.classList.add("active");

  try {
    const mod = await (id === "car"
      ? import("./modes/car/boot")
      : import("./modes/stroller/boot"));
    active = await mod.start({
      stage,
      onExit: showChooser,
      onSwitch: switchMode,
    });
  } catch (e) {
    showError(e);
  }
}

function switchMode(target: ModeId): void {
  active?.stop();
  active = null;
  void bootMode(target);
}

carCard.addEventListener("click", () => {
  void bootMode("car");
});

strollerCard.addEventListener("click", () => {
  void bootMode("stroller");
});

errorBackBtn.addEventListener("click", () => {
  showChooser();
});

window.addEventListener("beforeunload", () => {
  active?.stop();
});
