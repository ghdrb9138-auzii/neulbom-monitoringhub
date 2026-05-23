# Neulbom — Stroller Hazard PoC (Mode B)

유모차 모드에서 카메라가 외부를 향하도록 부착되었을 때, 뒤·옆으로 접근하는
사람·이동수단을 감지해 보호자에게 알리는 PoC.

`web_pwa/`(Mode A — 내부 영유아 모니터링)와 동일한 기술 스택을 쓰지만 카메라
방향과 추론 모델이 다르다. 임계값·클래스 필터는 `../shared/hazard.json`을
단일 출처로 공유한다.

## 단계별 진행

| Phase | 내용 | 상태 |
|---|---|---|
| 1 | Vite + TS + 사람 검출 + bbox 그리기 | ✅ |
| 2 | IOU 기반 트래커 + 트랙 ID 시각화 | ✅ |
| 3 | bbox 면적 증가율 → SAFE / WARN / DANGER 상태머신 | ✅ |
| 4 | TTS 알람 + 펄스 마스크 + 음성 mute | ✅ |
| — | 검출 모델 Pose Landmarker 교체 (head+shoulders bbox) | ✅ |
| 5 | PWA(manifest + service worker) + Vercel 배포 | ⬜ |

## 로컬 실행

```bash
cd stroller_pwa
npm install
npm run dev   # http://localhost:5174
```

`web_pwa/`와 포트 충돌을 피하려고 5174를 쓴다. 동시에 실행 가능.

## iPad / iPhone 테스트

`web_pwa/`와 동일하게 HTTPS 터널이 필요하다.

```bash
cloudflared tunnel --url http://localhost:5174
```

출력되는 `https://*.trycloudflare.com` URL을 모바일 Safari에서 연다. 유모차에
거치한다고 가정하므로 후면 카메라(`facingMode: "environment"`)가 기본이다.
노트북 웹캠은 단일 카메라이므로 자동으로 그것이 선택된다.

## 알고리즘

1. **검출** — MediaPipe **Pose Landmarker (Lite)**, `numPoses=5`. 사람당 33개
   랜드마크 중 **0–12번(얼굴 + 좌·우 어깨)** 만 사용해 타이트한 head+shoulders
   bbox 생성. 팔 벌림·옷차림·다리 움직임 등 전신 bbox 노이즈에서 해방됨.
2. **트래킹** — IOU 매칭 기반 간단 트래커 (PoC엔 DeepSORT 과함)
3. **접근 속도** — 트랙별 bbox 면적 증가율 (단안 카메라 TTC 대용 신호). 유모차
   전진 시 정지한 사람은 면적이 감소 → 능동적으로 접근하는 객체에만 트리거.
4. **상태머신** — `shared/hazard.json` 임계값 기반
   - **SAFE**: 사람 없음 / 면적·증가율 모두 낮음
   - **WARN**: 면적 ≥ 임계값(가까움) **또는** 증가율 ≥ 약한 임계값(천천히 접근)
   - **DANGER**: 면적 큼 **AND** 증가율 큼 (빠르게 접근)
5. **알람** — Web Speech 한국어 TTS + 펄스 마스크 (`web_pwa/` 패턴 재사용).
   DANGER만 풀스크린 마스크, WARN은 bbox 색상 + 음성만 (시야 유지).

> head+shoulders 면적은 전신 bbox의 약 1/10이므로 `shared/hazard.json`의
> `area_*_ratio`는 일반적인 person detector 임계값보다 작은 값을 사용한다
> (warn=1.5%, danger=5% 기본). 카메라 마운트 확정 후 재튜닝 권장.

## 파일 구조 (목표)

```
stroller_pwa/
├── index.html
├── package.json
├── tsconfig.json
├── vite.config.ts
├── README.md
└── src/
    ├── main.ts          # 카메라 + 추론 + 렌더 루프
    ├── detector.ts      # MediaPipe Object Detector 래퍼
    ├── tracker.ts       # IOU 트래커 (Phase 2)
    ├── approach.ts      # 면적 시계열 → 접근 속도 (Phase 3)
    ├── stateMachine.ts  # SAFE/WARN/DANGER (Phase 3)
    ├── overlay.ts       # 알람 마스크 (Phase 4)
    └── tts.ts           # 음성 알람 (Phase 4)
```

## 라이선스

Apache 2.0 (data_test/ · web_pwa/ 와 동일)
