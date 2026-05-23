# Neulbom — 통합 PWA (Mode A + Mode B)

`web_pwa/`(차량 모드)와 `stroller_pwa/`(유모차 모드)를 하나의 PWA로 통합한 PoC.
모드 선택 화면에서 두 모드 중 하나를 골라 부팅하며, 각 모드는 동적 import로
필요할 때만 로드된다 (사용 안 하는 모드의 MediaPipe 모델은 다운로드되지 않음).

## 폴더 구조

```
neulbom_pwa/
├── index.html               # 모드 선택 화면 + 스테이지 컨테이너
├── package.json
├── tsconfig.json
├── vite.config.ts           # @shared alias + PWA
├── public/                  # 아이콘 (web_pwa에서 복사)
└── src/
    ├── main.ts              # 모드 라우터 (동적 import)
    ├── style.css            # 공통 셸·패널·튠 CSS
    ├── modes/
    │   ├── car/             # Mode A — 차량 모드 (FaceLandmarker)
    │   │   ├── boot.ts      # start(opts) / stop() API
    │   │   ├── panel.html   # 인트로 + 패널 마크업 (?raw import)
    │   │   ├── ear.ts
    │   │   ├── headPose.ts
    │   │   ├── stateMachine.ts
    │   │   ├── overlay.ts
    │   │   └── tts.ts
    │   └── stroller/        # Mode B — 유모차 모드 (PoseLandmarker)
    │       ├── boot.ts
    │       ├── panel.html
    │       ├── detector.ts
    │       ├── tracker.ts
    │       ├── approach.ts
    │       ├── stateMachine.ts
    │       ├── overlay.ts
    │       └── tts.ts
    └── vite-env.d.ts
```

`shared/*.json`은 두 모드 모두 `@shared/*` 별칭으로 접근한다 (랜드마크 인덱스,
임계값, 알람 메시지 등). 기존 `web_pwa/`·`stroller_pwa/`는 손대지 않고 병행 유지.

## 로컬 실행

```bash
cd neulbom_pwa
npm install
npm run dev   # http://localhost:5175
```

(기존 web_pwa는 5173, stroller_pwa는 5174 — 세 앱 동시 실행 가능)

## 동작 흐름

1. 첫 화면: 모드 선택 (🚗 차량 / 🛒 유모차 두 카드)
2. 카드 클릭 → 해당 모드의 boot.ts가 동적 로드, 스테이지에 panel.html 렌더
3. 인트로 → Start 클릭 → 카메라 + 추론 시작
4. 모드 내 ⏹ 종료 또는 ← 모드 선택으로 클릭 → boot.stop() → 모드 선택 화면 복귀

## 모드별 카메라

- 차량 모드: `facingMode: "user"` (전면 카메라 — 카시트 위 아이 촬영 시뮬레이션)
- 유모차 모드: `facingMode: "environment"` (후면 카메라 — 유모차 외부 촬영)
- 노트북 웹캠 환경에서는 양쪽 모두 단일 웹캠으로 자동 폴백

## 영속 상태 (localStorage)

각 모드가 독립된 키를 사용해 다른 모드의 설정에 영향 주지 않음.

| 키 | 모드 | 내용 |
|---|---|---|
| `neulbom.config.v1` | 차량 | EAR/pitch/roll/frame 슬라이더 값 |
| `neulbom.debugVis.v1` | 차량 | 시각화 ON/OFF |
| `neulbom.alarm.muted` | 차량 | 음성 mute |
| `neulbom.stroller.config.v1` | 유모차 | 면적/증가율/프레임 슬라이더 값 |
| `neulbom.stroller.debugVis.v1` | 유모차 | 시각화 ON/OFF |
| `neulbom.stroller.muted` | 유모차 | 음성 mute |

## 다음 단계 후보

- 공유 모듈 추출: TTS, 카메라 헬퍼, 렌더 루프 → `src/shared/`
- Vercel Project Root Directory를 `neulbom_pwa`로 전환
- 기존 `web_pwa/`·`stroller_pwa/` 폴더 정리 (검증 완료 후)

## 라이선스

Apache 2.0 (data_test/ · web_pwa/ · stroller_pwa/ 와 동일)
