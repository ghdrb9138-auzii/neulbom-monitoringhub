# Neulbom — Web PoC (Drowsiness + Head Pose)

브라우저용 졸음 + 머리 자세 감지 PoC. `data_test/`의 Python PoC를 1:1 포팅했다.
**랜드마크 인덱스 · 3D 얼굴 모델 · 임계값은 `../shared/*.json`을 단일 출처로 공유**한다.

## 요구사항

- Node.js 20+ (개발 시)
- 카메라가 있는 디바이스: 맥북 / iPad / iPhone
- iOS Safari 15.4+ (Apple Silicon iPad·iPhone X 이상 권장)

## 로컬 실행 (macOS)

```bash
cd web_pwa
npm install
npm run dev
```

Vite가 `http://localhost:5173`를 연다. 브라우저에서 **Start Detection** 클릭 → 카메라 권한 허용.

상태 바에 EAR, Pitch / Yaw / Roll, FPS가 표시되며 `data_test/`의 CLI와 동일한
상태 머신(AWAKE / EYES CLOSED / HEAD BENT / DROWSY / DANGER)이 화면 상단에 출력된다.

## iPad / iPhone에서 테스트

브라우저 카메라 API는 **HTTPS** (또는 localhost) 환경에서만 동작한다. 같은 와이파이
상의 LAN IP(`192.168.x.x`)는 HTTP라 카메라 권한이 거부된다. HTTPS 터널이 필요하다.

### 옵션 A — cloudflared (계정 불필요, 가장 간편)

```bash
brew install cloudflared
# 다른 터미널에서 dev 서버를 띄운 상태로:
cloudflared tunnel --url http://localhost:5173
```

콘솔에 출력되는 `https://xxx.trycloudflare.com` URL을 iPhone Safari에서 열면 된다.
URL은 매번 바뀐다. 임시 데모 / 본인 디바이스 테스트용으로 충분.

### 옵션 B — ngrok (계정 필요, 무료 플랜으로 충분)

```bash
brew install ngrok/ngrok/ngrok
ngrok config add-authtoken <YOUR_TOKEN>     # 최초 1회
ngrok http 5173
```

`https://xxx.ngrok-free.app`이 출력된다. 같은 토큰을 쓰면 동일한 무료 도메인이
유지된다 (계정에 따라).

### 옵션 C — Tailscale + HTTPS (장기 데모용)

Tailscale Magic DNS + Let's Encrypt 자동 발급. 셋업이 가장 복잡하지만 같은 디바이스
세트 내에서 항상 같은 주소로 접근 가능. 본선 데모 환경에 적합.

## 디바이스 권장 설정

| 디바이스 | 메모 |
|---------|------|
| MacBook (Apple Silicon) | Chrome / Safari 모두 가능. GPU delegate가 WebGL2로 활성화돼 30 FPS+ |
| iPad (M-series) | Safari 권장. 전면 카메라 12MP에서도 추론 부담 적음 |
| iPhone (A15+ 이상) | Safari. 첫 모델 로드 1-3초 후 안정적인 ~30 FPS |

## 알고리즘

- **EAR** — `shared/landmarks.json`의 `eye.left` / `eye.right` 6점 인덱스를 그대로 사용.
  계산식은 Python과 동일. (`src/ear.ts`)
- **Head Pose** — MediaPipe `outputFacialTransformationMatrixes` 옵션을 켜서 SDK가
  직접 계산한 4×4 변환 행렬을 받고, ZYX 오일러 분해로 pitch/yaw/roll 추출.
  Python은 `cv2.solvePnP` + 우리 6점 모델로 풀지만, MediaPipe 내부 모델도 같은
  canonical face mesh를 기준으로 하므로 결과는 거의 동등하다.
  (`src/headPose.ts`)
- **상태머신** — `shared/thresholds.json`의 EAR 0.21, pitch 20°, roll 25°, frames 30
  을 그대로 적용. (`src/stateMachine.ts`)

## 파일 구조

```
web_pwa/
├── index.html
├── package.json
├── tsconfig.json
├── vite.config.ts            # @shared → ../shared 별칭
├── src/
│   ├── main.ts               # 카메라 + 추론 + 렌더 루프
│   ├── ear.ts                # EAR (shared 인덱스)
│   ├── headPose.ts           # 4×4 → 오일러 분해
│   ├── stateMachine.ts       # 상태머신 + 임계값 (shared)
│   └── overlay.ts            # canvas 시각화
└── README.md
```

## 알려진 제약 / TODO

- iOS Safari는 일반 페이지에서 카메라 OK. **PWA 홈스크린 추가 후 카메라 권한 재요청
  버그**(WebKit #185448)는 phase 2에서 manifest/service worker 추가 시 우회 검토.
- MediaPipe WASM + 모델(~6MB) 첫 로드 1-3초. HTTP cache로 두 번째부터 즉시.
- `data_test/` Python과 pitch 부호가 다를 가능성이 있다면 라이브 테스트 후
  `headPose.ts`의 정규화 분기에서 보정.
- 캘리브레이션(`todolist.md` 1순위)은 Python·Web 양쪽에 추가될 예정.

## 라이선스

Apache 2.0 (data_test/ 와 동일)
