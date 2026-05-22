# Drowsiness + Head-Pose Detector (MVP)

MediaPipe Face Mesh 기반 실시간 **졸음 감지** + **머리 자세 추정** 프로토타입.
**차량 카시트·유모차 유아 안전 모니터링**을 목표로 하지만 현 단계는 성인 검증 중.

## 검출 신호

| 신호 | 방법 | 의미 |
|------|------|------|
| **눈 감음** | 6점 EAR (Eye Aspect Ratio) | 졸음 후보 |
| **고개 떨어짐** | cv2.solvePnP 6점 머리 자세 | 자세성 질식 위험 (pitch ↑) |
| **결합 알람** | 위 두 신호 동시 + 시간 지속 | 진짜 위험 상태 |

## 상태 머신

| 상태 | 조건 | 화면 |
|------|------|------|
| AWAKE | 눈 뜸 + 머리 정상 | 일반 |
| EYES CLOSED | EAR < 임계값 (단발) | 일반 |
| HEAD FORWARD | pitch > 임계값 (단발) | 일반 |
| SLEEPING? | 눈 감음 + 머리 떨어짐 (단발) | 일반 |
| DROWSY | 눈 감음 N프레임 지속 | 🔴 빨강 마스크 |
| HEAD BENT | 머리 떨어짐 N프레임 지속 | 🟠 주황 마스크 |
| **DANGER** | 두 조건 동시 지속 | 🚨 빨강 마스크 + DANGER |

## 요구사항

- macOS / Linux / Windows
- Python 3.9 이상
- 웹캠 (내장 또는 USB 외장)

## 설치

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## 실행

### 1) 카메라 인덱스 확인

```bash
.venv/bin/python list_cameras.py
```

### 2) 감지기 실행

```bash
# 기본 카메라
.venv/bin/python drowsiness_detector.py

# 외장 카메라
.venv/bin/python drowsiness_detector.py --camera 1
```

### 키 조작

| 키 | 동작 |
|----|------|
| `q` | 종료 |
| `[` | EAR 임계값 ↓ (덜 민감) |
| `]` | EAR 임계값 ↑ (더 민감) |
| `,` | Pitch 임계값 ↓ (고개 떨어짐 더 빨리 감지) |
| `.` | Pitch 임계값 ↑ (덜 민감) |

### 화면 표시

- **눈 다각형**: 초록 = 뜸, 빨강 = 감음
- **3D 좌표축** (코 끝 기준): 빨강 = X(우), 초록 = Y(위), 파랑 = Z(전방)
- **상태 바**: EAR, Pitch/Yaw/Roll, FPS, 지속 프레임 수

## 폴더 이미지 일괄 테스트

```bash
.venv/bin/python test_on_photos.py path/to/folder
.venv/bin/python test_on_photos.py path/to/folder --threshold 0.21
```

## 구성

```
.
├── drowsiness_detector.py    # 메인 실시간 감지기
├── list_cameras.py           # 카메라 인덱스 탐색
├── test_on_photos.py         # 폴더 일괄 EAR 테스트
├── requirements.txt
└── src/
    ├── config.py             # shared/*.json 로더 (단일 출처)
    ├── ear.py                # EAR 계산
    ├── head_pose.py          # solvePnP 기반 머리 자세
    └── overlay.py            # 시각화 + 알람 마스크
```

랜드마크 인덱스 · 3D 얼굴 모델 · 임계값은 레포 루트의 `shared/` 디렉터리에서
JSON으로 관리되며, 향후 추가될 `web_pwa/`(브라우저 PoC) 및 Jetson Orin 본체
노드와 단일 출처로 공유된다.

```
../shared/
├── landmarks.json        # 눈 6점 + PnP 6점 인덱스
├── face_model_3d.json    # 6점 3D 얼굴 모델 (mm)
└── thresholds.json       # EAR / pitch / roll / frame 임계값 기본값
```

값을 바꾸려면 위 JSON을 편집하면 된다 — Python 소스에 하드코딩된 값은 없다.

## 파라미터 (기본값은 `shared/thresholds.json`)

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `ear.threshold` | 0.21 | EAR이 이 값 미만이면 눈 감음 |
| `ear.drowsy_frames` | 30 | 연속 N프레임 감음 → 졸음 (~30 FPS에서 1초) |
| `head_pose.pitch_threshold_deg` | 20.0 | pitch(도)가 이 값 초과면 고개 앞으로 떨어짐 |
| `head_pose.roll_threshold_deg` | 25.0 | \|roll\|이 이 값 초과면 옆으로 기울어짐 |
| `head_pose.bent_frames` | 30 | 연속 N프레임 떨어짐 → HEAD BENT |

## 알고리즘

### EAR (Soukupová & Čech, 2016)

```
EAR = (||p2 - p6|| + ||p3 - p5||) / (2 × ||p1 - p4||)
```

### 머리 자세 (cv2.solvePnP)

6개 얼굴 랜드마크 (코끝, 턱, 양 눈 외측, 양 입꼬리)와 표준 3D 얼굴 모델을
대응시켜 회전 벡터를 구하고 Euler 각도(pitch/yaw/roll)로 분해.

- **Pitch** > 0: 고개가 앞으로 (턱이 가슴 쪽) — **자세성 질식 위험 신호**
- **Yaw**: 좌우 시선 방향
- **Roll**: 좌우 기울임

## 한계 및 주의

- **MediaPipe Face Mesh는 성인 위주 학습**: 영아·신생아 얼굴 인식률 검증 필요
- 안경 반사·옆모습·저조도에서 정확도 저하
- 단일 카메라 + 단안이므로 절대 거리/3D 위치는 부정확
- **이 시스템은 부모 감독을 대체하지 않습니다.** 차량/유모차 환경 검증은 추가 작업 필요

## 향후 작업

- [ ] 개인 캘리브레이션 (사용자별 baseline EAR)
- [ ] PERCLOS (시간당 눈 감음 비율)
- [ ] 하품 감지 (MAR - Mouth Aspect Ratio)
- [ ] MediaPipe Pose 통합 (어깨 기준 머리 위치 보정)
- [ ] 시계열 모델 (LSTM 등)
- [ ] 유아 데이터 검증 (윤리 검토 후)
- [ ] IR/저조도 환경 강건성
- [ ] 알람 정교화 (단계별, 부모 알림 채널)

## 라이선스

Apache License 2.0 — [LICENSE](LICENSE) 참조.

## 의존성 라이선스

- MediaPipe — Apache 2.0
- OpenCV — Apache 2.0
- NumPy — BSD 3-Clause
