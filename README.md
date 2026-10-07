# MediaPipe Webcam Demos

[MediaPipe Tasks](https://developers.google.com/edge/mediapipe/solutions/guide) (Python) 기반 웹캠 실시간 비전 데모 모음입니다.

| 스크립트 | 기능 | 모델 |
|---|---|---|
| `hand_webcam.py` | 손 랜드마크 21점 검출, 좌/우 손 구분 | `hand_landmarker.task` |
| `gesture_webcam.py` | 손 제스처 인식 (7종) | `gesture_recognizer.task` |
| `face_webcam.py` | 얼굴 랜드마크 478점 + 표정(블렌드쉐이프) | `face_landmarker.task` |
| `gesture_studio.py` | **GUI 앱**: 내 제스처 수집 → 학습 → 실시간 인식 | `hand_landmarker.task` + 직접 학습한 모델 |
| `collect_gestures.py` / `train_gestures.py` / `custom_gesture_webcam.py` | 위 과정을 명령줄로 단계별 실행 | `hand_landmarker.task` + 직접 학습한 모델 |
| `export_web_model.py` + `docs/` | 학습한 모델을 웹 데모(GitHub Pages)로 내보내기 | 직접 학습한 모델 |

모델 파일(`.task`)은 저장소에 포함되어 있어 별도 다운로드 없이 바로 실행됩니다.

## 🌐 웹 데모 (설치 없이 바로 실행)

**https://taemin2000min-debug.github.io/mediapipe-webcam/**

직접 수집·학습한 가위바위보 모델(rock 🪨 / scissors ✂️ / paper ✋)을 브라우저에서 웹캠으로 실시간 인식합니다. 영상은 브라우저 안에서만 처리되고 서버로 전송되지 않습니다.

- 소스: `docs/` 폴더 (GitHub Pages로 배포)
- 손 인식: MediaPipe `@mediapipe/tasks-vision` (JavaScript)
- 분류: 학습한 scikit-learn 모델을 `export_web_model.py`로 `docs/model.json`에 내보내고, `docs/app.js`가 같은 계산을 수행합니다. 페이지 하단의 "모델 검증 통과"는 JS 결과가 Python 결과와 일치한다는 뜻입니다.

모델을 다시 학습했다면 웹에 반영하기:

```bash
python export_web_model.py      # models/custom_gesture.joblib -> docs/model.json
git add docs/model.json && git commit -m "Update web model" && git push
```

## 설치

```bash
pip install -r requirements.txt
```

- 테스트 환경: Windows 11, Python 3.12 / 3.14, mediapipe 1.1.0, opencv-python 5.0.0
- Windows에서 `python`과 `pip`가 서로 다른 Python을 가리킬 수 있으니 `python -m pip install -r requirements.txt` 사용을 권장합니다.

## 실행

```bash
python hand_webcam.py      [--camera 0] [--hands 2]
python gesture_webcam.py   [--camera 0] [--hands 2]
python face_webcam.py      [--camera 0] [--faces 1]
```

- `--camera`: 웹캠 번호 (기본 0, 외장 캠은 1 등)
- 화면은 거울 모드(좌우 반전)로 표시됩니다.
- 모든 창에서 `q` 또는 `ESC`로 종료합니다.

## 데모 상세

### 1. Hand Landmarker — `hand_webcam.py`

- 손 하나당 21개 키포인트(손목, 각 손가락 관절)를 검출하고 뼈대를 그립니다.
- 손가락 끝(4, 8, 12, 16, 20번)은 빨간 점으로 표시합니다.
- 손목 근처에 `Left` / `Right` 라벨과 신뢰도를 표시합니다.

### 2. Gesture Recognizer — `gesture_webcam.py`

손 랜드마크 위에 인식된 제스처를 라벨로 표시합니다 (예: `Right: Thumb_Up (0.87)`).

| 제스처 | 의미 |
|---|---|
| `Closed_Fist` | 주먹 |
| `Open_Palm` | 손바닥 펴기 (보) |
| `Pointing_Up` | 검지 위로 |
| `Thumb_Up` | 엄지 위로 |
| `Thumb_Down` | 엄지 아래로 |
| `Victory` | 브이 |
| `ILoveYou` | 사랑해 (🤟) |
| `None` | 해당 없음 |

OpenCV는 화면에 한글을 그릴 수 없어서, 한글 제스처 이름은 제스처가 바뀔 때마다 콘솔에 출력합니다.

### 3. Face Landmarker — `face_webcam.py`

- 얼굴 메시(478점)와 함께 얼굴형·입술·눈·눈썹·홍채 윤곽을 색상별로 그립니다.
- 좌측 상단에 블렌드쉐이프(표정 계수) 상위 8개를 막대그래프로 표시합니다.
  예: `eyeBlinkLeft`(눈 깜빡임), `mouthSmileRight`(미소), `jawOpen`(입 벌림)

| 키 | 동작 |
|---|---|
| `m` | 메시 표시 on/off |
| `b` | 블렌드쉐이프 표시 on/off |

### 4. 커스텀 제스처 학습 — `collect_gestures.py` → `train_gestures.py` → `custom_gesture_webcam.py`

원하는 제스처를 직접 수집·학습해서 실시간 인식합니다.
Hand Landmarker로 뽑은 21개 랜드마크를 손목 기준·크기 정규화한 63차원 특징으로 바꾸고(`gesture_features.py`), scikit-learn MLP 분류기로 학습합니다.

> MediaPipe 공식 Model Maker는 TensorFlow/tensorflow-text 의존성 때문에 Windows·Python 3.12+에서 설치되지 않아 이 방식을 사용합니다.

#### GUI로 한 번에: `gesture_studio.py` (추천)

```bash
python gesture_studio.py
```

한 창에서 탭으로 **수집 → 학습 → 실시간 인식**을 진행합니다.

| 탭 | 하는 일 |
|---|---|
| 1. 데이터 수집 | 제스처 이름 입력 후 `제스처 추가` → 목록에서 선택 → `녹화 시작` 또는 `Space`. 카운트다운 후 녹화되고, 목표 개수에 도달하면 자동으로 멈춥니다. 제스처별 개수와 진행률이 표시되고, 선택한 제스처의 데이터 삭제도 됩니다. |
| 2. 학습 | `학습 시작` 버튼 → 정확도·제스처별 결과·혼동 행렬이 로그로 출력되고 모델이 저장됩니다. 학습이 끝나면 모델이 인식 탭에 자동으로 로드됩니다. |
| 3. 실시간 인식 | 인식 결과를 크게 보여주고 클래스별 확률을 막대그래프로 표시합니다. 확신도 임계값·스무딩은 슬라이더로 조절합니다. |

- 영상 위에 한글 라벨도 표시됩니다(맑은 고딕).
- `rock` / `scissors` / `paper`로 인식되면 손 옆과 결과 패널에 🪨 / ✂️ / ✋ 이모지가 나타납니다. 다른 제스처에도 이모지를 붙이려면 `emoji_overlay.py`의 `GESTURE_EMOJI`에 `"라벨": "이모지"`를 추가하세요. (`custom_gesture_webcam.py`에도 똑같이 적용됩니다.)
- 수집 데이터는 `data/gestures.csv`, 모델은 `models/custom_gesture.joblib`에 저장되므로 아래 명령줄 스크립트와 그대로 호환됩니다.

#### 명령줄로 단계별 실행

```bash
pip install scikit-learn joblib

# 1) 수집: 1~9로 제스처 선택, SPACE로 녹화 시작/정지
python collect_gestures.py --labels none,ok,rock,call --target 300

# 2) 학습: 검증 정확도·혼동 행렬 출력 후 models/custom_gesture.joblib 저장
python train_gestures.py

# 3) 추론
python custom_gesture_webcam.py --threshold 0.7
```

| 단계 | 산출물 | 주요 옵션 |
|---|---|---|
| 수집 | `data/gestures.csv` (실행할 때마다 이어서 추가) | `--labels`, `--target`(제스처당 목표 개수, 도달 시 자동 정지), `--every`(N프레임마다 저장) |
| 학습 | `models/custom_gesture.joblib` | `--data`, `--out`, `--test-size` |
| 추론 | 화면에 `Right: ok (0.95)` 표시 | `--threshold`(미만이면 Unknown), `--smooth`(최근 N프레임 평균) |

잘 학습시키는 팁:
- **`none` 클래스를 꼭 넣으세요.** 아무 손 모양(편 손, 반쯤 쥔 손, 손 옮기는 중 등)을 모아두면 엉뚱한 제스처로 오인식하는 일이 크게 줄어듭니다.
- 녹화 중에 손을 **앞뒤·좌우로 움직이고 살짝 회전**시켜 다양한 각도를 담으세요. 제스처당 200~500개 정도면 충분합니다.
- 왼손은 자동으로 좌우 반전되므로 한 손으로만 수집해도 양손 모두 인식됩니다.
- 특정 제스처끼리 헷갈리면 혼동 행렬을 보고 그 제스처 데이터를 더 모은 뒤 다시 학습하세요.

## 구현 메모

- 세 스크립트 모두 MediaPipe Tasks API의 `RunningMode.VIDEO`를 사용합니다. 프레임마다 동기적으로 `detect_for_video` / `recognize_for_video`를 호출하고, 단조 증가 타임스탬프(ms)를 넘깁니다.
- mediapipe 1.x에서는 기존 `mp.solutions.drawing_utils`를 쓰지 않고, OpenCV로 직접 그립니다. 얼굴 연결선은 `vision.FaceLandmarksConnections`를 사용합니다.
- 시작할 때 출력되는 `XNNPACK delegate`, `inference_feedback_manager` 등의 로그는 정상 동작이므로 무시해도 됩니다.

## 모델 출처

Google AI Edge 공식 모델 (float16, latest)을 사용합니다.

- [Hand landmarks detection guide](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker)
- [Gesture recognition guide](https://developers.google.com/edge/mediapipe/solutions/vision/gesture_recognizer)
- [Face landmark detection guide](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker)
