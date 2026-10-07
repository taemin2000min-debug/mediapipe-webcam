# MediaPipe Webcam Demos

[MediaPipe Tasks](https://developers.google.com/edge/mediapipe/solutions/guide) (Python) 기반 웹캠 실시간 비전 데모 모음입니다.

| 스크립트 | 기능 | 모델 |
|---|---|---|
| `hand_webcam.py` | 손 랜드마크 21점 검출, 좌/우 손 구분 | `hand_landmarker.task` |
| `gesture_webcam.py` | 손 제스처 인식 (7종) | `gesture_recognizer.task` |
| `face_webcam.py` | 얼굴 랜드마크 478점 + 표정(블렌드쉐이프) | `face_landmarker.task` |

모델 파일(`.task`)은 저장소에 포함되어 있어 별도 다운로드 없이 바로 실행됩니다.

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

## 구현 메모

- 세 스크립트 모두 MediaPipe Tasks API의 `RunningMode.VIDEO`를 사용합니다. 프레임마다 동기적으로 `detect_for_video` / `recognize_for_video`를 호출하고, 단조 증가 타임스탬프(ms)를 넘깁니다.
- mediapipe 1.x에서는 기존 `mp.solutions.drawing_utils`를 쓰지 않고, OpenCV로 직접 그립니다. 얼굴 연결선은 `vision.FaceLandmarksConnections`를 사용합니다.
- 시작할 때 출력되는 `XNNPACK delegate`, `inference_feedback_manager` 등의 로그는 정상 동작이므로 무시해도 됩니다.

## 모델 출처

Google AI Edge 공식 모델 (float16, latest)을 사용합니다.

- [Hand landmarks detection guide](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker)
- [Gesture recognition guide](https://developers.google.com/edge/mediapipe/solutions/vision/gesture_recognizer)
- [Face landmark detection guide](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker)
