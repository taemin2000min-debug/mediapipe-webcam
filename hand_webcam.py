"""MediaPipe Hand Landmarker - 웹캠 실시간 손 랜드마크 검출.

실행: python hand_webcam.py [--camera 0] [--hands 2]
종료: q 또는 ESC
"""
import argparse
import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_PATH = Path(__file__).parent / "hand_landmarker.task"

# 21개 랜드마크 간 연결 (손목-엄지-검지-중지-약지-새끼 + 손바닥)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # 엄지
    (0, 5), (5, 6), (6, 7), (7, 8),          # 검지
    (5, 9), (9, 10), (10, 11), (11, 12),     # 중지
    (9, 13), (13, 14), (14, 15), (15, 16),   # 약지
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),  # 새끼 + 손바닥
]
FINGERTIPS = {4, 8, 12, 16, 20}


def draw_hands(frame, result):
    h, w = frame.shape[:2]
    for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)
        for i, p in enumerate(pts):
            color = (0, 0, 255) if i in FINGERTIPS else (255, 255, 255)
            cv2.circle(frame, p, 5, color, -1)

        # 손목 위에 좌/우 손 라벨 표시
        label = handedness[0]
        x, y = pts[0]
        cv2.putText(frame, f"{label.category_name} {label.score:.2f}",
                    (x - 30, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 128, 0), 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--hands", type=int, default=2)
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"모델 파일이 없습니다: {MODEL_PATH}")

    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=args.hands,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"카메라 {args.camera}를 열 수 없습니다.")

    start = time.monotonic()
    prev = start
    with vision.HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # 거울 모드

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((time.monotonic() - start) * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            draw_hands(frame, result)

            now = time.monotonic()
            fps = 1.0 / max(now - prev, 1e-6)
            prev = now
            cv2.putText(frame, f"FPS {fps:.1f}  hands {len(result.hand_landmarks)}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            cv2.imshow("Hand Landmarker", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
