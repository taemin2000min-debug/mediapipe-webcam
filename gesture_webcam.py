"""MediaPipe Gesture Recognizer - 웹캠 실시간 손 제스처 인식.

인식 제스처: Closed_Fist, Open_Palm, Pointing_Up, Thumb_Down, Thumb_Up,
            Victory, ILoveYou (그 외는 None)

실행: python gesture_webcam.py [--camera 0] [--hands 2]
종료: q 또는 ESC
"""
import argparse
import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_PATH = Path(__file__).parent / "gesture_recognizer.task"

# 21개 랜드마크 간 연결 (손목-엄지-검지-중지-약지-새끼 + 손바닥)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # 엄지
    (0, 5), (5, 6), (6, 7), (7, 8),          # 검지
    (5, 9), (9, 10), (10, 11), (11, 12),     # 중지
    (9, 13), (13, 14), (14, 15), (15, 16),   # 약지
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),  # 새끼 + 손바닥
]
FINGERTIPS = {4, 8, 12, 16, 20}

GESTURE_KO = {
    "None": "-",
    "Closed_Fist": "주먹",
    "Open_Palm": "보",
    "Pointing_Up": "검지 위",
    "Thumb_Down": "엄지 아래",
    "Thumb_Up": "엄지 위",
    "Victory": "브이",
    "ILoveYou": "사랑해",
}


def draw_hands(frame, result):
    h, w = frame.shape[:2]
    for landmarks, handedness, gestures in zip(
            result.hand_landmarks, result.handedness, result.gestures):
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)
        for i, p in enumerate(pts):
            color = (0, 0, 255) if i in FINGERTIPS else (255, 255, 255)
            cv2.circle(frame, p, 5, color, -1)

        # 손 바운딩박스 위에 제스처 + 좌/우 손 표시
        xs, ys = zip(*pts)
        x0, y0 = min(xs), min(ys)
        gesture = gestures[0]
        hand = handedness[0].category_name
        text = f"{hand}: {gesture.category_name} ({gesture.score:.2f})"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(frame, (x0, y0 - th - 16), (x0 + tw + 10, y0 - 4),
                      (40, 40, 40), -1)
        cv2.putText(frame, text, (x0 + 5, y0 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--hands", type=int, default=2)
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"모델 파일이 없습니다: {MODEL_PATH}")

    options = vision.GestureRecognizerOptions(
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
    last_printed = None
    with vision.GestureRecognizer.create_from_options(options) as recognizer:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # 거울 모드

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((time.monotonic() - start) * 1000)
            result = recognizer.recognize_for_video(mp_image, timestamp_ms)

            draw_hands(frame, result)

            # 제스처가 바뀔 때만 콘솔에 출력 (한글은 OpenCV putText로 못 그려서 콘솔로)
            current = tuple(
                f"{hd[0].category_name}:{GESTURE_KO.get(g[0].category_name, g[0].category_name)}"
                for hd, g in zip(result.handedness, result.gestures))
            if current != last_printed:
                print(" | ".join(current) if current else "(손 없음)")
                last_printed = current

            now = time.monotonic()
            fps = 1.0 / max(now - prev, 1e-6)
            prev = now
            cv2.putText(frame, f"FPS {fps:.1f}  hands {len(result.hand_landmarks)}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            cv2.imshow("Gesture Recognizer", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
