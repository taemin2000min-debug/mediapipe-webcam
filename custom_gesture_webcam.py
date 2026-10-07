"""학습한 커스텀 제스처 모델로 웹캠 실시간 추론.

실행 예:
    python custom_gesture_webcam.py
    python custom_gesture_webcam.py --model models/custom_gesture.joblib --threshold 0.8

확률이 --threshold 미만이면 'Unknown'으로 표시합니다.
종료: q 또는 ESC
"""
import argparse
import time
from collections import deque
from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from emoji_overlay import paste_emoji_bgr
from gesture_features import landmarks_to_row, row_to_feature

MODEL_PATH = Path(__file__).parent / "hand_landmarker.task"

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
    (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/custom_gesture.joblib")
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--smooth", type=int, default=5,
                        help="최근 N프레임 확률 평균으로 떨림 완화 (1 = 끔)")
    parser.add_argument("--hands", type=int, default=2)
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    bundle = joblib.load(args.model)
    clf, labels = bundle["model"], bundle["labels"]
    print(f"모델 로드: {args.model}  클래스: {labels}")

    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=args.hands,
    )
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"카메라 {args.camera}를 열 수 없습니다.")

    history = {}  # handedness -> 최근 확률들
    start = time.monotonic()
    prev = start
    with vision.HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(
                mp_image, int((time.monotonic() - start) * 1000))

            seen = set()
            for landmarks, hd in zip(result.hand_landmarks, result.handedness):
                hand = hd[0].category_name
                seen.add(hand)
                feat = row_to_feature(landmarks_to_row(landmarks, w, h), hand)
                proba = clf.predict_proba(feat[None, :])[0]

                hist = history.setdefault(hand, deque(maxlen=max(args.smooth, 1)))
                hist.append(proba)
                avg = np.mean(hist, axis=0)
                best = int(np.argmax(avg))
                name = labels[best] if avg[best] >= args.threshold else "Unknown"

                pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
                for a, b in HAND_CONNECTIONS:
                    cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)
                for p in pts:
                    cv2.circle(frame, p, 4, (255, 255, 255), -1)

                xs, ys = zip(*pts)
                x0, y0 = min(xs), min(ys)
                text = f"{hand}: {name} ({avg[best]:.2f})"
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(frame, (x0, y0 - th - 16), (x0 + tw + 10, y0 - 4),
                              (40, 40, 40), -1)
                cv2.putText(frame, text, (x0 + 5, y0 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                # rock 등 이모지가 지정된 제스처면 손 오른쪽에 이모지 표시
                paste_emoji_bgr(frame, name, min(max(xs) + 10, w - 110), max(0, y0), 110)

            for hand in list(history):  # 사라진 손의 기록은 초기화
                if hand not in seen:
                    del history[hand]

            now = time.monotonic()
            fps = 1.0 / max(now - prev, 1e-6)
            prev = now
            cv2.putText(frame, f"FPS {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            cv2.imshow("Custom Gesture", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
