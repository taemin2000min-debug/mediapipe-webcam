"""커스텀 제스처 학습 데이터 수집 (웹캠).

손 랜드마크 21점을 CSV에 저장합니다. 여러 번 실행하면 같은 파일에 이어서 추가됩니다.

실행 예:
    python collect_gestures.py --labels none,ok,rock,call --target 300

키:
    1~9     수집할 제스처 선택 (--labels 순서)
    SPACE   녹화 시작/정지 (녹화 중에는 손이 보이는 프레임마다 저장)
    q/ESC   종료
"""
import argparse
import csv
import time
from collections import Counter
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from gesture_features import CSV_COLUMNS, landmarks_to_row

MODEL_PATH = Path(__file__).parent / "hand_landmarker.task"

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
    (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]


def load_counts(csv_path):
    if not csv_path.exists():
        return Counter()
    with open(csv_path, newline="", encoding="utf-8") as f:
        return Counter(row["label"] for row in csv.DictReader(f))


def draw_hand(frame, landmarks, color):
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], color, 2)
    for p in pts:
        cv2.circle(frame, p, 4, (255, 255, 255), -1)


def draw_panel(frame, labels, current, counts, recording, target):
    y = 30
    status = "REC" if recording else "PAUSE"
    color = (0, 0, 255) if recording else (200, 200, 200)
    cv2.putText(frame, f"[{status}] SPACE: rec, 1-9: label, q: quit",
                (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    for i, label in enumerate(labels):
        y += 26
        mark = ">" if i == current else " "
        n = counts[label]
        done = target and n >= target
        c = (0, 255, 255) if i == current else ((0, 200, 0) if done else (255, 255, 255))
        goal = f"/{target}" if target else ""
        cv2.putText(frame, f"{mark} {i + 1}. {label}: {n}{goal}",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)
    if recording:
        cv2.circle(frame, (frame.shape[1] - 25, 25), 10, (0, 0, 255), -1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True,
                        help="쉼표로 구분한 제스처 이름 (최대 9개). 예: none,ok,rock")
    parser.add_argument("--out", default="data/gestures.csv")
    parser.add_argument("--target", type=int, default=300,
                        help="제스처당 목표 샘플 수. 도달하면 녹화 자동 정지 (0 = 제한 없음)")
    parser.add_argument("--every", type=int, default=2,
                        help="N프레임마다 1개 저장 (연속 프레임 중복 줄이기)")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    labels = [s.strip() for s in args.labels.split(",") if s.strip()]
    if not 1 <= len(labels) <= 9:
        raise SystemExit("--labels 는 1~9개여야 합니다.")

    csv_path = Path(args.out)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not csv_path.exists()
    counts = load_counts(csv_path)

    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
    )
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"카메라 {args.camera}를 열 수 없습니다.")

    current, recording, frame_idx = 0, False, 0
    start = time.monotonic()
    with open(csv_path, "a", newline="", encoding="utf-8") as f, \
            vision.HandLandmarker.create_from_options(options) as landmarker:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(CSV_COLUMNS)

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # 추론 스크립트와 동일하게 거울 모드
            h, w = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(
                mp_image, int((time.monotonic() - start) * 1000))

            frame_idx += 1
            if result.hand_landmarks:
                landmarks = result.hand_landmarks[0]
                handedness = result.handedness[0][0].category_name
                draw_hand(frame, landmarks, (0, 0, 255) if recording else (0, 255, 0))

                label = labels[current]
                if recording and frame_idx % args.every == 0:
                    writer.writerow([label, handedness] + landmarks_to_row(landmarks, w, h))
                    f.flush()
                    counts[label] += 1
                    if args.target and counts[label] >= args.target:
                        recording = False
                        print(f"'{label}' 목표 {args.target}개 도달 -> 녹화 정지")

            draw_panel(frame, labels, current, counts, recording, args.target)
            cv2.imshow("Collect Gestures", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                recording = not recording
            if ord("1") <= key < ord("1") + len(labels):
                current = key - ord("1")
                recording = False

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n저장 위치: {csv_path.resolve()}")
    for label in labels:
        print(f"  {label}: {counts[label]}")


if __name__ == "__main__":
    main()
