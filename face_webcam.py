"""MediaPipe Face Landmarker - 웹캠 실시간 얼굴 랜드마크(478점) 검출.

- 얼굴 메시 / 윤곽(눈, 눈썹, 입술, 얼굴형) / 홍채 그리기
- 블렌드쉐이프(표정 계수) 상위 항목 표시 (눈 깜빡임, 미소, 입 벌림 등)

실행: python face_webcam.py [--camera 0] [--faces 1]
키: m = 메시 on/off, b = 블렌드쉐이프 on/off, q/ESC = 종료
"""
import argparse
import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_PATH = Path(__file__).parent / "face_landmarker.task"

C = vision.FaceLandmarksConnections
CONTOUR_STYLES = [  # (연결 목록, BGR 색상)
    (C.FACE_LANDMARKS_FACE_OVAL, (224, 224, 224)),
    (C.FACE_LANDMARKS_LIPS, (80, 80, 255)),
    (C.FACE_LANDMARKS_LEFT_EYE, (80, 255, 80)),
    (C.FACE_LANDMARKS_RIGHT_EYE, (80, 255, 80)),
    (C.FACE_LANDMARKS_LEFT_EYEBROW, (80, 200, 255)),
    (C.FACE_LANDMARKS_RIGHT_EYEBROW, (80, 200, 255)),
    (C.FACE_LANDMARKS_LEFT_IRIS, (255, 200, 0)),
    (C.FACE_LANDMARKS_RIGHT_IRIS, (255, 200, 0)),
]


def draw_connections(frame, pts, connections, color, thickness):
    for c in connections:
        cv2.line(frame, pts[c.start], pts[c.end], color, thickness, cv2.LINE_AA)


def draw_faces(frame, result, show_mesh):
    h, w = frame.shape[:2]
    for landmarks in result.face_landmarks:
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
        if show_mesh:
            draw_connections(frame, pts, C.FACE_LANDMARKS_TESSELATION, (90, 90, 90), 1)
        for connections, color in CONTOUR_STYLES:
            draw_connections(frame, pts, connections, color, 2)


def draw_blendshapes(frame, result, top_k=8):
    if not result.face_blendshapes:
        return
    # 첫 번째 얼굴의 블렌드쉐이프 중 점수 높은 순 (_neutral 제외)
    shapes = sorted((b for b in result.face_blendshapes[0]
                     if b.category_name != "_neutral"),
                    key=lambda b: b.score, reverse=True)[:top_k]
    x, y = 10, 60
    for b in shapes:
        bar = int(b.score * 150)
        cv2.rectangle(frame, (x, y - 12), (x + bar, y + 2), (0, 180, 255), -1)
        cv2.putText(frame, f"{b.category_name} {b.score:.2f}", (x + 155, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        y += 22


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--faces", type=int, default=1)
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"모델 파일이 없습니다: {MODEL_PATH}")

    options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=args.faces,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=True,
    )

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"카메라 {args.camera}를 열 수 없습니다.")

    show_mesh, show_blend = True, True
    start = time.monotonic()
    prev = start
    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # 거울 모드

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((time.monotonic() - start) * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            draw_faces(frame, result, show_mesh)
            if show_blend:
                draw_blendshapes(frame, result)

            now = time.monotonic()
            fps = 1.0 / max(now - prev, 1e-6)
            prev = now
            cv2.putText(frame, f"FPS {fps:.1f}  faces {len(result.face_landmarks)}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            cv2.imshow("Face Landmarker", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("m"):
                show_mesh = not show_mesh
            if key == ord("b"):
                show_blend = not show_blend

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
