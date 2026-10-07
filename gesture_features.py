"""수집/학습/추론에서 공통으로 쓰는 손 랜드마크 -> 특징 벡터 변환."""
import numpy as np

NUM_LANDMARKS = 21
CSV_COLUMNS = ["label", "handedness"] + [
    f"{axis}{i}" for i in range(NUM_LANDMARKS) for axis in ("x", "y", "z")
]


def landmarks_to_row(landmarks, width, height):
    """MediaPipe 정규화 좌표(0~1)를 픽셀 비율 좌표로 바꿔 CSV 한 줄용 리스트로 반환.

    x, y를 각각 width, height로 곱해야 화면 비율(16:9 등)에 의한 찌그러짐이 없다.
    z는 MediaPipe 기준으로 x와 같은 스케일이므로 width를 곱한다.
    """
    row = []
    for lm in landmarks:
        row += [lm.x * width, lm.y * height, lm.z * width]
    return row


def row_to_feature(coords, handedness):
    """좌표 63개 -> 위치/크기/좌우손에 무관한 63차원 특징.

    1. 손목(0번) 기준으로 평행이동 -> 화면 위치 무관
    2. 왼손은 x축 반전 -> 한 손으로 수집해도 양손 모두 인식
    3. 손목에서 가장 먼 점까지 거리로 나눔 -> 손 크기/카메라 거리 무관
    """
    pts = np.asarray(coords, dtype=np.float32).reshape(NUM_LANDMARKS, 3)
    pts = pts - pts[0]
    if handedness == "Left":
        pts[:, 0] = -pts[:, 0]
    scale = np.linalg.norm(pts[:, :2], axis=1).max()
    if scale > 1e-6:
        pts = pts / scale
    return pts.flatten()
