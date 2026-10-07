"""수집한 손 랜드마크 CSV로 커스텀 제스처 분류기 학습.

실행 예:
    python train_gestures.py
    python train_gestures.py --data data/gestures.csv --out models/custom_gesture.joblib

결과:
    - 검증 정확도, 클래스별 precision/recall, 혼동 행렬 출력
    - 전체 데이터로 다시 학습한 모델을 joblib 파일로 저장
"""
import argparse
import csv
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from gesture_features import NUM_LANDMARKS, row_to_feature


def load_dataset(csv_path):
    X, y = [], []
    coord_cols = [f"{a}{i}" for i in range(NUM_LANDMARKS) for a in ("x", "y", "z")]
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            coords = [float(row[c]) for c in coord_cols]
            X.append(row_to_feature(coords, row["handedness"]))
            y.append(row["label"])
    return np.array(X), np.array(y)


def build_model(seed):
    return make_pipeline(
        StandardScaler(),
        # early_stopping(검증 정확도 기준)은 정확도가 금방 1.0에 도달하면 너무 일찍 멈춰서
        # 확률이 0.5 근처로 낮게 나온다 -> 학습 손실 기준으로 수렴할 때까지 학습
        MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-3,
                      max_iter=500, random_state=seed),
    )


def train_model(data, out, test_size=0.2, seed=42, log=print):
    """학습 + 검증 + 저장. 문제가 있으면 ValueError. GUI(gesture_studio.py)에서도 호출한다."""
    csv_path = Path(data)
    if not csv_path.exists():
        raise ValueError(f"데이터가 없습니다: {csv_path}  (먼저 데이터를 수집하세요)")

    X, y = load_dataset(csv_path)
    counts = Counter(y)
    log(f"샘플 {len(y)}개, 클래스 {len(counts)}개")
    for label, n in sorted(counts.items()):
        log(f"  {label}: {n}")

    if len(counts) < 2:
        raise ValueError("클래스가 2개 이상 필요합니다. (예: 원하는 제스처 + none)")
    too_few = [k for k, n in counts.items() if n < 10]
    if too_few:
        raise ValueError(f"샘플이 10개 미만인 클래스가 있습니다: {too_few}")

    # 1) 검증: 일부 데이터를 떼어서 성능 확인
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed)
    model = build_model(seed).fit(X_tr, y_tr)
    pred = model.predict(X_te)

    labels = sorted(counts)
    accuracy = accuracy_score(y_te, pred)
    log(f"\n검증 정확도: {accuracy:.3f}\n")
    log(classification_report(y_te, pred, labels=labels, digits=3))
    log("혼동 행렬 (행=정답, 열=예측):")
    cm = confusion_matrix(y_te, pred, labels=labels)
    width = max(len(s) for s in labels)
    log(" " * (width + 2) + " ".join(f"{s[:6]:>6}" for s in labels))
    for label, row in zip(labels, cm):
        log(f"{label:>{width}}  " + " ".join(f"{v:>6}" for v in row))

    # 2) 최종: 전체 데이터로 다시 학습해서 저장
    final = build_model(seed).fit(X, y)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": final, "labels": [str(c) for c in final.classes_]}, out)
    log(f"\n모델 저장: {out.resolve()}")
    return {"accuracy": accuracy, "path": out}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/gestures.csv")
    parser.add_argument("--out", default="models/custom_gesture.joblib")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        train_model(args.data, args.out, args.test_size, args.seed)
    except ValueError as e:
        raise SystemExit(str(e))


if __name__ == "__main__":
    main()
