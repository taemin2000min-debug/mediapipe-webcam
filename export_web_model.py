"""학습한 모델(joblib)을 웹 데모용 JSON으로 변환.

실행: python export_web_model.py [--model models/custom_gesture.joblib] [--out docs/model.json]

StandardScaler(평균/표준편차) + MLPClassifier(가중치/편향)를 그대로 내보내고,
docs/app.js 에서 같은 계산(정규화 -> ReLU 은닉층 -> softmax)을 수행한다.
검증용으로 학습 데이터 몇 개의 입력/정답 확률도 함께 저장한다.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from train_gestures import load_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/custom_gesture.joblib")
    parser.add_argument("--data", default="data/gestures.csv")
    parser.add_argument("--out", default="docs/model.json")
    args = parser.parse_args()

    bundle = joblib.load(args.model)
    pipe = bundle["model"]
    scaler, mlp = pipe.named_steps["standardscaler"], pipe.named_steps["mlpclassifier"]
    if mlp.activation != "relu" or mlp.out_activation_ != "softmax":
        raise SystemExit(f"지원하지 않는 구조: {mlp.activation}/{mlp.out_activation_}")

    def r(a):
        return np.round(np.asarray(a, dtype=np.float64), 7).tolist()

    out = {
        "labels": [str(c) for c in mlp.classes_],
        "mean": r(scaler.mean_),
        "scale": r(scaler.scale_),
        "layers": [{"W": r(W), "b": r(b)} for W, b in zip(mlp.coefs_, mlp.intercepts_)],
    }

    # JS 구현 검증용 샘플 (클래스별 2개씩)
    if Path(args.data).exists():
        X, y = load_dataset(args.data)
        idx = [i for c in mlp.classes_ for i in np.flatnonzero(y == c)[:2]]
        out["checks"] = [{"x": r(X[i]), "proba": r(pipe.predict_proba(X[i:i + 1])[0])} for i in idx]

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    print(f"저장: {path}  ({path.stat().st_size / 1024:.0f} KB, 클래스 {out['labels']})")


if __name__ == "__main__":
    main()
