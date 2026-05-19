"""
학습된 모델 평가

지표: Accuracy, F1-Score (클래스별 + macro), FPR, FNR, 혼동행렬
사용법:
  python model/evaluate.py
  python model/evaluate.py --data path/to/data.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from cnn_lstm import CNNLSTMModel
from dataset import SENSOR_COLUMNS, generate_dummy_data, get_dataloaders, preprocess

LABEL_NAMES = {0: "정상", 1: "주의", 2: "위험"}


# ─────────────────────────────────────────────
# 예측 수행
# ─────────────────────────────────────────────

def predict_all(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            logits = model(X)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())

    return np.array(all_labels), np.array(all_preds)


# ─────────────────────────────────────────────
# 지표 계산
# ─────────────────────────────────────────────

def compute_metrics(y_true, y_pred, n_classes=3):
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_per = f1_score(y_true, y_pred, average=None, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))

    # 클래스별 FPR, FNR
    fpr_list, fnr_list = [], []
    for i in range(n_classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp           # 실제 i인데 다른 클래스로 예측
        fp = cm[:, i].sum() - tp           # 실제 i 아닌데 i로 예측
        tn = cm.sum() - tp - fn - fp

        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0   # 오탐율
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0   # 미탐율
        fpr_list.append(fpr)
        fnr_list.append(fnr)

    return {
        "accuracy":  acc,
        "f1_macro":  f1_macro,
        "f1_per":    f1_per,
        "fpr":       fpr_list,
        "fnr":       fnr_list,
        "confusion": cm,
    }


# ─────────────────────────────────────────────
# 결과 출력
# ─────────────────────────────────────────────

def print_results(metrics, n_classes=3):
    print("\n" + "=" * 50)
    print("평가 결과")
    print("=" * 50)

    print(f"\n전체 정확도 (Accuracy): {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.1f}%)")
    print(f"F1-Score (macro):       {metrics['f1_macro']:.4f}")

    print("\n" + "-" * 50)
    print(f"{'클래스':<8} {'F1-Score':>10} {'오탐율(FPR)':>12} {'미탐율(FNR)':>12}")
    print("-" * 50)
    for i in range(n_classes):
        name = LABEL_NAMES[i]
        print(
            f"{name}({i})  "
            f"{metrics['f1_per'][i]:>10.4f} "
            f"{metrics['fpr'][i]:>12.4f} "
            f"{metrics['fnr'][i]:>12.4f}"
        )

    print("\n" + "-" * 50)
    print("혼동행렬 (행=실제, 열=예측)")
    print("-" * 50)
    header = f"{'':8}" + "".join(f"{LABEL_NAMES[i]:>8}" for i in range(n_classes))
    print(header)
    for i in range(n_classes):
        row = f"{LABEL_NAMES[i]}({i})  " + "".join(f"{metrics['confusion'][i,j]:>8}" for j in range(n_classes))
        print(row)

    print("\n[해석]")
    print("  오탐율(FPR): 정상인데 위험/주의로 잘못 판단한 비율 → 낮을수록 좋음")
    print("  미탐율(FNR): 위험인데 정상으로 놓친 비율           → 낮을수록 좋음")
    print("=" * 50)


# ─────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────

def evaluate(data_path=None, model_path=None):
    # 장치
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"사용 장치: {device}")

    # 모델 로드
    if model_path is None:
        model_path = Path(__file__).parent / "saved" / "best_model.pt"

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    print(f"모델 로드: {model_path}")
    print(f"  학습 에폭: {checkpoint['epoch']} | val_loss: {checkpoint['val_loss']:.4f}")

    model = CNNLSTMModel(
        n_features=config["n_features"],
        time_steps=config["time_steps"],
        n_classes=config["n_classes"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])

    # 데이터 로드
    if data_path:
        df = pd.read_csv(data_path)
        print(f"데이터 로드: {data_path} ({len(df):,}행)")
    else:
        df = generate_dummy_data(n_samples=3000)
        print("더미 데이터 사용")

    _, _, test_ds, _ = preprocess(df, window_size=config["time_steps"])
    _, _, test_loader = get_dataloaders(
        test_ds, test_ds, test_ds, batch_size=config["batch_size"]
    )

    print(f"테스트 샘플 수: {len(test_ds)}")

    # 예측 및 평가
    y_true, y_pred = predict_all(model, test_loader, device)
    metrics = compute_metrics(y_true, y_pred, n_classes=config["n_classes"])
    print_results(metrics)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",  type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    args = parser.parse_args()

    evaluate(data_path=args.data, model_path=args.model)
