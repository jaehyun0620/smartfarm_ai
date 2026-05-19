"""
비교 실험용 Baseline 모델

[봄철 딸기 재배 베이스라인 기준 — 스마트팜코리아 0000574 봄 데이터]
  정상 범위:
    내부 온도:  8 ~ 26°C     (봄철 IQR 기준)
    내부 습도:  60 ~ 87%     (정상 클래스 평균 66.7%)
    CO2:       400 ~ 700ppm  (봄철 평균 588ppm)

Baseline A: 고정 임계값 방식
Baseline B: Z-score 동적 임계값 방식

사용법:
  python model/baseline.py --data data/processed/baseline_data.csv
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler

from dataset import SENSOR_COLUMNS

LABEL_NAMES = {0: "정상", 1: "주의", 2: "위험"}

# ─────────────────────────────────────────────
# 봄철 딸기 재배 고정 임계값
# ─────────────────────────────────────────────

THRESHOLDS = {
    "danger": {
        "indoor_humid_max": 92,   # 잿빛곰팡이 위험
        "indoor_temp_max":  30,   # 열 스트레스
        "indoor_temp_min":   5,   # 냉해
    },
    "warning": {
        "indoor_humid_max": 87,   # 고습 주의
        "indoor_temp_max":  26,   # 고온 주의
        "indoor_temp_min":   8,   # 저온 주의
    },
}


# ─────────────────────────────────────────────
# Baseline A: 고정 임계값
# ─────────────────────────────────────────────

class FixedThresholdBaseline:
    """
    가장 단순한 기존 스마트팜 방식.
    현재 센서값 1개만 보고 임계값 초과 여부로 판단.
    """

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        preds = np.zeros(len(df), dtype=int)

        danger = (
            (df["indoor_humid"] >= THRESHOLDS["danger"]["indoor_humid_max"]) |
            (df["indoor_temp"]  >= THRESHOLDS["danger"]["indoor_temp_max"])  |
            (df["indoor_temp"]  <= THRESHOLDS["danger"]["indoor_temp_min"])
        )
        warning = (~danger) & (
            (df["indoor_humid"] >= THRESHOLDS["warning"]["indoor_humid_max"]) |
            (df["indoor_temp"]  >= THRESHOLDS["warning"]["indoor_temp_max"])  |
            (df["indoor_temp"]  <= THRESHOLDS["warning"]["indoor_temp_min"])
        )

        preds[danger]  = 2
        preds[warning] = 1
        return preds


# ─────────────────────────────────────────────
# Baseline B: Z-score 동적 임계값
# ─────────────────────────────────────────────

class ZScoreBaseline:
    """
    학습 데이터의 평균/표준편차를 기반으로 동적으로 임계값 설정.
    고정 임계값보다 데이터에 적응적이지만 시간 흐름은 고려 못 함.

    Z-score = (현재값 - 평균) / 표준편차
      |Z| >= 2.0 → 위험  (평균에서 2σ 이상 벗어남)
      |Z| >= 1.5 → 주의
    """

    def __init__(self, danger_z: float = 2.0, warning_z: float = 1.5):
        self.danger_z  = danger_z
        self.warning_z = warning_z
        self.scaler    = StandardScaler()
        self._fitted   = False

    def fit(self, df: pd.DataFrame):
        key_cols = ["indoor_temp", "indoor_humid"]
        self.scaler.fit(df[key_cols])
        self._fitted = True

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("fit()을 먼저 호출하세요.")

        key_cols = ["indoor_temp", "indoor_humid"]
        z = np.abs(self.scaler.transform(df[key_cols]))  # (N, 2)
        max_z = z.max(axis=1)                            # 각 행에서 최대 Z

        preds = np.zeros(len(df), dtype=int)
        preds[max_z >= self.warning_z] = 1
        preds[max_z >= self.danger_z]  = 2
        return preds


# ─────────────────────────────────────────────
# 평가 출력
# ─────────────────────────────────────────────

def print_metrics(name: str, y_true: np.ndarray, y_pred: np.ndarray):
    acc      = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_per   = f1_score(y_true, y_pred, average=None,    zero_division=0)
    cm       = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    fpr_list, fnr_list = [], []
    for i in range(3):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        fpr_list.append(fp / (fp + tn) if (fp + tn) > 0 else 0.0)
        fnr_list.append(fn / (fn + tp) if (fn + tp) > 0 else 0.0)

    print(f"\n{'='*50}")
    print(f"[{name}]")
    print(f"{'='*50}")
    print(f"정확도 (Accuracy): {acc:.4f} ({acc*100:.1f}%)")
    print(f"F1-Score (macro):  {f1_macro:.4f}")
    print(f"\n{'클래스':<8} {'F1':>8} {'FPR':>10} {'FNR':>10}")
    print("-" * 40)
    for i in range(3):
        print(f"{LABEL_NAMES[i]}({i})  {f1_per[i]:>8.4f} {fpr_list[i]:>10.4f} {fnr_list[i]:>10.4f}")

    print(f"\n혼동행렬 (행=실제, 열=예측)")
    header = f"{'':8}" + "".join(f"{LABEL_NAMES[i]:>8}" for i in range(3))
    print(header)
    for i in range(3):
        row = f"{LABEL_NAMES[i]}({i})  " + "".join(f"{cm[i,j]:>8}" for j in range(3))
        print(row)

    return {"accuracy": acc, "f1_macro": f1_macro, "f1_per": f1_per,
            "fpr": fpr_list, "fnr": fnr_list}


# ─────────────────────────────────────────────
# 메인 비교 실험
# ─────────────────────────────────────────────

def run_comparison(data_path: str):
    df = pd.read_csv(data_path)
    print(f"데이터 로드: {data_path} ({len(df):,}행)")

    # 시계열 순서 유지 분할 (70/15/15)
    n       = len(df)
    n_test  = int(n * 0.15)
    n_val   = int(n * 0.15)
    df_train = df.iloc[: n - n_test - n_val]
    df_test  = df.iloc[n - n_test :]

    y_true = df_test["risk_level"].values
    print(f"\n테스트셋: {len(df_test)}행")
    print(f"레이블 분포: { {LABEL_NAMES[k]: v for k, v in zip(*np.unique(y_true, return_counts=True))} }")

    results = {}

    # Baseline A: 고정 임계값
    model_a = FixedThresholdBaseline()
    pred_a  = model_a.predict(df_test)
    results["고정 임계값"] = print_metrics("Baseline A — 고정 임계값", y_true, pred_a)

    # Baseline B: Z-score
    model_b = ZScoreBaseline()
    model_b.fit(df_train)
    pred_b  = model_b.predict(df_test)
    results["Z-score"] = print_metrics("Baseline B — Z-score 동적 임계값", y_true, pred_b)

    # 비교 요약
    print(f"\n{'='*50}")
    print("비교 요약")
    print(f"{'='*50}")
    print(f"{'방법':<20} {'정확도':>10} {'F1 macro':>10}")
    print("-" * 42)
    for name, r in results.items():
        print(f"{name:<20} {r['accuracy']:>10.4f} {r['f1_macro']:>10.4f}")
    print()
    print("※ CNN+LSTM 결과는 evaluate.py 참고")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="../data/processed/baseline_data.csv")
    args = parser.parse_args()
    run_comparison(args.data)
