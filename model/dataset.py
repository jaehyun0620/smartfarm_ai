"""
데이터 전처리 파이프라인 및 Dataset 클래스

- Sliding window: window_size=30, stride=1
- MinMaxScaler 정규화 (센서별 독립 적용)
- Train 70% / Val 15% / Test 15% 분할
- 더미 데이터 생성 지원
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional

# 센서 컬럼 정의
# 현재: 스마트팜코리아 공개 데이터 기준 (6개)
# 추후: 테스트베드 수집 후 indoor_temp_bot, outdoor_humid, fan_rpm, fan_current 추가
SENSOR_COLUMNS = [
    "indoor_temp",
    "indoor_humid",
    "outdoor_temp",
    "light",
    "soil_moisture",
    "co2",
]

LABEL_COLUMN = "risk_level"  # 0:정상, 1:주의, 2:위험


# ─────────────────────────────────────────────
# Sliding Window
# ─────────────────────────────────────────────

def make_windows(
    data: np.ndarray,
    labels: np.ndarray,
    window_size: int = 30,
    stride: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    시계열 데이터를 sliding window로 분할.

    Returns:
        X: (n_windows, window_size, n_features)
        y: (n_windows,)  — 각 윈도우의 마지막 타임스텝 레이블
    """
    X, y = [], []
    for i in range(0, len(data) - window_size + 1, stride):
        X.append(data[i : i + window_size])
        y.append(labels[i + window_size - 1])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


# ─────────────────────────────────────────────
# Dataset 클래스
# ─────────────────────────────────────────────

class SmartFarmDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)  # (N, window_size, n_features)
        self.y = torch.from_numpy(y)  # (N,)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


# ─────────────────────────────────────────────
# 전처리 파이프라인
# ─────────────────────────────────────────────

def preprocess(
    df: pd.DataFrame,
    window_size: int = 30,
    stride: int = 1,
    test_size: float = 0.15,
    val_size: float = 0.15,
    scaler: Optional[MinMaxScaler] = None,
) -> Tuple[SmartFarmDataset, SmartFarmDataset, SmartFarmDataset, MinMaxScaler]:
    """
    DataFrame → 학습/검증/테스트 Dataset 반환.

    Args:
        df: SENSOR_COLUMNS + LABEL_COLUMN 포함한 DataFrame
        scaler: None이면 학습 데이터로 fit, 아니면 주어진 scaler로 transform만 수행
    Returns:
        train_ds, val_ds, test_ds, fitted_scaler
    """
    features = df[SENSOR_COLUMNS].values
    labels = df[LABEL_COLUMN].values

    # MinMaxScaler (학습 데이터로만 fit)
    if scaler is None:
        scaler = MinMaxScaler()
        features = scaler.fit_transform(features)
    else:
        features = scaler.transform(features)

    # Sliding window
    X, y = make_windows(features, labels, window_size, stride)

    # 시계열 순서 유지하면서 분할 (shuffle=False)
    n = len(X)
    n_test = int(n * test_size)
    n_val = int(n * val_size)

    X_train = X[: n - n_test - n_val]
    y_train = y[: n - n_test - n_val]
    X_val   = X[n - n_test - n_val : n - n_test]
    y_val   = y[n - n_test - n_val : n - n_test]
    X_test  = X[n - n_test :]
    y_test  = y[n - n_test :]

    return (
        SmartFarmDataset(X_train, y_train),
        SmartFarmDataset(X_val,   y_val),
        SmartFarmDataset(X_test,  y_test),
        scaler,
    )


def get_dataloaders(
    train_ds: SmartFarmDataset,
    val_ds: SmartFarmDataset,
    test_ds: SmartFarmDataset,
    batch_size: int = 64,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


# ─────────────────────────────────────────────
# 더미 데이터 생성
# ─────────────────────────────────────────────

def generate_dummy_data(n_samples: int = 2000, seed: int = 42) -> pd.DataFrame:
    """
    봄철(3~5월) 딸기 재배 베이스라인 기준 더미 데이터 생성.
    스마트팜코리아 0000574 봄 데이터 통계 기반.

    정상 범위:
      내부 온도:  8~26°C  (평균 15.3°C)
      내부 습도:  60~87%  (정상 클래스 평균 66.7%)
      외부 온도:  0~20°C  (봄철 평균 3.1°C)
      조도:       0~735 lux (평균 107 lux)
      CO2:        400~700 ppm (평균 588 ppm)

    시나리오:
      S1: 정상      (risk=0) 65%
      S2: 고습 위험  (risk=2) 20%  → 습도 92% 이상
      S3: 고온 주의  (risk=1) 10%  → 온도 26~30°C
      S5: 복합 이상  (risk=2)  5%  → 고온 + 고습 동시
    """
    rng = np.random.default_rng(seed)
    n = n_samples

    # S1 정상 기본값 (봄철 베이스라인 기준)
    df = pd.DataFrame({
        "indoor_temp":    rng.normal(15.3, 5.0, n),    # 봄철 평균 15.3°C
        "indoor_humid":   rng.normal(70.0, 8.0, n),    # 정상 클래스 평균 66.7%
        "outdoor_temp":   rng.normal(5.0,  6.0, n),    # 봄철 외부 평균 3.1°C
        "light":          rng.uniform(0, 400, n),       # 봄철 평균 107 lux
        "soil_moisture":  np.zeros(n),                  # 공개 데이터 대부분 0
        "co2":            rng.normal(588, 80, n),       # 봄철 평균 588 ppm
        "risk_level":     0,
    })

    # S2: 고습 위험 구간 (20%) → risk=2
    s2 = rng.choice(n, size=int(n * 0.2), replace=False)
    df.loc[s2, "indoor_humid"] = rng.uniform(92, 100, len(s2))
    df.loc[s2, "risk_level"]   = 2

    # S3: 고온 주의 구간 (10%) → risk=1
    remaining_s3 = np.setdiff1d(range(n), s2)
    s3 = rng.choice(remaining_s3, size=int(n * 0.1), replace=False)
    df.loc[s3, "indoor_temp"]  = rng.uniform(26, 30, len(s3))
    df.loc[s3, "risk_level"]   = 1

    # S5: 복합 이상 구간 (5%) → risk=2
    remaining_s5 = np.setdiff1d(remaining_s3, s3)
    s5 = rng.choice(remaining_s5, size=int(n * 0.05), replace=False)
    df.loc[s5, "indoor_temp"]  = rng.uniform(30, 35, len(s5))
    df.loc[s5, "indoor_humid"] = rng.uniform(92, 100, len(s5))
    df.loc[s5, "risk_level"]   = 2

    # 값 범위 클리핑 (센서 물리적 범위 초과 방지)
    df["indoor_temp"]  = df["indoor_temp"].clip(3, 40)
    df["indoor_humid"] = df["indoor_humid"].clip(10, 100)
    df["outdoor_temp"] = df["outdoor_temp"].clip(-5, 30)
    df["light"]        = df["light"].clip(0, 800)
    df["co2"]          = df["co2"].clip(200, 1500)

    return df


if __name__ == "__main__":
    df = generate_dummy_data(n_samples=2000)
    print("더미 데이터 shape:", df.shape)
    print(df[SENSOR_COLUMNS + [LABEL_COLUMN]].head())
    print("\n레이블 분포:\n", df["risk_level"].value_counts().sort_index())

    train_ds, val_ds, test_ds, scaler = preprocess(df)
    print(f"\nTrain: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    train_loader, val_loader, test_loader = get_dataloaders(train_ds, val_ds, test_ds)
    X_batch, y_batch = next(iter(train_loader))
    print(f"배치 X shape: {X_batch.shape}")  # (64, 30, 8)
    print(f"배치 y shape: {y_batch.shape}")  # (64,)
