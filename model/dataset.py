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

# 센서 컬럼 정의 (CLAUDE.md 기준)
SENSOR_COLUMNS = [
    "indoor_temp_top",
    "indoor_humid_top",
    "indoor_temp_bot",
    "indoor_humid_bot",
    "outdoor_temp",
    "outdoor_humid",
    "light",
    "soil_moisture",
    # "co2",      # 선택 항목 — 필요 시 활성화
    # "fan_rpm",  # 이상 감지 실험 시 활성화
    # "fan_current",
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
    5가지 시나리오를 포함한 더미 센서 데이터 생성.

    S1: 정상 (risk=0)
    S2: 고습 위험 humid > 85% (risk=2)
    S3: 환기 저하 fan_rpm 급감 (risk=1)
    S4: 과부하 fan_current 급등 (risk=2)
    S5: 복합 이상 (risk=2)
    """
    rng = np.random.default_rng(seed)
    n = n_samples

    # 기본 센서값 (정상 범위)
    df = pd.DataFrame({
        "indoor_temp_top":   rng.normal(24, 1.5, n),
        "indoor_humid_top":  rng.normal(60, 5,   n),
        "indoor_temp_bot":   rng.normal(23, 1.5, n),
        "indoor_humid_bot":  rng.normal(62, 5,   n),
        "outdoor_temp":      rng.normal(20, 3,   n),
        "outdoor_humid":     rng.normal(55, 8,   n),
        "light":             rng.uniform(1000, 5000, n),
        "soil_moisture":     rng.uniform(0.3, 0.7, n),
        "risk_level":        0,
    })

    # S2: 고습 위험 구간 (20%)
    s2 = rng.choice(n, size=int(n * 0.2), replace=False)
    df.loc[s2, "indoor_humid_top"]  = rng.uniform(85, 95, len(s2))
    df.loc[s2, "indoor_humid_bot"]  = rng.uniform(83, 93, len(s2))
    df.loc[s2, "risk_level"] = 2

    # S3: 환기 저하 구간 (10%) → 주의
    s3 = rng.choice(np.setdiff1d(range(n), s2), size=int(n * 0.1), replace=False)
    df.loc[s3, "indoor_temp_top"]  += rng.uniform(3, 6, len(s3))
    df.loc[s3, "risk_level"] = 1

    # S5: 복합 이상 구간 (5%) → 위험
    remaining = np.setdiff1d(np.setdiff1d(range(n), s2), s3)
    s5 = rng.choice(remaining, size=int(n * 0.05), replace=False)
    df.loc[s5, "indoor_temp_top"]   += rng.uniform(5, 8, len(s5))
    df.loc[s5, "indoor_humid_top"]  += rng.uniform(15, 25, len(s5))
    df.loc[s5, "risk_level"] = 2

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
