"""
학습 루프 + 체크포인트 저장

사용법:
  python model/train.py              # 더미 데이터로 동작 확인
  python model/train.py --data path/to/data.csv  # 실제 데이터
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from cnn_lstm import CNNLSTMModel
from dataset import (
    SENSOR_COLUMNS,
    generate_dummy_data,
    get_dataloaders,
    preprocess,
)


# ─────────────────────────────────────────────
# 학습 설정
# ─────────────────────────────────────────────

DEFAULT_CONFIG = {
    "n_features":  len(SENSOR_COLUMNS),  # 8
    "time_steps":  30,
    "n_classes":   3,
    "batch_size":  64,
    "lr":          1e-3,
    "epochs":      30,
    "patience":    5,       # Early stopping patience
    "save_dir":    Path(__file__).parent / "saved",
}


# ─────────────────────────────────────────────
# 유틸리티
# ─────────────────────────────────────────────

def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss, total_acc = 0.0, 0.0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = criterion(logits, y)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            total_acc  += accuracy(logits, y)

    n = len(loader)
    return total_loss / n, total_acc / n


# ─────────────────────────────────────────────
# 메인 학습 루프
# ─────────────────────────────────────────────

def train(config: dict, data_path: str = None):
    # 장치 설정 (MPS → CPU 순으로 자동 선택)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"사용 장치: {device}")

    # 데이터 로드
    if data_path:
        df = pd.read_csv(data_path)
        print(f"데이터 로드: {data_path} ({len(df):,}행)")
    else:
        df = generate_dummy_data(n_samples=3000)
        print(f"더미 데이터 생성: {len(df):,}행")

    print(f"레이블 분포: {df['risk_level'].value_counts().sort_index().to_dict()}")

    train_ds, val_ds, test_ds, scaler = preprocess(
        df,
        window_size=config["time_steps"],
    )
    train_loader, val_loader, test_loader = get_dataloaders(
        train_ds, val_ds, test_ds, config["batch_size"]
    )
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    # 모델 초기화
    model = CNNLSTMModel(
        n_features=config["n_features"],
        time_steps=config["time_steps"],
        n_classes=config["n_classes"],
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=config["lr"])
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)

    # 체크포인트 저장 경로
    save_dir = Path(config["save_dir"])
    save_dir.mkdir(exist_ok=True)
    best_path = save_dir / "best_model.pt"

    # 학습 루프
    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    print(f"\n{'에폭':>5} | {'학습 손실':>10} | {'학습 정확도':>10} | {'검증 손실':>10} | {'검증 정확도':>10}")
    print("-" * 60)

    for epoch in range(1, config["epochs"] + 1):
        t0 = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss,   val_acc   = run_epoch(model, val_loader,   criterion, optimizer, device, train=False)
        elapsed = time.time() - t0

        scheduler.step(val_loss)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"{epoch:>5} | {train_loss:>10.4f} | {train_acc:>10.4f} | {val_loss:>10.4f} | {val_acc:>10.4f}  ({elapsed:.1f}s)")

        # 최적 모델 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss": best_val_loss,
                "config": config,
            }, best_path)
            print(f"         → 체크포인트 저장 (val_loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]:
                print(f"\nEarly stopping (patience={config['patience']})")
                break

    print(f"\n학습 완료. 최적 모델: {best_path}")
    return history


# ─────────────────────────────────────────────
# 실행
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=None, help="CSV 데이터 경로 (없으면 더미 데이터 사용)")
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG["epochs"])
    parser.add_argument("--batch_size", type=int, default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--lr", type=float, default=DEFAULT_CONFIG["lr"])
    args = parser.parse_args()

    config = {**DEFAULT_CONFIG, "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr}
    train(config, data_path=args.data)
