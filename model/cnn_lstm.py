"""
1D-CNN + LSTM 모델 정의

입력: (batch_size, time_steps=30, n_features=8)
출력: 3클래스 (0:정상, 1:주의, 2:위험)

구조:
  Conv1D(32, k=3) → BN → ReLU → Dropout(0.2)
  Conv1D(64, k=3) → BN → ReLU → Dropout(0.2)
  LSTM(128)       → Dropout(0.3)
  FC(64)          → ReLU
  FC(3)           → Softmax
"""

import torch
import torch.nn as nn


class CNNLSTMModel(nn.Module):
    def __init__(self, n_features: int = 8, time_steps: int = 30, n_classes: int = 3):
        super().__init__()

        # 1D-CNN: (batch, n_features, time_steps) 입력
        self.cnn = nn.Sequential(
            nn.Conv1d(n_features, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # LSTM: (batch, time_steps, 64) 입력
        self.lstm = nn.LSTM(input_size=64, hidden_size=128, batch_first=True)
        self.dropout_lstm = nn.Dropout(0.3)

        # Fully Connected
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, time_steps, n_features)
        x = x.permute(0, 2, 1)           # → (batch, n_features, time_steps)
        x = self.cnn(x)                   # → (batch, 64, time_steps)
        x = x.permute(0, 2, 1)           # → (batch, time_steps, 64)
        lstm_out, _ = self.lstm(x)        # → (batch, time_steps, 128)
        x = lstm_out[:, -1, :]           # 마지막 타임스텝만 사용
        x = self.dropout_lstm(x)
        x = self.fc(x)                   # → (batch, n_classes)
        return x


if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"장치: {device}")

    model = CNNLSTMModel(n_features=8, time_steps=30, n_classes=3).to(device)
    print(model)

    # 구조 확인
    dummy = torch.randn(4, 30, 8).to(device)
    out = model(dummy)
    print(f"\n입력 shape: {dummy.shape}")
    print(f"출력 shape: {out.shape}")  # (4, 3)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"파라미터 수: {total_params:,}")
