"""
학습된 모델로 실시간 추론

사용법:
  # 단일 추론 (최근 30개 센서값 → 위험도 예측)
  from inference import SmartFarmPredictor
  predictor = SmartFarmPredictor()
  result = predictor.predict(sensor_window)

  # 커맨드라인 테스트
  python model/inference.py
"""

from pathlib import Path
from typing import Optional

import numpy as np
import torch

from cnn_lstm import CNNLSTMModel
from dataset import SENSOR_COLUMNS

LABEL_NAMES  = {0: "정상", 1: "주의", 2: "위험"}
LABEL_EMOJI  = {0: "✅",   1: "⚠️",  2: "🚨"}

DEFAULT_MODEL_PATH = Path(__file__).parent / "saved" / "best_model.pt"


# ─────────────────────────────────────────────
# Predictor 클래스
# ─────────────────────────────────────────────

class SmartFarmPredictor:
    """
    학습된 모델을 불러와서 실시간 추론을 수행하는 클래스.

    사용 예:
        predictor = SmartFarmPredictor()

        # 최근 30 타임스텝 센서값 (30, 6) 배열
        window = np.array([[24.1, 61.2, 19.8, 2300, 0.52, 495], ...])
        result = predictor.predict(window)

        print(result["risk_level"])   # 0 / 1 / 2
        print(result["label"])        # "정상" / "주의" / "위험"
        print(result["confidence"])   # 0.0 ~ 1.0
    """

    def __init__(self, model_path: Optional[Path] = None):
        if model_path is None:
            model_path = DEFAULT_MODEL_PATH

        # 장치 설정
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # 체크포인트 로드
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.config = checkpoint["config"]
        self.time_steps = self.config["time_steps"]   # 30
        self.n_features = self.config["n_features"]   # 6

        # 모델 초기화 및 가중치 로드
        self.model = CNNLSTMModel(
            n_features=self.config["n_features"],
            time_steps=self.config["time_steps"],
            n_classes=self.config["n_classes"],
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

        print(f"모델 로드 완료 | 장치: {self.device} | 에폭: {checkpoint['epoch']}")

    # ─────────────────────────────────────────────
    # 추론
    # ─────────────────────────────────────────────

    def predict(self, window: np.ndarray) -> dict:
        """
        Args:
            window: (time_steps, n_features) numpy 배열
                    컬럼 순서: SENSOR_COLUMNS 와 동일해야 함
                    [indoor_temp, indoor_humid, outdoor_temp,
                     light, soil_moisture, co2]

        Returns:
            {
                "risk_level":  int    (0/1/2),
                "label":       str    ("정상"/"주의"/"위험"),
                "confidence":  float  (0.0~1.0),
                "probabilities": list (클래스별 확률),
            }
        """
        # 입력 검증
        if window.shape != (self.time_steps, self.n_features):
            raise ValueError(
                f"입력 shape 오류: 기대={self.time_steps, self.n_features}, "
                f"실제={window.shape}\n"
                f"컬럼 순서: {SENSOR_COLUMNS}"
            )

        # 텐서 변환 및 배치 차원 추가 → (1, time_steps, n_features)
        x = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(x)                      # (1, n_classes)
            probs  = torch.softmax(logits, dim=1)[0]    # (n_classes,)

        risk_level  = probs.argmax().item()
        confidence  = probs[risk_level].item()
        prob_list   = probs.cpu().numpy().tolist()

        return {
            "risk_level":    risk_level,
            "label":         LABEL_NAMES[risk_level],
            "emoji":         LABEL_EMOJI[risk_level],
            "confidence":    round(confidence, 4),
            "probabilities": {
                LABEL_NAMES[i]: round(prob_list[i], 4)
                for i in range(len(prob_list))
            },
        }

    def predict_from_dict(self, sensor_dict: dict) -> dict:
        """
        Firebase에서 받은 딕셔너리 형태의 단일 센서값 1개로 추론.
        내부적으로 window 버퍼에 누적 후 30개 채워지면 예측.

        Args:
            sensor_dict: {
                "indoor_temp": 24.1,
                "indoor_humid": 61.2,
                "outdoor_temp": 19.8,
                "light": 2300,
                "soil_moisture": 0.52,
                "co2": 495,
            }

        Returns:
            dict | None  (버퍼가 30개 미만이면 None)
        """
        row = [sensor_dict.get(col, 0.0) for col in SENSOR_COLUMNS]

        if not hasattr(self, "_buffer"):
            self._buffer = []

        self._buffer.append(row)

        if len(self._buffer) < self.time_steps:
            return None  # 아직 window 미달

        # 최근 30개만 사용
        window = np.array(self._buffer[-self.time_steps:], dtype=np.float32)
        return self.predict(window)


# ─────────────────────────────────────────────
# 커맨드라인 테스트
# ─────────────────────────────────────────────

if __name__ == "__main__":
    predictor = SmartFarmPredictor()
    n_features = predictor.n_features
    time_steps = predictor.time_steps

    print(f"\n입력 컬럼 순서: {SENSOR_COLUMNS}")
    print(f"window shape: ({time_steps}, {n_features})\n")

    # ── 시나리오 1: 정상 ──────────────────────────
    print("=" * 45)
    print("시나리오 1: 정상 환경")
    window_normal = np.tile(
        [22.0, 70.0, 15.0, 2000.0, 0.5, 500.0],
        (time_steps, 1)
    ).astype(np.float32)
    result = predictor.predict(window_normal)
    print(f"  판정: {result['emoji']} {result['label']}  (신뢰도: {result['confidence']*100:.1f}%)")
    print(f"  확률: {result['probabilities']}")

    # ── 시나리오 2: 고습 위험 ─────────────────────
    print("\n" + "=" * 45)
    print("시나리오 2: 고습 위험 (습도 94%)")
    window_danger = np.tile(
        [24.0, 94.0, 15.0, 0.0, 0.5, 600.0],
        (time_steps, 1)
    ).astype(np.float32)
    result = predictor.predict(window_danger)
    print(f"  판정: {result['emoji']} {result['label']}  (신뢰도: {result['confidence']*100:.1f}%)")
    print(f"  확률: {result['probabilities']}")

    # ── 시나리오 3: 주의 (고온) ───────────────────
    print("\n" + "=" * 45)
    print("시나리오 3: 고온 주의 (온도 28°C)")
    window_warn = np.tile(
        [28.0, 75.0, 25.0, 4000.0, 0.4, 550.0],
        (time_steps, 1)
    ).astype(np.float32)
    result = predictor.predict(window_warn)
    print(f"  판정: {result['emoji']} {result['label']}  (신뢰도: {result['confidence']*100:.1f}%)")
    print(f"  확률: {result['probabilities']}")

    # ── 시나리오 4: Firebase 딕셔너리 형태 ────────
    print("\n" + "=" * 45)
    print("시나리오 4: Firebase 딕셔너리 누적 방식")
    predictor2 = SmartFarmPredictor()
    for i in range(32):
        sensor = {
            "indoor_temp":   22.0 + i * 0.1,
            "indoor_humid":  88.0 + i * 0.2,
            "outdoor_temp":  15.0,
            "light":         1500.0,
            "soil_moisture": 0.5,
            "co2":           500.0,
        }
        result = predictor2.predict_from_dict(sensor)
        if result:
            print(f"  {i+1}번째 입력 후 판정: {result['emoji']} {result['label']} "
                  f"(신뢰도: {result['confidence']*100:.1f}%)")
