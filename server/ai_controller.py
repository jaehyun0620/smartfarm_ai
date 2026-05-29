"""
AI 추론 — 학습된 모델로 기기 상태 예측
"""
import os
import pickle
import time as _time

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

FEATURES = ["temp1", "hum1", "soil_percent", "light_raw", "co2_raw", "water_raw", "hour"]
TARGETS  = ["fan1", "fan2", "window1", "window2", "led", "humidifier", "heater"]

# ── 아두이노 하드코딩 규칙 (auto 모드) ────────────────────────────────────────
AUTO_RULES = {
    "temp_high": 22.3,
    "temp_low":  14.9,
    "hum_high":  80.6,
    "hum_low":   66.6,
    "light_threshold": 900,   # raw > 이 값이면 어두움 → LED on
    "soil_dry":  30,
    "soil_wet":  70,
}

def auto_predict(sensors: dict) -> dict:
    """아두이노 규칙기반 로직을 파이썬으로 재현 (비교용)"""
    try:
        temp      = float(sensors.get("temp1",        0))
        hum       = float(sensors.get("hum1",         0))
        light_raw = float(sensors.get("light_raw",    0))
        soil_pct  = float(sensors.get("soil_percent", 0))
    except (TypeError, ValueError):
        return {t: 0 for t in TARGETS}

    temp_high = temp  > AUTO_RULES["temp_high"]
    temp_low  = temp  < AUTO_RULES["temp_low"]
    hum_high  = hum   > AUTO_RULES["hum_high"]
    hum_low   = hum   < AUTO_RULES["hum_low"]
    light_low = light_raw > AUTO_RULES["light_threshold"]

    fan = window = heater = humidifier = led = 0

    if temp_high:
        fan = window = 1
    elif temp_low:
        heater = 1
        humidifier = 1 if hum_low else 0
    elif hum_high:
        fan = window = 1
    elif hum_low:
        humidifier = 1

    if light_low:
        led = 1

    return {
        "fan1": fan, "fan2": fan,
        "window1": window, "window2": window,
        "heater": heater, "humidifier": humidifier,
        "led": led,
    }


class AIController:
    def __init__(self):
        self.models       = {}
        self.model_loaded = False
        self.train_acc    = {}
        self.reload()

    def reload(self):
        if not os.path.exists(MODEL_PATH):
            self.model_loaded = False
            return
        try:
            with open(MODEL_PATH, "rb") as f:
                data = pickle.load(f)

            if isinstance(data, dict) and "models" in data:
                # 신규 형식: {"models": {...}, "train_acc": {...}, ...}
                self.models    = data["models"]
                self.train_acc = data.get("train_acc", {})
            elif isinstance(data, tuple):
                # 중간 형식: (models_dict, train_acc_dict)
                self.models, self.train_acc = data
            elif isinstance(data, dict):
                # 구버전 형식: {target: clf, ...}
                self.models    = data
                self.train_acc = {}
            else:
                print("[AI] Unknown model format")
                self.model_loaded = False
                return

            self.model_loaded = True
            print(f"[AI] Model loaded ({len(self.models)} classifiers)")
        except Exception as e:
            print(f"[AI] Model load error: {e}")
            self.model_loaded = False

    def _build_x(self, record: dict):
        ts   = record.get("timestamp", "")
        hour = int(ts[11:13]) if len(ts) >= 13 else 0
        s    = record.get("sensors", record)
        return [[
            float(s.get("temp1",        0) or 0),
            float(s.get("hum1",         0) or 0),
            float(s.get("soil_percent",  0) or 0),
            float(s.get("light_raw",     0) or 0),
            float(s.get("co2_raw",       0) or 0),
            float(s.get("water_raw",     0) or 0),
            hour,
        ]]

    def predict(self, record: dict) -> dict:
        if not self.model_loaded:
            return {}
        try:
            x = self._build_x(record)
            return {target: int(clf.predict(x)[0]) for target, clf in self.models.items()}
        except ValueError as e:
            # feature 수 불일치 — 재학습 필요
            print(f"[AI] Predict skipped (재학습 필요): {e}")
            return {}
        except Exception as e:
            print(f"[AI] Predict error: {e}")
            return {}

    def compare(self, record: dict) -> dict:
        """auto 규칙 결과 vs AI 예측 결과 비교"""
        if not record:
            return {}
        try:
            s = record.get("sensors", record)
            auto_result = auto_predict(s)
            ai_result   = self.predict(record) if self.model_loaded else {}

            comparison = {}
            for target in TARGETS:
                a = auto_result.get(target, 0)
                b = ai_result.get(target, 0)
                comparison[target] = {"auto": a, "ai": b, "match": a == b}
            return comparison
        except Exception as e:
            print(f"[AI] Compare error: {e}")
            return {}

    def get_train_acc_display(self) -> dict:
        """
        train_acc를 UI 표시용으로 정규화.
        구버전: {target: float}
        중간버전: {target: {"train": float, "test": float}}
        신버전: {target: {"train": float, "val": float, "test": float}}
        → 항상 {"train", "val", "test"} 형태로 반환
        """
        result = {}
        for target, acc in self.train_acc.items():
            if isinstance(acc, dict):
                result[target] = {
                    "train": acc.get("train"),
                    "val":   acc.get("val"),
                    "test":  acc.get("test"),
                }
            else:
                result[target] = {"train": acc, "val": None, "test": None}
        return result

    def learned_thresholds(self) -> dict:
        """
        각 기기별 핵심 feature의 학습된 임계값 추출.
        Random Forest 전체 트리에서 해당 feature의 분기 임계값 중앙값을 반환.
        """
        if not self.model_loaded:
            return {}
        try:
            import numpy as np
        except ImportError:
            return {}

        # 기기별 핵심 feature 및 방향 (rule 기준값, 단위)
        KEY = {
            "fan1":       {"feature": "temp1",     "rule": AUTO_RULES["temp_high"], "unit": "°C", "dir": "이상"},
            "fan2":       {"feature": "temp1",     "rule": AUTO_RULES["temp_high"], "unit": "°C", "dir": "이상"},
            "window1":    {"feature": "temp1",     "rule": AUTO_RULES["temp_high"], "unit": "°C", "dir": "이상"},
            "window2":    {"feature": "temp1",     "rule": AUTO_RULES["temp_high"], "unit": "°C", "dir": "이상"},
            "heater":     {"feature": "temp1",     "rule": AUTO_RULES["temp_low"],  "unit": "°C", "dir": "이하"},
            "humidifier": {"feature": "hum1",      "rule": AUTO_RULES["hum_low"],   "unit": "%",  "dir": "이하"},
            "led":        {"feature": "light_raw", "rule": AUTO_RULES["light_threshold"], "unit": "", "dir": "이상"},
        }

        result = {}
        for target, clf in self.models.items():
            meta = KEY.get(target)
            if not meta or meta["feature"] not in FEATURES:
                continue
            feat_idx = FEATURES.index(meta["feature"])

            # 모든 트리에서 해당 feature의 분기 임계값 수집
            thresholds = []
            for tree in clf.estimators_:
                dt = tree.tree_
                for node in range(dt.node_count):
                    if dt.feature[node] == feat_idx and dt.threshold[node] > -2:
                        thresholds.append(dt.threshold[node])

            if thresholds:
                learned = round(float(np.median(thresholds)), 1)
                result[target] = {
                    "feature": meta["feature"],
                    "rule":    meta["rule"],
                    "learned": learned,
                    "unit":    meta["unit"],
                    "dir":     meta["dir"],
                    "diff":    round(learned - meta["rule"], 1),
                }
        return result

    def feature_importances(self) -> dict:
        """기기별 feature importance"""
        if not self.model_loaded:
            return {}
        result = {}
        for target, clf in self.models.items():
            try:
                imp = clf.feature_importances_
                result[target] = {f: round(float(v), 4) for f, v in zip(FEATURES, imp)}
            except Exception:
                pass
        return result
