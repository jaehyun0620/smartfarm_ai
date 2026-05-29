"""
모델 학습 — Random Forest (기기별 분류기)
"""
import os
import pickle
import shutil
from datetime import datetime

import numpy  as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

LOG_PATH   = os.path.join(os.path.dirname(__file__), "farm_log.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
MODEL_DIR  = os.path.join(os.path.dirname(__file__), "model_versions")

FEATURES = ["temp1", "hum1", "soil_percent", "light_raw", "co2_raw", "water_raw", "hour"]
TARGETS  = ["fan1", "fan2", "window1", "window2", "led", "humidifier", "heater"]


def _load_df() -> pd.DataFrame:
    # on_bad_lines='skip': 컬럼 수 불일치 행 건너뜀 (마이그레이션 전 안전망)
    try:
        df = pd.read_csv(LOG_PATH, on_bad_lines="skip")
    except TypeError:
        # pandas 구버전 호환
        df = pd.read_csv(LOG_PATH, error_bad_lines=False)
    except Exception as e:
        raise ValueError(f"CSV 읽기 실패: {e}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


def filter_df(df: pd.DataFrame,
              exclude_manual: bool = False,
              exclude_ranges: list = None) -> pd.DataFrame:
    """
    학습 데이터 필터링
    - exclude_manual : True이면 manual=1 행 제거
    - exclude_ranges : [{"start": "HH:MM", "end": "HH:MM"}, ...] 시간 구간 제거
    """
    try:
        if exclude_manual:
            df = df[df["manual"] != 1].copy()

        if exclude_ranges:
            for r in exclude_ranges:
                start = pd.to_datetime(r.get("start"), errors="coerce")
                end   = pd.to_datetime(r.get("end"),   errors="coerce")
                if pd.isna(start) or pd.isna(end):
                    continue
                mask = (df["timestamp"] >= start) & (df["timestamp"] <= end)
                df = df[~mask].copy()
    except Exception as e:
        print(f"[Trainer] filter_df error: {e}")

    return df


def preview_filter(exclude_manual: bool = False,
                   exclude_ranges: list = None) -> dict:
    """학습 전 데이터 건수 미리보기 (실제 학습 X)"""
    df       = _load_df()
    total    = len(df)
    manual   = int((df.get("manual", pd.Series(dtype=int)) == 1).sum()) if "manual" in df.columns else 0
    filtered = filter_df(df, exclude_manual, exclude_ranges)

    return {
        "total":          total,
        "manual":         manual,
        "excluded":       total - len(filtered),
        "will_train":     len(filtered),
        "exclude_manual": exclude_manual,
        "exclude_ranges": exclude_ranges or [],
    }


def train_models(exclude_manual: bool = False,
                 exclude_ranges: list = None) -> dict:
    """
    모델 학습 후 버전 파일로 저장하고 model.pkl 에 복사
    반환: {"version": "v3", "will_train": 77, "train_acc": {...}}
    """
    df = _load_df()

    if len(df) < 10:
        raise ValueError(f"데이터 부족: {len(df)}개 (최소 10개 필요)")

    df = filter_df(df, exclude_manual, exclude_ranges)

    if len(df) < 10:
        raise ValueError(f"필터 후 데이터 부족: {len(df)}개 (최소 10개 필요)")

    # 수동 제어 행 제외하지 않은 경우 2배 가중치
    if not exclude_manual and "manual" in df.columns:
        manual_rows = df[df["manual"] == 1]
        if len(manual_rows) > 0:
            df = pd.concat([df, manual_rows], ignore_index=True)

    # 필요한 컬럼만 숫자 변환
    available_features = [f for f in FEATURES if f in df.columns]
    available_targets  = [t for t in TARGETS  if t in df.columns]

    if not available_features:
        raise ValueError("학습에 필요한 feature 컬럼이 없습니다")
    if not available_targets:
        raise ValueError("학습에 필요한 target 컬럼이 없습니다")

    df[available_features] = df[available_features].apply(pd.to_numeric, errors="coerce")
    df[available_targets]  = df[available_targets].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=available_features + available_targets)

    if len(df) < 10:
        raise ValueError(f"결측값 제거 후 데이터 부족: {len(df)}개")

    X = df[available_features].values

    # 80:20 학습/검증 분리 (데이터 20개 미만이면 분리 생략)
    if len(df) >= 20:
        X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
        idx_train, idx_test = train_test_split(range(len(df)), test_size=0.2, random_state=42)
    else:
        X_train, X_test = X, X
        idx_train, idx_test = list(range(len(df))), list(range(len(df)))

    models    = {}
    train_acc = {}

    for target in available_targets:
        try:
            y       = df[target].values.astype(int)
            y_train = y[idx_train]
            y_test  = y[idx_test]

            clf = RandomForestClassifier(n_estimators=100, random_state=42)
            clf.fit(X_train, y_train)

            acc_train = clf.score(X_train, y_train)
            acc_test  = clf.score(X_test,  y_test)

            models[target] = clf
            train_acc[target] = {
                "train": round(acc_train, 4),
                "test":  round(acc_test,  4),
            }
            print(f"[Train] {target}: 학습 {acc_train:.1%} / 검증 {acc_test:.1%}")
        except Exception as e:
            print(f"[Train] {target} 학습 실패: {e}")

    if not models:
        raise ValueError("학습된 모델이 없습니다")

    # ── 버전 저장 ──────────────────────────────────────────────────────────────
    os.makedirs(MODEL_DIR, exist_ok=True)
    version      = _next_version()
    version_path = os.path.join(MODEL_DIR, f"{version}.pkl")

    payload = {
        "models":         models,
        "train_acc":      train_acc,
        "version":        version,
        "trained_at":     datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "data_count":     len(df),
        "train_count":    len(idx_train),
        "test_count":     len(idx_test),
        "features":       available_features,
        "exclude_manual": exclude_manual,
        "exclude_ranges": exclude_ranges or [],
    }

    try:
        with open(version_path, "wb") as f:
            pickle.dump(payload, f)
        shutil.copy(version_path, MODEL_PATH)
        print(f"[Train] Saved → {version_path}")
    except Exception as e:
        raise ValueError(f"모델 저장 실패: {e}")

    return {"version": version, "will_train": len(df), "train_acc": train_acc}


# ── 버전 유틸 ──────────────────────────────────────────────────────────────────

def _next_version() -> str:
    versions = list_versions()
    return f"v{len(versions) + 1}"


def list_versions() -> list:
    """저장된 모든 버전 목록 (최신순)"""
    os.makedirs(MODEL_DIR, exist_ok=True)
    files = sorted(
        [f for f in os.listdir(MODEL_DIR) if f.endswith(".pkl")],
        key=lambda f: os.path.getmtime(os.path.join(MODEL_DIR, f)),
        reverse=True,
    )
    result = []
    for f in files:
        path = os.path.join(MODEL_DIR, f)
        try:
            with open(path, "rb") as fh:
                payload = pickle.load(fh)
            result.append({
                "version":        payload.get("version", f.replace(".pkl", "")),
                "trained_at":     payload.get("trained_at", ""),
                "data_count":     payload.get("data_count", 0),
                "train_count":    payload.get("train_count", 0),
                "test_count":     payload.get("test_count", 0),
                "train_acc":      payload.get("train_acc", {}),
                "features":       payload.get("features", FEATURES),
                "exclude_manual": payload.get("exclude_manual", False),
                "exclude_ranges": payload.get("exclude_ranges", []),
                "filename":       f,
            })
        except Exception as e:
            print(f"[Trainer] list_versions skip {f}: {e}")
    return result


def get_current_version() -> str:
    """model.pkl 에서 직접 버전 읽기 — 롤백 후에도 정확한 현재 버전 반환"""
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        with open(MODEL_PATH, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, dict):
            return data.get("version")
    except Exception:
        pass
    return None


def rollback_to(version: str) -> bool:
    """특정 버전을 model.pkl 로 복원"""
    version_path = os.path.join(MODEL_DIR, f"{version}.pkl")
    if not os.path.exists(version_path):
        return False
    try:
        shutil.copy(version_path, MODEL_PATH)
        print(f"[Rollback] Restored → {version}")
        return True
    except Exception as e:
        print(f"[Rollback] Failed: {e}")
        return False
