"""
스마트팜코리아 공개 데이터 전처리 스크립트

Long format → Wide format 변환
규칙 기반 risk_level 레이블 생성 (Baseline 기준)
결과: data/processed/baseline_data.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ─────────────────────────────────────────────
# 경로 설정
# ─────────────────────────────────────────────

SOURCE_FILE = Path(__file__).parent.parent.parent / "smartfarm-ai" / "dataset" / "0000574" / "24_PF_0000574_01_7579_080400_env.csv"
PROCESSED_DIR = Path(__file__).parent / "processed"

# ─────────────────────────────────────────────
# 컬럼 매핑 (한국어 → 우리 변수명)
# ─────────────────────────────────────────────

COLUMN_MAP = {
    "내부온도":  "indoor_temp",
    "내부습도":  "indoor_humid",
    "외부온도":  "outdoor_temp",
    "내부일사량": "light",
    "지습":     "soil_moisture",
    "내부CO2":  "co2",
}

SENSOR_COLUMNS = list(COLUMN_MAP.values())

# ─────────────────────────────────────────────
# 규칙 기반 레이블 생성 (Baseline 기준값)
# ─────────────────────────────────────────────

def generate_labels(df: pd.DataFrame) -> np.ndarray:
    """
    딸기 재배 봄철(3~5월) 베이스라인 기준 risk_level 생성.

    [봄철 정상 범위 — 스마트팜코리아 0000574 데이터 분석 기준]
      내부 온도:  8 ~ 26°C     (봄철 평균 15.3°C, IQR 9.6~20.3°C)
      내부 습도:  60 ~ 87%     (정상 클래스 평균 66.7%)
      CO2:       400 ~ 700ppm  (봄철 평균 588ppm)

    위험(2):
      내부습도 >= 92%   → 잿빛곰팡이병 위험 수준
      내부온도 >= 30°C  → 딸기 열 스트레스
      내부온도 <=  5°C  → 냉해 위험

    주의(1):
      내부습도 >= 87%   → 고습 경보 구간
      내부온도 >= 26°C  → 고온 주의 구간
      내부온도 <=  8°C  → 저온 주의 구간

    정상(0): 나머지
    """
    risk = np.zeros(len(df), dtype=int)

    danger = (
        (df["indoor_humid"] >= 92) |
        (df["indoor_temp"]  >= 30) |
        (df["indoor_temp"]  <=  5)
    )
    warning = (~danger) & (
        (df["indoor_humid"] >= 87) |
        (df["indoor_temp"]  >= 26) |
        (df["indoor_temp"]  <=  8)
    )

    risk[danger]  = 2
    risk[warning] = 1

    return risk


# ─────────────────────────────────────────────
# 메인 전처리
# ─────────────────────────────────────────────

def preprocess():
    print(f"데이터 로드: {SOURCE_FILE.name}")
    df_raw = pd.read_csv(SOURCE_FILE)
    print(f"원본 행 수: {len(df_raw):,}")

    # 필요한 센서만 필터
    df_filtered = df_raw[df_raw["cd_nm"].isin(COLUMN_MAP.keys())].copy()

    # Long → Wide 피벗
    df_wide = df_filtered.pivot_table(
        index="measurmentdate",
        columns="cd_nm",
        values="snsr_mrslt",
        aggfunc="mean",
    ).reset_index()

    # 컬럼명 변환
    df_wide = df_wide.rename(columns={**COLUMN_MAP, "measurmentdate": "timestamp"})
    df_wide["timestamp"] = pd.to_datetime(df_wide["timestamp"])
    df_wide = df_wide.sort_values("timestamp").reset_index(drop=True)

    # 결측값 처리
    # 일사량: 밤엔 0이 정상 → 0으로 채우기
    df_wide["light"] = df_wide["light"].fillna(0)
    df_wide["soil_moisture"] = df_wide["soil_moisture"].fillna(0)

    # 온습도: 앞뒤 값으로 채우기
    df_wide[["indoor_temp", "indoor_humid", "outdoor_temp", "co2"]] = (
        df_wide[["indoor_temp", "indoor_humid", "outdoor_temp", "co2"]]
        .ffill()
        .bfill()
    )

    # 필수 컬럼 결측 행 제거
    df_wide = df_wide.dropna(subset=["indoor_temp", "indoor_humid"]).reset_index(drop=True)

    # 봄 데이터만 필터 (3~5월)
    # 이유: 현재 테스트베드 운영 시기와 동일한 계절 → 가장 현실적인 베이스라인
    df_wide = df_wide[df_wide["timestamp"].dt.month.isin([3, 4, 5])].reset_index(drop=True)
    print(f"봄 데이터 필터 후 (3~5월): {len(df_wide):,}행")

    # 레이블 생성
    df_wide["risk_level"] = generate_labels(df_wide)

    # 결과 출력
    print(f"전처리 후 행 수: {len(df_wide):,}")
    print(f"\n레이블 분포:")
    counts = df_wide["risk_level"].value_counts().sort_index()
    labels = {0: "정상", 1: "주의", 2: "위험"}
    for k, v in counts.items():
        print(f"  {labels[k]}({k}): {v}행 ({v/len(df_wide)*100:.1f}%)")

    print(f"\n기간: {df_wide['timestamp'].min()} ~ {df_wide['timestamp'].max()}")
    print(f"\n센서별 통계:")
    print(df_wide[SENSOR_COLUMNS].describe().round(2).to_string())

    # 저장
    PROCESSED_DIR.mkdir(exist_ok=True)
    out_path = PROCESSED_DIR / "baseline_data.csv"
    df_wide[["timestamp"] + SENSOR_COLUMNS + ["risk_level"]].to_csv(out_path, index=False)
    print(f"\n저장 완료: {out_path}")

    return df_wide


if __name__ == "__main__":
    preprocess()
