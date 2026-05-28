"""
데이터 로거 — 아두이노 JSON 레코드를 CSV에 저장
- 구버전 CSV(hour 컬럼 없음) 자동 마이그레이션
"""
import csv
import os
import time

LOG_PATH   = os.path.join(os.path.dirname(__file__), "farm_log.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

# 현재(신버전) 컬럼 — 16개
COLUMNS = [
    "timestamp",
    "temp1", "hum1", "soil_percent", "light_raw", "co2_raw", "water_raw",
    "hour",
    "fan1", "fan2", "window1", "window2", "led", "humidifier", "heater",
    "manual",
]

# 구버전 컬럼 — 15개 (hour 없음)
_OLD_COLUMNS = [
    "timestamp",
    "temp1", "hum1", "soil_percent", "light_raw", "co2_raw", "water_raw",
    "fan1", "fan2", "window1", "window2", "led", "humidifier", "heater",
    "manual",
]


def _migrate_csv_if_needed():
    """
    헤더가 구버전(15컬럼)이면 hour 컬럼을 삽입해 신버전(16컬럼)으로 변환.
    이미 최신이면 아무것도 안 함.
    """
    if not os.path.exists(LOG_PATH):
        return
    try:
        with open(LOG_PATH, "r", newline="") as f:
            first_line = f.readline().strip()

        existing_cols = [c.strip() for c in first_line.split(",")]

        if existing_cols == COLUMNS:
            return   # 이미 최신

        if "hour" in existing_cols:
            return   # hour 있으면 OK

        # ── 마이그레이션 필요 ──
        print("[Logger] CSV 마이그레이션 시작 (hour 컬럼 추가)…")
        rows = []
        with open(LOG_PATH, "r", newline="") as f:
            reader = csv.reader(f)
            next(reader)   # 헤더 스킵
            for line in reader:
                if len(line) == len(_OLD_COLUMNS):
                    ts   = line[0]
                    hour = int(ts[11:13]) if len(ts) >= 13 else 0
                    # 7번째 위치(water_raw 뒤, fan1 앞)에 hour 삽입
                    new_row = line[:7] + [hour] + line[7:]
                    rows.append(new_row)
                elif len(line) == len(COLUMNS):
                    rows.append(line)   # 이미 신버전
                # 그 외 깨진 행 건너뜀

        with open(LOG_PATH, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(COLUMNS)
            w.writerows(rows)

        print(f"[Logger] 마이그레이션 완료: {len(rows)}행 처리됨")

    except Exception as e:
        print(f"[Logger] 마이그레이션 실패 (무시하고 계속): {e}")


# 서버 시작 시 자동 실행
_migrate_csv_if_needed()


def _ensure_header():
    if not os.path.exists(LOG_PATH):
        try:
            with open(LOG_PATH, "w", newline="") as f:
                csv.writer(f).writerow(COLUMNS)
        except Exception as e:
            print(f"[Logger] Header write error: {e}")


def log_record(record: dict):
    """아두이노 JSON 레코드 한 줄을 CSV에 추가"""
    _ensure_header()
    try:
        s = record.get("sensors",   record)
        a = record.get("actuators", record)

        is_manual = int(record.get("manual", 0))
        ts        = record.get("timestamp", time.strftime("%Y-%m-%dT%H:%M:%S"))
        hour      = int(ts[11:13]) if len(ts) >= 13 else 0

        row = [
            ts,
            s.get("temp1",        ""),
            s.get("hum1",         ""),
            s.get("soil_percent", ""),
            s.get("light_raw",    ""),
            s.get("co2_raw",      ""),
            s.get("water_raw",    ""),
            hour,
            a.get("fan1",       0),
            a.get("fan2",       0),
            a.get("window1",    0),
            a.get("window2",    0),
            a.get("led",        0),
            a.get("humidifier", 0),
            a.get("heater",     0),
            is_manual,
        ]

        with open(LOG_PATH, "a", newline="") as f:
            csv.writer(f).writerow(row)

    except Exception as e:
        print(f"[Logger] log_record error: {e}")


def get_log_stats() -> dict:
    _ensure_header()
    total  = 0
    manual = 0
    try:
        with open(LOG_PATH, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                total  += 1
                if row.get("manual") == "1":
                    manual += 1
    except Exception as e:
        print(f"[Logger] get_log_stats error: {e}")

    model_exists = os.path.exists(MODEL_PATH)
    last_trained = None
    if model_exists:
        try:
            last_trained = time.strftime(
                "%Y-%m-%dT%H:%M:%S",
                time.localtime(os.path.getmtime(MODEL_PATH))
            )
        except Exception:
            pass

    return {
        "total":        total,
        "manual":       manual,
        "model_exists": model_exists,
        "last_trained": last_trained,
    }
