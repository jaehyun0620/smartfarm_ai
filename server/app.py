"""
FastAPI 서버 — 젯슨 나노에서 실행
$ uvicorn app:app --host 0.0.0.0 --port 8000
"""
import json, os, threading, time, uuid
from collections import deque
from datetime import datetime, timedelta

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── 로컬 모듈 ──────────────────────────────────────────────────────────────────
from serial_client import SerialClient
from data_logger   import log_record, get_log_stats
from model_trainer import train_models, preview_filter, list_versions, rollback_to, get_current_version
from ai_controller import AIController

# ── 전역 상태 ──────────────────────────────────────────────────────────────────
SERIAL_PORT   = os.getenv("SERIAL_PORT", "/dev/ttyUSB0")
SERIAL_BAUD   = int(os.getenv("SERIAL_BAUD", "9600"))
RECORD_BUFFER = deque(maxlen=120)

serial_client: SerialClient = None
ai_controller               = AIController()
control_mode                = "auto"   # "auto" | "ai"
train_lock                  = threading.Lock()
is_training                 = False

# ── 타이머 상태 ────────────────────────────────────────────────────────────────
timers      = {}
timers_lock = threading.Lock()

# ── FastAPI ────────────────────────────────────────────────────────────────────
app = FastAPI(title="SmartFarm API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 디바이스 이름 매핑 ─────────────────────────────────────────────────────────
DEVICE_MAP = {
    "fan 1":    "fan1",
    "fan 2":    "fan2",
    "window 1": "window1",
    "window 2": "window2",
    "grow_led": "led",
    "humid":    "humidifier",
    "heater":   "heater",
}

# ── 시리얼 수신 콜백 ───────────────────────────────────────────────────────────
def on_serial_data(record: dict):
    try:
        if "timestamp" not in record:
            record["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        RECORD_BUFFER.append(record)
        log_record(record)

        if control_mode == "ai" and ai_controller.model_loaded:
            commands = ai_controller.predict(record)
            for field, value in commands.items():
                if serial_client:
                    serial_client.send_control(field, value)
                    print(f"[AI] Control → {field}={value}")
    except Exception as e:
        print(f"[Server] on_serial_data error: {e}")

# ── 앱 시작 / 종료 ─────────────────────────────────────────────────────────────
@app.on_event("startup")
def startup():
    global serial_client
    try:
        serial_client = SerialClient(
            port=SERIAL_PORT,
            baud=SERIAL_BAUD,
            on_data=on_serial_data,
        )
        serial_client.start()
        print(f"[Server] Serial client started on {SERIAL_PORT}@{SERIAL_BAUD}")
    except Exception as e:
        print(f"[Server] Serial start error: {e}")

@app.on_event("shutdown")
def shutdown():
    if serial_client:
        serial_client.stop()

# ── API ────────────────────────────────────────────────────────────────────────

@app.get("/api/sensor-logs")
def get_sensor_logs():
    return list(RECORD_BUFFER)


class ControlRequest(BaseModel):
    device: str
    state:  bool

@app.post("/api/control")
def control_device(req: ControlRequest):
    field = DEVICE_MAP.get(req.device)
    if not field:
        return {"ok": False, "error": f"Unknown device: {req.device}"}
    value = 1 if req.state else 0
    print(f"[Control] {req.device!r} → {field!r}={value}")
    if serial_client:
        serial_client.send_control(field, value)
    return {"ok": True, "device": req.device, "field": field, "value": value}


class ModeRequest(BaseModel):
    mode: str

@app.post("/api/mode")
def set_mode(req: ModeRequest):
    global control_mode
    if req.mode not in ("auto", "ai"):
        return {"ok": False, "error": "mode must be 'auto' or 'ai'"}
    control_mode = req.mode
    print(f"[Mode] Changed to {control_mode}")
    if req.mode == "auto" and serial_client:
        serial_client.send_control("manual", 0)
        print("[Mode] Sent {manual:0} → Arduino rule-based resumed")
    return {"ok": True, "mode": control_mode}


@app.get("/api/train-status")
def train_status():
    try:
        stats = get_log_stats()
        return {
            "total":        stats["total"],
            "manual":       stats["manual"],
            "model_exists": stats["model_exists"],
            "last_trained": stats.get("last_trained"),
        }
    except Exception as e:
        return {"total": 0, "manual": 0, "model_exists": False, "last_trained": None, "error": str(e)}


@app.get("/api/model-info")
def model_info():
    try:
        from ai_controller import AUTO_RULES
        latest      = RECORD_BUFFER[-1] if RECORD_BUFFER else {}
        comparison  = ai_controller.compare(latest) if latest else {}
        importances = ai_controller.feature_importances()
        stats       = get_log_stats()

        return {
            "model_loaded":    ai_controller.model_loaded,
            "train_acc":       ai_controller.get_train_acc_display(),
            "auto_rules":      AUTO_RULES,
            "comparison":      comparison,
            "importances":     importances,
            "data_stats": {
                "total":        stats["total"],
                "manual":       stats["manual"],
                "last_trained": stats.get("last_trained"),
            },
            "current_sensors": latest.get("sensors", {}),
            "current_mode":    control_mode,
        }
    except Exception as e:
        return {"error": str(e)}


class TrainRequest(BaseModel):
    exclude_manual: bool = False
    exclude_ranges: list = []

@app.post("/api/train-preview")
def train_preview(req: TrainRequest):
    try:
        result = preview_filter(
            exclude_manual=req.exclude_manual,
            exclude_ranges=req.exclude_ranges,
        )
        return {"ok": True, **result}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/api/train")
def trigger_train(req: TrainRequest = None):
    global is_training
    if is_training:
        return {"ok": False, "error": "Already training"}
    if req is None:
        req = TrainRequest()

    def _train():
        global is_training
        with train_lock:
            is_training = True
            try:
                result = train_models(
                    exclude_manual=req.exclude_manual,
                    exclude_ranges=req.exclude_ranges,
                )
                ai_controller.reload()
                print(f"[Train] Done — {result['version']} reloaded")
            except Exception as e:
                print(f"[Train] Error: {e}")
            finally:
                is_training = False

    threading.Thread(target=_train, daemon=True).start()
    try:
        stats = get_log_stats()
        return {
            "ok":             True,
            "total":          stats["total"],
            "manual":         stats["manual"],
            "exclude_manual": req.exclude_manual,
            "exclude_ranges": req.exclude_ranges,
        }
    except Exception as e:
        return {"ok": True, "error": str(e)}


# ── 모델 버전 관리 ─────────────────────────────────────────────────────────────

@app.get("/api/model-versions")
def get_model_versions():
    try:
        versions = list_versions()
        current  = get_current_version()   # model.pkl 직접 읽어서 정확한 버전 반환
        return {"versions": versions, "current": current}
    except Exception as e:
        return {"versions": [], "current": None, "error": str(e)}


class RollbackRequest(BaseModel):
    version: str

@app.post("/api/model-rollback")
def model_rollback(req: RollbackRequest):
    try:
        ok = rollback_to(req.version)
        if not ok:
            return {"ok": False, "error": f"버전을 찾을 수 없음: {req.version}"}
        ai_controller.reload()
        print(f"[Rollback] Active model → {req.version}")
        return {"ok": True, "version": req.version}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ── 타이머 헬퍼 ───────────────────────────────────────────────────────────────

def _send_device(device: str, state: bool):
    field = DEVICE_MAP.get(device)
    if not field or not serial_client:
        return
    try:
        serial_client.send_control(field, 1 if state else 0)
        print(f"[Timer] {device} → {'ON' if state else 'OFF'}")
    except Exception as e:
        print(f"[Timer] send_device error: {e}")


def _duration_thread(timer_id: str, device: str, duration_sec: int,
                     cancel_event: threading.Event):
    try:
        _send_device(device, True)
        with timers_lock:
            if timer_id in timers:
                timers[timer_id]["status"] = "running"

        cancelled = cancel_event.wait(timeout=duration_sec)

        with timers_lock:
            if timer_id in timers:
                timers[timer_id]["status"] = "cancelled" if cancelled else "done"

        if not cancelled:
            _send_device(device, False)
    except Exception as e:
        print(f"[Timer] _duration_thread error: {e}")
    finally:
        time.sleep(5)
        with timers_lock:
            timers.pop(timer_id, None)


def _schedule_thread(timer_id: str, device: str, on_time_str: str,
                     off_time_str: str, cancel_event: threading.Event):
    try:
        def _next_dt(hhmm: str) -> datetime:
            now = datetime.now()
            t = datetime.strptime(hhmm, "%H:%M").replace(
                year=now.year, month=now.month, day=now.day, second=0, microsecond=0
            )
            if t <= now:
                t += timedelta(days=1)
            return t

        on_dt  = _next_dt(on_time_str)
        off_dt = datetime.strptime(off_time_str, "%H:%M").replace(
            year=on_dt.year, month=on_dt.month, day=on_dt.day, second=0, microsecond=0
        )
        if off_dt <= on_dt:
            off_dt += timedelta(days=1)

        with timers_lock:
            if timer_id in timers:
                timers[timer_id].update({
                    "status": "waiting",
                    "on_at":  on_dt.strftime("%Y-%m-%dT%H:%M:%S"),
                    "off_at": off_dt.strftime("%Y-%m-%dT%H:%M:%S"),
                })

        wait = (on_dt - datetime.now()).total_seconds()
        if wait > 0 and cancel_event.wait(timeout=wait):
            with timers_lock:
                if timer_id in timers:
                    timers[timer_id]["status"] = "cancelled"
            time.sleep(5)
            with timers_lock:
                timers.pop(timer_id, None)
            return

        _send_device(device, True)
        with timers_lock:
            if timer_id in timers:
                timers[timer_id]["status"] = "on"

        wait = (off_dt - datetime.now()).total_seconds()
        cancelled = cancel_event.wait(timeout=max(0, wait))

        with timers_lock:
            if timer_id in timers:
                timers[timer_id]["status"] = "cancelled" if cancelled else "done"

        if not cancelled:
            _send_device(device, False)

    except Exception as e:
        print(f"[Timer] _schedule_thread error: {e}")
    finally:
        time.sleep(5)
        with timers_lock:
            timers.pop(timer_id, None)


# ── 타이머 API ────────────────────────────────────────────────────────────────

class TimerRequest(BaseModel):
    device:           str
    timer_type:       str
    duration_minutes: int = 0
    on_time:          str = ""
    off_time:         str = ""

@app.post("/api/timer")
def create_timer(req: TimerRequest):
    if req.device not in DEVICE_MAP:
        return {"ok": False, "error": f"Unknown device: {req.device}"}

    tid          = uuid.uuid4().hex[:8]
    cancel_event = threading.Event()

    if req.timer_type == "duration":
        if req.duration_minutes <= 0:
            return {"ok": False, "error": "duration_minutes must be > 0"}
        dur_sec = req.duration_minutes * 60
        ends_at = (datetime.now() + timedelta(seconds=dur_sec)).strftime("%Y-%m-%dT%H:%M:%S")
        info = {
            "id": tid, "device": req.device, "type": "duration",
            "duration_minutes": req.duration_minutes,
            "started_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "ends_at": ends_at, "status": "starting",
            "cancel_event": cancel_event,
        }
        with timers_lock:
            timers[tid] = info
        threading.Thread(target=_duration_thread,
                         args=(tid, req.device, dur_sec, cancel_event),
                         daemon=True).start()

    elif req.timer_type == "schedule":
        if not req.on_time or not req.off_time:
            return {"ok": False, "error": "on_time and off_time required"}
        try:
            datetime.strptime(req.on_time,  "%H:%M")
            datetime.strptime(req.off_time, "%H:%M")
        except ValueError:
            return {"ok": False, "error": "Time format must be HH:MM"}
        info = {
            "id": tid, "device": req.device, "type": "schedule",
            "on_time": req.on_time, "off_time": req.off_time,
            "status": "starting", "cancel_event": cancel_event,
        }
        with timers_lock:
            timers[tid] = info
        threading.Thread(target=_schedule_thread,
                         args=(tid, req.device, req.on_time, req.off_time, cancel_event),
                         daemon=True).start()
    else:
        return {"ok": False, "error": "timer_type must be 'duration' or 'schedule'"}

    print(f"[Timer] Created {tid} — {req.timer_type} for {req.device}")
    return {"ok": True, "timer_id": tid}


@app.get("/api/timers")
def get_timers():
    now = datetime.now()
    with timers_lock:
        result = []
        for info in timers.values():
            d = {k: v for k, v in info.items() if k != "cancel_event"}
            if d.get("type") == "duration" and d.get("status") == "running":
                try:
                    ends = datetime.fromisoformat(d["ends_at"])
                    d["remaining_seconds"] = max(0, int((ends - now).total_seconds()))
                except Exception:
                    pass
            result.append(d)
    return {"timers": result}


@app.delete("/api/timer/{timer_id}")
def cancel_timer(timer_id: str):
    with timers_lock:
        info = timers.get(timer_id)
    if not info:
        return {"ok": False, "error": "Timer not found"}
    info["cancel_event"].set()
    print(f"[Timer] Cancelled {timer_id}")
    return {"ok": True, "timer_id": timer_id}
