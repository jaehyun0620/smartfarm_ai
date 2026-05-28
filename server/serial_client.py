"""
시리얼 클라이언트 — 아두이노와 JSON 통신
- 아두이노가 보내는 JSON을 읽어 on_data 콜백 호출
- send_control(field, value) 로 제어 명령 전송
- 연결 끊기면 5초마다 자동 재연결
"""
import json
import serial
import threading
import time


class SerialClient:
    def __init__(self, port: str, baud: int, on_data=None):
        self.port     = port
        self.baud     = baud
        self.on_data  = on_data
        self._ser     = None
        self._thread  = None
        self._running = False
        self._lock    = threading.Lock()   # write 동시 호출 보호 전용

    def start(self):
        self._running = True
        self._connect()
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        try:
            if self._ser and self._ser.is_open:
                self._ser.close()
        except Exception:
            pass

    def _connect(self):
        """시리얼 포트 연결 시도 (실패해도 서버 계속 실행)"""
        try:
            if self._ser and self._ser.is_open:
                self._ser.close()
            self._ser = serial.Serial(self.port, self.baud, timeout=1)
            time.sleep(2)   # 아두이노 리셋 대기
            print(f"[Serial] Opened {self.port} @ {self.baud}")
        except serial.SerialException as e:
            print(f"[Serial] Connect failed: {e}")
            self._ser = None
        except Exception as e:
            print(f"[Serial] Unexpected connect error: {e}")
            self._ser = None

    def send_control(self, field: str, value: int):
        """
        아두이노로 제어 명령 전송
        예: send_control("window1", 1)  →  {"window1":1}\n
        """
        if not self._ser or not self._ser.is_open:
            print(f"[Serial] send_control SKIP — port not open  field={field} value={value}")
            return

        payload = json.dumps({field: value}) + "\n"
        with self._lock:   # write 끼리만 직렬화 (readline 과 lock 공유 안 함)
            try:
                self._ser.write(payload.encode("utf-8"))
                self._ser.flush()
                print(f"[Serial] Sent: {payload.strip()}")
            except serial.SerialException as e:
                print(f"[Serial] Write error: {e}")
            except Exception as e:
                print(f"[Serial] Write unexpected: {e}")

    # ── 내부: 수신 루프 ──────────────────────────────────────────────────────
    def _read_loop(self):
        while self._running:

            # ── 포트 없으면 5초 후 재연결 ──
            if not self._ser or not self._ser.is_open:
                print("[Serial] Reconnecting in 5s…")
                time.sleep(5)
                self._connect()
                continue

            try:
                # lock 없이 readline — pyserial read/write 는 별개 버퍼라 안전
                raw = self._ser.readline()

                if not raw:
                    continue

                line = raw.decode("utf-8", errors="ignore").strip()
                if not line:
                    continue

                # JSON 파싱
                try:
                    record = json.loads(line)
                    if self.on_data:
                        self.on_data(record)
                except json.JSONDecodeError:
                    print(f"[Arduino] {line}")

            except serial.SerialException as e:
                print(f"[Serial] Read error: {e} — reconnecting")
                try:
                    self._ser.close()
                except Exception:
                    pass
                self._ser = None   # 루프 상단에서 재연결

            except Exception as e:
                print(f"[Serial] Unexpected: {e}")
                time.sleep(0.1)
