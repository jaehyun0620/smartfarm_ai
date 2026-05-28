# 스마트팜 AI 자동제어 시스템

센서 데이터를 기반으로 **규칙 기반 자동제어**와 **사용자 행동 학습 AI**를 함께 운용하는 IoT 스마트팜 시스템.

> **핵심 차별점 3가지**
> 1. **사용자 행동 학습** — 수동 조작 이력을 학습 데이터로 활용해 쓸수록 사용자 취향에 맞게 진화
> 2. **규칙 vs AI 실시간 비교** — 같은 센서값에 대해 규칙기반과 AI 모델의 판단을 동시에 시각화
> 3. **학습 데이터 품질 제어** — 재학습 전 특정 시간 구간 또는 수동 조작 구간을 직접 제외 가능

---

## 시스템 구조

```
아두이노 (센서 + 구동부)
    ↕ USB 시리얼 (JSON, 9600baud)
젯슨 나노 (FastAPI 서버)
    ├── auto 모드: 하드코딩 규칙 기반 제어
    ├── AI 모드: Random Forest 모델 제어
    └── ngrok 터널
         ↕ HTTP
대시보드 (React, 스마트폰 브라우저)
```

---

## 주요 기능

### 제어 모드
| 모드 | 동작 |
|---|---|
| **auto** | 온도·습도·조도 기준값으로 아두이노가 직접 판단 |
| **AI** | 젯슨에서 Random Forest 모델이 예측 후 시리얼 명령 전송 |
| **수동** | 대시보드에서 기기별 직접 ON/OFF |

### 타이머 제어
- **타이머 모드**: 기기를 켠 뒤 N분 후 자동으로 끄기
- **스케줄 모드**: HH:MM에 켜고 HH:MM에 끄기 (당일 또는 익일 자동 판단)
- 1초 단위 실시간 카운트다운 + 프로그레스 바

### AI 모델
- 알고리즘: Random Forest (기기별 7개 분류기)
- 학습 feature: 온도·습도·토양수분·조도·CO₂·수위·시간대 (7개)
- 예측 target: fan1, fan2, window1, window2, led, humidifier, heater
- 수동 조작 데이터에 **2배 가중치** 적용
- **모델 버전 관리**: 학습할 때마다 자동 저장, 이전 버전으로 롤백 가능

### 데이터 관리
- 아두이노 JSON → `farm_log.csv` 자동 저장
- 재학습 전 미리보기: 전체 / 제외 / 학습 예정 건수 확인
- 수동 조작 구간 제외 옵션
- 특정 시간 구간 제외 옵션 (datetime-local 입력)

---

## 폴더 구조

```
smartfarm_ai/
├── server/
│   ├── app.py             # FastAPI 서버 (젯슨에서 실행)
│   ├── ai_controller.py   # AI 추론 + auto 규칙 비교
│   ├── model_trainer.py   # Random Forest 학습 + 버전관리
│   ├── data_logger.py     # CSV 로그 저장 + 자동 마이그레이션
│   ├── serial_client.py   # 아두이노 시리얼 통신 + 자동 재연결
│   ├── farm_log.csv       # 센서·구동부 기록 (gitignore)
│   ├── model.pkl          # 현재 활성 모델 (gitignore)
│   └── model_versions/    # 버전별 모델 저장소 (gitignore)
├── dashboard/
│   └── src/App.jsx        # React 대시보드 (단일 파일)
├── hardware/
│   └── smartfarm/
│       └── smartfarm.ino  # 아두이노 펌웨어
└── requirements.txt
```

---

## 실행 방법

### 젯슨 나노 (서버)

```bash
cd server
pip install -r ../requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

외부 접속이 필요한 경우 ngrok 사용:
```bash
ngrok http 8000
```

환경변수로 시리얼 포트 변경:
```bash
SERIAL_PORT=/dev/ttyUSB1 uvicorn app:app --host 0.0.0.0 --port 8000
```

### 대시보드 (로컬 개발)

```bash
cd dashboard
npm install
npm run dev
```

`src/App.jsx` 상단의 `API_BASE_URL`을 젯슨 IP 또는 ngrok 주소로 변경:
```js
const API_BASE_URL = 'http://192.168.x.x:8000'   // 같은 와이파이
// 또는
const API_BASE_URL = 'https://xxxx.ngrok-free.app' // ngrok
```

---

## API 엔드포인트

| 메서드 | 경로 | 설명 |
|---|---|---|
| GET | `/api/sensor-logs` | 최근 120개 센서 로그 |
| POST | `/api/control` | 기기 ON/OFF 제어 |
| POST | `/api/mode` | auto / AI 모드 전환 |
| GET | `/api/model-info` | 모델 정보 + auto vs AI 비교 |
| POST | `/api/train` | 모델 재학습 |
| POST | `/api/train-preview` | 학습 전 데이터 건수 미리보기 |
| GET | `/api/model-versions` | 저장된 모델 버전 목록 |
| POST | `/api/model-rollback` | 특정 버전으로 롤백 |
| POST | `/api/timer` | 타이머 생성 (duration / schedule) |
| GET | `/api/timers` | 활성 타이머 목록 |
| DELETE | `/api/timer/{id}` | 타이머 취소 |

---

## 하드웨어 구성

| 부품 | 핀 | 용도 |
|---|---|---|
| DHT11 × 2 | D11, D12 | 온도·습도 측정 |
| 토양 수분 센서 | A0 | 토양 수분 (raw → %) |
| 조도 센서 | A1 | 광량 측정 |
| 수위 센서 | A2 | 물통 수위 |
| CO₂ 센서 | A3 | 이산화탄소 농도 |
| 릴레이 × 5 | D3,4,8,9,10 | 팬·히터·LED·가습기 제어 |
| 서보모터 × 2 | D5, D6 | 창문 개폐 (비블로킹) |
| 마그네틱 센서 × 2 | D2, D7 | 창문 실제 상태 감지 |

---

## 기술 스택

- **하드웨어**: Arduino Uno, Jetson Nano
- **서버**: Python 3.9+, FastAPI, scikit-learn, pandas
- **AI**: Random Forest Classifier (sklearn)
- **프론트엔드**: React 18, Vite, Recharts
- **통신**: USB Serial (JSON, 9600baud), ngrok (외부 접속)
