# 스마트팜 AI 자동제어 프로젝트 — CLAUDE.md

> 이 파일은 Claude Code가 프로젝트 컨텍스트를 이해하기 위한 문서입니다.
> 코드 작업 시작 전 반드시 이 파일을 먼저 읽어주세요.

---

## 프로젝트 한 줄 요약

```
가정용 스마트팜에서 AI가 환경 변화를 예측하고
환기·관수·조명을 선제적으로 자동 제어하는 적응형 IoT 시스템
```

---

## 배경 및 목적

현재 스마트팜은 센서값이 임계값을 넘으면 그때 반응하는 **규칙 기반 제어** 방식이 대부분이다.
이 프로젝트는 이를 넘어서:

- 환경 변화 **추세**를 분석해 위험을 **미리** 예측
- 외부 온습도까지 고려한 **적응형** 제어
- 제어 후 실제로 효과가 있었는지 **결과 검증**
- 구동 장치(환기팬) 상태를 **실시간 감시**하는 기능

을 구현하는 것이 목표다.

---

## 시스템 아키텍처

```
[센서들]
  SHT31 (내부 온습도 ×2)
  DHT22 (외부 온습도)
  BH1750 (조도)
  Capacitive (토양수분 ×2)
  MH-Z19B (CO₂, 선택)
  WJFAN FG센서 (팬 RPM)
  ACS712 (팬 전류)
       │
       ▼
[ESP32 WROOM-32]
  - 센서 데이터 수집 (30초/1분 주기)
  - 1차 규칙 기반 판단
  - Wi-Fi → Firebase 전송
  - 제어 명령 수신 → 팬/창/펌프/LED 제어
       │
       ▼
[Firebase Realtime Database]
  /sensors/
    indoor_temp_top      내부 온도 상단
    indoor_humid_top     내부 습도 상단
    indoor_temp_bot      내부 온도 하단
    indoor_humid_bot     내부 습도 하단
    outdoor_temp         외부 온도
    outdoor_humid        외부 습도
    light                조도 (lux)
    soil_moisture        토양 수분
    co2                  CO₂ (선택)
    fan_rpm              팬 RPM
    fan_current          팬 전류
  /control/
    fan_status           팬 ON/OFF
    window_angle         환기창 각도 (0~180)
    pump_status          워터펌프 ON/OFF
    led_r                LED 적색값 (0~255)
    led_b                LED 청색값 (0~255)
  /prediction/
    risk_level           예측 위험도 (0:정상 1:주의 2:위험)
    action               권장 제어 명령
    confidence           모델 신뢰도
       │
       ▼
[FastAPI 서버 (Python)]
  - Firebase 데이터 구독
  - 1D-CNN+LSTM 모델 추론
  - 예측 결과 Firebase 저장
  - 제어 명령 생성
       │
       ▼
[웹 대시보드 (React)]
  - 실시간 센서 차트
  - 심각도 알림 (정상/주의/위험)
  - 제어 현황 표시
  - 장치 이상 경고
```

---

## AI 모델 상세

### 모델 구조: 1D-CNN + LSTM

```python
입력 shape: (batch_size, time_steps, n_features)
            time_steps = 30  (30 스텝 sliding window)
            n_features = 8   (센서 변수 수, CO₂ 선택)

구조:
  Conv1D(32, kernel=3) → ReLU → BatchNorm → Dropout(0.2)
  Conv1D(64, kernel=3) → ReLU → BatchNorm → Dropout(0.2)
  LSTM(128, return_sequences=False) → Dropout(0.3)
  FC(64) → ReLU
  FC(3)  → Softmax

출력: 3클래스 분류
  0: 정상
  1: 주의 (위험 가능성)
  2: 위험 (즉각 제어 필요)
```

### 비교 실험 구성

```
Baseline  고정 임계값 방식 (기존 스마트팜)
Method A  Z-score 동적 임계값
Method B  1D-CNN+LSTM (제안 방식)  ← 우리 모델
```

### 평가 지표

```
정확도 (Accuracy)
F1-Score (macro)
오탐율 FPR (False Positive Rate)
미탐율 FNR (False Negative Rate)
```

---

## 데이터 전략

### 1. 공개 데이터 (스마트팜코리아)

```
URL: https://www.smartfarmkorea.net/datamart
품목: 딸기, 토마토, 파프리카, 오이 등
활용: baseline 패턴 파악, 작물별 최적 환경 기준값 설정
저장: data/raw/smartfarmkorea/
```

### 2. 자체 수집 데이터 (테스트베드)

```
수집 시나리오 5가지:
  S1: 정상 동작
  S2: 고습 위험 (습도 85% 이상)
  S3: 환기 기능 저하 (팬 RPM 급감)
  S4: 과부하 (팬 전류 급등)
  S5: 복합 이상 (온습도 + 장치 이상)

저장: data/collected/
```

### 3. 데이터 전처리

```python
sliding window 구성:
  window_size = 30
  stride = 1
  n분 후 위험 레이블 부여

정규화: MinMaxScaler (센서별 독립 적용)
분할: train 70% / val 15% / test 15%
```

---

## 폴더 구조

```
smartfarm-ai/
├── CLAUDE.md                 ← 이 파일
├── README.md
├── requirements.txt
│
├── data/
│   ├── raw/                  공개 데이터셋 원본
│   │   └── smartfarmkorea/
│   ├── processed/            전처리된 데이터 (.npy, .csv)
│   └── collected/            테스트베드 수집 데이터
│
├── model/
│   ├── cnn_lstm.py           1D-CNN+LSTM 모델 정의
│   ├── baseline.py           고정 임계값 / Z-score 방법
│   ├── dataset.py            데이터셋 클래스 / 로더
│   ├── train.py              학습 루프
│   ├── evaluate.py           평가 및 비교 실험
│   ├── predict.py            단일 추론
│   └── saved/                학습된 모델 체크포인트
│
├── server/
│   ├── app.py                FastAPI 서버 메인
│   ├── inference.py          모델 추론 로직
│   ├── firebase_client.py    Firebase 연동
│   └── controller.py         제어 명령 생성 로직
│
├── dashboard/                웹 대시보드 (React)
│   ├── src/
│   └── public/
│
├── hardware/
│   └── esp32/
│       ├── main.ino          ESP32 메인 펌웨어
│       ├── sensors.h         센서 읽기 함수
│       ├── firebase.h        Firebase 전송
│       └── control.h         제어 명령 실행
│
└── notebooks/
    ├── 01_eda.ipynb           공개 데이터 탐색
    ├── 02_baseline.ipynb      baseline 실험
    └── 03_model_dev.ipynb     모델 개발 과정
```

---

## 개발 환경

```
OS:     macOS (맥북 M5)
GPU:    Apple Silicon MPS 백엔드
Python: 3.11+
IDE:    VSCode

PyTorch MPS 설정:
  import torch
  device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
  print(f"사용 장치: {device}")  # → mps
```

### requirements.txt

```
torch
torchvision
numpy
pandas
scikit-learn
matplotlib
seaborn
firebase-admin
fastapi
uvicorn
python-dotenv
jupyter
```

---

## GitHub 브랜치 전략

```
main          최종 안정 버전 (발표용)
dev           통합 테스트용
feature/hw    하드웨어/펌웨어 (팀원 A)
feature/data  데이터 분석 (팀원 B)
feature/ai    AI 모델 (재현)
feature/server 서버/대시보드 (재현)
```

---

## 역할 분담

| 담당 | 팀원 | 현재 상태 |
|------|------|----------|
| 하드웨어 박스 설계 + ESP32 펌웨어 | 팀원 A | 진행중 |
| 스마트팜코리아 데이터셋 분석 + 변수 매핑 | 팀원 B | 진행중 |
| 1D-CNN+LSTM 모델 + FastAPI 서버 | 재현 (김재현) | 진행중 |

---

## 구매 부품 목록

### 환경 센서
- SHT31 온습도 ×2
- DHT22 온습도 ×1 (외부)
- BH1750 조도 ×1
- Capacitive 토양수분 v1.2 ×2
- MH-Z19B CO₂ ×1 (선택)

### 구동 감지
- WJFAN WJB2501005H-F50 환기팬 ×1 (FG센서 내장)
- ACS712 5A 전류 모듈 ×1
- 리미트 스위치 ×2

### 제어부
- ESP32 WROOM-32 ×1
- 4채널 릴레이 모듈 ×1
- SG90 서보모터 ×1

### 관수
- DC 5V 미니 워터펌프 ×1
- 실리콘 호스 + 점적 노즐
- 물탱크 500ml~1L ×1

### 조명
- WS2812B NeoPixel 60LED/m × 0.5m
- 5V 2A 전원 어댑터 (LED 전용) ×1

### 구조
- 아크릴 3mm 판 (400×300×530mm 기준)
- 아크릴 전용 본드
- 브레드보드, 점퍼선 세트
- 5V 2A USB 어댑터 ×1
- 12V 어댑터 ×1 (선택)

---

## 마일스톤

### 중간발표 (8주차) 목표
- [ ] 하드웨어 테스트베드 동작
- [ ] 센서 → ESP32 → Firebase 파이프라인
- [ ] 데이터 수집 완료
- [ ] 1D-CNN+LSTM 1차 학습 결과
- [ ] FastAPI 서버 AI 추론 동작
- [ ] 웹 대시보드 기본 시각화

### 최종발표 (15주차) 목표
- [ ] AI 적응형 제어 완성
- [ ] 3가지 방법 비교 실험 결과
- [ ] 구동 장치 이상 감지 동작
- [ ] 웹 대시보드 완성
- [ ] 라이브 시연

---

## Claude Code 작업 시작 가이드

이 파일을 읽었다면 아래 순서로 진행:

```
1단계  프로젝트 폴더 구조 생성
       model/ server/ data/ hardware/ notebooks/ dashboard/

2단계  requirements.txt 생성 및 가상환경 세팅
       python -m venv venv && source venv/bin/activate
       pip install -r requirements.txt

3단계  model/cnn_lstm.py 작성
       1D-CNN+LSTM 기본 구조 (MPS 백엔드 적용)

4단계  model/dataset.py 작성
       sliding window 데이터 로더

5단계  model/train.py 작성
       학습 루프 + 체크포인트 저장

6단계  더미 데이터로 동작 확인
       팀원 B 데이터 오면 바로 연결
```
