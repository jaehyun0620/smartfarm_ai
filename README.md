# 스마트팜 AI 자동제어 시스템

가정용 스마트팜에서 AI가 환경 변화를 예측하고 환기·관수·조명을 선제적으로 자동 제어하는 적응형 IoT 시스템.

## 기술 스택

- **AI/서버**: Python 3.11+, PyTorch (MPS), FastAPI, Firebase
- **하드웨어**: ESP32 WROOM-32
- **프론트엔드**: React + Firebase SDK

## 폴더 구조

```
smartfarm_ai/
├── data/           데이터셋 (공개 + 자체 수집)
├── model/          1D-CNN+LSTM 모델
├── server/         FastAPI 서버
├── dashboard/      React 웹 대시보드
├── hardware/       ESP32 펌웨어
└── notebooks/      EDA 및 실험
```

## 환경 세팅

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
