# 한국어 LLM 파인튜닝 프로젝트

이 프로젝트는 한국어 LLM(Large Language Model)을 특정 목적에 맞게 파인튜닝하는 코드를 제공합니다.

## 프로젝트 구조

```
LLM_SFT/
├── QA/                     # 질의응답 모델 학습
│   └── train_qa.py        # KorQuAD 데이터셋을 이용한 QA 모델 학습
├── Domain_Specific/        # 도메인 특화 모델 학습
│   └── train_domain.py    # 사용자 정의 데이터셋을 이용한 도메인 특화 모델 학습
└── Resources/             # 프로젝트 리소스
    ├── requirements.txt   # 필요한 패키지 목록
    └── README.md         # 프로젝트 설명
```

## 설치 방법

1. 필요한 패키지 설치:
```bash
pip install -r Resources/requirements.txt
```

2. (선택사항) CUDA 설치:
- NVIDIA GPU를 사용하는 경우 CUDA 11.7 이상 설치 필요

## 사용 방법

### QA 모델 학습

KorQuAD 데이터셋을 사용하여 질의응답 모델을 학습합니다:

```bash
cd QA
python train_qa.py
```

### 도메인 특화 모델 학습

사용자 정의 데이터셋을 사용하여 특정 도메인에 특화된 모델을 학습합니다:

1. 데이터 준비:
   - CSV 형식: instruction,input,output
   - 예시:
     ```
     instruction,input,output
     "이 문장을 분석해주세요","텍스트 내용","분석 결과"
     ```

2. 학습 실행:
```bash
cd Domain_Specific
python train_domain.py
```

## 주요 기능

1. QA 모델 학습
   - KorQuAD 데이터셋 사용
   - 4비트 양자화로 메모리 효율성 개선
   - LoRA를 통한 효율적인 파인튜닝

2. 도메인 특화 모델 학습
   - 사용자 정의 데이터셋 지원
   - instruction 튜닝 방식 적용
   - 4비트 양자화 및 LoRA 적용

## 참고 사항

- 기본 모델로 beomi/llama-2-ko-7b를 사용
- 학습에는 최소 16GB 이상의 GPU 메모리 권장
- 4비트 양자화를 통해 메모리 사용량 최적화
- LoRA를 통해 효율적인 파인튜닝 가능

## 라이선스

이 프로젝트는 Apache 2.0 라이선스를 따릅니다. 