# QA 모델 학습

이 디렉토리는 한국어 질의응답(QA) 모델 학습을 위한 코드를 포함하고 있습니다.

## 데이터셋

다음 두 가지 데이터셋을 사용합니다:

1. KorQuAD 1.0
   - 한국어 기계독해 데이터셋
   - 약 70,000개의 질문-답변 쌍
   - Hugging Face에서 자동으로 다운로드됨

2. AI Hub 기계독해
   - AI Hub의 기계독해 데이터셋
   - 데이터 다운로드 필요: https://aihub.or.kr/
   - 다운로드 후 다음 파일명으로 저장:
     - aihub_mrc_train.json
     - aihub_mrc_valid.json

## 사용 방법

1. 데이터셋 준비:
```bash
python prepare_qa_dataset.py
```
- KorQuAD 데이터셋 자동 다운로드
- AI Hub 데이터셋이 있는 경우 함께 통합
- 전처리된 데이터는 다음 파일로 저장됨:
  - qa_train.json
  - qa_valid.json

2. 모델 학습:

a) LoRA 방식 (권장):
```bash
python train_qa.py
```
- 적은 메모리로 효율적인 학습 가능
- 기본 모델 파라미터는 유지하면서 작은 크기의 어댑터만 학습
- 학습된 모델은 `qa_model_output` 디렉토리에 저장

b) Full Fine-tuning 방식:
```bash
python train_qa_full.py
```
- 모든 모델 파라미터를 학습
- 더 많은 GPU 메모리 필요 (최소 24GB 이상 권장)
- 더 나은 성능을 얻을 수 있지만 학습이 더 오래 걸림
- 학습된 모델은 `qa_full_model_output` 디렉토리에 저장

## 데이터 형식

전처리된 데이터는 다음 형식을 따릅니다:
```json
{
    "question": "질문 내용",
    "context": "지문 내용",
    "answer": "답변 내용",
    "answer_start": 답변시작위치
}
```

## 전처리 과정

1. 텍스트 클리닝
   - 중복 공백 제거
   - 앞뒤 공백 제거
   - 특수문자 정규화

2. 데이터셋 통합
   - KorQuAD와 AI Hub 데이터 형식 통일
   - 중복 제거
   - 학습/검증 데이터 분리

## 학습 방식 비교

1. LoRA (Low-Rank Adaptation)
   - 장점:
     - 적은 메모리 사용량 (8-16GB GPU 메모리로 충분)
     - 빠른 학습 속도
     - 기존 모델 파라미터 보존
   - 단점:
     - 기본 모델 대비 약간의 성능 저하 가능성

2. Full Fine-tuning
   - 장점:
     - 최대의 성능 달성 가능
     - 모델의 모든 파라미터 최적화
   - 단점:
     - 많은 메모리 필요 (24GB 이상 권장)
     - 긴 학습 시간
     - 과적합 위험이 더 큼

## 참고사항

- KorQuAD 데이터셋은 자동으로 다운로드됩니다.
- AI Hub 데이터셋은 별도 다운로드가 필요합니다.
- 메모리 사용량을 고려하여 데이터를 배치로 처리합니다.
- GPU 메모리가 제한적인 경우 LoRA 방식을 권장합니다. 