import json
from datasets import load_dataset, Dataset
import pandas as pd
from tqdm import tqdm
import re

def clean_text(text):
    """텍스트 클리닝"""
    text = re.sub(r'\s+', ' ', text)  # 중복 공백 제거
    text = text.strip()  # 앞뒤 공백 제거
    return text

def prepare_korquad():
    """KorQuAD 1.0 데이터셋 준비"""
    print("KorQuAD 1.0 데이터셋 다운로드 중...")
    dataset = load_dataset("squad_kor_v1")
    
    def preprocess_korquad(example):
        """KorQuAD 예제 전처리"""
        return {
            "question": clean_text(example["question"]),
            "context": clean_text(example["context"]),
            "answer": clean_text(example["answers"]["text"][0]),
            "answer_start": example["answers"]["answer_start"][0]
        }
    
    print("데이터 전처리 중...")
    processed_dataset = dataset.map(preprocess_korquad)
    
    # 학습/검증 데이터 분리
    train_data = processed_dataset["train"]
    valid_data = processed_dataset["validation"]
    
    print(f"학습 데이터 크기: {len(train_data)}")
    print(f"검증 데이터 크기: {len(valid_data)}")
    
    return train_data, valid_data

def prepare_aihub_mrc():
    """AI Hub 기계독해 데이터셋 준비 (데이터 다운로드 필요)"""
    try:
        with open("aihub_mrc_train.json", "r", encoding="utf-8") as f:
            train_data = json.load(f)
        with open("aihub_mrc_valid.json", "r", encoding="utf-8") as f:
            valid_data = json.load(f)
    except FileNotFoundError:
        print("AI Hub 기계독해 데이터셋을 찾을 수 없습니다.")
        print("https://aihub.or.kr/에서 데이터셋을 다운로드 받으세요.")
        return None, None
    
    def convert_aihub_format(data):
        converted = []
        for article in tqdm(data["data"]):
            for paragraph in article["paragraphs"]:
                context = clean_text(paragraph["context"])
                for qa in paragraph["qas"]:
                    if not qa["answers"]:  # 답변이 없는 경우 건너뛰기
                        continue
                    converted.append({
                        "question": clean_text(qa["question"]),
                        "context": context,
                        "answer": clean_text(qa["answers"][0]["text"]),
                        "answer_start": qa["answers"][0]["answer_start"]
                    })
        return Dataset.from_pandas(pd.DataFrame(converted))
    
    print("AI Hub 데이터 전처리 중...")
    train_dataset = convert_aihub_format(train_data)
    valid_dataset = convert_aihub_format(valid_data)
    
    print(f"AI Hub 학습 데이터 크기: {len(train_dataset)}")
    print(f"AI Hub 검증 데이터 크기: {len(valid_dataset)}")
    
    return train_dataset, valid_dataset

def combine_datasets():
    """KorQuAD와 AI Hub 데이터셋 통합"""
    # KorQuAD 데이터 로드
    korquad_train, korquad_valid = prepare_korquad()
    
    # AI Hub 데이터 로드
    aihub_train, aihub_valid = prepare_aihub_mrc()
    
    # 데이터셋 통합
    if aihub_train is not None and aihub_valid is not None:
        train_dataset = Dataset.from_pandas(pd.concat([
            pd.DataFrame(korquad_train),
            pd.DataFrame(aihub_train)
        ]))
        valid_dataset = Dataset.from_pandas(pd.concat([
            pd.DataFrame(korquad_valid),
            pd.DataFrame(aihub_valid)
        ]))
    else:
        train_dataset = korquad_train
        valid_dataset = korquad_valid
    
    # 최종 데이터셋을 파일로 저장
    train_dataset.to_json("qa_train.json")
    valid_dataset.to_json("qa_valid.json")
    
    print("\n최종 데이터셋 통계:")
    print(f"학습 데이터 크기: {len(train_dataset)}")
    print(f"검증 데이터 크기: {len(valid_dataset)}")
    
    return train_dataset, valid_dataset

if __name__ == "__main__":
    print("QA 데이터셋 준비 시작...")
    train_dataset, valid_dataset = combine_datasets()
    print("데이터셋 준비 완료!") 