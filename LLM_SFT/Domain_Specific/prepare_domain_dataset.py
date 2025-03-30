import json
import pandas as pd
from datasets import load_dataset, Dataset
from tqdm import tqdm
import re
import os

def clean_text(text):
    """텍스트 클리닝"""
    text = re.sub(r'\s+', ' ', text)  # 중복 공백 제거
    text = text.strip()  # 앞뒤 공백 제거
    return text

def prepare_nsmc_dataset():
    """네이버 영화 리뷰 감성분석 데이터셋 준비"""
    print("NSMC 데이터셋 다운로드 중...")
    dataset = load_dataset("nsmc")
    
    def convert_nsmc_format(example):
        return {
            "instruction": "이 영화 리뷰의 감정을 분석해주세요.",
            "input": clean_text(example["text"]),
            "output": "긍정적입니다." if example["label"] == 1 else "부정적입니다."
        }
    
    print("NSMC 데이터 전처리 중...")
    train_data = dataset["train"].map(convert_nsmc_format)
    valid_data = dataset["test"].map(convert_nsmc_format)
    
    print(f"NSMC 학습 데이터 크기: {len(train_data)}")
    print(f"NSMC 검증 데이터 크기: {len(valid_data)}")
    
    return train_data, valid_data

def prepare_klue_nli():
    """KLUE-NLI 데이터셋 준비"""
    print("KLUE-NLI 데이터셋 다운로드 중...")
    dataset = load_dataset("klue", "nli")
    
    label_map = {
        0: "두 문장은 서로 모순됩니다.",
        1: "두 문장은 서로 관련이 없습니다.",
        2: "두 문장은 서로 동일한 의미입니다."
    }
    
    def convert_nli_format(example):
        return {
            "instruction": "다음 두 문장의 관계를 분석해주세요.",
            "input": f"문장1: {clean_text(example['premise'])}\n문장2: {clean_text(example['hypothesis'])}",
            "output": label_map[example["label"]]
        }
    
    print("KLUE-NLI 데이터 전처리 중...")
    train_data = dataset["train"].map(convert_nli_format)
    valid_data = dataset["validation"].map(convert_nli_format)
    
    print(f"KLUE-NLI 학습 데이터 크기: {len(train_data)}")
    print(f"KLUE-NLI 검증 데이터 크기: {len(valid_data)}")
    
    return train_data, valid_data

def prepare_aihub_dialog():
    """AI Hub 한국어 대화 데이터셋 준비 (데이터 다운로드 필요)"""
    try:
        with open("aihub_dialog_train.json", "r", encoding="utf-8") as f:
            train_data = json.load(f)
        with open("aihub_dialog_valid.json", "r", encoding="utf-8") as f:
            valid_data = json.load(f)
    except FileNotFoundError:
        print("AI Hub 대화 데이터셋을 찾을 수 없습니다.")
        print("https://aihub.or.kr/에서 데이터셋을 다운로드 받으세요.")
        return None, None
    
    def convert_dialog_format(data):
        converted = []
        for dialog in tqdm(data):
            context = ""
            for i, utterance in enumerate(dialog["dialog"]):
                if i > 0:  # 첫 발화가 아닌 경우
                    converted.append({
                        "instruction": "주어진 대화 맥락에 대한 적절한 응답을 생성해주세요.",
                        "input": context.strip(),
                        "output": clean_text(utterance["text"])
                    })
                context += f"{utterance['speaker']}: {clean_text(utterance['text'])}\n"
        
        return Dataset.from_pandas(pd.DataFrame(converted))
    
    print("AI Hub 대화 데이터 전처리 중...")
    train_dataset = convert_dialog_format(train_data)
    valid_dataset = convert_dialog_format(valid_data)
    
    print(f"AI Hub 대화 학습 데이터 크기: {len(train_dataset)}")
    print(f"AI Hub 대화 검증 데이터 크기: {len(valid_dataset)}")
    
    return train_dataset, valid_dataset

def combine_datasets():
    """모든 데이터셋 통합"""
    datasets_train = []
    datasets_valid = []
    
    # NSMC 데이터셋
    nsmc_train, nsmc_valid = prepare_nsmc_dataset()
    datasets_train.append(pd.DataFrame(nsmc_train))
    datasets_valid.append(pd.DataFrame(nsmc_valid))
    
    # KLUE-NLI 데이터셋
    nli_train, nli_valid = prepare_klue_nli()
    datasets_train.append(pd.DataFrame(nli_train))
    datasets_valid.append(pd.DataFrame(nli_valid))
    
    # AI Hub 대화 데이터셋
    dialog_train, dialog_valid = prepare_aihub_dialog()
    if dialog_train is not None and dialog_valid is not None:
        datasets_train.append(pd.DataFrame(dialog_train))
        datasets_valid.append(pd.DataFrame(dialog_valid))
    
    # 데이터셋 통합
    train_dataset = Dataset.from_pandas(pd.concat(datasets_train, ignore_index=True))
    valid_dataset = Dataset.from_pandas(pd.concat(datasets_valid, ignore_index=True))
    
    # 최종 데이터셋을 CSV 파일로 저장
    train_dataset.to_csv("domain_train.csv", index=False)
    valid_dataset.to_csv("domain_valid.csv", index=False)
    
    print("\n최종 데이터셋 통계:")
    print(f"학습 데이터 크기: {len(train_dataset)}")
    print(f"검증 데이터 크기: {len(valid_dataset)}")
    
    return train_dataset, valid_dataset

if __name__ == "__main__":
    print("도메인 특화 데이터셋 준비 시작...")
    train_dataset, valid_dataset = combine_datasets()
    print("데이터셋 준비 완료!") 