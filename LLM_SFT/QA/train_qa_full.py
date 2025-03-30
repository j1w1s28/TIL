import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
import json
from datasets import Dataset
import os
from typing import Dict, Sequence

def load_dataset(file_path: str) -> Dataset:
    """데이터셋 로드"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return Dataset.from_dict(data)

def format_qa_prompt(example: Dict) -> str:
    """QA 형식으로 프롬프트 포맷팅"""
    return f"""### 질문: {example['question']}
### 컨텍스트: {example['context']}
### 답변: {example['answer']}"""

class QADataset(torch.utils.data.Dataset):
    """QA 데이터셋 클래스"""
    def __init__(self, dataset: Dataset, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        example = self.dataset[idx]
        prompt = format_qa_prompt(example)
        
        # 토크나이징
        encoded = self.tokenizer(
            prompt,
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoded["input_ids"][0],
            "attention_mask": encoded["attention_mask"][0],
            "labels": encoded["input_ids"][0].clone()
        }

def main():
    # 기본 설정
    MODEL_NAME = "beomi/llama-2-ko-7b"  # 또는 다른 한국어 LLM
    OUTPUT_DIR = "./qa_full_model_output"
    TRAIN_FILE = "qa_train.json"
    VALID_FILE = "qa_valid.json"
    
    print("모델과 토크나이저 로드 중...")
    # 모델과 토크나이저 로드
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    print("데이터셋 로드 중...")
    # 데이터셋 로드
    train_dataset = load_dataset(TRAIN_FILE)
    valid_dataset = load_dataset(VALID_FILE)
    
    # 데이터셋 준비
    train_dataset = QADataset(train_dataset, tokenizer)
    valid_dataset = QADataset(valid_dataset, tokenizer)
    
    # 학습 인자 설정
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        evaluation_strategy="steps",
        eval_steps=500,
        save_steps=500,
        warmup_steps=100,
        learning_rate=2e-5,
        fp16=True,
        logging_steps=100,
        save_total_limit=3,
        report_to="tensorboard",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )
    
    # 데이터 콜레이터 설정
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Trainer 초기화
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
    )
    
    print("학습 시작...")
    # 학습 시작
    trainer.train()
    
    print("모델 저장 중...")
    # 모델 저장
    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print("학습 완료!")

if __name__ == "__main__":
    main() 