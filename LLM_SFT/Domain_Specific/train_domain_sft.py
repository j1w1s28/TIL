import os
import torch
import pandas as pd
from typing import Dict
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model

def load_dataset(file_path: str) -> Dataset:
    """CSV 파일에서 데이터셋을 로드합니다."""
    df = pd.read_csv(file_path)
    return Dataset.from_pandas(df)

def format_instruction_prompt(example: Dict) -> str:
    """지시사항 형식으로 프롬프트를 포맷팅합니다."""
    instruction = example['instruction']
    input_text = example['input']
    output = example['output']
    
    # SFT Trainer 형식에 맞게 포맷팅
    prompt = f"### 지시사항:\n{instruction}\n\n### 입력:\n{input_text}\n\n### 응답:\n{output}"
    return prompt

class InstructionDataset(torch.utils.data.Dataset):
    """SFT 학습을 위한 데이터셋 클래스"""
    def __init__(self, dataset: Dataset, tokenizer, max_length: int = 512):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        prompt = format_instruction_prompt(example)
        
        # 토크나이징 시 응답 부분을 labels로 구분
        encoded = self.tokenizer(
            prompt,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # 응답 시작 위치 찾기
        response_start = prompt.find("### 응답:")
        response_encoded = self.tokenizer(
            prompt[response_start:],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # 응답 부분만 labels로 설정, 나머지는 -100으로 마스킹
        labels = encoded["input_ids"].clone()
        response_length = response_encoded["input_ids"].shape[1]
        non_response_length = labels.shape[1] - response_length
        labels[0, :non_response_length] = -100
        
        return {
            "input_ids": encoded["input_ids"].squeeze(),
            "attention_mask": encoded["attention_mask"].squeeze(),
            "labels": labels.squeeze()
        }

def main():
    # 모델과 토크나이저 설정
    model_name = "beomi/llama-2-ko-7b"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 4비트 양자화 설정
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False
    )
    
    # 모델 로드
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # LoRA 설정
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    
    # 데이터셋 로드
    train_dataset = load_dataset("domain_train.csv")
    valid_dataset = load_dataset("domain_valid.csv")
    
    # 학습 인자 설정
    training_args = TrainingArguments(
        output_dir="domain_sft_model_output",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        evaluation_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=100,
        optim="paged_adamw_32bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        report_to="none"
    )
    
    # SFT Trainer 초기화
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        packing=False,
        max_seq_length=512,
        formatting_func=format_instruction_prompt
    )
    
    # 학습 시작
    trainer.train()
    
    # 모델 저장
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)

if __name__ == "__main__":
    main() 