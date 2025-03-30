import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset, Dataset
import pandas as pd
import os

def setup_model_and_tokenizer(model_name, quantization_config):
    """모델과 토크나이저 설정"""
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def prepare_custom_dataset(data_path):
    """사용자 정의 데이터셋 준비"""
    # CSV 파일에서 데이터 로드 (예시)
    # CSV 형식: instruction,input,output
    df = pd.read_csv(data_path)
    
    def format_instruction(row):
        instruction = row['instruction']
        input_text = row['input'] if pd.notna(row['input']) else ""
        output = row['output']
        
        if input_text:
            return f"### 지시사항: {instruction}\n### 입력: {input_text}\n### 출력: {output}"
        else:
            return f"### 지시사항: {instruction}\n### 출력: {output}"
    
    df['text'] = df.apply(format_instruction, axis=1)
    dataset = Dataset.from_pandas(df[['text']])
    return dataset

def main():
    # 기본 설정
    MODEL_NAME = "beomi/llama-2-ko-7b"  # 또는 다른 한국어 LLM
    OUTPUT_DIR = "./domain_model_output"
    DATA_PATH = "your_domain_data.csv"  # 도메인 특화 데이터 경로
    
    # 4비트 양자화 설정
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    
    # LoRA 설정
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"]
    )
    
    # 모델과 토크나이저 설정
    model, tokenizer = setup_model_and_tokenizer(MODEL_NAME, quantization_config)
    model = get_peft_model(model, lora_config)
    
    # 데이터셋 준비
    dataset = prepare_custom_dataset(DATA_PATH)
    
    # 토크나이징 함수
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
    
    # 데이터셋 토크나이징
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    # 학습 인자 설정
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,
        fp16=True,
    )
    
    # 학습 시작
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=lambda data: {'input_ids': torch.stack([x['input_ids'] for x in data]),
                                  'attention_mask': torch.stack([x['attention_mask'] for x in data])},
    )
    
    trainer.train()
    
    # 모델 저장
    trainer.save_model()

if __name__ == "__main__":
    main() 