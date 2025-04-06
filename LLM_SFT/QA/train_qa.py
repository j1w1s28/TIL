import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import os

# 프로젝트 루트 디렉토리 설정
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_CACHE_DIR = os.path.join(ROOT_DIR, "model")
DATASET_DIR = os.path.join(ROOT_DIR, "dataset")

def setup_model_and_tokenizer(model_name, quantization_config):
    """모델과 토크나이저 설정"""
    cache_dir = os.path.join(MODEL_CACHE_DIR, "pretrained")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        cache_dir=cache_dir
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir
    )
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def prepare_korquad_dataset():
    """KorQuAD 데이터셋 준비"""
    cache_dir = os.path.join(DATASET_DIR, "korquad")
    dataset = load_dataset("squad_kor_v1", cache_dir=cache_dir)
    
    def format_qa(example):
        return {
            "text": f"질문: {example['question']}\n컨텍스트: {example['context']}\n답변: {example['answers']['text'][0]}"
        }
    
    return dataset.map(format_qa)

def main():
    # 기본 설정
    MODEL_NAME = "beomi/llama-2-ko-7b"  # 또는 다른 한국어 LLM
    OUTPUT_DIR = os.path.join(MODEL_CACHE_DIR, "qa")
    
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
    dataset = prepare_korquad_dataset()
    
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
    )
    
    # 학습 시작
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        data_collator=lambda data: {'input_ids': torch.stack([x['input_ids'] for x in data]),
                                  'attention_mask': torch.stack([x['attention_mask'] for x in data])},
    )
    
    trainer.train()
    
    # 모델 저장
    trainer.save_model()

if __name__ == "__main__":
    main() 