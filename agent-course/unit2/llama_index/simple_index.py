from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms import HuggingFaceLLM
import os
from dotenv import load_dotenv

def main():
    # 환경 변수 로드
    load_dotenv()
    
    # 문서 로드
    documents = SimpleDirectoryReader('data').load_data()
    
    # LLM 설정
    llm = HuggingFaceLLM(
        model_name="huggingface/CodeLlama-7b-Instruct-hf",
        tokenizer_name="huggingface/CodeLlama-7b-Instruct-hf",
    )
    
    # 인덱스 생성
    index = VectorStoreIndex.from_documents(documents, llm=llm)
    
    # 쿼리 엔진 생성
    query_engine = index.as_query_engine()
    
    # 쿼리 실행
    response = query_engine.query("문서의 주요 내용은 무엇인가요?")
    print(response)

if __name__ == "__main__":
    main() 