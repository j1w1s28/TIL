from smolagents import Agent
from transformers import pipeline

def main():
    # 기본 파이프라인 설정
    pipe = pipeline("text-generation", model="huggingface/CodeLlama-7b-Instruct-hf")
    
    # 에이전트 생성
    agent = Agent(
        llm=pipe,
        system_prompt="당신은 도움이 되는 AI 어시스턴트입니다."
    )
    
    # 에이전트와 대화
    response = agent.chat("안녕하세요! 오늘 날씨에 대해 이야기해볼까요?")
    print(response)

if __name__ == "__main__":
    main() 