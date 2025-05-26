# langchain_basic.py
"""
LangChain 기본 사용법 예시
"""
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
import os

# 환경 변수 설정 (OpenAI API 키 필요)
os.environ["OPENAI_API_KEY"] = "your-api-key"

# LLM 모델 생성
llm = OpenAI(temperature=0.7)

# 프롬프트 템플릿
template = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?"
)

# 체인 생성
chain = LLMChain(llm=llm, prompt=template)
result = chain.run("colorful socks")
print(f"체인 결과: {result}")

# 대화 메모리 예시
memory = ConversationBufferMemory()
chain_with_memory = LLMChain(llm=llm, prompt=template, memory=memory)
chain_with_memory.run("colorful shoes") 