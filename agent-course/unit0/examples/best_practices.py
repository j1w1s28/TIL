"""
에이전트 개발 모범 사례 예시 (에러 처리, 비용 관리, 성능 최적화)
"""
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.callbacks import get_openai_callback
import os

os.environ["OPENAI_API_KEY"] = "your-api-key"

llm = OpenAI(temperature=0)
template = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?"
)
chain = LLMChain(llm=llm, prompt=template)

# 에러 처리 예시
try:
    response = chain.run("shoes")
    print(f"응답: {response}")
except Exception as e:
    print(f"에러 발생: {e}")

# 비용 관리 예시
with get_openai_callback() as cb:
    result = chain.run("socks")
    print(f"토큰 사용량: {cb.total_tokens}")

# 성능 최적화 예시 (배치 처리)
inputs = [{"product": "socks"}, {"product": "shoes"}]
results = chain.apply(inputs)
print(f"배치 처리 결과: {results}") 