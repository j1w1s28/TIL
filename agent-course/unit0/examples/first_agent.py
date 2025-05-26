# first_agent.py
"""
LangChain으로 첫 에이전트 구현 예시 (검색 도구 포함)
"""
from langchain.agents import initialize_agent, Tool
from langchain.tools import DuckDuckGoSearchTool
from langchain.llms import OpenAI
import os

os.environ["OPENAI_API_KEY"] = "your-api-key"

# 도구 정의
search = DuckDuckGoSearchTool()
tools = [
    Tool(
        name="Search",
        func=search.run,
        description="인터넷 검색이 필요할 때 사용"
    )
]

llm = OpenAI(temperature=0.7)

# 에이전트 초기화
agent = initialize_agent(tools, llm, agent="zero-shot-react-description")

# 에이전트 실행
result = agent.run("최근 AI 기술 동향에 대해 알려줘")
print(f"에이전트 응답: {result}") 