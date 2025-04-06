# Agent Types and Tools

## 에이전트 타입

LangChain에서 제공하는 주요 에이전트 타입들을 살펴보겠습니다.

### 1. Zero-shot ReAct 에이전트
```python
from langchain.agents import initialize_agent, AgentType

# Zero-shot ReAct 에이전트 초기화
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)
```

### 2. Structured Chat 에이전트
```python
# Structured Chat 에이전트 초기화
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)
```

### 3. Conversational 에이전트
```python
from langchain.memory import ConversationBufferMemory

# 대화 메모리 생성
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Conversational 에이전트 초기화
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True
)
```

## 도구(Tools) 종류

### 1. 기본 도구
```python
from langchain.tools import (
    DuckDuckGoSearchTool,
    Calculator,
    WikipediaQueryRun,
    WolframAlphaQueryRun
)

# 검색 도구
search = DuckDuckGoSearchTool()

# 계산기 도구
calculator = Calculator()

# 위키피디아 도구
wikipedia = WikipediaQueryRun()

# Wolfram Alpha 도구
wolfram = WolframAlphaQueryRun()
```

### 2. 커스텀 도구
```python
from langchain.tools import BaseTool

class CustomTool(BaseTool):
    name = "custom_tool"
    description = "커스텀 도구 설명"

    def _run(self, query: str) -> str:
        # 도구 로직 구현
        return f"결과: {query}"

    def _arun(self, query: str) -> str:
        # 비동기 실행 로직
        raise NotImplementedError("비동기 실행은 지원하지 않습니다")
```

## 도구 조합 예시

### 1. 검색 및 계산 도구
```python
tools = [
    Tool(
        name="Search",
        func=search.run,
        description="인터넷 검색이 필요할 때 사용"
    ),
    Tool(
        name="Calculator",
        func=calculator.run,
        description="수학 계산이 필요할 때 사용"
    )
]

agent = initialize_agent(tools, llm, agent="zero-shot-react-description")
```

### 2. 위키피디아 및 Wolfram Alpha 도구
```python
tools = [
    Tool(
        name="Wikipedia",
        func=wikipedia.run,
        description="위키피디아 검색이 필요할 때 사용"
    ),
    Tool(
        name="Wolfram Alpha",
        func=wolfram.run,
        description="과학적 계산이 필요할 때 사용"
    )
]

agent = initialize_agent(tools, llm, agent="zero-shot-react-description")
```

## 도구 선택 가이드

1. **검색 관련**
   - DuckDuckGo: 일반 웹 검색
   - Wikipedia: 지식 검색
   - Wolfram Alpha: 과학적 계산

2. **계산 관련**
   - Calculator: 기본 수학 계산
   - Wolfram Alpha: 복잡한 수학 계산

3. **데이터 처리**
   - CSV Reader: CSV 파일 처리
   - JSON Reader: JSON 파일 처리
   - Text Reader: 텍스트 파일 처리

## 에이전트 타입 선택 가이드

1. **Zero-shot ReAct**
   - 단순한 작업
   - 빠른 응답 필요
   - 기본적인 도구 사용

2. **Structured Chat**
   - 구조화된 대화 필요
   - 복잡한 작업
   - 여러 단계의 상호작용

3. **Conversational**
   - 대화 기록 유지
   - 맥락 기반 응답
   - 장기 상호작용

## 모범 사례

1. **도구 선택**
   - 작업에 맞는 적절한 도구 선택
   - 도구 중복 방지
   - 명확한 설명 제공

2. **에이전트 타입 선택**
   - 작업 복잡도 고려
   - 응답 시간 요구사항
   - 메모리 관리 필요성

3. **성능 최적화**
   - 도구 실행 순서 최적화
   - 캐싱 활용
   - 에러 처리 