# LLM과 메시지

## LLM의 역할

LLM(Large Language Model)은 에이전트 시스템에서 다음과 같은 핵심 역할을 수행합니다:

1. **자연어 이해**
   - 사용자 입력 해석
   - 컨텍스트 파악
   - 의도 분석

2. **의사결정**
   - 도구 선택
   - 행동 계획 수립
   - 우선순위 결정

3. **응답 생성**
   - 자연어 응답 생성
   - 결과 요약
   - 설명 제공

## 메시지 시스템

### 1. 메시지 구조
```python
from smolagents import Message

# 기본 메시지 구조
message = Message(
    role="user",           # 메시지 작성자 역할
    content="안녕하세요",   # 메시지 내용
    name=None,            # 선택적 이름
    function_call=None    # 함수 호출 정보
)
```

### 2. 메시지 역할
```python
# 시스템 메시지
system_message = Message(
    role="system",
    content="당신은 도움이 되는 AI 비서입니다."
)

# 사용자 메시지
user_message = Message(
    role="user",
    content="오늘 날씨는 어때?"
)

# 어시스턴트 메시지
assistant_message = Message(
    role="assistant",
    content="서울의 현재 날씨를 확인해보겠습니다."
)

# 함수 메시지
function_message = Message(
    role="function",
    name="get_weather",
    content="{'temperature': 20, 'condition': 'sunny'}"
)
```

### 3. 대화 기록 관리
```python
class Conversation:
    def __init__(self):
        self.messages = []
    
    def add_message(self, message: Message):
        self.messages.append(message)
    
    def get_context(self) -> list:
        return self.messages[-5:]  # 최근 5개 메시지만 유지
```

## 특수 토큰의 활용

### 1. 시스템 프롬프트
```python
SYSTEM_PROMPT = """당신은 도움이 되는 AI 비서입니다.
다음 규칙을 따라주세요:
1. 명확하고 간단하게 응답하기
2. 필요한 경우에만 도구 사용하기
3. 안전하고 윤리적인 응답하기
"""

system_message = Message(
    role="system",
    content=SYSTEM_PROMPT
)
```

### 2. 함수 호출
```python
# 함수 정의
function_spec = {
    "name": "get_weather",
    "description": "특정 도시의 현재 날씨 정보를 조회합니다.",
    "parameters": {
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "날씨를 조회할 도시 이름"
            }
        },
        "required": ["city"]
    }
}

# 함수 호출 메시지
function_call_message = Message(
    role="assistant",
    content=None,
    function_call={
        "name": "get_weather",
        "arguments": '{"city": "서울"}'
    }
)
```

### 3. 컨텍스트 관리
```python
class ContextManager:
    def __init__(self, max_tokens=4000):
        self.max_tokens = max_tokens
        self.messages = []
    
    def add_message(self, message: Message):
        self.messages.append(message)
        self._trim_context()
    
    def _trim_context(self):
        """토큰 제한을 초과하지 않도록 컨텍스트 관리"""
        while self._count_tokens() > self.max_tokens:
            # 시스템 메시지는 유지
            for i, msg in enumerate(self.messages):
                if msg.role != "system":
                    self.messages.pop(i)
                    break
```

## 모범 사례

### 1. 메시지 구조화
```python
def create_structured_message(
    role: str,
    content: str,
    metadata: dict = None
) -> Message:
    """구조화된 메시지 생성"""
    return Message(
        role=role,
        content=content,
        name=metadata.get("name") if metadata else None,
        function_call=metadata.get("function_call") if metadata else None
    )
```

### 2. 에러 처리
```python
def safe_message_handling(message: Message) -> Message:
    """안전한 메시지 처리"""
    try:
        # 메시지 유효성 검사
        if not message.content and not message.function_call:
            raise ValueError("메시지 내용이 비어있습니다.")
        
        # 특수 문자 이스케이프
        if message.content:
            message.content = escape_special_chars(message.content)
        
        return message
    except Exception as e:
        logger.error(f"메시지 처리 중 오류 발생: {e}")
        return Message(
            role="system",
            content="메시지 처리 중 오류가 발생했습니다."
        )
```

### 3. 메시지 최적화
```python
def optimize_message_chain(messages: list) -> list:
    """메시지 체인 최적화"""
    # 중복 제거
    seen = set()
    unique_messages = []
    for msg in messages:
        key = (msg.role, msg.content)
        if key not in seen:
            seen.add(key)
            unique_messages.append(msg)
    
    # 컨텍스트 유지를 위한 메시지 선택
    return select_relevant_messages(unique_messages)
```

## 다음 단계

- 도구와 액션의 구현
- 에이전트 워크플로우 이해
- 실제 에이전트 개발 