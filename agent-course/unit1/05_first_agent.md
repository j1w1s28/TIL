# 첫 번째 에이전트 만들기

## smolagents 라이브러리 활용

### 1. 환경 설정
```python
# 필요한 패키지 설치
!pip install smolagents
!pip install huggingface_hub
!pip install python-dotenv

# 환경 변수 설정
from dotenv import load_dotenv
import os

load_dotenv()

# Hugging Face 토큰 설정
os.environ["HUGGINGFACE_TOKEN"] = "your-token-here"
```

### 2. 기본 구조 설정
```python
from smolagents import Agent, Message, Tool
from typing import List, Dict, Any

class AlfredAgent(Agent):
    def __init__(self):
        super().__init__()
        self.name = "Alfred"
        self.description = "당신의 개인 AI 비서"
        self.tools = self._initialize_tools()
    
    def _initialize_tools(self) -> List[Tool]:
        """도구 초기화"""
        return [
            WeatherTool(),
            ScheduleTool(),
            SearchTool(),
            CalculatorTool()
        ]
```

## Alfred 에이전트 구현

### 1. 도구 구현
```python
class WeatherTool(Tool):
    name = "weather"
    description = "날씨 정보 조회"
    
    def _run(self, city: str) -> Dict:
        # 날씨 API 호출
        return {
            "temperature": 20,
            "condition": "sunny"
        }

class ScheduleTool(Tool):
    name = "schedule"
    description = "일정 관리"
    
    def _run(self, action: str, event: Dict) -> Dict:
        # 일정 관리 로직
        return {
            "status": "success",
            "message": f"일정 {action} 완료"
        }
```

### 2. 메시지 처리
```python
class AlfredAgent(Agent):
    def process_message(self, message: Message) -> Message:
        """메시지 처리"""
        # 의도 분석
        intent = self._analyze_intent(message.content)
        
        # 도구 선택
        tool = self._select_tool(intent)
        
        # 도구 실행
        result = tool.run(intent.parameters)
        
        # 응답 생성
        return self._create_response(result)
    
    def _analyze_intent(self, content: str) -> Dict:
        """의도 분석"""
        return {
            "action": "get_weather",
            "parameters": {"city": "서울"}
        }
    
    def _create_response(self, result: Dict) -> Message:
        """응답 생성"""
        return Message(
            role="assistant",
            content=f"날씨 정보입니다: {result}"
        )
```

### 3. 대화 관리
```python
class Conversation:
    def __init__(self, agent: Agent):
        self.agent = agent
        self.history = []
    
    def add_message(self, message: Message):
        """메시지 추가"""
        self.history.append(message)
    
    def get_response(self, message: Message) -> Message:
        """응답 생성"""
        self.add_message(message)
        response = self.agent.process_message(message)
        self.add_message(response)
        return response
```

## Hugging Face Spaces 배포

### 1. 앱 구성
```python
# app.py
import gradio as gr
from smolagents import Message
from alfred_agent import AlfredAgent, Conversation

# 에이전트 초기화
agent = AlfredAgent()
conversation = Conversation(agent)

def chat(message: str) -> str:
    """채팅 인터페이스"""
    # 메시지 생성
    user_message = Message(
        role="user",
        content=message
    )
    
    # 응답 생성
    response = conversation.get_response(user_message)
    
    return response.content

# Gradio 인터페이스 생성
iface = gr.Interface(
    fn=chat,
    inputs="text",
    outputs="text",
    title="Alfred - Your AI Assistant",
    description="무엇을 도와드릴까요?"
)

# 앱 실행
iface.launch()
```

### 2. 배포 설정
```yaml
# .github/workflows/deploy.yml
name: Deploy to Spaces
on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Deploy to Spaces
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
        run: |
          pip install huggingface_hub
          huggingface-cli login --token $HF_TOKEN
          python deploy.py
```

### 3. 배포 스크립트
```python
# deploy.py
from huggingface_hub import HfApi
import os

def deploy_to_spaces():
    """Hugging Face Spaces 배포"""
    api = HfApi()
    
    # 저장소 생성
    api.create_repo(
        repo_id="your-username/alfred-agent",
        repo_type="space",
        space_sdk="gradio"
    )
    
    # 파일 업로드
    api.upload_file(
        path_or_fileobj="app.py",
        path_in_repo="app.py",
        repo_id="your-username/alfred-agent",
        repo_type="space"
    )
    
    print("배포 완료!")

if __name__ == "__main__":
    deploy_to_spaces()
```

## 테스트 및 모니터링

### 1. 단위 테스트
```python
# test_alfred.py
import pytest
from alfred_agent import AlfredAgent, Message

def test_weather_query():
    agent = AlfredAgent()
    message = Message(
        role="user",
        content="서울의 날씨는 어때?"
    )
    
    response = agent.process_message(message)
    assert response.role == "assistant"
    assert "날씨" in response.content

def test_schedule_management():
    agent = AlfredAgent()
    message = Message(
        role="user",
        content="내일 오후 2시에 회의 일정 추가해줘"
    )
    
    response = agent.process_message(message)
    assert "일정" in response.content
    assert "success" in response.content.lower()
```

### 2. 성능 모니터링
```python
import logging
from datetime import datetime

# 로깅 설정
logging.basicConfig(
    filename=f"logs/alfred_{datetime.now().strftime('%Y%m%d')}.log",
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class MonitoredAlfredAgent(AlfredAgent):
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
    
    def process_message(self, message: Message) -> Message:
        start_time = datetime.now()
        
        try:
            response = super().process_message(message)
            
            # 처리 시간 기록
            processing_time = (datetime.now() - start_time).total_seconds()
            self.logger.info(f"Message processed in {processing_time}s")
            
            return response
        except Exception as e:
            self.logger.error(f"Error processing message: {str(e)}")
            raise
```

### 3. 에러 처리
```python
class ErrorHandler:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def handle_error(self, error: Exception) -> Message:
        """에러 처리 및 사용자 친화적 메시지 생성"""
        self.logger.error(f"Error: {str(error)}")
        
        if isinstance(error, APIError):
            return Message(
                role="assistant",
                content="죄송합니다. 서비스에 일시적인 문제가 있습니다."
            )
        elif isinstance(error, ValidationError):
            return Message(
                role="assistant",
                content="입력 형식이 올바르지 않습니다."
            )
        else:
            return Message(
                role="assistant",
                content="죄송합니다. 요청을 처리하는 중 문제가 발생했습니다."
            )
```

## 다음 단계

1. **기능 확장**
   - 새로운 도구 추가
   - 대화 컨텍스트 개선
   - 자연어 처리 향상

2. **성능 최적화**
   - 응답 시간 개선
   - 메모리 사용량 최적화
   - 캐싱 구현

3. **사용자 경험 개선**
   - 더 자연스러운 대화
   - 오류 메시지 개선
   - 사용자 피드백 수집 