# 도구와 액션

## 도구의 개념

도구(Tool)는 에이전트가 특정 작업을 수행하기 위해 사용하는 기능 단위입니다. 도구는 다음과 같은 특징을 가집니다:

1. **목적성**: 특정 작업 수행을 위한 명확한 목적
2. **재사용성**: 여러 에이전트에서 공유 가능
3. **인터페이스**: 표준화된 입출력 인터페이스
4. **독립성**: 다른 도구와 독립적으로 동작

## 도구 구현 방법

### 1. 기본 도구 구조
```python
from smolagents import BaseTool
from typing import Optional

class WeatherTool(BaseTool):
    name = "weather"
    description = "특정 도시의 현재 날씨 정보를 조회합니다."
    
    def _run(self, city: str) -> dict:
        """도구 실행 로직"""
        # API 호출 등 실제 구현
        return {
            "temperature": 20,
            "condition": "sunny"
        }
    
    def _arun(self, city: str) -> dict:
        """비동기 실행 로직"""
        raise NotImplementedError("비동기 실행은 지원하지 않습니다")
```

### 2. 도구 매개변수 정의
```python
from pydantic import BaseModel, Field

class WeatherInput(BaseModel):
    city: str = Field(..., description="날씨를 조회할 도시 이름")
    units: str = Field(default="celsius", description="온도 단위")

class WeatherTool(BaseTool):
    name = "weather"
    description = "날씨 정보 조회"
    args_schema = WeatherInput
    
    def _run(self, city: str, units: str = "celsius") -> dict:
        # 구현 로직
        pass
```

### 3. 에러 처리
```python
class ToolError(Exception):
    """도구 실행 중 발생하는 에러"""
    pass

class WeatherTool(BaseTool):
    def _run(self, city: str) -> dict:
        try:
            # API 호출
            response = self._call_weather_api(city)
            
            # 응답 검증
            if not response:
                raise ToolError("날씨 정보를 가져올 수 없습니다")
            
            return response
        except Exception as e:
            raise ToolError(f"날씨 조회 중 오류 발생: {str(e)}")
```

## 도구 통합하기

### 1. 도구 모음 생성
```python
from typing import List
from smolagents import Tool

def create_tool_set() -> List[Tool]:
    """기본 도구 모음 생성"""
    return [
        WeatherTool(),
        CalculatorTool(),
        SearchTool(),
        TimeTool()
    ]
```

### 2. 도구 선택 로직
```python
class ToolSelector:
    def __init__(self, tools: List[Tool]):
        self.tools = tools
    
    def select_tool(self, task_description: str) -> Optional[Tool]:
        """작업에 적합한 도구 선택"""
        # 도구 설명과 작업 설명 간의 유사도 계산
        similarities = [
            (tool, self._calculate_similarity(task_description, tool.description))
            for tool in self.tools
        ]
        
        # 가장 적합한 도구 선택
        if similarities:
            best_tool, score = max(similarities, key=lambda x: x[1])
            if score > 0.5:  # 임계값
                return best_tool
        
        return None
```

### 3. 도구 체인
```python
class ToolChain:
    def __init__(self, tools: List[Tool]):
        self.tools = tools
    
    def execute_chain(self, inputs: dict) -> dict:
        """여러 도구를 순차적으로 실행"""
        results = {}
        for tool in self.tools:
            try:
                # 이전 결과를 다음 도구의 입력으로 사용
                tool_input = self._prepare_input(tool, inputs, results)
                result = tool.run(tool_input)
                results[tool.name] = result
            except Exception as e:
                results[tool.name] = {"error": str(e)}
        
        return results
```

## 액션 구현

### 1. 기본 액션 구조
```python
from dataclasses import dataclass
from typing import Any, Dict

@dataclass
class Action:
    tool_name: str
    parameters: Dict[str, Any]
    
    def execute(self, tools: Dict[str, Tool]) -> Any:
        """액션 실행"""
        if self.tool_name not in tools:
            raise ValueError(f"도구를 찾을 수 없음: {self.tool_name}")
        
        tool = tools[self.tool_name]
        return tool.run(**self.parameters)
```

### 2. 액션 시퀀스
```python
class ActionSequence:
    def __init__(self, actions: List[Action]):
        self.actions = actions
    
    def execute(self, tools: Dict[str, Tool]) -> List[Any]:
        """액션 시퀀스 실행"""
        results = []
        for action in self.actions:
            result = action.execute(tools)
            results.append(result)
        return results
```

### 3. 조건부 액션
```python
@dataclass
class ConditionalAction:
    condition: Callable[[Dict[str, Any]], bool]
    action: Action
    else_action: Optional[Action] = None
    
    def execute(self, tools: Dict[str, Tool], context: Dict[str, Any]) -> Any:
        """조건에 따른 액션 실행"""
        if self.condition(context):
            return self.action.execute(tools)
        elif self.else_action:
            return self.else_action.execute(tools)
        return None
```

## 모범 사례

### 1. 도구 문서화
```python
class WeatherTool(BaseTool):
    """날씨 정보를 조회하는 도구
    
    이 도구는 OpenWeatherMap API를 사용하여 특정 도시의
    현재 날씨 정보를 조회합니다.
    
    Attributes:
        name: 도구 이름
        description: 도구 설명
        api_key: OpenWeatherMap API 키
    
    Example:
        >>> tool = WeatherTool()
        >>> result = tool.run("서울")
        >>> print(result)
        {'temperature': 20, 'condition': 'sunny'}
    """
    
    def _run(self, city: str) -> dict:
        """도구 실행
        
        Args:
            city: 날씨를 조회할 도시 이름
        
        Returns:
            dict: 날씨 정보를 담은 딕셔너리
        
        Raises:
            ToolError: API 호출 실패 시
        """
        pass
```

### 2. 도구 테스트
```python
import pytest

def test_weather_tool():
    tool = WeatherTool()
    
    # 기본 기능 테스트
    result = tool.run("서울")
    assert "temperature" in result
    assert "condition" in result
    
    # 에러 처리 테스트
    with pytest.raises(ToolError):
        tool.run("")  # 빈 도시 이름
    
    # 경계값 테스트
    result = tool.run("뉴욕")
    assert isinstance(result["temperature"], (int, float))
```

### 3. 성능 최적화
```python
from functools import lru_cache
from typing import Dict

class CachedWeatherTool(BaseTool):
    @lru_cache(maxsize=100)
    def _get_weather(self, city: str) -> Dict:
        """날씨 정보 캐싱"""
        return self._call_weather_api(city)
    
    def _run(self, city: str) -> Dict:
        return self._get_weather(city)
```

## 다음 단계

- 에이전트 워크플로우 구현
- 실제 에이전트 개발
- 도구와 액션의 최적화 