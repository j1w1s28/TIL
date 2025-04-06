# 에이전트 워크플로우

## Think (사고) 단계

Think 단계는 에이전트가 주어진 상황을 분석하고 행동 계획을 수립하는 단계입니다.

### 1. 상황 분석
```python
from smolagents import Agent, Message
from typing import List, Dict

class ThinkingAgent(Agent):
    def analyze_situation(self, messages: List[Message]) -> Dict:
        """현재 상황 분석"""
        # 최근 메시지 추출
        current_message = messages[-1]
        
        # 컨텍스트 분석
        context = self._extract_context(messages)
        
        # 의도 파악
        intent = self._identify_intent(current_message)
        
        return {
            "context": context,
            "intent": intent,
            "constraints": self._identify_constraints(messages)
        }
```

### 2. 계획 수립
```python
from dataclasses import dataclass
from typing import List

@dataclass
class Step:
    action: str
    tool: str
    parameters: Dict[str, Any]

@dataclass
class Plan:
    steps: List[Step]
    fallback: Optional[Step] = None

class PlanningAgent(Agent):
    def create_plan(self, analysis: Dict) -> Plan:
        """행동 계획 수립"""
        steps = []
        
        # 의도에 따른 단계 생성
        for action in self._break_down_intent(analysis["intent"]):
            tool = self._select_tool(action)
            parameters = self._prepare_parameters(action)
            steps.append(Step(action, tool, parameters))
        
        # 대체 계획 수립
        fallback = self._create_fallback_step()
        
        return Plan(steps=steps, fallback=fallback)
```

## Act (행동) 단계

Act 단계는 수립된 계획을 실행하는 단계입니다.

### 1. 도구 실행
```python
class ActionExecutor:
    def __init__(self, tools: Dict[str, Tool]):
        self.tools = tools
    
    def execute_step(self, step: Step) -> Any:
        """단계 실행"""
        if step.tool not in self.tools:
            raise ValueError(f"도구를 찾을 수 없음: {step.tool}")
        
        tool = self.tools[step.tool]
        return tool.run(**step.parameters)
    
    def execute_plan(self, plan: Plan) -> List[Any]:
        """계획 실행"""
        results = []
        for step in plan.steps:
            try:
                result = self.execute_step(step)
                results.append(result)
            except Exception as e:
                if plan.fallback:
                    result = self.execute_step(plan.fallback)
                    results.append(result)
                else:
                    raise e
        return results
```

### 2. 병렬 실행
```python
import asyncio
from typing import List

class AsyncActionExecutor:
    def __init__(self, tools: Dict[str, Tool]):
        self.tools = tools
    
    async def execute_step_async(self, step: Step) -> Any:
        """비동기 단계 실행"""
        tool = self.tools[step.tool]
        return await tool.arun(**step.parameters)
    
    async def execute_plan_async(self, plan: Plan) -> List[Any]:
        """계획 병렬 실행"""
        tasks = [
            self.execute_step_async(step)
            for step in plan.steps
        ]
        return await asyncio.gather(*tasks)
```

## Observe (관찰) 단계

Observe 단계는 실행 결과를 분석하고 다음 행동을 결정하는 단계입니다.

### 1. 결과 분석
```python
class ResultAnalyzer:
    def analyze_results(self, results: List[Any]) -> Dict:
        """실행 결과 분석"""
        return {
            "success": self._check_success(results),
            "errors": self._collect_errors(results),
            "metrics": self._calculate_metrics(results),
            "summary": self._create_summary(results)
        }
    
    def _check_success(self, results: List[Any]) -> bool:
        """성공 여부 확인"""
        return all(
            isinstance(result, dict) and
            not result.get("error")
            for result in results
        )
```

### 2. 피드백 처리
```python
class FeedbackProcessor:
    def process_feedback(
        self,
        results: List[Any],
        analysis: Dict
    ) -> List[Message]:
        """피드백 처리 및 메시지 생성"""
        messages = []
        
        # 성공/실패 메시지
        if analysis["success"]:
            messages.append(self._create_success_message(results))
        else:
            messages.append(self._create_error_message(analysis["errors"]))
        
        # 추가 정보 요청
        if self._needs_clarification(results):
            messages.append(self._create_clarification_request())
        
        return messages
```

## Re-Act 접근 방식

Re-Act는 Reasoning과 Acting을 반복하는 접근 방식입니다.

### 1. 기본 구조
```python
class ReActAgent(Agent):
    def __init__(self):
        self.memory = []
        self.tools = create_tool_set()
        self.executor = ActionExecutor(self.tools)
    
    def react(self, message: Message) -> Message:
        """Re-Act 루프 실행"""
        while True:
            # Think: 상황 분석 및 계획 수립
            analysis = self.analyze_situation(self.memory + [message])
            plan = self.create_plan(analysis)
            
            # Act: 계획 실행
            results = self.executor.execute_plan(plan)
            
            # Observe: 결과 분석
            observation = self.analyze_results(results)
            
            # 메모리 업데이트
            self.update_memory(message, plan, results, observation)
            
            # 종료 조건 확인
            if self.should_stop(observation):
                break
            
            # 다음 단계 준비
            message = self.prepare_next_step(observation)
        
        return self.generate_final_response()
```

### 2. 메모리 관리
```python
class Memory:
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.items = []
    
    def add(self, item: Any):
        """메모리 항목 추가"""
        self.items.append(item)
        if len(self.items) > self.max_size:
            self.items.pop(0)
    
    def get_recent(self, n: int) -> List[Any]:
        """최근 n개 항목 조회"""
        return self.items[-n:]
    
    def clear(self):
        """메모리 초기화"""
        self.items = []
```

### 3. 종료 조건
```python
class StopCondition:
    def __init__(self, max_steps: int = 10):
        self.max_steps = max_steps
        self.step_count = 0
    
    def should_stop(self, observation: Dict) -> bool:
        """종료 여부 확인"""
        self.step_count += 1
        
        # 최대 단계 수 초과
        if self.step_count >= self.max_steps:
            return True
        
        # 목표 달성
        if observation["success"]:
            return True
        
        # 복구 불가능한 오류
        if observation.get("fatal_error"):
            return True
        
        return False
```

## 모범 사례

### 1. 로깅
```python
import logging

class LoggedAgent(Agent):
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def think(self, message: Message):
        self.logger.info(f"Thinking about message: {message.content}")
        result = super().think(message)
        self.logger.info(f"Thought result: {result}")
        return result
```

### 2. 성능 모니터링
```python
from time import time
from typing import Dict

class MonitoredAgent(Agent):
    def __init__(self):
        self.metrics = {
            "think_time": [],
            "act_time": [],
            "observe_time": []
        }
    
    def _measure_time(self, func: Callable, phase: str) -> Any:
        """함수 실행 시간 측정"""
        start_time = time()
        result = func()
        end_time = time()
        
        self.metrics[f"{phase}_time"].append(end_time - start_time)
        return result
```

### 3. 에러 복구
```python
class ResilientAgent(Agent):
    def execute_with_retry(
        self,
        func: Callable,
        max_retries: int = 3,
        delay: float = 1.0
    ) -> Any:
        """재시도 로직"""
        for attempt in range(max_retries):
            try:
                return func()
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                time.sleep(delay * (attempt + 1))
```

## 다음 단계

- 실제 에이전트 구현
- 성능 최적화
- 에러 처리 개선 