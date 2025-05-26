# agent_basic.py
"""
에이전트의 기본 구조 예시 (상태, 행동, 관찰, 도구)
"""

class Agent:
    def __init__(self, name):
        self.name = name
        self.state = {}

    def observe(self, observation):
        print(f"[{self.name}] 관찰: {observation}")
        self.state['last_observation'] = observation

    def decide(self):
        # 간단한 의사결정 예시
        if self.state.get('last_observation') == '문제 발생':
            return '문제 해결'
        return '대기'

    def act(self, action):
        print(f"[{self.name}] 행동: {action}")
        self.state['last_action'] = action

    def use_tool(self, tool, *args, **kwargs):
        print(f"[{self.name}] 도구 사용: {tool.__name__}")
        return tool(*args, **kwargs)

# 도구 예시 함수
def calculator(a, b):
    return a + b

if __name__ == "__main__":
    agent = Agent("SimpleAgent")
    agent.observe("문제 발생")
    action = agent.decide()
    agent.act(action)
    result = agent.use_tool(calculator, 3, 5)
    print(f"도구 결과: {result}") 