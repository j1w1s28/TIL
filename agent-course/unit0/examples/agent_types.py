"""
여러 타입의 에이전트 예시 (단순, 모델 기반, 목표 기반, 학습형)
"""

class SimpleReactiveAgent:
    def act(self, observation):
        if observation == "위험":
            return "회피"
        return "대기"

class ModelBasedAgent:
    def __init__(self):
        self.model = {}
    def act(self, observation):
        self.model['last'] = observation
        return f"모델 기반 행동({observation})"

class GoalBasedAgent:
    def __init__(self, goal):
        self.goal = goal
    def act(self, observation):
        if self.goal in observation:
            return "목표 달성!"
        return "계속 탐색"

class LearningAgent:
    def __init__(self):
        self.memory = []
    def act(self, observation):
        self.memory.append(observation)
        return f"학습 중: {observation}"

if __name__ == "__main__":
    agents = [
        SimpleReactiveAgent(),
        ModelBasedAgent(),
        GoalBasedAgent("보물"),
        LearningAgent()
    ]
    obs = "보물 발견"
    for agent in agents:
        print(agent.act(obs)) 