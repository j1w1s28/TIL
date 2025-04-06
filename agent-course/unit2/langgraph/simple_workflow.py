from langgraph.graph import Graph, END
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from typing import Dict, TypedDict

class AgentState(TypedDict):
    messages: list
    next_step: str

def chat_with_user(state: AgentState) -> AgentState:
    messages = state["messages"]
    # 사용자 입력 처리 로직
    return {"messages": messages, "next_step": "process"}

def process_message(state: AgentState) -> AgentState:
    messages = state["messages"]
    # 메시지 처리 로직
    return {"messages": messages, "next_step": "respond"}

def respond_to_user(state: AgentState) -> AgentState:
    messages = state["messages"]
    # 응답 생성 로직
    return {"messages": messages, "next_step": END}

def main():
    # 워크플로우 그래프 생성
    workflow = Graph()
    
    # 노드 추가
    workflow.add_node("chat", chat_with_user)
    workflow.add_node("process", process_message)
    workflow.add_node("respond", respond_to_user)
    
    # 엣지 연결
    workflow.add_edge("chat", "process")
    workflow.add_edge("process", "respond")
    
    # 그래프 컴파일
    app = workflow.compile()
    
    # 실행
    config = {"messages": [], "next_step": "chat"}
    result = app.invoke(config)
    print(result)

if __name__ == "__main__":
    main() 