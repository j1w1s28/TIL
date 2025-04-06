# Unit 1: Introduction to Agents

이 유닛에서는 AI 에이전트의 기본 개념과 구현 방법에 대해 학습합니다.

## 학습 목표

- AI 에이전트의 기본 개념 이해
- LLM(Large Language Models)의 역할 파악
- 도구와 액션의 활용 방법 학습
- 에이전트 워크플로우 이해 (Think → Act → Observe)
- 첫 번째 에이전트 구현 및 배포

## 목차

1. [에이전트란 무엇인가?](./01_what_is_agent.md)
   - 에이전트의 정의
   - 에이전트의 동작 방식
   - 의사결정과 계획 수립

2. [LLM과 메시지](./02_llm_and_messages.md)
   - LLM의 역할
   - 메시지 시스템
   - 특수 토큰의 활용

3. [도구와 액션](./03_tools_and_actions.md)
   - 도구의 개념
   - 도구 구현 방법
   - 도구 통합하기

4. [에이전트 워크플로우](./04_agent_workflow.md)
   - Think (사고) 단계
   - Act (행동) 단계
   - Observe (관찰) 단계
   - Re-Act 접근 방식

5. [첫 번째 에이전트 만들기](./05_first_agent.md)
   - smolagents 라이브러리 활용
   - Alfred 에이전트 구현
   - Hugging Face Spaces 배포

## 실습 환경

```bash
pip install smolagents
pip install langchain
pip install huggingface_hub
```

## 평가

이 유닛을 완료하면 "Certificate of Fundamentals of Agents" 인증서를 획득할 수 있습니다.

## 참고 자료

- [Hugging Face Agents Course](https://huggingface.co/learn/agents-course/unit1/introduction)
- [smolagents 문서](https://huggingface.co/docs/smolagents) 