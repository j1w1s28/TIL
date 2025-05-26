# Unit 0: Onboarding

이 유닛은 Hugging Face Agents Course의 온보딩(입문) 파트입니다.

## 코스 개요
- AI 에이전트 이론, 설계, 실습을 단계별로 학습
- smolagents, LlamaIndex, LangGraph 등 주요 라이브러리 실습
- 실전 과제 및 챌린지, 인증서 발급

## 수강 방법
- 각 유닛의 마크다운 파일을 순서대로 학습
- 실습 코드는 각 섹션의 코드 블록 또는 `examples/` 폴더 참고
- [공식 코스 페이지](https://huggingface.co/learn/agents-course/unit0/introduction)에서 최신 정보 확인

## 인증 과정
- Unit 1 완료 시 Fundamentals 인증
- 전체 과제 및 챌린지 완료 시 Completion 인증 (2025년 7월 1일 마감)

## 커뮤니티
- [Hugging Face Discord](https://discord.gg/huggingface) 참여 권장
- 질문은 디스코드 #agents-course-questions 채널 이용

## 기여/버그 리포트
- 개선 제안, 버그 리포트는 PR 또는 이슈로 환영

---

## 학습 내용

1. [Introduction to Agents](./01_introduction.md)
   - AI 에이전트의 정의와 개념
   - 에이전트의 기본 구성 요소
   - 에이전트의 작동 방식

2. [Getting Started with LangChain](./02_langchain.md)
   - LangChain 프레임워크 소개
   - 기본 컴포넌트 이해
   - 환경 설정 및 설치

3. [First Agent Implementation](./03_first_agent.md)
   - 간단한 에이전트 구현
   - 기본 도구 사용
   - 에이전트 실행 및 테스트

4. [Agent Types and Tools](./04_agent_types.md)
   - 다양한 에이전트 타입
   - 도구(Tools)의 종류와 사용법
   - 에이전트 선택 가이드

5. [Best Practices](./05_best_practices.md)
   - 에이전트 개발 모범 사례
   - 성능 최적화
   - 보안 고려사항

## 실습 환경

이 유닛의 실습을 위해서는 다음 패키지들이 필요합니다:

```bash
pip install langchain
pip install openai
pip install python-dotenv
```

## 시작하기

각 섹션의 마크다운 파일을 순서대로 읽으면서 학습을 진행하시면 됩니다.

실습 코드는 `examples/` 디렉토리에서 확인할 수 있습니다. 