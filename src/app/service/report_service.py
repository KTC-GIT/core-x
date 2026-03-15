# report_service.py
# 대화 로그 기반 일일 리포트 생성 기능.
# 현재는 자주 호출되지는 않지만,
# 요약이 필요할 때 수동으로 사용할 목적으로 남겨둠.

from sqlmodel import Session , select


from ..core.database import engine
from ..models.models import Conversation, DailyReport

import os
from dotenv import load_dotenv
from datetime import datetime, timedelta


load_dotenv()

MODEL_NAME = os.getenv("REPORT_MODEL_NAME")


async def generate_daily_report(user_id: str, model_strategy:str, target_date: str):
    """오늘 나눈 대화를 정리하는 기능"""
    try:
        start_date = datetime.strptime(target_date, "%Y%m%d")     # 20260109
    except ValueError:
        # 혹시 2026-01-09 형식으로 들어올 것을 대비.
        start_date = datetime.strptime(target_date, "%Y-%m-%d")

    end_date = start_date + timedelta(days=1)                       # 20260110
    # 오늘 나눈 대화 가져오기 (시간순 정렬)
    # 실제로는 DB 쿼리로 '오늘 날짜' 필터링 필요
    with Session(engine) as db:
        today_chats = db.exec(
            select(Conversation)
            .where(Conversation.user_id == user_id)
            .where(Conversation.created_at >= start_date)
            .where(Conversation.created_at < end_date)
            .order_by(Conversation.created_at)
        ).all()

        if not today_chats:
            return None

        # 대화 내용을 하나의 텍스트로 합치기
        full_transcript = "\n".join([f"- User: {c.message}\n- AI: {c.response}" for c in today_chats])

        # 프롬프트
        system_prompt = """
        너는 사용자의 하루 대화 로그를 분석하여, '일일 개발 리포트(Daily Dev Report)'를 작성하라
        제공된 대화 내역을 보고, 정리하여 객관적인 사실로 상세히 정리할 것.
        
        [작성 목표]
        사용자가 나중에 보고 백지 상태에서 그대로 따라할 수 있는 **기술문서(Technical Report)** 내지 **매뉴얼(Manual)** 형태로 요약해.
        
        [중요 조건]
        1. 조언은 붙이지 않고 내용을 정리한다.
        2. 코드 등은 빼지말고 최종적인 결과물을 정리하며, 트러블슈팅 내역은 별도로 적는다.
        3. 대화 내용을 보고 사용자가 **'실제로 수행한 작업'**과 **'단순히 학습/질문한 내용'**을 명확히 구분할 것
        4. AI가 일방적으로 설명한 내용은 "수행함"이라고 적지 말고, **학습함(Studied)** 또는 **조사함(Researched)**으로 기록해.
        
        [출력 필드 가이드]
        1. title: 제목
        2. summary: 오늘 대화의 핵심 주제 1줄 요약.
        3. tags: 다루었던 주요 기술 키워드 3~5개 (예: Kubernetes, Ollama, RAG 등)
        4. category: 다음 중 하나 선택 [Study, Debugging, Feature, Architecture, General]
        5. content: 상세 리포트. (마크다운 포맷 사용. ## Key Topics, ## Issues 등으로 구조화)
        
        [출력 포맷 가이드]
        content 섹션은 아래와 같이 구분해서 작성해 (내용은 보고 그대로 따라할 수 있을 정도로):
        ## 1. 구현 상세 (실제로 수행한 작업)
        - 작업의 흐름과 구체적인 명령어 기록.
        - (없으면 '없음'으로 기록)
        
        ## 2. Trouble Shooting & Study (새롭게 알게 된 기술 / AI의 제안)
        - 발생했던 에러 메시지와 해결방법.
        - AI가 설명해준 주요 개념 정리.
        
        ## 3. Next Step (다음에 해야 할 일)
        - 향후 진행 계획 
        
        [출력 형식 - JSON Only]
        예) {
                "title": "레포트 제목"
                "summary": "핵심요약 1줄",
                "tags": ["K8s", "Model Serving", ... ],
                "category": "...",
                "content": "상세 리포트 내용 (마크다운 형식)",
            }
        """

        user_prompt = f"[오늘의 대화]\n{full_transcript}"

        # 순환참조 방지
        from .llm import CoreXAgent
        agent = CoreXAgent()
        # AI 호출
        response = await agent.grok_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"}     # JSON 강제 (파싱편하게..)
        )

        # 결과 파싱 및 저장
        import json
        result_json = json.loads(response.choices[0].message.content)
        print(result_json)

        # 임베딩 생성 (일기내용으로)
        report_embedding = await agent.get_embedding(result_json['content'],False)

        # 임베딩 생성 (일기 내용으로)
        new_report = DailyReport(
            user_id = user_id,
            date=target_date,
            title=result_json["title"],
            summary=result_json["summary"],
            tags=result_json["tags"],
            category=result_json["category"],
            content=result_json["content"],
            embedding=report_embedding
        )
        db.add(new_report)
        db.commit()
        db.refresh(new_report)

        return new_report
