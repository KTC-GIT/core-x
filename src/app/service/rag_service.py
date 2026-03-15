# rag_service.py
# 초기 RAG 프로토타입에서 사용하던 벡터유사도 검색 코드
# 현재 일부 경로에서만 사용 중.
# 추후 확인 후 제거

import numpy as np
from sqlmodel import Session, select

from ..core.database import engine
from ..models.models import DailyReport


def cosine_similarity(vec_a, vec_b):
    """두 벡터 간 코사인 유사도 계산 (1에 가까울수록 유사)"""
    a = np.array(vec_a)
    b = np.array(vec_b)

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(np.dot(a, b) / (norm_a * norm_b))


async def retrieve_relevant_reports(
    user_query: str,
    agent,
    user_id: str = "default_user",
    top_k: int = 3
):
    """사용자 질문과 가장 관련된 과거 report Top-K 조회"""
    query_embedding = await agent.get_embedding(user_query, is_query=False)

    with Session(engine) as db:
        reports = db.exec(
            select(DailyReport)
            .where(DailyReport.user_id == user_id)
            .where(DailyReport.embedding.is_not(None))
        ).all()

    if not reports:
        return []

    scored_reports = []
    for report in reports:
        if report.embedding:
            score = cosine_similarity(query_embedding, report.embedding)
            scored_reports.append((score, report))

    scored_reports.sort(key=lambda x: x[0], reverse=True)
    return [report for _, report in scored_reports[:top_k]]
