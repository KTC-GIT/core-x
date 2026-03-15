from sqlmodel import Session, select
from ..models.models import Conversation, ChatCreate
from ..core.database import engine

def search_similar_memory(embedding: list[float], limit: int= 10):
    """
    현재 질문의 임베딩과 가장 유사한 과거 대화 찾기 (RAG 핵심)
    """
    if not embedding:
        return []

    with Session(engine) as session:
        # [핵심] l2_distance로 정렬 (거리가 가까울수록 유사함)
        # Conversation.embedding이 벡터 컬럼
        statement = select(Conversation).order_by(
            Conversation.embedding.cosine_distance(embedding)
        ).limit(limit)

        result = session.exec(statement).all()
        return result

def search_recent_chats(request:ChatCreate,limit:int=5):
    """
    최근 대화내용을 가져옴
    """
    with (Session(engine) as session):
        result = session.exec(
            select(Conversation)
            .where(Conversation.user_id == request.user_id)
            .where(Conversation.response.is_not(None))
            .order_by(Conversation.created_at.desc())
            .limit(limit)
        ).all()

        result = list(reversed(result))

        return result