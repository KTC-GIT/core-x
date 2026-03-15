# chat_to_es_service.py
import os

from elasticsearch import Elasticsearch
import datetime

ELASTICSEARCH_URL = os.getenv("ELASTICSEARCH_URL")
# ES 클라이언트
es_client = Elasticsearch(ELASTICSEARCH_URL, request_timeout=5)

def index_chat_to_es(user_id: str, message: str, response: str, model_name: str):
    """
    LLM 대화 종료 직후 ES에 실시간 색인하는 함수
    """
    # 쓰레기 데이터 감지 필터
    error_keywords = ["죄송합니다", "검색결과가 없습니다."]

    if not es_client:
        return

    if any(keyword in response for keyword in error_keywords):
        print("⚠️ [ES] 에러 응답 감지됨. 인덱싱을 건너뜁니다.")
        return

    try:
        doc = {
            "user_id": user_id,
            "message": message,
            "response": response,
            "model_name": model_name,
            "created_at": datetime.datetime.now().isoformat(),
        }

        # ID를 안 주면 ES가 알아서 자동 생성함 (DB PK랑 맞추려면 id=str(db_id) 추가)
        es_client.index(index="chat_history", document=doc)
        print(f"📝 [ES] 대화 실시간 색인 완료 (Model: {model_name})")

    except Exception as e:
        print(f"⚠️ [ES] 대화 색인 실패: {e}")