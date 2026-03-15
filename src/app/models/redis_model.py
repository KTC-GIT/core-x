from redis.commands.search.field import TextField, TagField, VectorField, NumericField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query
import numpy as np

import time

from redis.exceptions import ResponseError

from ..core.database import rd

INDEX_NAME = "core-x:chat-idx"
PREFIX = "core-x:ChatCache:"

class ChatCache:
    """
    순수 Python 클래스로 데이터 구조만 정의
    """
    def __init__(self, user_id: str, model_name: str, message: str, response: str, embedding: list[float], created_at: float):
        self.user_id = user_id
        self.model_name = model_name
        self.message = message
        self.response = response
        self.embedding = embedding
        self.created_at = created_at

    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "model_name": self.model_name,
            "message": self.message,
            "response": self.response,
            "embedding": np.array(self.embedding, dtype=np.float32).tobytes(),  # 벡터는 bytes로 변환
            "created_at": self.created_at
        }

async def create_redis_index():
    """
    [수동 모드] Redis 명령어로 직접 인덱스를 생성.
    기존 redis 인덱스가 존재하더라도 재기동 시 삭제하고 다시 만들기.
    """
    try:
        # 기존 인덱스 삭제
        # delete_document=False: 인덱스만 지우고, 데이터(JSON)은 남김 (True면 데이터도 삭제)
        await rd.ft(INDEX_NAME).dropindex(delete_documents=True)
        print(f"🗑️ [Redis] 기존 인덱스 삭제 완료.")
    except ResponseError as e:
        # 인덱스가 없어서 삭제 실패한 거면 정상 진행
        if "Unknown Index name" in str(e):
            print("🌟 [Redis] 기존 인덱스 없음. 신규 생성합니다.")
        else:
            print(f"⚠️ [Redis] 인덱스 삭제 중 예기치 못한 에러 발생: {e}")

    try:
        # 스키마 정의 (JSON 타입에 맞춤)
        # JSON Path 문법 ('$.field')을 사용해야 함.
        schema = (
            TagField("$.user_id", as_name="user_id"),
            TagField("$.model_name", as_name="model_name"),
            TextField("$.message", as_name="message"),
            NumericField("$.created_at", as_name="created_at", sortable=True),

            # 벡터 필드 정의 (HNSW 알고리즘 사용)
            VectorField(
                "$.embedding",
                "HNSW",     # FLAT보다 속도/성능 균형이 좋음
                {
                    "TYPE": "FLOAT32",
                    "DIM": 1024,
                    "DISTANCE_METRIC": "COSINE"
                },
                as_name="embedding"
            )
        )

        # 인덱스 정의 (JSON 타입 지정)
        definition = IndexDefinition(prefix=[PREFIX], index_type=IndexType.JSON)

        # 생성 실행
        await rd.ft(INDEX_NAME).create_index(fields=schema, definition=definition)
        print("✅ [Redis] 인덱스 생성 완료 (Manual Mode)")

    except Exception as e:
        print(f"❌ [Redis] 인덱스 생성 실패: {e}")

async def save_chat_to_redis(chat_data: ChatCache):
    """
    데이터 저장 도우미 함수 (JSON.SET 사용)
    """
    # Key 생성 (유니크 하게)
    key = f"{PREFIX}{chat_data.user_id}:{chat_data.created_at}"

    # JSON으로 저장 (embedding은 리스트 형태 그대로 저장해야 RedisJSON이 인식함)
    # 주의: to_dict()에서 bytes로 바꾼 건 검색 쿼리용이고, 저장할 땐 list[float]이어야 함.
    # 따라서 저장용 딕셔너리를 따로 만들기.
    json_data = {
        "user_id": chat_data.user_id,
        "model_name": chat_data.model_name,
        "message": chat_data.message,
        "response": chat_data.response,
        "embedding": chat_data.embedding,
        "created_at": chat_data.created_at
    }

    await rd.json().set(key,"$", json_data)
