# redis_service.py
# Redis 기반 단기 대화 기억 및 유사 질의 캐시 관리
# 현재 최근 대화 조회 용도로 주로 사용 중이며,
# 유사 질의 캐시는 사용자 수/재질문 패턴상 hit 빈도가 낮을 수 있음.

import time
import numpy as np
from redis.commands.search.query import Query
from ..core.database import rd
from ..models.redis_model import INDEX_NAME, save_chat_to_redis, ChatCache

class RedisMemoryManager:
    def __init__(self):
        self.index_name = INDEX_NAME

    async def get_recent_history(self, user_id: str, k: int = 2) -> str:
        """
        [단기 기억] 최근 k개의 대화를 시간순으로 가져와서 프롬프트용 텍스트로 전환
        """
        # 최근 시간순 정렬 쿼리
        query = (
            Query(f"@userid:{{{user_id}}}")
            .sort_by("created_at", asc=False)
            .paging(0,k)
            .return_fields("message","response")
        )

        try:
            res = await rd.ft(self.index_name).search(query)

            history_text = []

            # 최신 순으로 오기 때문에 뒤집어서 과거 -> 현재 순으로 만들어야 맥락이해를 잘함.
            for doc in reversed(res.docs):
                msg = getattr(doc,"message", "")
                resp = getattr(doc,"response", "")
                history_text.append(f"User: {msg}\nAI: {resp}")

            return "\n".join(history_text)

        except Exception as e:
            print(f"⚠️ 기억 조회 실패: {e}")
            return ""


    async def find_similar_cache(self, embedding: list[float], threshold: float = 0.20) -> str | None:
        """
        [캐시 검색] 유사한 질문이 있었는지 확인 (Distance가 threshold보다 작으면 Hit)
        """
        vec_bytes = np.array(embedding, dtype=np.float32).tobytes()

        query = (
            Query("(*)=>[KNN 1 @embedding $vec_param AS score]")
            .sort_by("score")
            .return_fields("message","response","score")
            .dialect(2)
        )

        try:
            res = await rd.ft(self.index_name).search(query, query_params={"vec_param":vec_bytes})

            if res.docs:
                top_doc = res.docs[0]
                score = float(top_doc.score)

                # 점수가 낮을수록 유사함 (0 = 똑같음)
                # 0.15 정도면 "거의 같은 문장"
                if score < threshold:
                    print(f"⚡ [Cache Hit] 유사도: {score:.4f} (기준: {threshold})")
                    return top_doc.response
                else:
                    print(f"💨 [Cache Miss] 유사도: {score:.4f} (기준: {threshold})")

        except Exception as e:
            print(f"⚠️ 캐시 검색 에러: {e}")

        return None

    async def save_interaction(self, user_id: str, message: str, response: str, embedding: list[float], model_name: str):
        """
        [기억 저장] 대화 내용을 Redis에 저장
        """
        chat_data = ChatCache(
            user_id=user_id,
            model_name=model_name,
            message=message,
            response=response,
            embedding=embedding,
            created_at=time.time()
        )

        await save_chat_to_redis(chat_data)