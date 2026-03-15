# tools.py
import asyncio

from ddgs import DDGS
from sqlalchemy import text
from sqlmodel import Session
from kiwipiepy import Kiwi
from elasticsearch import Elasticsearch
from opentelemetry import trace
from openinference.instrumentation import using_attributes

import pandas as pd
import numpy as np
import json
import requests
import os
import redis


# Kiwi는 로딩시간이 좀 걸리기 때문에 전역변수로 한번 띄우기
kiwi = Kiwi()

RERANK_URL = os.getenv("RERANK_URL", "")

SEARXNG_URL = os.getenv("SEARXNG_URL", "")
FIRECRAWL_URL = os.getenv("FIRECRAWL_URL", "")

REDIS_HOST = os.getenv("REDIS_HOST", "redis-service")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")

ES_HOST = os.getenv("ES_HOST")

tracer = trace.get_tracer(__name__)

try:
    es_client = Elasticsearch(ES_HOST, request_timeout=30)
    if es_client.ping():
        print(f"✅ [ES] 연결 성공: {ES_HOST}")
    else:
        print(f"⚠️ [ES] 핑 실패: {ES_HOST}")
except Exception as e:
    print(f"⚠️ [ES] 연결 설정 에러: {e}")
    es_client = None

try:
    rd = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASSWORD ,db=0, decode_responses=True)
    # 연결 테스트 (0.1초 안에 응답이 안 오면 에러로 간주)
    rd.ping()
    print(f"✅ [Redis] 연결 성공 ({REDIS_HOST}:{REDIS_PORT})")
except Exception as e:
    print(f"⚠️ [Redis] 연결 실패: {e}")
    rd = None


def index_to_es(url, title, content, summary=""):
    """
    ES에 데이터 색인 (ID는 URL로 지정해서 중복방지)
    """
    if not es_client:
        return

    try:
        doc = {
            "url": url,
            "title": title,
            "content": content,
            "summary": summary,
            "created_at": "now"     # ES가 알아서 시간 변환
        }

        # ID를 URL로 주면, 같은 URL 크롤링 시 덮어쓰기(Update) 효과
        res = es_client.index(index="web_knowledge", id=url, document=doc)
        print(f"🚀 [ES] 인덱싱 완료: {res['result']} (Title: {title[:10]}...)")

    except Exception as e:
        print(f"⚠️ [ES] 인덱싱 실패: {e}")

@tracer.start_as_current_span("save_web_knowledge")
def save_web_knowledge(url:str, title:str, raw_content:str):
    """
    [지식축적] Firecrawl이 긁어온 데이터를 DB에 저장 (UPSERT)
    이미 있는 URL이면 내용을 최신으로 업데이트 합니다.
    """
    if not raw_content or len(raw_content) < 50:
        return  # 내용이 너무 부실하면 저장 안함.

    try:
        from .database import engine

        # SQL 작성: URL이 겹치면 내용과 날짜를 업데이트 (On Conflict)
        query = text("""
            INSERT INTO web_knowledge (url, title, content, updated_at)
            VALUES (:url, :title, :content, NOW())
            ON CONFLICT (url)
            DO UPDATE SET
                title = EXCLUDED.title,
                content = EXCLUDED.content,
                updated_at = NOW();
        """)

        with engine.begin() as conn:
            conn.execute(query, {
                "url": url,
                "title": title,
                "content": raw_content
            })
            # 로그에 저장 성공 메시지 출력
            print(f"💾 [Knowledge] 지식 저장 완료 (Length: {len(raw_content)})")

    except Exception as e:
        print(f"⚠️ 지식 저장 실패: {e}")

@tracer.start_as_current_span("search_web")
def search_web(query: str) -> str:
    """
    1. SearXNG로 정보를 찾고, FireCrawl로 본문을 정밀 타격
    2. DB에 검색결과 저장
    3. 요약.

    Redis 캐싱 적용
    """
    # 캐시 키 생성 (구분하기 쉽도록 prefix 붙임)
    cache_key = f"search_v1:{query.strip()}"

    # [Redis] 캐시 확인
    if rd:
        try:
            cached_data = rd.get(cache_key)
            if cached_data:
                print(f"⚡ [Redis] 캐시 히트! (검색 생략): {query}")
                return cached_data
        except Exception as e:
            print(f"⚠️ Redis 조회 에러: {e}")

    print(f"🔎 [Tool] 웹 검색 중: {query}")

    try:
        # SearXNG 검색 (JSON 요청)
        resp = requests.get(
            f"{SEARXNG_URL}/search",
            params={
                "q": query,
                "format": "json",
                "categories": "general",
                "language": "ko-KR"
            },
            timeout=5
        )

        if resp.status_code != 200:
            return f"⚠️ 검색 엔진 연결 실패 ({resp.status_code})"

        # 검색 결과 상위 2개만 추출 (속도 고려)
        results = resp.json().get('results', [])[:2]
        if not results:
            return "검색 결과가 없습니다."

        final_summary = []

        for idx, res in enumerate(results):
            url = res.get('url')
            title = res.get('title')
            print(f"    🚀 Firecrawler 출동 ({idx+1}/2): {title[:15]}...")

            # Firecrawler로 본문 긁어오기
            try:
                crawl_resp = requests.post(
                    FIRECRAWL_URL,
                    json={
                        "url": url,
                        "pageOptions": {"onlyMainContent": True},   # 광고/메뉴 제거
                    },
                    timeout=20  # 브라우저 렌더링 시간 고려
                )

                if crawl_resp.status_code == 200:
                    data = crawl_resp.json()
                    # 마크다운 확보
                    content = data.get('data',{}).get('markdown', '')

                    if not content:
                        content = "(본문 추출 실패, 요약 사용)"

                    # 긁어온 데이터를 DB에 저장
                    save_web_knowledge(url, title, content)

                    # ES에 넣기
                    index_to_es(url,title, content)

                    # 너무 길면 자르기 (Context 보호)
                    if len(content) > 2000:
                        content = content[:2000] + "\n... (내용이 길어 중략됨)"

                    final_summary.append(f"### {idx+1}. {title}\n🔗 Link: {url}\n📝 Content:\n{content}\n")

                else:
                    # 실패 시 SearXNG가 준 기본 요약(snippet) 사용
                    final_summary.append(f"### {idx+1}. {title}\n🔗 Link: {url}\n(Firecrawl 접근 불가)\n{res.get('content')}\n")

            except Exception as e:
                print(f"    ❌ Crawl Error: {e}")
                final_summary.append(f"### {idx+1}. {title}\n🔗 Link: {url}\n(크롤링 에러)\n{res.get('content')}\n")

        # [Redis] 결과 저장 (TTL: 24시간 = 86400초)
        if rd and final_summary:
            try:
                rd.setex(cache_key, 86400, "\n".join(final_summary))
                print(f"💾 [Redis] 캐시 저장 완료 (TTL: 24h)")
            except Exception as e:
                print(f"⚠️ Redis 저장 에러: {e}")

        return "\n".join(final_summary)


    except Exception as e:
        return f"통신 에러 발생: {str(e)}"



def save_daily_report(title: str, content: str, summary: str, tags: str, category: str = "General"):
    """
    AI가 요약한 내용을 dailyreport 테이블에 저장합니다.
    tags "['tag1','tag2']" 형태의 문자열로 받아서 JSON으로 변환합니다.
    """
    from .database import engine
    try:
        # 태그 문자열을 리스트로 변환 (안전장치)
        if isinstance(tags, str):
            # 대괄호가 있다면 리스트로 변환 시도, 없으면 콤마로 분리
            if "[" in tags:
                import ast
                try:
                    tags_list = ast.literal_eval(tags)
                except:
                    tags_list = [tags]  # 에러나면 그냥 통으로 넣기
            else:
                tags_list = [t.strip() for t in tags.split(",")]
        else:
            tags_list = tags

        # Json 직렬화
        tags_json = json.dumps(tags_list, ensure_ascii=False)

        # 쿼리 실행
        query = text("""
            INSERT INTO dailyreport (user_id, title, content, summary, tags, category, date, created_at)
                VALUES (:user_id, :title, :content, :summary, :tags, :category, TO_CHAR(NOW(), 'YYYYMMDD'), NOW())
        """)

        with engine.begin() as conn:
            conn.execute(query,{
                "user_id": "ktc",
                "title": title,
                "content": content,
                "summary": summary,
                "tags": tags_json,
                "category": category,
            })

        return f"✅ 리포트 저장 완료! (제목: {title}"

    except Exception as e:
        return f"❌ 저장 실패: {str(e)}"

async def search_from_es(query: str):
    """
    Elasticsearch를 이용한 키워드(N-gram) 기반 검색
    """
    if not es_client:
        return []

    target_indices = "chat_history,web-knowledge"

    body = {
        "query": {
            "multi_match": {
                "query": query,
                # 두 인덱스에 있는 모든 필드 때려넣기.
                # (없는 필드는 ES가 알아서 스킵함)
                "fields": ["message^2","response","title","content"]
            }
        },
        "size": 10,
    }

    try:
        res = es_client.search(index=target_indices, body=body)
        results = []
        for hit in res['hits']['hits']:
            source = hit["_source"]
            index_name = hit["_index"]
            doc_id = hit["_id"]

            # Reranker가 읽기 좋게 하나의 'text' 포맷으로 통일.
            if index_name == "chat_history":
                content_text = f"user: {source.get('message', '')}\nAI: {source.get('response', '')}"
            else:
                content_text = f"Title: {source.get('title', '')}\nContent: {source.get('content', '')}"

            results.append({
                "id": f"{index_name}_{doc_id}",     # ID 중복 방지 (chat_123, web_456)
                "text": content_text,
                "created_at": source.get("created_at", "")
            })
        return results
    except Exception as e:
        print(f"⚠️ [ES] 검색 실패: {e}")
        return []



async def get_pg_vector_candidates(query_vector: str):
    sql = text(f"""
        SELECT id, message, response, created_at
        FROM conversation
        ORDER BY embedding <=> :vector_str
        LIMIT 10
    """)

    results = []
    try:
        from .database import engine
        session = Session(engine)
        rows = session.exec(statement=sql,
                            params={"vector_str": query_vector}
                            ).fetchall()

        for row in rows:
            context_text = f"user: {row.message}\nAI: {row.response}"
            results.append({
                "id": row.id,
                "text": context_text,
                "created_at": row.created_at
            })
        return results
    except Exception as e:
        print(f"❌ postgresql vector 검색 에러: {str(e)}")
        return []

@tracer.start_as_current_span("search_memory")
async def search_memory(agent, original_query: str, queries: list[str], limit: int = 30):
    """
    멀티 쿼리를 병렬 검색함.
    """
    print(f"🚀 [Multi-Query] {len(queries)}개의 쿼리를 병렬처리 중: {queries}")

    # 병렬 실행을 위한 Task 생성
    # 이전 로직을 단일 검색로직으로 뺐다.
    tasks = [_search_single_query(agent,q) for q in queries]

    # asyncio.gather로 동시에 실행
    result_docs_nested = await asyncio.gather(*tasks)

    # 이중 리스트 평탄화 (Flatten)
    # [[A,B], [C,D]] -> [A,B,C,D]
    all_docs = [doc for sublist in result_docs_nested for doc in sublist]

    reranker_query = queries[0] if queries else original_query

    print(f"📝 [Reranker] 질문 교체: '{original_query}' -> '{reranker_query}'")

    # 최종 Reranking
    return await _rerank_docs(original_query=reranker_query,result_docs=all_docs)

async def _search_single_query(agent, query_text: str):
    """
    [Hybrid Search]
    1. 의미 기반 검색(Vector): "벡터 망함" -> "Deployment Failed" 찾음
    2. 키워드 검색 (Keyword): "NullPointerException" -> 정확한 에러 로그 찾음
    3. 결과 통합 (Ranking): 중복 제거 후 반환

    multi-query로 들어온 것을 하나씩 검색해서 반환함.
    """

    # 임베딩 생성 (Agent의 기능을 빌려씀)
    query_vector = await agent.get_embedding(query_text, is_query=True)
    vector_str = str(query_vector)  # pgvector는 문자열로 쿼리에 넣어야함.

    try:
        pg_docs, es_docs = await asyncio.gather(
            get_pg_vector_candidates(vector_str),
            search_from_es(query_text)
        )

        return pg_docs + es_docs

    except Exception as e:
        print(f"Error: {e}")
        return f"Search Error: {str(e)}"

async def _rerank_docs(original_query:str, result_docs:list):
    """
    Reranker 돌리는 부분 분리.
    """
    # [reranker 투입]

    # Reranker 입력용 쌍 만들기
    candidate_docs = []
    doc_texts_for_api = []
    deduplicate_docs = []


    for doc in result_docs:
        candidate_docs.append({
            "row": doc,
            "text": doc['text'],
            "score": 0.0
        })
        doc_texts_for_api.append({'text':doc['text']})

    # 결과 취합 및 중복 제거 (Deduplication)
    # 문서 ID나 내용을 기준으로 중복을 제거해야 함.
    unique_docs = {}
    for doc in doc_texts_for_api:
        key = hash(doc['text'])
        unique_docs[key] = doc

        deduplicate_docs = list(unique_docs.values())

    print(f"✂️ Deduplication: {len(result_docs)} -> {len(deduplicate_docs)}")

    docs_payload = [doc['text'] for doc in deduplicate_docs]

    resp = requests.post(RERANK_URL, json={"query": original_query, "docs": docs_payload})
    # Reranker가 있다면 예측 실행
    if resp.status_code == 200:
        print(f"📊 Reranking {len(deduplicate_docs)} candidates...")

        # 예측 (Logits 값 반환)
        scores = resp.json()['scores']

        # 점수 주입
        for i, score in enumerate(scores):
            candidate_docs[i]['score'] = float(score)

        # 점수 높은 순 정렬 (내림차순)
        candidate_docs.sort(key=lambda x: x['score'], reverse=True)

        # [격차 필터링] - Reranker 점수 기준
        # BGE-Reranker 점수는 음수~양수(Logits)이므로 비율(%)이 아니라 '차이(Diff)'로 자름

        top_score = candidate_docs[0]['score']

        # [튜닝 포인트] 1등과 4점 이상 차이나면 관련 없는 것으로 간주
        # BGE-M3 기준 4~5점 차이면 꽤 큰 격차
        cutoff_threshold = top_score - 4.0

        # 절대 필터링 (0점 밑으로는 취급 안함 - BGE-Reranker 기준)
        # *중요: BGE-M3는 관련 없으면 음수 점수가 나옴.
        # 관련없음을 0.0을 기준으로 잡고 감.
        min_score_limit = 0.1

        # cutoff와 min_score 중 더 높은 기준을 적용 (깐깐한 쪽 기준)
        final_cutoff = max(cutoff_threshold, min_score_limit)

        final_candidates = [d for d in candidate_docs if d['score'] > final_cutoff]

        # 상위 5개만 최종 선택
        final_candidates = final_candidates[:5]

        print(f"🔎 [Rerank Result] 1등: {top_score:.2f} / Cutoff: {final_cutoff:.2f}")
        print(f"🗑️ [Filter] {len(deduplicate_docs)}개 -> {len(final_candidates)}개로 압축됨.")

    else:
        # Reranker가 없는 경우 (Fallback) -> 빈값 return하기
        print("⚠️ Agent has no reranker. Empty list return.")
        final_candidates = []

    if not final_candidates:
        return "검색 결과가 없습니다."

    final_docs_str = []
    for item in final_candidates:
        row = item["row"]
        score = item["score"]
        text = item["text"]

        # PG(객체)냐 ES(딕셔너리)냐에 따라 created_at 꺼내는 방법이 다름.
        if isinstance(row, dict):
            # ES 데이터인 경우 (딕셔너리)
            created_at = row.get("created_at", "Unknown")
        else:
            # PG 데이터인 경우 (SQLModel 객체)
            created_at = row.created_at

        final_docs_str.append(
            f"[{created_at}](Rel_Score: {score:.2f})\n"
            f"{text}\n"
        )

    if not final_docs_str:
        return "검색 결과가 없습니다."

    # Reranker는 정확도가 높아서 가장 정확한 걸 맨 위로 두는게 낫지만,
    # LLM의 'Recency Bias(마지막 거 잘 봄)'를 이용하려면 역순 정렬도 고려 가능.
    # 일단은 점수 높은 순(Top 1이 맨 위)으로 리턴.
    return "\n\n".join(final_docs_str)