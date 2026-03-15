import time
import json
import asyncio
import os
import numpy as np

from fastapi import FastAPI, Depends, HTTPException, Request
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from sqlmodel import Session, select
from fastapi.responses import StreamingResponse
from phoenix.otel import register
from openinference.instrumentation.openai import OpenAIInstrumentor
from redis.commands.search.query import Query

from .core.database import init_db, get_session, rd, engine
from .models.models import Conversation, ChatCreate, ChatResponse, DailyReportResponse
from .models.redis_model import  ChatCache, save_chat_to_redis, INDEX_NAME, create_redis_index
from .service.chat_service import  search_recent_chats
from .service.llm import CoreXAgent
from .service.report_service import  generate_daily_report

# .env 파일 로드 (로컬 개발용)
load_dotenv()


# Phoenix Tracer EndPoint
PHOENIX_ENDPOINT = os.getenv("PHOENIX_ENDPOINT")

tracer_provider = register(
    project_name="core-x",
    endpoint=PHOENIX_ENDPOINT,
    protocol="grpc"
)

# OpenAI 호출 감시
OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)

# 서버 시작 시 테이블 자동 생성 (LifeSpan)
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()

    # redis 인덱스 생성
    try:
        await create_redis_index()
    except Exception as e:
        print(f"⚠️ Redis 인덱스 생성 실패: {e}")
    yield
app = FastAPI(
    title="Core-X: Personal AI Assistant",
    lifespan=lifespan
)

agent = CoreXAgent()

@app.post("/chat/")
async def chat(request: ChatCreate, session: Session = Depends(get_session)):

    # DTO -> Entity 변환 마법
    # request 객체의 필드(user_id, message)를 그대로 풀어서 Conversation 생성자에 넣음
    # 사용자 메시지 우선 저장 (User Log)
    # (일단 AI 응답은 비워두고 저장)
    memory = Conversation(**request.model_dump())
    session.add(memory)
    session.commit()
    session.refresh(memory)

    user_embedding = await agent.get_embedding(text=request.message, is_query=True)

    # Redis 뒤지기
    from .service import redis_service
    rmm = redis_service.RedisMemoryManager()
    cached_answer = await rmm.find_similar_cache(embedding=user_embedding)

    # 있으면 바로 리턴
    if cached_answer:
        print(f"🚀 [Cache Hit] LLM 패스하고 바로 응답")

        # 캐시된 답을 DB에도 업데이트 (그래야 히스토리에 남음)
        memory.response = cached_answer
        memory.model_name = "RedisCache"    # 모델명 대신 표시
        session.add(memory)
        session.commit()

        async def cache_stream():
            yield cached_answer

        return StreamingResponse(
            cache_stream(),
            media_type="text/plain",
        )

    # related_memories = search_similar_memory(user_embedding,limit=2)
    recent_chats = search_recent_chats(request,limit=2)

    # 프롬프트 강화 (Augmentation)
    # 과거 기억이 있으면 LLM에게 힌트로 줌.
    system_prompt = "### [Context Injection]"
    # if related_memories:
    #     memory_text = "\n".join([f"- User: {m.message} / AI: {m.response}" for m in related_memories])
    #     print(f"memory_text: \n{memory_text}")
    #     system_prompt += f"\n\n**Related Past Conversation History**\n\n{memory_text}\n"
        # print(f"🌟 RAG Activated! 참고할 기억 {len(related_memories)}개를 찾았습니다.")

    # Recent History(단기 기억) 주입
    if recent_chats:
        history_text = "\n".join([f"- User: {chat.message} \n AI: {chat.response}" for chat in recent_chats])
        print(f"history_text: \n{history_text}")
        system_prompt += f"\n\n**Recent Conversation Flow**\n\n{history_text}\n"

    system_prompt += """
    
    ### [Instruction]
    Based on the past history and recent flow, please answer current question naturally in Korean.
    Ensure seamless continuity in the conversation.
    """

    async def response_generator():
        full_response_text = ""

        try:
            # LLM이 한 글자씩 줄 때마다 바로바로 클라이언트로 쏘고 변수에 모음
            async for token in agent.chat_with_ai(
                prompt=memory.message,
                model_strategy=memory.model_name,
                system_role=system_prompt
            ):
                full_response_text += token
                yield token


            # [검증 및 업데이트] 응답 성공 여부에 따른 분기 처리
            # AI 응답이 유효한지 체크 (빈 값이나 에러 메시지인지 확인)

            is_valid_response = full_response_text and not full_response_text.startswith("[Error]")

            if is_valid_response:
                # ✅ 성공 시 : Q+A 묶어서 벡터화 (High Quality Data)
                # 저장할 벡터 생성 (질문 + 답변)
                # 나중에 검색될 때는 "이런 질문에 이런 답을 했었지"라는 맥락이 통재로 필요함.
                # is_query=False (저장용 문서이므로)
                combined_text = f"User Question: {request.message} \n AI Answer: {full_response_text}"
                combined_vector = await agent.get_embedding(text=combined_text, is_query=False)
                memory.embedding = combined_vector
                memory.response = full_response_text

                print(f"✅ [ID: {memory.id}] 벡터화 및 저장 완료")

                # [Redis] 응답 저장
                await  rmm.save_interaction(
                    user_id=memory.user_id,
                    message=memory.message,
                    response=memory.response,
                    embedding=user_embedding,  # 질문의 벡터를 넣는다.
                    model_name=memory.model_name
                )

                print(f"💾 [Redis] 캐시 등록 됨.")

            else:
                # ❌ 실패 시: 벡터화 건너뜀 (Low Quality Data)
                # 답변 필드에 에러 메시지나 원본을 넣되, 검색에는 걸리지 않게 함
                memory.response = full_response_text if full_response_text else "[No Response]"
                memory.embedding = None     # 벡터 저장 X (검색 안 됨)
                print(f"⚠️ [ID:{memory.id}] AI 응답 실패/에러로 인한 벡터화 건너뜀.")


            # AI 응답을 DB에 업데이트 (Update)
            session.add(memory)
            session.commit()
            session.refresh(memory)

            # ES에도 indexing
            from .service.chat_to_es_service import index_chat_to_es
            index_chat_to_es(
                user_id=memory.user_id,
                message=memory.message,
                response=memory.response,
                model_name=memory.model_name
            )
        except Exception as e:
            print(f"⚠️ 스트리밍 중 에러: {e}")
            yield f"\n[System Error] {str(e)}"

    return StreamingResponse(
        response_generator(),
        media_type="text/event-stream",
    )


@app.post("/v1/chat/completions")
async def chat_completions(request: Request, session: Session = Depends(get_session)):
    # SillyTavern용

    body = await request.json()

    provider = body.get("model", "grok-4-1-fast-reasoning").strip()

    messages = body.get("messages", [])
    prompt = messages[-1]["content"] if messages and messages[-1].get("role") == "user" else ""

    chat_create = ChatCreate(
        user_id='ktc',
        model_name=provider,
        message=prompt
    )

    memory = Conversation(
        user_id='ktc',
        model_name=provider,
        message=prompt
    )

    session.add(memory)
    session.commit()
    session.refresh(memory)

    user_embedding = await agent.get_embedding(text=prompt, is_query=True)

    # Redis 뒤지기
    from .service import redis_service
    rmm = redis_service.RedisMemoryManager()
    cached_answer = await rmm.find_similar_cache(embedding=user_embedding)

    # 있으면 바로 리턴
    if cached_answer:
        print(f"🚀 [Cache Hit] LLM 패스하고 바로 응답")

        # 캐시된 답을 DB에도 업데이트 (그래야 히스토리에 남음)
        memory.response = cached_answer
        memory.model_name = "RedisCache"  # 모델명 대신 표시
        session.add(memory)
        session.commit()

        async def cache_stream():
            yield cached_answer

        return StreamingResponse(
            cache_stream(),
            media_type="text/event-stream",
        )

    # related_memories = search_similar_memory(user_embedding,limit=2)
    recent_chats = search_recent_chats(chat_create, limit=2)

    # 프롬프트 강화 (Augmentation)
    # 과거 기억이 있으면 LLM에게 힌트로 줌.
    system_prompt = "### [Context Injection]"
    # if related_memories:
    #     memory_text = "\n".join([f"- User: {m.message} / AI: {m.response}" for m in related_memories])
    #     print(f"memory_text: \n{memory_text}")
    #     system_prompt += f"\n\n**Related Past Conversation History**\n\n{memory_text}\n"
    # print(f"🌟 RAG Activated! 참고할 기억 {len(related_memories)}개를 찾았습니다.")

    # Recent History(단기 기억) 주입
    if recent_chats:
        history_text = "\n".join([f"- User: {chat.message} \n AI: {chat.response}" for chat in recent_chats])
        print(f"history_text: \n{history_text}")
        system_prompt += f"\n\n**Recent Conversation Flow**\n\n{history_text}\n"

    system_prompt += """

    ### [Instruction]
    Based on the past history and recent flow, please answer current question naturally in Korean.
    Ensure seamless continuity in the conversation.
    """

    async def response_generator():
        full_response_text = ""

        try:
            # LLM이 한 글자씩 줄 때마다 바로바로 클라이언트로 쏘고 변수에 모음
            async for token in agent.chat_with_ai(
                    prompt=memory.message,
                    model_strategy=memory.model_name,
                    system_role=system_prompt
            ):
                full_response_text += token
                if token:
                    chunk_data = {
                        "id": f"chatcmpl-Core-x",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": provider,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": token},
                            "finish_reason": None
                        }]
                    }
                    yield f'data: {json.dumps(chunk_data)}\n\n'
                    await asyncio.sleep(0.018)

            # 마무리에 꼭 필요함.
            yield 'data: {"choices": [{"delta": {}, "finish_reason": "stop"}]}\n\n'
            yield 'data: [DONE]\n\n'

            # [검증 및 업데이트] 응답 성공 여부에 따른 분기 처리
            # AI 응답이 유효한지 체크 (빈 값이나 에러 메시지인지 확인)

            is_valid_response = full_response_text and not full_response_text.startswith("[Error]")

            if is_valid_response:
                # ✅ 성공 시 : Q+A 묶어서 벡터화 (High Quality Data)
                # 저장할 벡터 생성 (질문 + 답변)
                # 나중에 검색될 때는 "이런 질문에 이런 답을 했었지"라는 맥락이 통재로 필요함.
                # is_query=False (저장용 문서이므로)
                combined_text = f"User Question: {prompt} \n AI Answer: {full_response_text}"
                combined_vector = await agent.get_embedding(text=combined_text, is_query=False)
                memory.embedding = combined_vector
                memory.response = full_response_text

                print(f"✅ [ID: {memory.id}] 벡터화 및 저장 완료")

                # [Redis] 응답 저장
                await  rmm.save_interaction(
                    user_id=memory.user_id,
                    message=memory.message,
                    response=memory.response,
                    embedding=user_embedding,  # 질문의 벡터를 넣는다.
                    model_name=memory.model_name
                )

                print(f"💾 [Redis] 캐시 등록 됨.")

            else:
                # ❌ 실패 시: 벡터화 건너뜀 (Low Quality Data)
                # 답변 필드에 에러 메시지나 원본을 넣되, 검색에는 걸리지 않게 함
                memory.response = full_response_text if full_response_text else "[No Response]"
                memory.embedding = None  # 벡터 저장 X (검색 안 됨)
                print(f"⚠️ [ID:{memory.id}] AI 응답 실패/에러로 인한 벡터화 건너뜀.")

            # AI 응답을 DB에 업데이트 (Update)
            session.add(memory)
            session.commit()
            session.refresh(memory)

            # ES에도 indexing
            from .service.chat_to_es_service import index_chat_to_es
            index_chat_to_es(
                user_id=memory.user_id,
                message=memory.message,
                response=memory.response,
                model_name=memory.model_name
            )
        except Exception as e:
            print(f"⚠️ 스트리밍 중 에러: {e}")
            error_chunk = {"choices": [{"delta": {"content": f"\n\n[오류] {str(e)}"}, "finish_reason": "stop"}]}
            yield f"data: {json.dumps(error_chunk)}\n\n"
            yield 'data: [DONE]\n\n'

    return StreamingResponse(
        response_generator(),
        media_type="text/event-stream",
    )


@app.post("/report/",response_model=DailyReportResponse)
async def create_daily_report(user_id: str, model_strategy: str, target_date: str):
    """특정 날짜의 대화 로그를 기반으로 일일 리포트를 생성한다."""
    report = await generate_daily_report(user_id, model_strategy, target_date)
    if report is None:
        raise HTTPException(status_code=404, detail="해당 날짜에 대화 기록이 없습니다.")
    return report

@app.get("/memories/")
def read_memories(session: Session = Depends(get_session)):
    memories = session.exec(select(Conversation)).all()
    return memories

@app.get("/")
def read_root():
    return {
        "system": "Core-X Online",
        "module": "Hippocampus (Memory)"
    }

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/redis-health")
async def redis_health():
    try:
        # ping으로 연결확인
        pong = await rd.ping()
        module = await rd.module_list()
        return {"status": "online", "response": pong, "module": module}
    except Exception as e:
        return {"status": "offline", "response": str(e)}


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "qwen3:14b",
                "object": "model",
                "created": 1720000000,
                "owned_by": "user"
            },
            {
                "id": "qwen3:4b-thinking-2507-fp16",
                "object": "model",
                "created": 1720000000,
                "owned_by": "user"
            },
            {
                "id": "qwen3:32b-q4_k_m",
                "object": "model",
                "created": 1720000000,
                "owned_by": "user"
            },
            {
                "id": "qwen3:32b",
                "object": "model",
                "created": 1720000000,
                "owned_by": "user"
            },
            {
                "id": "llama3.3:70b",
                "object": "model",
                "created": 1720000000,
                "owned_by": "user"
            },
            {
                "id": "grok-4-1-fast-reasoning",
                "object": "model",
                "created": 1720000000,
                "owned_by": "user"
            },
            {
                "id": "grok-4-1-fast-non-reasoning",
                "object": "model",
                "created": 1720000000,
                "owned_by": "user"
            },
            {
                "id": "gpt-5.4",
                "object": "model",
                "created": 1720000000,
                "owned_by": "user"
            }
        ]
    }