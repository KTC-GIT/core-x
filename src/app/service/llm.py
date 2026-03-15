#llm.py
from datetime import datetime
from typing import AsyncGenerator

from langsmith.wrappers import wrap_openai
from openai import AsyncOpenAI
from dotenv import load_dotenv
from opentelemetry import trace
from openinference.instrumentation import using_attributes

import httpx
import os
import asyncio
import numpy as np

from ..service.rag_service import retrieve_relevant_reports

#로컬용 환경변수 로드
load_dotenv(override=True)

# 트레이서 생성
tracer = trace.get_tracer(__name__)

class CoreXAgent:
    def __init__(self):
        """
        초기화 시점에 로컬용, 외부용, 임베딩용 클라이언트를 각각 생성하여
        self 변수에 담기.
        """

        # 로컬 LLM(Qwen 등) - 주력/임베딩용
        self.local_client = wrap_openai(AsyncOpenAI(
            base_url=os.getenv("OPENAI_BASE_URL"),
            api_key=os.getenv("OPENAI_API_KEY"),
        ))

        # 로컬 LLM(Qwen 등) - 원격 실행용
        # 공개용 코드에서는 실제 엔드포인트를 직접 노출하지 않고 환경변수로 관리
        self.runpod_client = wrap_openai(AsyncOpenAI(
            base_url=os.getenv("REMOTE_LLM_BASE_URL"),
            api_key=os.getenv("REMOTE_LLM_API_KEY", "dummy"),
        ))

        # 외부 LLM (Grok)
        # API key가 없으면 None으로 두어 에러방지
        xai_key = os.getenv("XAI_API_KEY")
        self.grok_client = None
        if xai_key:
            self.grok_client = wrap_openai(AsyncOpenAI(
                base_url="https://api.x.ai/v1",
                api_key=xai_key,
            ))

        # 외부 LLM (GPT)
        # API key가 없으면 None으로 두어 에러방지
        gpt_key = os.getenv("GPT_API_KEY")
        self.gpt_client = None
        if gpt_key:
            self.gpt_client = wrap_openai(AsyncOpenAI(
                base_url="https://api.openai.com/v1",
                api_key=gpt_key,
            ))

        # 모델명 설정
        self.local_model_name = os.getenv("MODEL_NAME", "qwen3:14b")
        self.external_model_name = "grok-4-1-fast-reasoning"

        self.tools_schema = [
            {
                "type": "function",
                "function": {
                    "name": "search_web",
                    "description": (
                        "- 설명:"
                        "필수 도구입니다."
                        "사용자의 질문이 날씨, 최신 뉴스, 주가, 혹은 AI 내부에 없는 정보를 포함할 때"
                        "**반드시 이 도구를 사용하여 정보를 찾아햡니다.** 거짓말이나 모른다는 답변을 하지 않기 위해서 사용하세요."
                        "- 주의:"
                        "검색어는 문장이 아닌 **핵심 키워드 위주**로 작성하세요"
                        "(Bad: '로봇 MLOps와 일반 MLOps의 차이점은 무엇인가요?')"
                        "(Good: 'Robot MLOps vs General MLOps difference')"
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "검색할 키워드 또는 문장"
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_memory",
                    "description": (
                        "Use this tool when you search user history, conversation, practice content"
                        "Use this tool when user say 'Do you remember?', 'What did I say?', 'Search DB'"
                        "You Don't need make SQL Query. Just input 'Main Keyword' or 'sentence'"
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "array",
                                "items": {
                                    "type": "string"
                                },
                                "description": "A list of specific search queries. Example: ['error automatic', 'report last week', 'RAG plan']"
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "save_daily_report",
                    "description": "대화 내용을 요약하여 DB에 저장합니다. '정리해줘', '일기로 써줘' 요청 시 사용합니다. **반드시 먼저 execute_sql로 대화내용을 조회 후 이 도구를 사용해야합니다.**",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string", "description": "레포트 제목"},
                            "content": {"type": "string", "description": "전체 내용을 마크다운으로 정리"},
                            "summary": {"type": "string", "description": "3줄 요약"},
                            "tags": {"type": "string", "description": "태그 리스트 (예: ['ReAct', 'Error'])"},
                            "category": {"type": "string", "description": "카테고리 (IT, Study, Life 등)"}
                        },
                        "required": ["title", "content", "summary", "tags", "category"]
                    }
                }
            }
        ]


    @tracer.start_as_current_span("chat_with_ai")
    async def chat_with_ai(
        self,
        prompt: str,
        model_strategy: str,
        system_role: str = "당신은 유능한 AI 비서 Core-X 입니다."
    ) -> AsyncGenerator[str, None]:
        """
        OpenAI 호환 API를 비동기(Async)로 호출합니다.
        Main 스레드를 차단하지 않으므로, 생각하는 동안 다른 요청을 받을 수 있음.

        Args:
            prompt: 사용자 질문
            model_strategy: 'local' | 'grok' (UI에서 선택된 값)

        Returns:
            (답변 텍스트, 실제 사용된 모델명) -> DB 저장용

        [2026-01-31] 스트리밍 타입으로 변경
        1. return 타입이 str -> AsyncGenerator[str,None]
        2. 도구 실행 단계는 내부에서 처리 (None-Streaming)
        3. 최종 답변 단계에서 stream=True로 호출하여 yield
        """

        # 디버깅
        # 공개용 코드에서는 내부 객체 정보 전체를 출력하지 않도록 축약
        print(f"[LLM] requested strategy: {model_strategy}")

        # 클라이언트 및 모델 결정
        if 'grok' in model_strategy and self.grok_client:
            active_client = self.grok_client
            active_model = model_strategy
            print(f"🚀 grok이 선택됨. {active_model}")
        elif 'gpt' in model_strategy and self.gpt_client:
            active_client = self.gpt_client
            active_model = model_strategy
            print(f"🧠 gpt가 선택됨. {active_model}")
        elif '32b' in model_strategy or '70b' in model_strategy:
            active_client = self.runpod_client
            active_model = model_strategy
            print(f"📦 원격 모델이 선택됨. {active_model}")
        else:
            active_client = self.local_client
            active_model = model_strategy
            print(f"🏠 로컬 모델이 선택됨. {active_model}")

        # 날짜 가져오기
        current_time_str = datetime.now().strftime("%Y%m%d-%H%M%S")

        # 최대 재시도 횟수 (무한 루프 방지용)
        max_turn_limit = 5

        # 과거 기억 검색 (RAG)
        relevant_reports = await retrieve_relevant_reports(
            prompt,
            agent=self,
            user_id="ktc"
        )

        context_str = ""
        if relevant_reports:
            context_str = "\n".join([
                f"[{d.date} report] (category: {d.category})\n(tags:{d.tags})\n(내용: {d.content})"
                for d in relevant_reports
            ])

        # 시스템 프롬프트 (영어 지시 + 한국어 출력)
        system_role += f"""
        You are Core-X, a highly capable AI assistant.
        Current Time: {current_time_str}

        ### [Core Principles]
        1. **Time Awareness:** The current time provided above is absolute truth. Do not doubt it.
        2. **Language Protocol:** - **Input:** User will speak in Korean.
            - **Thinking:** You MUST think and reason in **English** (Chain of Thought).
            - **OutputL** You MUST generate the final answer in **Natural Korean**.
            - **Translation:** If tool results (Search/SQL) are in English, translate them into Korean for the final answer.
        3. **Readability:** Use line breaks and bullet points actively.
        4. **User Persona:** Assume the user cannot read English well. Everything provided to the user must be in Korean.

        ### [Tool Usage Guidelines]
        Evaluate the situation and select the appropriate tool.

        ## Parallel Execution & Query Expansion
        1. **Analyze First, Execute Once:**
            Before calling any retrieval tools, analyze the user's request to identify all independent pieces of information required.
        2. **Multi-Query Strategy:**
            Do NOT call the search tool multiple times sequentially for independent facts. Instead, generate **list of distinct, specific search queries** covering all necessary aspect and pass them in a **single tool call**.
            * *Bad:* Search "deployment" -> Wait for result -> Search "plans" -> Wait for result.
            * *Good:* Search `["deployment automation log", "plans from 2026-01-10", "current RAG status"]` in one call.
        3. **Handling Dependencies:**
            Only use sequential steps (multiple turns) if the result of the first search is strictly required to formulate the second search query. Otherwise, prioritize parallel retrieval.
        4. **Completeness:**
            Ensure the retrieved documents provide sufficient context to answer the user's question comprehensively without needing follow-up searches.

        ## Tools
        **1. External Info (Weather, Stock, News, General Knowledge)** -> Use `search_web`
            - Your internal knowledge cutoff is in the past. For 2024+ info, REAL-TIME weather, or stock prices, DO NOT rely on your memory.
            - Always try to search before saying "I don't know".

        **2. Internal Info (User Diary, Past Conversations, Reports)** -> Use `search_memory`
            - If user asks "Do you remember?", "what did I say?", "Find my diary", query the database.
            - If SQL result is empty, report "No data found" immediately. Do NOT retry with hallucinated queries.
            - Translate the query into Korean before using `search_memory`.

        **3. Conversation Summary** -> Use `save_daily_report`
            - Step 1: `execute_sql` to fetch today's conversation.
            - Step 2: Summarize based on the fetched data.
            - Step 3: `save_daily_report` to save.

        ### [Memory Retrieval]
        - If you need to recall past conversation or facts, use the `search_memory` tool.
        - Just provide the search query in natural language. The system will handle the database lookup.

        ### [Thinking Process & Strict Output Format]
        For general chat (no tools needed), use the following format:

        [THOUGHT]
        (Write your reasoning here in **English**. Analyze intent, check context, decide response strategy.)

        [ANSWER]
        (Write your final answer here in **Korean**.)

        **CRITICAL:** Never output the [THOUGHT] part in the final answer if possible, but the code will handle parsing.
        The text after `[ANSWER]` MUST be in Korean.

        ### [Retrieved Long-term Memories (RAG Context)]
        The following text contains relevant past reports or conversation logs retrieved from your database.
        **You MUST prioritize this information over your general knowledge.**

        {context_str}
        """

        messages = [
            {"role": "system", "content": system_role},
            {"role": "user", "content": prompt}
        ]

        # ReAct 루프 시작
        # LLM이 도구를 쓰고 싶어하면 돌고, 답변을 내놓으면 멈춤
        for turn in range(max_turn_limit):
            try:
                response = await active_client.chat.completions.create(
                    model=active_model,
                    messages=messages,
                    tools=self.tools_schema,
                    tool_choice="auto",
                    temperature=0.7,
                )
                response_msg = response.choices[0].message

                # [탈출 조건] 도구를 안 쓰고 그냥 말했다면 -> 탈출 조건
                if not response_msg.tool_calls:
                    import re
                    full_content = response_msg.content or ""

                    # 진짜 답변 발라내기
                    final_answer = ""

                    # Grok/Reasoning 모델이 [ANSWER] 태그를 사용한 경우
                    if '[ANSWER]' in full_content:
                        # [ANSWER] 뒤에 있는 텍스트만 답변으로 간주
                        final_answer = full_content.split('[ANSWER]')[-1].strip()

                    # <response> 대응
                    elif "<response>" in full_content:
                        match = re.search(r'<response>(.*?)</response>', full_content, re.DOTALL)
                        final_answer = match.group(1).strip() if match else full_content

                    # 태그가 아예 없는 경우
                    else:
                        final_answer = full_content

                    # 불필요한 태그 잔재 청소
                    final_answer = final_answer.replace("<response>", "").replace("</response>", "").strip()

                    # 한글 비율 체크
                    korean_char_count = len(re.findall(r'[가-힣]', final_answer))
                    total_char_count = len(final_answer) + 1  # 분모 0 방지
                    korean_ratio = korean_char_count / total_char_count

                    # 판결 : 다시 시킬지? 그냥 보낼지?
                    # 조건 : 한글이 너무 적거나(20% 미만)
                    is_bad_response = (korean_ratio < 0.2)

                    # 예외: 코드 블록(```)이 포함된 기술 답변은 영어가 많을 수 있으므로 False처리
                    if "```" in final_answer:
                        is_bad_response = False

                    if is_bad_response:
                        print(f"🤖 [Auto-Fix] 답변 품질 미달 (Ratio: {korean_ratio:.2f})")

                        # 모델에게 재요청
                        messages.append(response_msg)
                        messages.append({
                            "role": "user",
                            "content": (
                                "System Warning: You used too much English in final answer.\n"
                                "Please rewrite the part after '[ANSWER]' in natural Korean. "
                                "If it is technical term, keep it in English but explain in Korean.\n"
                                f"Your answer:\n{final_answer}"
                            )
                        })
                        continue

                    # 통과
                    chunk_size = 4
                    for i in range(0, len(final_answer), chunk_size):
                        yield final_answer[i:i + chunk_size]
                        await asyncio.sleep(0.01)
                    return

                # 도구를 쓰겠다고 했으면 -> 기록에 남기고 실행
                # 도구쓸래요 요청을 대화 내역에 추가해야 문맥이 이어짐.
                messages.append(response_msg)

                # AI가 "검색하고 싶어요!" 라고 신호를 보냈는지 확인
                for tool_call in response_msg.tool_calls:
                    import json
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)

                    # 공개용 코드에서는 상세 인자/결과 전체를 로그로 남기지 않도록 축약
                    print(f"🔧 [{active_model}][Turn {turn+1}] 도구 호출: {function_name}")

                    tool_result = ""

                    # 도구 실행 (에러가 나도 에러 메시지를 tool_result에 담아서 LLM에게 줘야 함!)
                    try:
                        if function_name == "search_web":
                            # 실제 파이썬 함수 실행
                            from ..core.tools import search_web

                            # 방어코드: LLM이 지시어를 잘못보고 list로 던져주는 것을 방지하는 코드
                            raw_query = function_args.get("query") or function_args.get("queries")

                            if isinstance(raw_query, list):
                                query_input = raw_query[0] if raw_query else ""
                            elif isinstance(raw_query, str):
                                query_input = raw_query.strip()
                            else:
                                query_input = ""

                            tool_result = search_web(query=query_input)

                        elif function_name == "search_memory":
                            from ..core.tools import search_memory

                            # 방어코드: queries가 없으면 query를 찾고, 그것도 없으면 빈 리스트
                            queries_input = function_args.get("queries")
                            if not queries_input:
                                single_query = function_args.get("query")
                                if isinstance(single_query, list):
                                    queries_input = single_query
                                elif single_query:
                                    queries_input = [single_query]
                                else:
                                    queries_input = []

                            tool_result = await search_memory(
                                self,
                                original_query=prompt,
                                queries=queries_input
                            )

                        elif function_name == "save_daily_report":
                            from ..core.tools import save_daily_report
                            # 인자가 여러 개이므로 **kwargs로 풀어서 전달
                            tool_result = save_daily_report(**function_args)

                    except Exception as tool_err:
                        # 파이썬 에러를 LLM에게 그대로 알려줌 -> LLM이 보고 고칠 수 있도록!
                        tool_result = f"Error executing tool: {str(tool_err)}. please fix your query."
                        print(f"❌ 도구 에러 발생: {type(tool_err).__name__}")

                    # 검색결과 확인
                    print("tool_result received")

                    if isinstance(tool_result, str) and '검색 결과가 없습니다.' in tool_result:
                        chunk_size = 4
                        for i in range(0, len(tool_result), chunk_size):
                            yield tool_result[i:i + chunk_size]
                            await asyncio.sleep(0.01)
                        return

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": function_name,
                        "content": str(tool_result)
                    })

            except Exception as e:
                print(f"Error detail: {type(e).__name__}")
                yield "응답 생성 중 오류가 발생했습니다."
                return

        # 최종답변 생성
        final_system_prompt = """
        당신은 유능한 개인비서 Core-X입니다. 위 대화 내용(History)을 바탕으로 사용자에게 **최종 답변** 하세요.

        [규칙]
        1. 생각(Reasoning) 과정을 출력하지 마세요.
        2. 오직 사용자를 위한 답변만 **질문이 한국어로 되어 있다면 한국어**로 정중하게 작성하세요.
        3. 앞선 대화에서 도구를 사용했다면, 그 결과를 잘 요약해서 전달하세요.
        """

        # 기존 messages(도구 실행 로그 포함) 뒤에 "이제 정리해"라는 명령을 붙임.
        messages.append({
            "role": "system",
            "content": final_system_prompt
        })

        final_response = await active_client.chat.completions.create(
            model=active_model,
            messages=messages,
            stream=True,
            temperature=0.7,
        )

        async for chunk in final_response:
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content

    async def get_embedding(self, text: str, is_query: bool = False) -> list[float]:
        """
        텍스트를 입력받아 벡터(숫자 리스트)로 변환.
        사용 모델: bge-m3

        is_query=True 면 검색용 접두어(Instruction)를 붙여서 임베딩 품질을 높임
        """
        try:
            # 텍스트가 비어있으면 에러 나므로 예외 처리
            # 빈 값 대신 0으로 채운 벡터 반환 (에러 방지)
            if not text:
                return [0.0] * 1024

            # 길이 제한 로직 추가 (Truncation)
            # 모델 컨텍스트 한계를 넘지 않도록 안전하게 자름
            MAX_CHARS = 1000
            if len(text) > MAX_CHARS:
                text = text[:MAX_CHARS]
                print(f"[Warn] 텍스트가 너무 길어서 잘랐습니다. ({len(text)} -> {MAX_CHARS})")

            # 검색할 때 이 접두어를 붙여야 질문의 의도를 파악하고 답을 찾아옴
            if is_query:
                text = f"Represent this sentence for searching relevant passages: {text}"

            response = await self.local_client.embeddings.create(
                model="bge-m3",
                input=text,
            )
            embedding_data = response.data[0].embedding

            # 만약 numpy array라면 리스트로 변환, 아니면 그대로 반환
            if hasattr(embedding_data, "tolist"):
                return embedding_data.tolist()

            return list(embedding_data)

        except Exception as e:
            print(f"[Error] 임베딩 생성 실패: {str(e)}")
            # 실패 시 크래시 방지를 위해 0으로 채운 '더미 벡터' 반환
            # 이렇게 하면 DB 저장 에러는 안 나고, 검색엔 안 걸리게 됨.
            return [0.0] * 1024
