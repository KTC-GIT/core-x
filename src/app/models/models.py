from pydantic import field_serializer
from sqlmodel import Field, SQLModel, JSON, Column
from datetime import datetime
from pgvector.sqlalchemy import Vector
from typing import Any


# Base 클래스 정의
# DTO와 Entity 양쪽 공통으로 사용하는 필드만 정의
class ConversationBase(SQLModel):
    user_id: str = Field(index=True)
    message: str
    model_name: str | None

# [DTO] 입력용 클래스 (Request Body)
# -Base를 상속받으므로 user_id, message가 자동으로 포함됨.
class ChatCreate(ConversationBase):
    pass

# [DTO] 응답용 클래스 (Response Body)
class ChatResponse(ConversationBase):
    id: int | None = Field(default=None)
    response: str | None = None
    created_at: datetime = Field(default_factory=datetime.now)

# [ENTITY] DB 테이블 정의
class Conversation(ConversationBase, table=True):
    # 이미 테이블이 정의되어 있어도 에러내지 말고 확장하라.
    __table_args__ = {"extend_existing": True}

    id: int | None = Field(default=None, primary_key=True)
    response: str | None = None
    created_at: datetime = Field(default_factory=datetime.now)

    # 임베딩 벡터 저장 컬럼 (1024차원)
    # sa_column을 써서 SQLAlchemy의 Vector 타입을 직접 지정
    embedding: list[float] | None = Field(default=None, sa_column=Column(Vector(1024)))

    # 직렬화(JSON 변환) 할 때 실행되는 함수
    @field_serializer("embedding")
    def serialize_embedding(self, v: Any):
        # 만약 v가 numpy 배열이거나 tolist 메서드가 있다면 list로 변환
        if v is not None and hasattr(v,"tolist"):
            return v.tolist()
        return v

    # Pydantic 설정: 임의의 타입(numpy)이 들어와도 일단 OK해라
    model_config = {
        "arbitrary_types_allowed": True,
    }

class DailyReportBase(SQLModel):
    # AI가 요약해준 제목과 내용
    title: str
    summary: str
    content: str

    # 태그 (주제들)
    tags: list[str] | None = Field(default=None, sa_column=Column(JSON))
    # 카테고리 [Study, Debugging, Feature, Architecture, General] 중 1
    category: str | None = None


class DailyReport(DailyReportBase, table=True):
    # 이미 테이블이 정의되어 있어도 에러내지 말고 확장하라.
    __table_args__ = {"extend_existing": True}

    id: int | None = Field(default=None, primary_key=True)
    user_id: str = Field(index=True)
    date: str | None = Field(index=True)  # YYYYMMDD 형식
    created_at: datetime = Field(default_factory=datetime.now)

    # 벡터 임베딩
    embedding: list[float] | None = Field(default=None, sa_column=Column(Vector(1024)))

    # 직렬화 로직은 Conversation과 동일하게..
    @field_serializer("embedding")
    def serialize_embedding(self, v: Any):
        if v is not None and hasattr(v,"tolist"):
            return v.tolist()
        return v

    model_config = {
        "arbitrary_types_allowed": True,
    }

class DailyReportResponse(DailyReportBase):
    pass


class WebKnowledge(SQLModel,  table=True):
    __tablename__ = "web_knowledge"
    id: int  = Field(default=None, primary_key=True)
    url: str = Field(index=True)
    title: str | None
    summary: str | None
    content: str | None
    source_type: str | None = Field(default='web')
    embedding: list[float] | None = Field(default=None, sa_column=Column(Vector(1024)))
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)