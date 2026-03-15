from sqlmodel import SQLModel, create_engine, Session
import os
from dotenv import load_dotenv
import redis.asyncio as redis

# .env 파일 로드 (로컬 개발용)
load_dotenv()

# REDIS
REDIS_HOST = os.getenv("REDIS_HOST")
REDIS_PORT = int(os.getenv("REDIS_PORT"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")

# 환경변수에서 DB 주소 조합 (postgresql://user:pass@host:port/db)
DATABASE_URL = f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASS')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
print(DATABASE_URL)

# 엔진 생성 (Echo=True로 해서 SQL로그를 다 찍도록 해서 디버깅을 용이하도록 한다.)
engine = create_engine(DATABASE_URL, echo=True)




def init_db():
    # 테이블이 없으면 자동으로 생성
    SQLModel.metadata.create_all(engine)

def get_session():
    with Session(engine) as session:
        yield session


def get_redis_client():
    return redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        password=REDIS_PASSWORD,
        decode_responses=True,
    )

rd = get_redis_client()