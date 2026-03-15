import os
import psycopg2
from psycopg2 import OperationalError

def get_db_connection():
    try:
        # K8s가 찔러준 환경변수를 os.getenv로 받아온다.
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASS"),
            database=os.getenv("DB_NAME"),
        )
        return conn
    except OperationalError as e:
        print(f"❌ DB 연결 실패: {e}")
        return None