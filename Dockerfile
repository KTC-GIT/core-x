# 1. 가벼운 파이썬 이미지 사용
FROM python:3.11-slim

# 환경변수 설정
# PYTHONUNBUFFERED=1: 로그가 버퍼링 없이 즉시 출려됨 (K8s 로그 확인 시 필수! 이거 없으면 한참 뒤에 로그가 뜸)
# PYTHONDONTWRITEBYTECODE=1: .pyc 파일 생성 방지 (이미지 크기 미세하게 절약 및 불필요한 I/O 방지)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# 2. 작업 디렉토리 생성
WORKDIR /code

# 3. 필수 패키지 설치 (캐싱 활용을 위해 requirements 먼저 복사)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. 소스코드 전체 복사 (.dockerignore에 venv 등은 빠져있어야 함)
COPY . .

# [핵심] 파이썬이 'src' 폴더를 패키지 경로로 인식하게 함.
ENV PYTHONPATH=/code/src

# 문서화 목적 (실제 포트 개방은 K8s Service가 하지만 명시해두면 좋음)
EXPOSE 8000

# 5. 실행 명령 (src/app/main.py를 실행)
# 호스트 0.0.0.0으로 열어야 컨테이너 외부에서 접속 가능
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]