# reranker_server.py
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForSequenceClassification
import uvicorn
import torch
app = FastAPI()

# Reranker 모델 로드 (GPU 있으면 cuda, 없으면 cpu)
device = "cpu"
MODEL_ID = 'BAAI/bge-reranker-v2-m3'
print(f"🔄️ Loading ONNX Reranker model on {device}")

# 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# 모델 로드 (ONNX 변환 적용)
# export=True: 최초 실행 시 Pytorch 모델을 ONNX로 변환해서 로컬 캐시에 저장함.
# 두 번째 실행부터는 변환된 것을 가져오므로 빠름.
model = ORTModelForSequenceClassification.from_pretrained(
    MODEL_ID,
    export=True
)
print("✅ Reranker loaded!")


class Payload(BaseModel):
    query: str
    docs: list[str]


@app.post("/rerank")
def rerank(payload: Payload):
    if not payload.docs: return {"scores": []}
    pairs = [[payload.query, doc] for doc in payload.docs]

    # 토크나이징
    inputs = tokenizer(
        pairs,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )

    # ONNX Runtime 추론 (No Grad 불필요하지만 명시적으로)
    with torch.no_grad():
        outputs = model(**inputs)

    scores = outputs.logits.view(-1).tolist()
    return {"scores": scores}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8010)