from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException

from news_rag.factory import build_pipeline

app = FastAPI(title="AI News Research Assistant", version="0.1.0")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=3, ge=1, le=10)


class AskResponse(BaseModel):
    answer: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    try:
        pipeline = build_pipeline()
        answer = pipeline.answer(request.question, top_k=request.top_k)
        return AskResponse(answer=answer)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
