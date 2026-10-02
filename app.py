import time
from typing import Literal

import ollama
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

DEFAULT_MODEL = "llama3.2:3b-instruct-q4_K_M"
ModelName = Literal[
    "llama3.2:3b-instruct-q4_K_M",
    "llama3.2:3b-instruct-q5_K_M",
    "phi4-mini:latest",
    "mistral:latest",
]
SETTINGS = {"temperature": 0, "seed": 42, "num_predict": 300}
SYSTEM_PROMPT = (
    "You are a helpful assistant for bank employees. "
    "Answer clearly and briefly. If you are not sure, say so."
)

app = FastAPI(
    title="Local SLM Assistant",
    description="A small language model running 100% locally on a CPU-only laptop.",
    version="0.1.0",
)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    model: ModelName = DEFAULT_MODEL


class AskResponse(BaseModel):
    answer: str
    model: str
    tokens: int
    tokens_per_second: float
    seconds: float


@app.get("/health")
def health():
    try:
        installed = [m.model for m in ollama.list().models]
    except Exception:
        raise HTTPException(status_code=503, detail="Ollama is not running")
    return {
        "status": "ok",
        "default_model": DEFAULT_MODEL,
        "default_model_installed": DEFAULT_MODEL in installed,
    }


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    start = time.perf_counter()
    try:
        r = ollama.chat(
            model=req.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": req.question},
            ],
            options=SETTINGS,
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Model call failed: {e}")

    seconds = time.perf_counter() - start
    tokens = r["eval_count"]
    speed = tokens / (r["eval_duration"] / 1e9)
    return AskResponse(
        answer=r["message"]["content"].strip(),
        model=req.model,
        tokens=tokens,
        tokens_per_second=round(speed, 1),
        seconds=round(seconds, 1),
    )