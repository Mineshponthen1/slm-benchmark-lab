import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import ollama
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ValidationError
from leave import LeaveResult, extract

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
LOG_FILE = Path("logs/requests.jsonl")

app = FastAPI(
    title="Local SLM Assistant",
    description="A small language model running 100% locally on a CPU-only laptop.",
    version="0.2.0",
)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    model: ModelName = DEFAULT_MODEL


class AskResponse(BaseModel):
    answer: str
    model: str
    tokens: int
    ttft_seconds: float
    tokens_per_second: float
    seconds: float


def log_request(record: dict) -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


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
    ttft = None
    final = None
    parts = []
    try:
        stream = ollama.chat(
            model=req.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": req.question},
            ],
            options=SETTINGS,
            stream=True,
        )
        for chunk in stream:
            text = chunk["message"]["content"]
            if text:
                if ttft is None:
                    ttft = time.perf_counter() - start
                parts.append(text)
            if chunk["done"]:
                final = chunk
        if final is None:
            raise RuntimeError("Ollama never sent the final summary")
    except Exception as e:
        log_request({
            "timestamp": now_utc(),
            "status": "error",
            "model": req.model,
            "question_chars": len(req.question),
            "error": type(e).__name__,
            "seconds": round(time.perf_counter() - start, 2),
        })
        raise HTTPException(status_code=503, detail=f"Model call failed: {e}")

    seconds = time.perf_counter() - start
    tokens = final["eval_count"]
    speed = tokens / (final["eval_duration"] / 1e9)

    log_request({
        "timestamp": now_utc(),
        "status": "ok",
        "model": req.model,
        "question_chars": len(req.question),
        "tokens": tokens,
        "ttft_seconds": round(ttft or 0.0, 3),
        "tokens_per_second": round(speed, 1),
        "seconds": round(seconds, 2),
    })

    return AskResponse(
        answer="".join(parts).strip(),
        model=req.model,
        tokens=tokens,
        ttft_seconds=round(ttft or 0.0, 3),
        tokens_per_second=round(speed, 1),
        seconds=round(seconds, 2),
    )
class LeaveMessage(BaseModel):
    message: str = Field(min_length=1, max_length=1000)
    model: ModelName = DEFAULT_MODEL


@app.post("/leave-request", response_model=LeaveResult)
def leave_request(req: LeaveMessage):
    try:
        result = extract(req.message, req.model)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Model call failed: {e}")
    if result.status == "failed":
        raise HTTPException(status_code=502, detail=result.model_dump(mode="json"))
    return result