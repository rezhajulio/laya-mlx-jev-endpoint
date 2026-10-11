import asyncio
from contextlib import asynccontextmanager
import logging
import os
from typing import Any, Literal

import anyio
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import laya_mlx as laya
from pydantic import BaseModel, ConfigDict, Field

MODEL_PATH = os.getenv("LAYA_MODEL", "aac6fef/laya-mlx")
MODEL_DTYPE = os.getenv("LAYA_DTYPE", "float16")
PORT = int(os.getenv("PORT", "8000"))
HOST = os.getenv("HOST", "127.0.0.1")
ACCEPTED_MODELS = {MODEL_PATH, "jev-latest", "laya-mlx"}


class QuestionSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["choice", "score", "noul"]
    instructions: str | dict[str, Any] | list[Any] | int | float
    criteria: dict[str, Any] | list[Any] | None = None
    labels: dict[str, str] | None = None


class SystemOneRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str | None = None
    state: str | dict[str, Any] | list[Any]
    questions: dict[str, QuestionSpec] = Field(..., min_length=1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    def _load_agent():
        return laya.load(MODEL_PATH, dtype=MODEL_DTYPE)

    agent = await anyio.to_thread.run_sync(_load_agent)
    # warmup graph compilation
    await anyio.to_thread.run_sync(
        agent.predict,
        "warmup",
        {"ping": {"type": "noul", "instructions": "System operational?"}},
    )
    app.state.agent = agent
    app.state.lock = asyncio.Lock()
    yield
    app.state.agent = None


app = FastAPI(title="Laya MLX JEV Endpoint", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=400, content={"detail": exc.errors()})

@app.get("/health")
async def health():
    if not getattr(app.state, "agent", None):
        raise HTTPException(status_code=503, detail="Model runtime not initialized")
    return {"status": "ready", "model": MODEL_PATH, "engine": "mlx"}


@app.get("/v1/models")
async def list_models():
    return {
        "models": [
            {
                "id": MODEL_PATH,
                "name": MODEL_PATH,
                "description": "Laya MLX System-1 Decision Model",
            },
            {
                "id": "jev-latest",
                "name": "jev-latest",
                "description": "Compatibility alias for active System-1 model",
            },
        ]
    }


@app.post("/v1/systemone")
async def predict_decision(payload: SystemOneRequest):
    agent = getattr(app.state, "agent", None)
    if agent is None:
        raise HTTPException(status_code=503, detail="Model runtime not initialized")

    if payload.model and payload.model not in ACCEPTED_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model: {payload.model}. Available: {sorted(ACCEPTED_MODELS)}",
        )

    # Convert questions to raw dicts, dropping None keys
    questions_dict = {
        qid: q.model_dump(exclude_none=True)
        for qid, q in payload.questions.items()
    }

    lock: asyncio.Lock = app.state.lock
    try:
        # MLX utilizes Apple Silicon unified memory directly; serialize forward passes
        async with lock:
            result = await anyio.to_thread.run_sync(
                agent.predict, payload.state, questions_dict
            )
        return {
            "model": MODEL_PATH,
            "answers": result["answers"],
            "usage": result.get("usage", {}),
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except FloatingPointError as e:
        raise HTTPException(status_code=500, detail=f"Numerical error: {e}") from e
    except Exception as e:
        logging.exception("Unexpected inference failure")
        raise HTTPException(status_code=500, detail="Internal server error") from e


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host=HOST, port=PORT, workers=1)
