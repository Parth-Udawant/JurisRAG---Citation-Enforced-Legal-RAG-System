from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from ..core.config import get_settings
from ..models.schemas import (
    ActOption,
    ActsResponse,
    ChatRequest,
    ChatResponse,
    RetrievalInfo,
    Usage,
)
from ..observability.telemetry import RequestTelemetry
from ..services.generation_service import DISCLAIMER
from ..services.rag_service import RAGService

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
settings = get_settings()
telemetry = RequestTelemetry()
rag = RAGService(settings)


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(
    title="Legal RAG API",
    version="1.0.0",
    description="Grounded legal RAG backend for BNS and Indian Contract Act.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_observability(request: Request, call_next):
    request_id = telemetry.new_request_id()
    request.state.request_id = request_id
    started = telemetry.start()
    try:
        response = await call_next(request)
        return response
    finally:
        count = telemetry.record_request()
        telemetry.log(
            event="http_request",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            status=getattr(locals().get("response"), "status_code", 500),
            latency_ms=telemetry.elapsed_ms(started),
            requests_today=count,
        )


def check_api_key(x_api_key: str | None) -> None:
    if settings.api_key and x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "legal-rag-api"}


@app.get("/api/v1/acts", response_model=ActsResponse)
def acts(x_api_key: str | None = Header(default=None)) -> ActsResponse:
    check_api_key(x_api_key)
    return ActsResponse(acts=[
        ActOption(id="all", name="All legal sources"),
        ActOption(id="bns", name="Bharatiya Nyaya Sanhita, 2023"),
        ActOption(id="ica", name="Indian Contract Act, 1872"),
    ])


@app.post("/api/v1/chat", response_model=ChatResponse)
def chat(payload: ChatRequest, request: Request, x_api_key: str | None = Header(default=None)) -> ChatResponse:
    check_api_key(x_api_key)
    request_id = request.state.request_id
    started = time.perf_counter()

    result = rag.answer(payload.question.strip(), payload.act)
    requests_today = telemetry._requests

    telemetry.log(
        event="rag_request",
        request_id=request_id,
        conversation_id=payload.conversation_id,
        act=payload.act,
        retrieved_sources=len(result.sources),
        embedding_input_tokens=result.embedding_input_tokens,
        gpt_input_tokens=result.gpt_input_tokens,
        gpt_output_tokens=result.gpt_output_tokens,
        content_filtered=result.content_filtered,
        latency_ms=round((time.perf_counter() - started) * 1000, 2),
        estimated_cost_usd=round(
            result.embedding_input_tokens / 1_000_000 * settings.embedding_usd_per_1m_tokens
            + result.gpt_input_tokens / 1_000_000 * settings.gpt_input_usd_per_1m_tokens
            + result.gpt_output_tokens / 1_000_000 * settings.gpt_output_usd_per_1m_tokens
            + settings.search_usd_per_operation,
            8,
        ),
    )

    return ChatResponse(
        request_id=request_id,
        conversation_id=payload.conversation_id,
        answer=result.answer,
        insufficient_evidence=result.insufficient_evidence,
        citations=result.citations,
        sources=result.sources,
        retrieval=RetrievalInfo(
            candidate_k=settings.candidate_k,
            final_k=settings.final_k,
            act_filter=payload.act,
            returned_sources=len(result.sources),
        ),
        usage=Usage(
            request_id=request_id,
            embedding_input_tokens=result.embedding_input_tokens,
            gpt_input_tokens=result.gpt_input_tokens,
            gpt_output_tokens=result.gpt_output_tokens,
        content_filtered=result.content_filtered,
            requests_today=requests_today,
        ),
        disclaimer=DISCLAIMER,
    )
