from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    act: Literal["all", "bns", "ica"] = "all"
    conversation_id: str | None = Field(default=None, max_length=128)


class Citation(BaseModel):
    citation: str
    citation_id: str
    act: str
    section_number: str | None = None
    section_title: str | None = None


class SourceCard(BaseModel):
    source_id: str
    citation: str
    act: str
    chapter_number: str | None = None
    chapter_title: str | None = None
    section_number: str | None = None
    section_title: str | None = None
    subsection: str | None = None
    content_type: str | None = None
    content: str
    citation_text: str | None = None


class Usage(BaseModel):
    request_id: str
    embedding_input_tokens: int = 0
    gpt_input_tokens: int = 0
    gpt_output_tokens: int = 0
    requests_today: int = 0


class RetrievalInfo(BaseModel):
    strategy: str = "hybrid_rrf"
    candidate_k: int
    final_k: int
    act_filter: str
    returned_sources: int


class ChatResponse(BaseModel):
    request_id: str
    conversation_id: str | None = None
    answer: str
    insufficient_evidence: bool = False
    citations: list[Citation] = []
    sources: list[SourceCard] = []
    retrieval: RetrievalInfo
    disclaimer: str


class ActOption(BaseModel):
    id: str
    name: str


class ActsResponse(BaseModel):
    acts: list[ActOption]
