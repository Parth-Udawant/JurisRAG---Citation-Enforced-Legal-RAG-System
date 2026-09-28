from __future__ import annotations

from dataclasses import dataclass

from openai import AzureOpenAI, BadRequestError

from ..core.config import Settings
from ..models.schemas import Citation, SourceCard
from .generation_service import CITATION_PATTERN, DISCLAIMER, GenerationService, citation_label
from .search_service import SearchService


@dataclass
class RAGResult:
    answer: str
    citations: list[Citation]
    sources: list[SourceCard]
    insufficient_evidence: bool
    embedding_input_tokens: int
    gpt_input_tokens: int
    gpt_output_tokens: int
    content_filtered: bool = False


class RAGService:
    def __init__(self, settings: Settings):
        client = AzureOpenAI(
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key,
            api_version=settings.azure_openai_api_version,
        )
        self.settings = settings
        self.search = SearchService(settings, client)
        self.generation = GenerationService(settings, client)

    def answer(self, question: str, act: str) -> RAGResult:
        documents, embedding_tokens = self.search.retrieve(question, act)
        try:
            answer, gpt_input, gpt_output = self.generation.generate(question, documents)
        except BadRequestError as exc:
            text = str(exc).lower()
            if "content_filter" in text or "responsibleaipolicyviolation" in text or "content management policy" in text:
                return RAGResult(
                    answer="The requested legal response could not be generated because the model safety filter blocked this request. No legal conclusion is provided.",
                    citations=[],
                    sources=[
                        SourceCard(
                            source_id=str(d.get("citation_id") or d.get("chunk_id")),
                            citation=citation_label(d),
                            act=str(d.get("act") or ""),
                            chapter_number=str(d.get("chapter_number")) if d.get("chapter_number") not in (None, "") else None,
                            chapter_title=d.get("chapter_title"),
                            section_number=str(d.get("section_number")) if d.get("section_number") not in (None, "") else None,
                            section_title=d.get("section_title"),
                            subsection=d.get("subsection"),
                            content_type=d.get("content_type"),
                            content=str(d.get("content") or ""),
                            citation_text=d.get("citation_text"),
                        )
                        for d in documents
                    ],
                    insufficient_evidence=False,
                    embedding_input_tokens=embedding_tokens,
                    gpt_input_tokens=0,
                    gpt_output_tokens=0,
                    content_filtered=True,
                )
            raise

        allowed = {citation_label(d): d for d in documents}
        citations: list[Citation] = []
        seen: set[str] = set()
        for act_code, section in CITATION_PATTERN.findall(answer):
            label = f"[{act_code} §{section}]"
            if label in seen or label not in allowed:
                continue
            seen.add(label)
            d = allowed[label]
            citations.append(
                Citation(
                    citation=label,
                    citation_id=str(d.get("citation_id") or d.get("chunk_id")),
                    act=str(d.get("act") or ""),
                    section_number=str(d.get("section_number")) if d.get("section_number") not in (None, "") else None,
                    section_title=d.get("section_title"),
                )
            )

        all_answer_citations = {f"[{a} §{s}]" for a, s in CITATION_PATTERN.findall(answer)}
        unsupported = all_answer_citations - set(allowed)
        if unsupported:
            answer = "I could not provide a grounded answer from the supplied legal evidence because the generated response contained a citation that was not among the retrieved sources."
            citations = []
            insufficient_evidence = True
        else:
            insufficient_evidence = (
                not documents
                or answer.lower().startswith("i could not retrieve relevant legal provisions")
                or "retrieved evidence is insufficient" in answer.lower()
                or "cannot provide a grounded answer" in answer.lower()
            )

        sources = [
            SourceCard(
                source_id=str(d.get("citation_id") or d.get("chunk_id")),
                citation=citation_label(d),
                act=str(d.get("act") or ""),
                chapter_number=str(d.get("chapter_number")) if d.get("chapter_number") not in (None, "") else None,
                chapter_title=d.get("chapter_title"),
                section_number=str(d.get("section_number")) if d.get("section_number") not in (None, "") else None,
                section_title=d.get("section_title"),
                subsection=d.get("subsection"),
                content_type=d.get("content_type"),
                content=str(d.get("content") or ""),
                citation_text=d.get("citation_text"),
            )
            for d in documents
        ]

        return RAGResult(answer, citations, sources, insufficient_evidence, embedding_tokens, gpt_input, gpt_output, False)
