from __future__ import annotations

import re
from openai import AzureOpenAI

from ..core.config import Settings

CITATION_PATTERN = re.compile(r"\[(BNS|ICA)\s+§([0-9A-Za-z-]+)\]")

SYSTEM_PROMPT = """You are a legal RAG answer generator for a system containing the Bharatiya Nyaya Sanhita, 2023 (BNS) and the Indian Contract Act, 1872 (ICA).

Answer the user's question using ONLY the retrieved legal evidence supplied in the prompt.

GROUNDING RULES
1. Do not use outside legal knowledge to fill gaps.
2. Do not invent, infer, or fabricate a statutory provision.
3. If the retrieved evidence is insufficient, say so clearly. Do not guess.
4. You may synthesize multiple retrieved sources when they support that synthesis.
5. Preserve the legal meaning of the supplied text.
6. Do not confuse BNS provisions with ICA provisions.
7. Do not claim that a provision says something unless the retrieved text supports it.

CITATION RULES
1. Every substantive legal claim must have an inline citation.
2. Use ONLY citation labels supplied with the retrieved sources.
3. Citation format must be exactly like [BNS §103] or [ICA §73].
4. Do not invent section numbers or citation labels.
5. Put citations immediately after the claim they support.
6. If multiple provisions support a statement, cite each relevant provision.

ANSWER STYLE
- Answer the actual question directly.
- Be concise but complete.
- Use short bullets when useful.
- Do not mention retrieval, embeddings, vector databases, prompts, or internal mechanics.
- The retrieved text is authoritative for this response. If it is insufficient, say so.
"""

DISCLAIMER = "This application provides answers grounded in the supplied legal documents for informational and research purposes and is not a substitute for professional legal advice."


def citation_label(doc: dict) -> str:
    act = str(doc.get("act") or "")
    short = "BNS" if "Bharatiya Nyaya" in act else "ICA" if "Contract Act" in act else act or "SOURCE"
    section = doc.get("section_number")
    return f"[{short} §{section}]" if section not in (None, "") else f"[{short}]"


def build_context(documents: list[dict]) -> str:
    blocks = []
    for i, doc in enumerate(documents, start=1):
        blocks.append(
            f"""SOURCE {i}
Citation: {citation_label(doc)}
Act: {doc.get('act') or ''}
Chapter: {doc.get('chapter_number') or ''} — {doc.get('chapter_title') or ''}
Section: {doc.get('section_number') or ''} — {doc.get('section_title') or ''}
Subsection: {doc.get('subsection') or ''}
Content type: {doc.get('content_type') or ''}

Legal text:
{doc.get('content') or ''}
"""
        )
    return "\n---\n".join(blocks)


class GenerationService:
    def __init__(self, settings: Settings, client: AzureOpenAI):
        self.settings = settings
        self.client = client

    def generate(self, question: str, documents: list[dict]) -> tuple[str, int, int]:
        if not documents:
            return (
                "I could not retrieve relevant legal provisions from the supplied legal sources, so I cannot provide a grounded answer.",
                0,
                0,
            )
        user_prompt = f"""USER QUESTION\n{question}\n\nRETRIEVED LEGAL EVIDENCE\n{build_context(documents)}\n\nINSTRUCTIONS\nAnswer using only the retrieved evidence above. Cite every substantive legal claim using the exact citation labels supplied above."""
        response = self.client.chat.completions.create(
            model=self.settings.gpt_deployment,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            max_completion_tokens=self.settings.gpt_max_completion_tokens,
        )
        answer = response.choices[0].message.content
        if not answer:
            raise RuntimeError("GPT returned an empty response")
        usage = getattr(response, "usage", None)
        input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        return answer.strip(), input_tokens, output_tokens
