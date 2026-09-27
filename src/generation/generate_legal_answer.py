from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

CANDIDATE_K = 10
FINAL_K = 5
EMBEDDING_DIMENSIONS = int(os.getenv("AZURE_OPENAI_EMBEDDING_DIMENSIONS", "1536"))
OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")
GPT_TEMPERATURE = float(os.getenv("GPT_TEMPERATURE", "0"))
GPT_MAX_TOKENS = int(os.getenv("GPT_MAX_TOKENS", "1800"))


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing {name} in .env")
    return value


AZURE_OPENAI_ENDPOINT = require_env("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = require_env("AZURE_OPENAI_API_KEY")
EMBEDDING_DEPLOYMENT = require_env("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
EMBEDDING_MODEL = os.getenv("AZURE_OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
GPT_DEPLOYMENT = require_env("AZURE_OPENAI_GPT_DEPLOYMENT")
AZURE_SEARCH_ENDPOINT = require_env("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_ADMIN_KEY = require_env("AZURE_SEARCH_ADMIN_KEY")
AZURE_SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME", "legal-rag-search-vectors")

openai_client = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version=OPENAI_API_VERSION,
)

search_client = SearchClient(
    endpoint=AZURE_SEARCH_ENDPOINT,
    index_name=AZURE_SEARCH_INDEX_NAME,
    credential=AzureKeyCredential(AZURE_SEARCH_ADMIN_KEY),
)

SEARCH_SELECT_FIELDS = [
    "chunk_id", "act", "chapter_number", "chapter_title", "section_number",
    "section_title", "subsection", "content_type", "content", "citation_id",
    "citation_text", "source_file",
]


def embed_query(question: str) -> list[float]:
    response = openai_client.embeddings.create(
        model=EMBEDDING_DEPLOYMENT,
        input=question,
        dimensions=EMBEDDING_DIMENSIONS,
    )
    vector = response.data[0].embedding
    if len(vector) != EMBEDDING_DIMENSIONS:
        raise RuntimeError(
            f"Query embedding dimension mismatch: expected {EMBEDDING_DIMENSIONS}, got {len(vector)}"
        )
    return vector


def hybrid_retrieve(question: str, candidate_k: int = CANDIDATE_K, final_k: int = FINAL_K) -> list[dict[str, Any]]:
    if candidate_k < final_k:
        raise ValueError("candidate_k must be >= final_k")

    query_vector = embed_query(question)
    vector_query = VectorizedQuery(
        vector=query_vector,
        k_nearest_neighbors=candidate_k,
        fields="content_vector",
    )

    results = search_client.search(
        search_text=question,
        vector_queries=[vector_query],
        select=SEARCH_SELECT_FIELDS,
        top=candidate_k,
    )

    candidates = []
    for rank, result in enumerate(results, start=1):
        candidates.append({
            "rank": rank,
            "search_score": result.get("@search.score"),
            "chunk_id": result.get("chunk_id"),
            "act": result.get("act"),
            "chapter_number": result.get("chapter_number"),
            "chapter_title": result.get("chapter_title"),
            "section_number": result.get("section_number"),
            "section_title": result.get("section_title"),
            "subsection": result.get("subsection"),
            "content_type": result.get("content_type"),
            "content": result.get("content"),
            "citation_id": result.get("citation_id"),
            "citation_text": result.get("citation_text"),
            "source_file": result.get("source_file"),
        })

    return candidates[:final_k]


def citation_label(document: dict[str, Any]) -> str:
    act = str(document.get("act") or "").strip()
    if "Bharatiya Nyaya" in act:
        act_short = "BNS"
    elif "Contract Act" in act:
        act_short = "ICA"
    else:
        act_short = act or "SOURCE"

    section = document.get("section_number")
    return f"[{act_short} §{section}]" if section not in (None, "") else f"[{act_short}]"


def build_grounded_context(documents: list[dict[str, Any]]) -> str:
    blocks = []
    for i, doc in enumerate(documents, start=1):
        blocks.append(f"""SOURCE {i}
Citation: {citation_label(doc)}
Act: {doc.get('act') or ''}
Chapter: {doc.get('chapter_number') or ''} — {doc.get('chapter_title') or ''}
Section: {doc.get('section_number') or ''} — {doc.get('section_title') or ''}
Subsection: {doc.get('subsection') or ''}
Content type: {doc.get('content_type') or ''}

Legal text:
{doc.get('content') or ''}
""")
    return "\n---\n".join(blocks)


SYSTEM_PROMPT = """You are am expert legal RAG answer generator for a system containing the Bharatiya Nyaya Sanhita, 2023 (BNS) and the Indian Contract Act, 1872 (ICA).

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


def generate_answer(question: str, documents: list[dict[str, Any]]) -> str:
    if not documents:
        return "I could not retrieve relevant legal provisions from the supplied legal sources, so I cannot provide a grounded answer."

    user_prompt = f"""USER QUESTION
{question}

RETRIEVED LEGAL EVIDENCE
{build_grounded_context(documents)}

INSTRUCTIONS
Answer using only the retrieved evidence above. Cite every substantive legal claim using the exact citation labels supplied above.
"""

    response = openai_client.chat.completions.create(
        model=GPT_DEPLOYMENT,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        reasoning_effort="none",
        temperature=GPT_TEMPERATURE,
        max_completion_tokens=GPT_MAX_TOKENS,
    )
    answer = response.choices[0].message.content
    if not answer:
        raise RuntimeError("GPT returned an empty response.")
    return answer.strip()


CITATION_PATTERN = re.compile(r"\[(BNS|ICA)\s+§([0-9A-Za-z-]+)\]")


def validate_citations(answer: str, documents: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    allowed = {citation_label(doc) for doc in documents}
    found = CITATION_PATTERN.findall(answer)
    citations = [f"[{act} §{section}]" for act, section in found]
    valid = list(dict.fromkeys(c for c in citations if c in allowed))
    unsupported = list(dict.fromkeys(c for c in citations if c not in allowed))
    return valid, unsupported


def print_sources(documents: list[dict[str, Any]]) -> None:
    print("\n" + "=" * 80)
    print("RETRIEVED SOURCES")
    print("=" * 80)
    for doc in documents:
        print(
            f"{doc['rank']}. {citation_label(doc)} | "
            f"{doc.get('section_title') or ''} | score={doc.get('search_score')}"
        )


def run(question: str) -> None:
    print("\nRetrieving legal evidence...")
    print(f"Strategy: Hybrid + RRF | candidate_k={CANDIDATE_K} | final_k={FINAL_K}")
    documents = hybrid_retrieve(question)

    if not documents:
        print("\nNo relevant documents were retrieved.")
        return

    print_sources(documents)
    print("\nGenerating grounded answer...")
    answer = generate_answer(question, documents)
    valid_citations, unsupported_citations = validate_citations(answer, documents)

    print("\n" + "=" * 80)
    print("ANSWER")
    print("=" * 80)
    print(answer)

    print("\n" + "=" * 80)
    print("CITATION CHECK")
    print("=" * 80)
    if valid_citations:
        print("Citations used:")
        for citation in valid_citations:
            print(f"  ✓ {citation}")
    else:
        print("  No recognized citations found in the answer.")
    if unsupported_citations:
        print("\nWARNING: Unsupported citations detected:")
        for citation in unsupported_citations:
            print(f"  ! {citation}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate grounded legal answers using Azure AI Search + GPT.")
    parser.add_argument("--question", type=str, help="Legal question to answer.")
    args = parser.parse_args()

    try:
        question = args.question or input("\nEnter your legal question: ").strip()
        if not question:
            raise ValueError("Question cannot be empty.")
        run(question)
    except KeyboardInterrupt:
        print("\nCancelled.")
        sys.exit(130)
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
