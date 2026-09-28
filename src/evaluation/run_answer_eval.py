from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
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
VECTOR_DIMENSIONS = int(os.getenv("AZURE_OPENAI_EMBEDDING_DIMENSIONS", "1536"))
OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")
GPT_MAX_TOKENS = int(os.getenv("GPT_MAX_TOKENS", "1800"))
JUDGE_MAX_TOKENS = int(os.getenv("ANSWER_JUDGE_MAX_TOKENS", "1600"))

GOLDEN_SET_PATH = PROJECT_ROOT / "src" / "evaluation" / "retrieval_golden_dataset_v1.json"
OUTPUT_DIR = PROJECT_ROOT / "src" / "evaluation" / "answer_eval"
DEFAULT_RESULTS = OUTPUT_DIR / "answer_eval_results.jsonl"
SUMMARY_JSON = OUTPUT_DIR / "answer_eval_summary.json"
SUMMARY_CSV = OUTPUT_DIR / "answer_eval_summary.csv"

GOLDEN_SET_CANDIDATES = [
    GOLDEN_SET_PATH,
    PROJECT_ROOT / "data" / "retrieval_golden_set" / "retrieval_golden_dataset_v1.json",
    PROJECT_ROOT / "retrieval_golden_set" / "retrieval_golden_dataset_v1.json",
]


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing {name} in .env")
    return value


AZURE_OPENAI_ENDPOINT = require_env("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = require_env("AZURE_OPENAI_API_KEY")
EMBEDDING_DEPLOYMENT = require_env("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
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

SEARCH_FIELDS = [
    "act",
    "chapter_title",
    "section_number",
    "section_title",
    "subsection",
    "content",
    "citation_text",
]

SELECT_FIELDS = [
    "chunk_id",
    "act",
    "chapter_number",
    "chapter_title",
    "section_number",
    "section_title",
    "subsection",
    "content_type",
    "content",
    "citation_id",
    "citation_text",
    "source_file",
]

CITATION_PATTERN = re.compile(r"\[(BNS|ICA)\s+§([0-9A-Za-z-]+)\]")


def resolve_golden_path(explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Golden set not found: {path}")
        return path

    for path in GOLDEN_SET_CANDIDATES:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Could not find retrieval_golden_dataset_v1.json. "
        "Pass --golden-set /path/to/retrieval_golden_dataset_v1.json."
    )


def load_questions(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    questions: list[dict[str, Any]] = []

    if isinstance(data.get("questions"), list):
        questions.extend(data["questions"])
    elif isinstance(data.get("acts"), dict):
        for act_block in data["acts"].values():
            questions.extend(act_block.get("questions", []))
    else:
        raise ValueError("Unsupported golden dataset structure.")

    if not questions:
        raise ValueError("Golden dataset contains no questions.")

    for q in questions:
        for key in ("id", "act", "query", "gold"):
            if key not in q:
                raise ValueError(f"Golden question missing '{key}': {q}")

    ids = [q["id"] for q in questions]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate golden question IDs found.")

    return questions


def gold_sections(question: dict[str, Any]) -> set[str]:
    return {str(item["section_number"]) for item in question["gold"]}


def direct_gold_sections(question: dict[str, Any]) -> set[str]:
    return {
        str(item["section_number"])
        for item in question["gold"]
        if int(item.get("relevance_grade", 0)) == 2
    }


def embed_query(question: str) -> list[float]:
    response = openai_client.embeddings.create(
        model=EMBEDDING_DEPLOYMENT,
        input=question,
        dimensions=VECTOR_DIMENSIONS,
    )
    vector = response.data[0].embedding
    if len(vector) != VECTOR_DIMENSIONS:
        raise RuntimeError(
            f"Query embedding dimension mismatch: expected {VECTOR_DIMENSIONS}, got {len(vector)}"
        )
    return vector


def citation_label(doc: dict[str, Any]) -> str:
    act = str(doc.get("act") or "").strip()
    if "Bharatiya Nyaya" in act:
        short = "BNS"
    elif "Contract Act" in act:
        short = "ICA"
    else:
        short = act or "SOURCE"
    section = doc.get("section_number")
    return f"[{short} §{section}]" if section not in (None, "") else f"[{short}]"


def hybrid_retrieve(question: str) -> list[dict[str, Any]]:
    query_vector = embed_query(question)
    vector_query = VectorizedQuery(
        vector=query_vector,
        k_nearest_neighbors=CANDIDATE_K,
        fields="content_vector",
    )

    raw_results = search_client.search(
        search_text=question,
        vector_queries=[vector_query],
        select=SELECT_FIELDS,
        top=CANDIDATE_K,
    )

    results: list[dict[str, Any]] = []
    for rank, result in enumerate(raw_results, start=1):
        item = dict(result)
        item["rank"] = rank
        item["search_score"] = result.get("@search.score")
        results.append(item)

    return results[:FINAL_K]


def build_context(documents: list[dict[str, Any]]) -> str:
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


def generate_answer(question: str, documents: list[dict[str, Any]]) -> str:
    if not documents:
        return "I could not retrieve relevant legal provisions from the supplied legal sources, so I cannot provide a grounded answer."

    user_prompt = f"""USER QUESTION
{question}

RETRIEVED LEGAL EVIDENCE
{build_context(documents)}

INSTRUCTIONS
Answer using only the retrieved evidence above. Cite every substantive legal claim using the exact citation labels supplied above.
"""

    response = openai_client.chat.completions.create(
        model=GPT_DEPLOYMENT,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        max_completion_tokens=GPT_MAX_TOKENS,
    )
    answer = response.choices[0].message.content
    if not answer:
        raise RuntimeError("GPT returned an empty response.")
    return answer.strip()

def answer_citations(answer: str) -> list[str]:
    return list(dict.fromkeys(
        f"[{act} §{section}]" for act, section in CITATION_PATTERN.findall(answer)
    ))


def allowed_citations(documents: list[dict[str, Any]]) -> set[str]:
    return {citation_label(doc) for doc in documents}


def citation_section_set(citations: list[str]) -> set[str]:
    return {m.group(2) for c in citations if (m := CITATION_PATTERN.fullmatch(c))}


def deterministic_metrics(
    question: dict[str, Any],
    documents: list[dict[str, Any]],
    answer: str,
) -> dict[str, Any]:
    gold = gold_sections(question)
    direct_gold = direct_gold_sections(question)
    retrieved = {
        str(doc.get("section_number"))
        for doc in documents
        if doc.get("section_number") not in (None, "")
    }

    citations = answer_citations(answer)
    allowed = allowed_citations(documents)
    unsupported = [c for c in citations if c not in allowed]
    cited_sections = citation_section_set(citations)

    substantive_claim_text = re.sub(r"\[[A-Z]+\s+§[0-9A-Za-z-]+\]", "", answer).strip()
    citation_near_claim = bool(citations) and bool(substantive_claim_text)

    return {
        "retrieved_section_ids": sorted(retrieved),
        "gold_section_ids": sorted(gold),
        "direct_gold_section_ids": sorted(direct_gold),
        "gold_section_coverage": round(len(retrieved & gold) / len(gold), 4) if gold else 1.0,
        "direct_gold_coverage": round(len(retrieved & direct_gold) / len(direct_gold), 4) if direct_gold else 1.0,
        "all_direct_gold_retrieved": direct_gold <= retrieved,
        "citations": citations,
        "unsupported_citations": unsupported,
        "citation_syntax_valid": all(c in allowed for c in citations),
        "all_citations_from_retrieved_sources": all(c in allowed for c in citations),
        "cited_retrieved_section_ids": sorted(cited_sections),
        "citation_count": len(citations),
        "has_substantive_answer_text": bool(substantive_claim_text),
        "has_citation_with_answer": citation_near_claim,
    }

JUDGE_SYSTEM_PROMPT = """You are an evaluator for a legal RAG system.

Evaluate the generated answer ONLY against:
1. the user's question,
2. the retrieved legal evidence, and
3. the supplied gold section labels.

Do not use outside legal knowledge to fill gaps. The gold labels identify sections expected to be relevant; they are not themselves proof of what the law says.

Score each dimension from 0 to 2:

answer_relevance:
0 = does not answer the question / off-topic
1 = partially answers or materially misses the target
2 = directly answers the question

answer_correctness:
0 = materially incorrect legal statement
1 = mostly correct but has a material omission, ambiguity, or minor error
2 = correct based on the supplied evidence

groundedness:
0 = substantive claims are unsupported by retrieved text or rely on outside knowledge
1 = partly grounded but includes unsupported/weakly supported claims
2 = substantive claims are supported by retrieved text

answer_completeness:
0 = misses the central requested information
1 = covers the main point but misses important supported details
2 = covers the important answerable points in the evidence

citation_correctness:
0 = citations are absent, fabricated, or materially mismatch the claims
1 = citations are generally relevant but one or more claims are weakly/mismatched
2 = citations used in the answer point to retrieved sources that support the cited claims

citation_completeness:
0 = important substantive claims lack citations
1 = most substantive claims are cited, but some important claims are not
2 = substantive legal claims are appropriately cited

context_utilization:
0 = ignores or contradicts useful retrieved evidence
1 = uses some useful evidence but misses important retrieved support
2 = appropriately uses the retrieved evidence needed to answer

abstention_quality:
0 = should have abstained from an unsupported/insufficient answer but did not, or abstained despite sufficient evidence
1 = cautious but imperfect handling of evidence limits
2 = correctly answers when supported or clearly abstains when evidence is insufficient

hallucination:
0 = no material unsupported legal claims
1 = minor unsupported/invented detail
2 = material unsupported/invented legal claim

For abstention_quality, judge whether the answer handled evidence sufficiency correctly. Do not penalize an answer for refusing to use outside knowledge.

Return ONLY valid JSON matching the requested schema. Keep reasons concise and evidence-based.
"""


def is_content_filter_error(exc: Exception) -> bool:
    """Return True for Azure OpenAI content-management policy rejections."""
    text = str(exc)
    return (
        "content_filter" in text
        or "ResponsibleAIPolicyViolation" in text
        or "content management policy" in text.lower()
    )


def judge_answer(
    question: dict[str, Any],
    documents: list[dict[str, Any]],
    answer: str,
    deterministic: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        "question_id": question["id"],
        "question": question["query"],
        "act": question["act"],
        "gold_sections": question["gold"],
        "retrieved_sources": [
            {
                "rank": d.get("rank"),
                "citation": citation_label(d),
                "section_number": d.get("section_number"),
                "section_title": d.get("section_title"),
                "content": d.get("content"),
            }
            for d in documents
        ],
        "generated_answer": answer,
        "deterministic_checks": deterministic,
    }

    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "answer_relevance": {"type": "integer", "enum": [0, 1, 2]},
            "answer_correctness": {"type": "integer", "enum": [0, 1, 2]},
            "groundedness": {"type": "integer", "enum": [0, 1, 2]},
            "answer_completeness": {"type": "integer", "enum": [0, 1, 2]},
            "citation_correctness": {"type": "integer", "enum": [0, 1, 2]},
            "citation_completeness": {"type": "integer", "enum": [0, 1, 2]},
            "context_utilization": {"type": "integer", "enum": [0, 1, 2]},
            "abstention_quality": {"type": "integer", "enum": [0, 1, 2]},
            "hallucination": {"type": "integer", "enum": [0, 1, 2]},
            "overall_note": {"type": "string"},
            "error_tags": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "answer_relevance",
            "answer_correctness",
            "groundedness",
            "answer_completeness",
            "citation_correctness",
            "citation_completeness",
            "context_utilization",
            "abstention_quality",
            "hallucination",
            "overall_note",
            "error_tags",
        ],
    }

    try:
        response = openai_client.chat.completions.create(
            model=GPT_DEPLOYMENT,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "Evaluate this single answer.\n\n"
                        + json.dumps(payload, ensure_ascii=False)
                    ),
                },
            ],
            max_completion_tokens=JUDGE_MAX_TOKENS,
            response_format={"type": "json_schema", "json_schema": {"name": "answer_eval", "strict": True, "schema": schema}},
        )
    except Exception as exc:
        if is_content_filter_error(exc):
            return {
                "judge_status": "content_filtered",
                "judge_error": "Azure OpenAI content management policy filtered the judge prompt.",
                "judge_error_detail": str(exc),
            }
        raise

    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("Judge returned an empty response.")

    try:
        result = json.loads(content)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Judge returned invalid JSON: {content}") from exc

    return result


def load_completed(path: Path) -> dict[str, dict[str, Any]]:
    completed: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return completed
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            completed[row["question_id"]] = row
    return completed


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    numeric_metrics = [
        "answer_relevance",
        "answer_correctness",
        "groundedness",
        "answer_completeness",
        "citation_correctness",
        "citation_completeness",
        "context_utilization",
        "abstention_quality",
        "hallucination",
    ]

    summary: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "questions_evaluated": len(rows),
        "baseline": {
            "retrieval": "Hybrid + RRF",
            "candidate_k": CANDIDATE_K,
            "final_k": FINAL_K,
            "mmr": False,
            "generator_deployment": GPT_DEPLOYMENT,
        },
        "metrics": {},
        "flags": {},
    }

    for metric in numeric_metrics:
        values = [r["judge"][metric] for r in rows if metric in r.get("judge", {})]
        summary["metrics"][metric] = {
            "mean": mean(values),
            "max": 2,
            "mean_percent_of_max": round(100 * sum(values) / (2 * len(values)), 2) if values else None,
        }

    summary["metrics"]["gold_section_coverage"] = {
        "mean": mean([r["deterministic"]["gold_section_coverage"] for r in rows])
    }
    summary["metrics"]["direct_gold_coverage"] = {
        "mean": mean([r["deterministic"]["direct_gold_coverage"] for r in rows])
    }

    judged_rows = [
        r for r in rows
        if r.get("judge", {}).get("answer_correctness") is not None
    ]
    answer_rows = [
        r for r in rows
        if r.get("generation", {}).get("status") == "success"
    ]

    summary["flags"] = {
        "retrieval_direct_gold_not_fully_retrieved": sum(
            not r["deterministic"]["all_direct_gold_retrieved"] for r in rows
        ),
        "unsupported_citations": sum(
            bool(r["deterministic"].get("unsupported_citations", [])) for r in answer_rows
        ),
        "material_hallucination": sum(
            r.get("judge", {}).get("hallucination") == 2 for r in judged_rows
        ),
        "correctness_zero": sum(
            r.get("judge", {}).get("answer_correctness") == 0 for r in judged_rows
        ),
        "groundedness_zero": sum(
            r.get("judge", {}).get("groundedness") == 0 for r in judged_rows
        ),
        "generation_content_filtered": sum(
            r.get("generation", {}).get("status") == "content_filtered" for r in rows
        ),
        "generation_successful": sum(
            r.get("generation", {}).get("status") == "success" for r in rows
        ),
        "judge_content_filtered": sum(
            r.get("judge", {}).get("judge_status") == "content_filtered" for r in rows
        ),
        "judge_successful": len(judged_rows),
    }

    sufficient = [
        r for r in rows
        if r["deterministic"]["all_direct_gold_retrieved"]
        and r.get("judge", {}).get("answer_correctness") is not None
    ]
    insufficient = [
        r for r in rows
        if not r["deterministic"]["all_direct_gold_retrieved"]
        and r.get("judge", {}).get("answer_correctness") is not None
    ]
    summary["diagnostics"] = {
        "retrieval_sufficient_judged_questions": len(sufficient),
        "retrieval_insufficient_judged_questions": len(insufficient),
        "mean_correctness_when_direct_gold_retrieved": mean(
            [r["judge"]["answer_correctness"] for r in sufficient]
        ),
        "mean_groundedness_when_direct_gold_retrieved": mean(
            [r["judge"]["groundedness"] for r in sufficient]
        ),
        "mean_correctness_when_direct_gold_missing": mean(
            [r["judge"]["answer_correctness"] for r in insufficient]
        ),
        "mean_groundedness_when_direct_gold_missing": mean(
            [r["judge"]["groundedness"] for r in insufficient]
        ),
    }

    return summary


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "question_id", "act", "query",
        "generation_status", "judge_status",
        "gold_section_coverage", "direct_gold_coverage",
        "all_direct_gold_retrieved", "citation_count",
        "unsupported_citations",
        "answer_relevance", "answer_correctness", "groundedness",
        "answer_completeness", "citation_correctness", "citation_completeness",
        "context_utilization", "abstention_quality", "hallucination",
        "overall_note", "error_tags",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            j = r.get("judge", {})
            d = r["deterministic"]
            writer.writerow({
                "question_id": r["question_id"],
                "act": r["act"],
                "query": r["query"],
                "generation_status": r.get("generation", {}).get("status", "unknown"),
                "judge_status": j.get("judge_status", "success"),
                "gold_section_coverage": d["gold_section_coverage"],
                "direct_gold_coverage": d["direct_gold_coverage"],
                "all_direct_gold_retrieved": d["all_direct_gold_retrieved"],
                "citation_count": d["citation_count"],
                "unsupported_citations": ";".join(d["unsupported_citations"]),
                **{k: j.get(k) for k in fields if k in j},
                "overall_note": j.get("overall_note", ""),
                "error_tags": ";".join(j.get("error_tags", [])),
            })


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate answer quality for the frozen Legal RAG baseline."
    )
    parser.add_argument("--golden-set", help="Path to retrieval_golden_dataset_v1.json")
    parser.add_argument("--output", default=str(DEFAULT_RESULTS), help="JSONL result path")
    parser.add_argument("--summary-json", default=str(SUMMARY_JSON))
    parser.add_argument("--summary-csv", default=str(SUMMARY_CSV))
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N questions")
    parser.add_argument("--question-id", action="append", help="Evaluate only this question ID; repeatable")
    parser.add_argument("--no-resume", action="store_true", help="Ignore existing JSONL results and rerun selected questions")
    args = parser.parse_args()

    golden_path = resolve_golden_path(args.golden_set)
    questions = load_questions(golden_path)

    if args.question_id:
        wanted = set(args.question_id)
        questions = [q for q in questions if q["id"] in wanted]
        missing = wanted - {q["id"] for q in questions}
        if missing:
            raise ValueError(f"Question IDs not found: {sorted(missing)}")

    if args.limit is not None:
        questions = questions[:args.limit]

    results_path = Path(args.output).resolve()
    completed = {} if args.no_resume else load_completed(results_path)

    print("=" * 80)
    print("LEGAL RAG ANSWER-LEVEL EVALUATION")
    print("=" * 80)
    print(f"Golden set       : {golden_path}")
    print(f"Questions        : {len(questions)}")
    print(f"Retrieval        : Hybrid + RRF")
    print(f"Candidate K      : {CANDIDATE_K}")
    print(f"Final K          : {FINAL_K}")
    print(f"MMR              : False")
    print(f"Generator        : {GPT_DEPLOYMENT}")
    print(f"Results          : {results_path}")
    print()

    evaluated_rows: list[dict[str, Any]] = []

    for index, question in enumerate(questions, start=1):
        qid = question["id"]
        if qid in completed:
            print(f"[{index}/{len(questions)}] {qid} — resume: already evaluated")
            evaluated_rows.append(completed[qid])
            continue

        print(f"[{index}/{len(questions)}] {qid} — {question['query']}")
        try:
            documents = hybrid_retrieve(question["query"])

            try:
                answer = generate_answer(question["query"], documents)
            except Exception as exc:
                if is_content_filter_error(exc):
                    deterministic = deterministic_metrics(question, documents, "")
                    row = {
                        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
                        "question_id": qid,
                        "act": question["act"],
                        "query": question["query"],
                        "gold": question["gold"],
                        "retrieved": [
                            {
                                "rank": d.get("rank"),
                                "chunk_id": d.get("chunk_id"),
                                "citation": citation_label(d),
                                "section_number": d.get("section_number"),
                                "section_title": d.get("section_title"),
                                "search_score": d.get("search_score"),
                                "content": d.get("content"),
                            }
                            for d in documents
                        ],
                        "answer": None,
                        "generation": {
                            "status": "content_filtered",
                            "error": str(exc),
                        },
                        "deterministic": deterministic,
                        "judge": {
                            "judge_status": "not_run",
                            "judge_error": "Generation was content-filtered; no answer was available to judge.",
                        },
                        "baseline_config": {
                            "retrieval": "hybrid_rrf",
                            "candidate_k": CANDIDATE_K,
                            "final_k": FINAL_K,
                            "mmr": False,
                        },
                    }
                    append_jsonl(results_path, row)
                    evaluated_rows.append(row)
                    print(
                        "  generation=CONTENT_FILTERED (recorded as N/A; continuing) "
                        f"direct_gold_coverage={deterministic['direct_gold_coverage']}"
                    )
                    continue
                raise

            deterministic = deterministic_metrics(question, documents, answer)
            judge = judge_answer(question, documents, answer, deterministic)

            row = {
                "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
                "question_id": qid,
                "act": question["act"],
                "query": question["query"],
                "gold": question["gold"],
                "retrieved": [
                    {
                        "rank": d.get("rank"),
                        "chunk_id": d.get("chunk_id"),
                        "citation": citation_label(d),
                        "section_number": d.get("section_number"),
                        "section_title": d.get("section_title"),
                        "search_score": d.get("search_score"),
                        "content": d.get("content"),
                    }
                    for d in documents
                ],
                "answer": answer,
                "generation": {"status": "success"},
                "deterministic": deterministic,
                "judge": judge,
                "baseline_config": {
                    "retrieval": "hybrid_rrf",
                    "candidate_k": CANDIDATE_K,
                    "final_k": FINAL_K,
                    "mmr": False,
                },
            }
            append_jsonl(results_path, row)
            evaluated_rows.append(row)

            if judge.get("judge_status") == "content_filtered":
                print(
                    "  judge=CONTENT_FILTERED (recorded as N/A; continuing) "
                    f"direct_gold_coverage={deterministic['direct_gold_coverage']}"
                )
            else:
                print(
                    f"  correctness={judge['answer_correctness']} "
                    f"groundedness={judge['groundedness']} "
                    f"citation={judge['citation_correctness']} "
                    f"hallucination={judge['hallucination']} "
                    f"direct_gold_coverage={deterministic['direct_gold_coverage']}"
                )
        except Exception as exc:
            print(f"  ERROR: {exc}", file=sys.stderr)
            raise


    summary = aggregate(evaluated_rows)
    summary["golden_set_path"] = str(golden_path)
    summary["results_path"] = str(results_path)

    summary_json_path = Path(args.summary_json).resolve()
    summary_json_path.parent.mkdir(parents=True, exist_ok=True)
    summary_json_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    summary_csv_path = Path(args.summary_csv).resolve()
    write_summary_csv(summary_csv_path, evaluated_rows)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nWrote: {results_path}")
    print(f"Wrote: {summary_json_path}")
    print(f"Wrote: {summary_csv_path}")


if __name__ == "__main__":
    main()
