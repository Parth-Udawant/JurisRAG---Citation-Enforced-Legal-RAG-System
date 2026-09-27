from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from dotenv import load_dotenv
from openai import AzureOpenAI


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

GOLDEN_SET_PATH = Path(
    os.getenv(
        "RETRIEVAL_GOLDEN_SET",
        str(PROJECT_ROOT / "src" / "evaluation" / "retrieval_golden_dataset_v1.json"),
    )
)

BNS_EMBEDDINGS_PATH = Path(
    os.getenv(
        "BNS_EMBEDDINGS_PATH",
        str(PROJECT_ROOT / "data" / "processed" / "embeddings" / "bns_embeddings.jsonl"),
    )
)

ICA_EMBEDDINGS_PATH = Path(
    os.getenv(
        "ICA_EMBEDDINGS_PATH",
        str(PROJECT_ROOT / "data" / "processed" / "embeddings" / "ica_embeddings.jsonl"),
    )
)

OUTPUT_DIR = Path(
    os.getenv(
        "RETRIEVAL_EVAL_OUTPUT_DIR",
        str(PROJECT_ROOT / "src" / "evaluation" / "retrieval_results"),
    )
)

SEARCH_ENDPOINT = os.environ["AZURE_SEARCH_ENDPOINT"]
SEARCH_ADMIN_KEY = os.environ["AZURE_SEARCH_ADMIN_KEY"]
INDEX_NAME = os.environ["AZURE_SEARCH_INDEX_NAME"]

OPENAI_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]
OPENAI_API_KEY = os.environ["AZURE_OPENAI_API_KEY"]
EMBEDDING_DEPLOYMENT = os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"]

VECTOR_DIMENSIONS = int(
    os.getenv("AZURE_OPENAI_EMBEDDING_DIMENSIONS", "1536")
)

API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")

CANDIDATE_K_VALUES = [10, 20, 50]
FINAL_K_VALUES = [1, 3, 5, 10]

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

search_client = SearchClient(
    endpoint=SEARCH_ENDPOINT,
    index_name=INDEX_NAME,
    credential=AzureKeyCredential(SEARCH_ADMIN_KEY),
)

embedding_client = AzureOpenAI(
    api_key=OPENAI_API_KEY,
    azure_endpoint=OPENAI_ENDPOINT,
    api_version=API_VERSION,
)


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_embeddings(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Embedding file not found: {path}")

    vectors: dict[str, np.ndarray] = {}

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)

            chunk_id = record.get("chunk_id")
            embedding = record.get("embedding")

            if not chunk_id:
                raise ValueError(
                    f"{path}:{line_number}: missing chunk_id"
                )

            if embedding is None:
                raise ValueError(
                    f"{path}:{line_number}: missing embedding"
                )

            if len(embedding) != VECTOR_DIMENSIONS:
                raise ValueError(
                    f"{path}:{line_number}: chunk {chunk_id} has "
                    f"{len(embedding)} dimensions; expected "
                    f"{VECTOR_DIMENSIONS}"
                )

            if chunk_id in vectors:
                raise ValueError(
                    f"{path}:{line_number}: duplicate chunk_id {chunk_id}"
                )

            vectors[chunk_id] = np.asarray(embedding, dtype=np.float32)

    return vectors


def load_golden_questions(path: Path) -> list[dict[str, Any]]:
    dataset = load_json(path)

    questions: list[dict[str, Any]] = []

    acts = dataset.get("acts")
    if not isinstance(acts, dict):
        raise ValueError("Golden dataset must contain an 'acts' object.")

    for act_key, act_data in acts.items():
        for question in act_data.get("questions", []):
            q = dict(question)
            q["_act_key"] = act_key
            questions.append(q)

    if len(questions) != 50:
        raise ValueError(
            f"Expected 50 golden questions, found {len(questions)}"
        )

    return questions


def validate_golden_questions(questions: list[dict[str, Any]]) -> None:
    ids = set()

    for q in questions:
        qid = q.get("id")
        act = q.get("act")
        query = q.get("query")
        gold = q.get("gold")

        if not qid or not act or not query or not isinstance(gold, list):
            raise ValueError(
                f"Invalid golden question record: {q!r}"
            )

        if qid in ids:
            raise ValueError(f"Duplicate golden question id: {qid}")
        ids.add(qid)

        for gold_item in gold:
            if not gold_item.get("section_number"):
                raise ValueError(
                    f"{qid}: gold item missing section_number"
                )

            grade = gold_item.get("relevance_grade")
            if grade not in (1, 2):
                raise ValueError(
                    f"{qid}: relevance_grade must be 1 or 2, got {grade}"
                )

            chunk_ids = gold_item.get("chunk_ids")
            if not isinstance(chunk_ids, list) or not chunk_ids:
                raise ValueError(
                    f"{qid}: gold item must contain non-empty chunk_ids"
                )


def odata_escape(value: str) -> str:
    return value.replace("'", "''")


def section_key(result: dict[str, Any]) -> tuple[str, str]:
    return (
        str(result.get("act", "")),
        str(result.get("section_number", "")),
    )


def gold_chunk_grades(question: dict[str, Any]) -> dict[str, int]:
    grades: dict[str, int] = {}

    for item in question["gold"]:
        grade = int(item["relevance_grade"])
        for chunk_id in item["chunk_ids"]:
            grades[chunk_id] = max(grades.get(chunk_id, 0), grade)

    return grades


def gold_section_grades(question: dict[str, Any]) -> dict[tuple[str, str], int]:
    grades: dict[tuple[str, str], int] = {}

    for item in question["gold"]:
        key = (question["act"], str(item["section_number"]))
        grade = int(item["relevance_grade"])
        grades[key] = max(grades.get(key, 0), grade)

    return grades

def embed_queries(
    questions: list[dict[str, Any]],
) -> dict[str, list[float]]:
    ids = [q["id"] for q in questions]
    texts = [q["query"] for q in questions]

    print(f"Embedding {len(texts)} golden queries...")

    response = embedding_client.embeddings.create(
        model=EMBEDDING_DEPLOYMENT,
        input=texts,
        dimensions=VECTOR_DIMENSIONS,
    )

    if len(response.data) != len(texts):
        raise RuntimeError(
            f"Expected {len(texts)} query embeddings, "
            f"received {len(response.data)}"
        )

    ordered = sorted(response.data, key=lambda item: item.index)

    embeddings: dict[str, list[float]] = {}

    for qid, item in zip(ids, ordered):
        vector = item.embedding

        if len(vector) != VECTOR_DIMENSIONS:
            raise RuntimeError(
                f"{qid}: expected {VECTOR_DIMENSIONS} dimensions, "
                f"got {len(vector)}"
            )

        embeddings[qid] = vector

    print("Query embedding complete.")
    return embeddings

def run_search(
    *,
    query: str,
    act: str,
    strategy: str,
    candidate_k: int,
    query_vector: list[float] | None,
) -> list[dict[str, Any]]:
    filter_expression = f"act eq '{odata_escape(act)}'"

    vector_query = None

    if strategy in {"vector", "hybrid_rrf"}:
        if query_vector is None:
            raise ValueError(
                f"{strategy} retrieval requires a query vector."
            )

        vector_query = VectorizedQuery(
            vector=query_vector,
            k_nearest_neighbors=candidate_k,
            fields="content_vector",
        )

    if strategy == "bm25":
        raw_results = search_client.search(
            search_text=query,
            search_fields=SEARCH_FIELDS,
            filter=filter_expression,
            select=SELECT_FIELDS,
            top=candidate_k,
        )

    elif strategy == "vector":
        raw_results = search_client.search(
            search_text=None,
            vector_queries=[vector_query],
            filter=filter_expression,
            select=SELECT_FIELDS,
            top=candidate_k,
        )

    elif strategy == "hybrid_rrf":
        raw_results = search_client.search(
            search_text=query,
            search_fields=SEARCH_FIELDS,
            vector_queries=[vector_query],
            filter=filter_expression,
            select=SELECT_FIELDS,
            top=candidate_k,
        )

    else:
        raise ValueError(f"Unknown retrieval strategy: {strategy}")

    results = []

    for rank, result in enumerate(raw_results, start=1):
        item = dict(result)

        item["_rank"] = rank
        item["_score"] = result.get("@search.score")

        results.append(item)

    return results

def deduplicate_sections(
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    seen = set()
    section_results = []

    for result in results:
        key = section_key(result)

        if key in seen:
            continue

        seen.add(key)

        item = dict(result)
        item["_section_key"] = key
        item["_section_rank"] = len(section_results) + 1

        section_results.append(item)

    return section_results


def get_relevance_list(
    results: list[dict[str, Any]],
    question: dict[str, Any],
    level: str,
) -> list[int]:
    if level == "chunk_level":
        gold = gold_chunk_grades(question)

        return [
            int(gold.get(str(result.get("chunk_id")), 0))
            for result in results
        ]

    if level == "section_level":
        gold = gold_section_grades(question)

        return [
            int(gold.get(section_key(result), 0))
            for result in results
        ]

    raise ValueError(f"Unknown evaluation level: {level}")


def recall_at_k(
    relevance: list[int],
    total_gold_relevant: int,
) -> float:
    if total_gold_relevant == 0:
        return 0.0

    retrieved_relevant = sum(1 for grade in relevance if grade > 0)

    return min(retrieved_relevant / total_gold_relevant, 1.0)


def precision_at_k(relevance: list[int]) -> float:
    if not relevance:
        return 0.0

    return sum(1 for grade in relevance if grade > 0) / len(relevance)


def hit_rate_at_k(relevance: list[int]) -> float:
    return 1.0 if any(grade > 0 for grade in relevance) else 0.0


def reciprocal_rank(relevance: list[int]) -> float:
    for rank, grade in enumerate(relevance, start=1):
        if grade > 0:
            return 1.0 / rank

    return 0.0


def ndcg_at_k(
    relevance: list[int],
    total_gold_grades: dict[Any, int],
) -> float:
    if not relevance:
        return 0.0

    dcg = 0.0

    for rank, grade in enumerate(relevance, start=1):
        gain = (2**grade) - 1
        dcg += gain / math.log2(rank + 1)

    ideal = sorted(total_gold_grades.values(), reverse=True)

    idcg = 0.0

    for rank, grade in enumerate(ideal[: len(relevance)], start=1):
        gain = (2**grade) - 1
        idcg += gain / math.log2(rank + 1)

    if idcg == 0.0:
        return 0.0

    return dcg / idcg


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denominator = np.linalg.norm(a) * np.linalg.norm(b)

    if denominator == 0.0:
        return 0.0

    return float(np.dot(a, b) / denominator)


def average_pairwise_cosine_similarity(
    results: list[dict[str, Any]],
    embedding_map: dict[str, np.ndarray],
) -> float | None:
    vectors = []

    for result in results:
        chunk_id = str(result.get("chunk_id"))

        vector = embedding_map.get(chunk_id)

        if vector is None:
            continue

        vectors.append(vector)

    if len(vectors) < 2:
        return None

    similarities = []

    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            similarities.append(
                cosine_similarity(vectors[i], vectors[j])
            )

    if not similarities:
        return None

    return float(np.mean(similarities))


def unique_section_ratio(results: list[dict[str, Any]]) -> float:
    if not results:
        return 0.0

    unique_sections = len(
        {
            section_key(result)
            for result in results
        }
    )

    return unique_sections / len(results)


def redundancy_rate(results: list[dict[str, Any]]) -> float:
    if not results:
        return 0.0

    return 1.0 - unique_section_ratio(results)


def evaluate_result_set(
    *,
    question: dict[str, Any],
    results: list[dict[str, Any]],
    candidate_k: int,
    final_k: int,
    strategy: str,
    embedding_map: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    rows = []

    for level in ("chunk_level", "section_level"):
        if level == "chunk_level":
            evaluation_results = results
        else:
            evaluation_results = deduplicate_sections(results)

        gold_grades = (
            gold_chunk_grades(question)
            if level == "chunk_level"
            else gold_section_grades(question)
        )

        for k in [k for k in FINAL_K_VALUES if k <= candidate_k]:
            top_results = evaluation_results[:k]

            relevance = get_relevance_list(
                top_results,
                question,
                level,
            )

            avg_cosine = average_pairwise_cosine_similarity(
                top_results,
                embedding_map,
            )

            row = {
                "question_id": question["id"],
                "act": question["act"],
                "strategy": strategy,
                "candidate_k": candidate_k,
                "final_k": k,
                "evaluation_level": level,
                "recall_at_k": recall_at_k(
                    relevance,
                    len(gold_grades),
                ),
                "precision_at_k": precision_at_k(relevance),
                "hit_rate_at_k": hit_rate_at_k(relevance),
                "mrr": reciprocal_rank(relevance),
                "ndcg_at_k": ndcg_at_k(
                    relevance,
                    gold_grades,
                ),
                "average_pairwise_cosine_similarity": avg_cosine,
                "redundancy_rate": redundancy_rate(top_results),
                "unique_section_ratio": unique_section_ratio(top_results),
                "retrieved_count": len(top_results),
                "relevant_retrieved_count": sum(
                    1 for grade in relevance if grade > 0
                ),
                "retrieved_chunk_ids": [
                    result.get("chunk_id")
                    for result in top_results
                ],
                "retrieved_sections": [
                    f"{result.get('act')} §{result.get('section_number')}"
                    for result in top_results
                ],
                "relevance_grades": relevance,
            }

            rows.append(row)

    return rows

def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        print(f"No rows to write: {path}")
        return

    fieldnames: list[str] = []
    seen_fields: set[str] = set()

    for row in rows:
        for key in row.keys():
            if key not in seen_fields:
                seen_fields.add(key)
                fieldnames.append(key)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()

        for row in rows:
            normalized = {}

            for field in fieldnames:
                value = row.get(field)

                if isinstance(value, (list, dict)):
                    normalized[field] = json.dumps(
                        value,
                        ensure_ascii=False,
                    )
                elif value is None:
                    normalized[field] = ""
                else:
                    normalized[field] = value

            writer.writerow(normalized)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    rows: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            record = json.loads(line)

            if not isinstance(record, dict):
                raise ValueError(
                    f"{path}:{line_number}: expected a JSON object"
                )

            rows.append(record)

    return rows


def load_result_files(
    detailed_path: Path,
    summary_json_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    detailed_rows = load_jsonl(detailed_path)
    summary_data = load_json(summary_json_path)

    if not isinstance(summary_data, list):
        raise ValueError(
            f"{summary_json_path}: expected a JSON list of summary rows"
        )

    summary_rows = []

    for index, row in enumerate(summary_data):
        if not isinstance(row, dict):
            raise ValueError(
                f"{summary_json_path}: row {index} is not a JSON object"
            )

        summary_rows.append(row)

    if not detailed_rows:
        raise ValueError(f"{detailed_path}: no detailed result rows found")

    if not summary_rows:
        raise ValueError(f"{summary_json_path}: no summary result rows found")

    return detailed_rows, summary_rows


def write_evaluation_csv_outputs(
    *,
    detailed_path: Path,
    summary_json_path: Path,
    detailed_csv_path: Path,
    summary_csv_path: Path,
) -> None:
    detailed_rows, summary_rows = load_result_files(
        detailed_path,
        summary_json_path,
    )

    write_csv(detailed_csv_path, detailed_rows)
    write_csv(summary_csv_path, summary_rows)

    print("\nCSV regeneration complete.")
    print(f"Detailed JSONL : {detailed_path}")
    print(f"Summary JSON   : {summary_json_path}")
    print(f"Detailed CSV   : {detailed_csv_path}")
    print(f"Summary CSV    : {summary_csv_path}")
    print(f"Detailed rows  : {len(detailed_rows)}")
    print(f"Summary rows   : {len(summary_rows)}")


def aggregate_metrics(
    detailed_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)

    for row in detailed_rows:
        key = (
            row["strategy"],
            row["candidate_k"],
            row["final_k"],
            row["evaluation_level"],
        )
        groups[key].append(row)

    metric_names = [
        "recall_at_k",
        "precision_at_k",
        "hit_rate_at_k",
        "mrr",
        "ndcg_at_k",
        "average_pairwise_cosine_similarity",
        "redundancy_rate",
        "unique_section_ratio",
    ]

    summary = []

    for key, rows in sorted(groups.items()):
        strategy, candidate_k, final_k, level = key

        row = {
            "strategy": strategy,
            "candidate_k": candidate_k,
            "final_k": final_k,
            "evaluation_level": level,
            "question_count": len(rows),
        }

        for metric in metric_names:
            values = [
                row_item[metric]
                for row_item in rows
                if row_item[metric] is not None
            ]

            row[metric] = (
                float(np.mean(values))
                if values
                else None
            )

        summary.append(row)

    return summary


def print_summary(summary: list[dict[str, Any]]) -> None:
    print("\n" + "=" * 120)
    print("RETRIEVAL EVALUATION SUMMARY")
    print("=" * 120)

    for level in ("chunk_level", "section_level"):
        print(f"\n--- {level} ---")

        rows = [
            row
            for row in summary
            if row["evaluation_level"] == level
        ]

        print(
            f"{'strategy':<14}"
            f"{'candK':>7}"
            f"{'finalK':>8}"
            f"{'Recall':>10}"
            f"{'Precision':>11}"
            f"{'HitRate':>10}"
            f"{'MRR':>9}"
            f"{'nDCG':>9}"
            f"{'CosSim':>10}"
            f"{'Redund.':>10}"
            f"{'UniqueSec':>11}"
        )

        print("-" * 120)

        for row in rows:
            def fmt(value: Any) -> str:
                if value is None:
                    return "N/A"
                return f"{value:.4f}"

            print(
                f"{row['strategy']:<14}"
                f"{row['candidate_k']:>7}"
                f"{row['final_k']:>8}"
                f"{fmt(row['recall_at_k']):>10}"
                f"{fmt(row['precision_at_k']):>11}"
                f"{fmt(row['hit_rate_at_k']):>10}"
                f"{fmt(row['mrr']):>9}"
                f"{fmt(row['ndcg_at_k']):>9}"
                f"{fmt(row['average_pairwise_cosine_similarity']):>10}"
                f"{fmt(row['redundancy_rate']):>10}"
                f"{fmt(row['unique_section_ratio']):>11}"
            )

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate legal RAG retrieval, or regenerate CSV outputs "
            "from existing retrieval evaluation JSON/JSONL files."
        )
    )

    parser.add_argument(
        "--from-results",
        action="store_true",
        help=(
            "Regenerate CSV files from existing retrieval_eval_detailed.jsonl "
            "and retrieval_eval_summary.json without calling Azure."
        ),
    )

    parser.add_argument(
        "--detailed-jsonl",
        type=Path,
        default=None,
        help="Path to an existing detailed JSONL result file.",
    )

    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Path to an existing summary JSON result file.",
    )

    parser.add_argument(
        "--detailed-csv",
        type=Path,
        default=None,
        help="Output path for the detailed CSV.",
    )

    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help="Output path for the summary CSV.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    detailed_path = (
        args.detailed_jsonl
        or OUTPUT_DIR / "retrieval_eval_detailed.jsonl"
    )
    summary_json_path = (
        args.summary_json
        or OUTPUT_DIR / "retrieval_eval_summary.json"
    )
    detailed_csv_path = (
        args.detailed_csv
        or OUTPUT_DIR / "retrieval_eval_detailed.csv"
    )
    summary_csv_path = (
        args.summary_csv
        or OUTPUT_DIR / "retrieval_eval_summary.csv"
    )

    if args.from_results:
        write_evaluation_csv_outputs(
            detailed_path=detailed_path,
            summary_json_path=summary_json_path,
            detailed_csv_path=detailed_csv_path,
            summary_csv_path=summary_csv_path,
        )
        return

    print("=" * 80)
    print("LEGAL RAG RETRIEVAL EVALUATION")
    print("=" * 80)
    print(f"Search endpoint : {SEARCH_ENDPOINT}")
    print(f"Index           : {INDEX_NAME}")
    print(f"Golden set      : {GOLDEN_SET_PATH}")
    print(f"Candidate K     : {CANDIDATE_K_VALUES}")
    print(f"Final K         : {FINAL_K_VALUES}")
    print("Strategies      : BM25, Vector, Hybrid + RRF")
    print("MMR             : EXCLUDED")
    print()


    questions = load_golden_questions(GOLDEN_SET_PATH)
    validate_golden_questions(questions)

    print(f"Golden questions: {len(questions)}")

    embedding_map = {}
    embedding_map.update(load_embeddings(BNS_EMBEDDINGS_PATH))
    embedding_map.update(load_embeddings(ICA_EMBEDDINGS_PATH))

    print(f"Local embeddings: {len(embedding_map)}")

    if len(embedding_map) != 642:
        raise ValueError(
            f"Expected 642 local embeddings, found {len(embedding_map)}"
        )

    search_document_count = search_client.get_document_count()

    print(f"Azure Search documents: {search_document_count}")

    if search_document_count != 642:
        raise RuntimeError(
            f"Expected 642 Azure Search documents, "
            f"found {search_document_count}"
        )

    query_embeddings = embed_queries(questions)

    strategies = [
        "bm25",
        "vector",
        "hybrid_rrf",
    ]

    detailed_rows: list[dict[str, Any]] = []

    total_searches = (
        len(questions)
        * len(strategies)
        * len(CANDIDATE_K_VALUES)
    )

    search_number = 0

    for question in questions:
        qid = question["id"]

        print(
            f"\n[{qid}] {question['query']}"
        )

        for strategy in strategies:
            for candidate_k in CANDIDATE_K_VALUES:
                search_number += 1

                print(
                    f"  [{search_number}/{total_searches}] "
                    f"{strategy} candidate_k={candidate_k}"
                )

                query_vector = (
                    query_embeddings[qid]
                    if strategy in {"vector", "hybrid_rrf"}
                    else None
                )

                results = run_search(
                    query=question["query"],
                    act=question["act"],
                    strategy=strategy,
                    candidate_k=candidate_k,
                    query_vector=query_vector,
                )

                if len(results) > candidate_k:
                    raise RuntimeError(
                        f"{qid}/{strategy}/K={candidate_k}: "
                        f"received {len(results)} results"
                    )

                rows = evaluate_result_set(
                    question=question,
                    results=results,
                    candidate_k=candidate_k,
                    final_k=max(
                        k
                        for k in FINAL_K_VALUES
                        if k <= candidate_k
                    ),
                    strategy=strategy,
                    embedding_map=embedding_map,
                )

                detailed_rows.extend(rows)


    summary_rows = aggregate_metrics(detailed_rows)

    config_path = OUTPUT_DIR / "retrieval_eval_config.json"

    write_jsonl(detailed_path, detailed_rows)
    write_json(summary_json_path, summary_rows)
    write_csv(detailed_csv_path, detailed_rows)
    write_csv(summary_csv_path, summary_rows)

    config = {
        "golden_set": str(GOLDEN_SET_PATH),
        "index_name": INDEX_NAME,
        "search_endpoint": SEARCH_ENDPOINT,
        "strategies": strategies,
        "mmr": False,
        "candidate_k_values": CANDIDATE_K_VALUES,
        "final_k_values": FINAL_K_VALUES,
        "metrics": [
            "recall_at_k",
            "precision_at_k",
            "hit_rate_at_k",
            "mrr",
            "ndcg_at_k",
            "average_pairwise_cosine_similarity",
            "redundancy_rate",
            "unique_section_ratio",
        ],
        "evaluation_levels": [
            "chunk_level",
            "section_level",
        ],
        "gold_relevance_definition": {
            "2": "directly answers the query",
            "1": "useful supporting legal context",
            "0": "not in the gold set / not relevant",
        },
        "redundancy_definition": (
            "1 - unique_section_ratio, measured on raw chunk retrieval"
        ),
        "document_count_verified": search_document_count,
        "local_embedding_count": len(embedding_map),
        "vector_dimensions": VECTOR_DIMENSIONS,
    }

    write_json(config_path, config)

    print_summary(summary_rows)

    print("\n" + "=" * 80)
    print("OUTPUT FILES")
    print("=" * 80)
    print(f"Detailed results : {detailed_path}")
    print(f"Summary JSON     : {summary_json_path}")
    print(f"Detailed CSV     : {detailed_csv_path}")
    print(f"Summary CSV      : {summary_csv_path}")
    print(f"Config           : {config_path}")
    print()
    print("Evaluation complete.")
    print("No MMR was used.")


if __name__ == "__main__":
    main()
