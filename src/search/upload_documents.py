import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError, ServiceRequestError
from azure.search.documents import SearchClient
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
SEARCH_ADMIN_KEY = os.getenv("AZURE_SEARCH_ADMIN_KEY")
INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME")

VECTOR_DIMENSIONS = int(os.getenv("AZURE_SEARCH_VECTOR_DIMENSIONS", "1536"))
BATCH_SIZE = int(os.getenv("AZURE_SEARCH_UPLOAD_BATCH_SIZE", "100"))
MAX_RETRIES = int(os.getenv("AZURE_SEARCH_UPLOAD_RETRIES", "3"))

EMBEDDING_DIR = PROJECT_ROOT / "data" / "processed" / "embeddings"
INPUT_FILES = [
    EMBEDDING_DIR / "bns_embeddings.jsonl",
    EMBEDDING_DIR / "ica_embeddings.jsonl",
]

REQUIRED_FIELDS = [
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


def require_env(name: str, value: str | None) -> str:
    if not value:
        raise RuntimeError(f"Missing {name} in .env")
    return value


SEARCH_ENDPOINT = require_env("AZURE_SEARCH_ENDPOINT", SEARCH_ENDPOINT)
SEARCH_ADMIN_KEY = require_env("AZURE_SEARCH_ADMIN_KEY", SEARCH_ADMIN_KEY)
INDEX_NAME = require_env("AZURE_SEARCH_INDEX_NAME", INDEX_NAME)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Embedding file not found: {path}")

    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_number}: {exc}"
                ) from exc

            records.append(record)

    return records


def build_search_document(record: dict[str, Any]) -> dict[str, Any]:
    missing = [field for field in REQUIRED_FIELDS if field not in record]

    if missing:
        raise ValueError(
            f"{record.get('chunk_id', '<unknown>')}: "
            f"missing fields: {missing}"
        )

    chunk_id = record["chunk_id"]
    embedding = record.get("embedding")

    if not isinstance(chunk_id, str) or not chunk_id:
        raise ValueError("chunk_id must be a non-empty string")

    if not isinstance(embedding, list):
        raise ValueError(f"{chunk_id}: embedding must be a list")

    if len(embedding) != VECTOR_DIMENSIONS:
        raise ValueError(
            f"{chunk_id}: expected {VECTOR_DIMENSIONS} dimensions, "
            f"got {len(embedding)}"
        )

    if not all(isinstance(value, (int, float)) for value in embedding):
        raise ValueError(f"{chunk_id}: embedding contains non-numeric values")

    document = {field: record[field] for field in REQUIRED_FIELDS}
    document["content_vector"] = embedding

    return document


def chunked(items: list[dict[str, Any]], size: int):
    for start in range(0, len(items), size):
        yield items[start:start + size]


def upload_batch(
    search_client: SearchClient,
    documents: list[dict[str, Any]],
    batch_number: int,
    total_batches: int,
) -> None:
    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            results = search_client.merge_or_upload_documents(
                documents=documents
            )

            failed = [result for result in results if not result.succeeded]

            if not failed:
                print(
                    f"Batch {batch_number}/{total_batches}: "
                    f"{len(documents)} succeeded"
                )
                return

            failed_ids = {result.key for result in failed}
            failed_docs = [
                document
                for document in documents
                if document["chunk_id"] in failed_ids
            ]

            print(
                f"Batch {batch_number}/{total_batches}: "
                f"{len(documents) - len(failed_docs)} succeeded, "
                f"{len(failed_docs)} failed"
            )

            for document in failed_docs:
                last_doc_error: Exception | None = None

                for doc_attempt in range(1, MAX_RETRIES + 1):
                    try:
                        result = search_client.merge_or_upload_documents(
                            documents=[document]
                        )[0]

                        if result.succeeded:
                            print(
                                f"  recovered: {document['chunk_id']}"
                            )
                            last_doc_error = None
                            break

                        last_doc_error = RuntimeError(
                            f"{document['chunk_id']}: "
                            f"{result.error_message}"
                        )

                    except (HttpResponseError, ServiceRequestError) as exc:
                        last_doc_error = exc

                    if doc_attempt < MAX_RETRIES:
                        time.sleep(2 ** (doc_attempt - 1))

                if last_doc_error is not None:
                    raise RuntimeError(
                        f"Failed to index {document['chunk_id']} after "
                        f"{MAX_RETRIES} retries: {last_doc_error}"
                    )

            return

        except (HttpResponseError, ServiceRequestError) as exc:
            last_error = exc

            if attempt < MAX_RETRIES:
                delay = 2 ** (attempt - 1)

                print(
                    f"Batch {batch_number}/{total_batches} failed "
                    f"(attempt {attempt}/{MAX_RETRIES}): {exc}"
                )
                print(f"Retrying in {delay}s...")
                time.sleep(delay)

    raise RuntimeError(
        f"Batch {batch_number}/{total_batches} failed after "
        f"{MAX_RETRIES} retries: {last_error}"
    )


def verify_index(
    search_client: SearchClient,
    expected_ids: set[str],
) -> None:
    timeout_seconds = 60
    poll_seconds = 3
    deadline = time.time() + timeout_seconds

    while True:
        count = search_client.get_document_count()

        if count == len(expected_ids):
            break

        if time.time() >= deadline:
            raise RuntimeError(
                f"Document count mismatch after {timeout_seconds}s: "
                f"expected {len(expected_ids)}, got {count}"
            )

        print(
            f"Waiting for index consistency: "
            f"expected {len(expected_ids)}, current {count}"
        )
        time.sleep(poll_seconds)

    print(f"\nAzure AI Search document count: {count}")

    actual_ids: set[str] = set()

    results = search_client.search(
        search_text="*",
        select=["chunk_id"],
        top=1000,
    )

    for result in results:
        actual_ids.add(result["chunk_id"])

    missing = expected_ids - actual_ids
    extra = actual_ids - expected_ids

    if missing or extra:
        raise RuntimeError(
            "Index key verification failed.\n"
            f"Missing IDs: {sorted(missing)[:20]}"
            f"{' ...' if len(missing) > 20 else ''}\n"
            f"Extra IDs: {sorted(extra)[:20]}"
            f"{' ...' if len(extra) > 20 else ''}"
        )

    print("Index verification: PASS")
    print(f"Verified exact document set: {len(actual_ids)} documents")


def main() -> None:
    print("=== Azure AI Search document upload ===")
    print(f"Endpoint : {SEARCH_ENDPOINT}")
    print(f"Index    : {INDEX_NAME}")
    print(f"Dimension: {VECTOR_DIMENSIONS}")
    print(f"Batch    : {BATCH_SIZE}")

    all_documents: list[dict[str, Any]] = []

    for path in INPUT_FILES:
        records = load_jsonl(path)

        print(f"Loaded {len(records):>3} records from {path.name}")

        documents = [build_search_document(record) for record in records]
        all_documents.extend(documents)

    id_counts = Counter(document["chunk_id"] for document in all_documents)
    duplicate_ids = sorted(
        chunk_id
        for chunk_id, count in id_counts.items()
        if count > 1
    )

    if duplicate_ids:
        raise RuntimeError(
            f"Duplicate chunk_id values found: {duplicate_ids[:20]}"
        )

    expected_ids = set(id_counts)

    print(f"\nTotal documents prepared: {len(all_documents)}")

    if len(all_documents) != 642:
        raise RuntimeError(
            f"Expected exactly 642 embedding records, "
            f"got {len(all_documents)}"
        )

    client = SearchClient(
        endpoint=SEARCH_ENDPOINT,
        index_name=INDEX_NAME,
        credential=AzureKeyCredential(SEARCH_ADMIN_KEY),
    )

    batches = list(chunked(all_documents, BATCH_SIZE))

    for batch_number, batch in enumerate(batches, start=1):
        upload_batch(
            search_client=client,
            documents=batch,
            batch_number=batch_number,
            total_batches=len(batches),
        )

    verify_index(client, expected_ids)

    print("\n=== Upload complete ===")
    print("642/642 documents indexed successfully.")


if __name__ == "__main__":
    main()