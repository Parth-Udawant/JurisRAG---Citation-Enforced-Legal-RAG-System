import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHUNK_DIR = PROJECT_ROOT / "data" / "processed" / "chunks"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "embeddings"

INPUT_FILES = {
    "bns": CHUNK_DIR / "bns_chunks.jsonl",
    "ica": CHUNK_DIR / "ica_chunks.jsonl",
}

OUTPUT_FILES = {
    "bns": OUTPUT_DIR / "bns_embeddings.jsonl",
    "ica": OUTPUT_DIR / "ica_embeddings.jsonl",
}

load_dotenv(PROJECT_ROOT / ".env")

ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
MODEL = os.getenv("AZURE_OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
DIMENSIONS = int(os.getenv("AZURE_OPENAI_EMBEDDING_DIMENSIONS", "1536"))
BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))

if not ENDPOINT:
    raise RuntimeError("Missing AZURE_OPENAI_ENDPOINT")

if not API_KEY:
    raise RuntimeError("Missing AZURE_OPENAI_API_KEY")

if not DEPLOYMENT:
    raise RuntimeError(
        "Missing AZURE_OPENAI_EMBEDDING_DEPLOYMENT"
    )

client = OpenAI(
    api_key=API_KEY,
    base_url=f"{ENDPOINT.rstrip('/')}/openai/v1/",
)

def load_jsonl(path: Path) -> list[dict]:
    records = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Invalid JSON on line {line_number} of {path}"
                ) from exc

            records.append(record)

    return records


def load_existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()

    existing_ids = set()

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Invalid JSON on line {line_number} of {path}"
                ) from exc

            chunk_id = record.get("chunk_id")

            if not chunk_id:
                raise RuntimeError(
                    f"Missing chunk_id in existing embedding file "
                    f"{path}, line {line_number}"
                )

            existing_ids.add(chunk_id)

    return existing_ids


def append_records(path: Path, records: list[dict]) -> None:
    with path.open("a", encoding="utf-8") as f:
        for record in records:
            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )

        f.flush()
        os.fsync(f.fileno())


def embed_batch(texts: list[str], max_retries: int = 5) -> list[list[float]]:
    for attempt in range(1, max_retries + 1):
        try:
            response = client.embeddings.create(
                model=DEPLOYMENT,
                input=texts,
                dimensions=DIMENSIONS,
            )

            ordered = sorted(response.data, key=lambda item: item.index)

            embeddings = [item.embedding for item in ordered]

            if len(embeddings) != len(texts):
                raise RuntimeError(
                    f"Expected {len(texts)} embeddings, "
                    f"received {len(embeddings)}"
                )

            return embeddings

        except Exception as exc:
            if attempt == max_retries:
                raise

            wait_seconds = 2 ** (attempt - 1)

            print(
                f"  API request failed "
                f"(attempt {attempt}/{max_retries}): {exc}"
            )
            print(f"  Retrying in {wait_seconds}s...")

            time.sleep(wait_seconds)

    raise RuntimeError("Embedding request failed unexpectedly.")

def validate_chunk(record: dict) -> None:
    chunk_id = record.get("chunk_id")

    if not chunk_id:
        raise ValueError("Chunk is missing chunk_id")

    content = record.get("content")

    if not isinstance(content, str):
        raise ValueError(
            f"{chunk_id}: content must be a string"
        )

    if not content.strip():
        raise ValueError(
            f"{chunk_id}: content is empty"
        )


def build_embedding_record(
    chunk: dict,
    embedding: list[float],
) -> dict:
    if len(embedding) != DIMENSIONS:
        raise ValueError(
            f"{chunk['chunk_id']}: expected {DIMENSIONS} dimensions, "
            f"received {len(embedding)}"
        )

    record = dict(chunk)

    record["embedding_model"] = MODEL
    record["embedding_dimensions"] = DIMENSIONS
    record["embedding"] = embedding

    return record

def process_dataset(name: str) -> None:
    input_path = INPUT_FILES[name]
    output_path = OUTPUT_FILES[name]

    print()
    print("=" * 70)
    print(f"Processing: {name.upper()}")
    print("=" * 70)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    chunks = load_jsonl(input_path)

    print(f"Input file : {input_path}")
    print(f"Input chunks: {len(chunks)}")

    for chunk in chunks:
        validate_chunk(chunk)

    input_ids = [chunk["chunk_id"] for chunk in chunks]

    if len(input_ids) != len(set(input_ids)):
        raise ValueError(
            f"{name}: duplicate chunk_id detected in input"
        )

    existing_ids = load_existing_ids(output_path)

    unknown_existing_ids = existing_ids - set(input_ids)

    if unknown_existing_ids:
        raise ValueError(
            f"{name}: output contains {len(unknown_existing_ids)} "
            f"chunk IDs that do not exist in the current input"
        )

    pending = [
        chunk
        for chunk in chunks
        if chunk["chunk_id"] not in existing_ids
    ]

    print(f"Already embedded: {len(existing_ids)}")
    print(f"Remaining       : {len(pending)}")
    print(f"Batch size      : {BATCH_SIZE}")
    print(f"Dimensions      : {DIMENSIONS}")
    print(f"Deployment      : {DEPLOYMENT}")

    if not pending:
        print("Nothing to embed.")
        return

    completed = len(existing_ids)

    for start in range(0, len(pending), BATCH_SIZE):
        batch = pending[start:start + BATCH_SIZE]

        texts = [chunk["content"] for chunk in batch]

        print(
            f"\nEmbedding batch "
            f"{start // BATCH_SIZE + 1}/"
            f"{(len(pending) + BATCH_SIZE - 1) // BATCH_SIZE}"
        )

        print(
            f"Chunks: {completed + 1}"
            f"-{completed + len(batch)} / {len(chunks)}"
        )

        embeddings = embed_batch(texts)

        output_records = []

        for chunk, embedding in zip(batch, embeddings):
            record = build_embedding_record(
                chunk,
                embedding,
            )

            output_records.append(record)

        append_records(output_path, output_records)

        completed += len(batch)

        print(
            f"Progress: {completed}/{len(chunks)} "
            f"({completed / len(chunks) * 100:.1f}%)"
        )

    print(f"\nCompleted: {name.upper()}")
    print(f"Output: {output_path}")

def verify_output(name: str) -> None:
    input_path = INPUT_FILES[name]
    output_path = OUTPUT_FILES[name]

    input_chunks = load_jsonl(input_path)
    output_chunks = load_jsonl(output_path)

    input_ids = {chunk["chunk_id"] for chunk in input_chunks}
    output_ids = [chunk["chunk_id"] for chunk in output_chunks]
    output_id_set = set(output_ids)

    print()
    print("=" * 70)
    print(f"VERIFYING: {name.upper()}")
    print("=" * 70)

    print(f"Input chunks : {len(input_chunks)}")
    print(f"Output chunks: {len(output_chunks)}")

    if len(output_ids) != len(output_id_set):
        raise RuntimeError(
            f"{name}: duplicate chunk IDs found in output"
        )

    missing = input_ids - output_id_set
    extra = output_id_set - input_ids

    if missing:
        raise RuntimeError(
            f"{name}: missing embeddings for {len(missing)} chunks"
        )

    if extra:
        raise RuntimeError(
            f"{name}: output contains {len(extra)} unexpected chunks"
        )

    for record in output_chunks:
        embedding = record.get("embedding")

        if not isinstance(embedding, list):
            raise RuntimeError(
                f"{record['chunk_id']}: embedding is not a list"
            )

        if len(embedding) != DIMENSIONS:
            raise RuntimeError(
                f"{record['chunk_id']}: "
                f"expected {DIMENSIONS} dimensions, "
                f"got {len(embedding)}"
            )

        if record.get("embedding_model") != MODEL:
            raise RuntimeError(
                f"{record['chunk_id']}: incorrect embedding model"
            )

        if record.get("embedding_dimensions") != DIMENSIONS:
            raise RuntimeError(
                f"{record['chunk_id']}: incorrect embedding dimensions"
            )

    print("✓ No duplicate chunk IDs")
    print("✓ No missing embeddings")
    print("✓ No unexpected chunk IDs")
    print(f"✓ Every vector has exactly {DIMENSIONS} dimensions")
    print(f"✓ Every record uses {MODEL}")
    print("✓ Verification PASSED")


def main() -> None:
    print("Legal RAG — Azure Embedding Pipeline")
    print()
    print(f"Model       : {MODEL}")
    print(f"Deployment  : {DEPLOYMENT}")
    print(f"Dimensions  : {DIMENSIONS}")
    print(f"Batch size  : {BATCH_SIZE}")
    print(f"Output dir  : {OUTPUT_DIR}")

    process_dataset("bns")
    process_dataset("ica")

    verify_output("bns")
    verify_output("ica")

    print()
    print("=" * 70)
    print("ALL EMBEDDINGS COMPLETED AND VERIFIED")
    print("=" * 70)


if __name__ == "__main__":
    main()