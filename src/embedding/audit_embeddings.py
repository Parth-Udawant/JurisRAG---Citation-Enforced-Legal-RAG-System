import json
import math
from pathlib import Path


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHUNK_DIR = PROJECT_ROOT / "data" / "processed" / "chunks"
EMBEDDING_DIR = PROJECT_ROOT / "data" / "processed" / "embeddings"

DATASETS = {
    "BNS": {
        "chunks": CHUNK_DIR / "bns_chunks.jsonl",
        "embeddings": EMBEDDING_DIR / "bns_embeddings.jsonl",
    },
    "ICA": {
        "chunks": CHUNK_DIR / "ica_chunks.jsonl",
        "embeddings": EMBEDDING_DIR / "ica_embeddings.jsonl",
    },
}

EXPECTED_MODEL = "text-embedding-3-large"
EXPECTED_DIMENSIONS = 1536


# ============================================================
# Helpers
# ============================================================

def load_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file without modifying it."""
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
                    f"Invalid JSON on line {line_number}: {path}"
                ) from exc

            records.append(record)

    return records


def compare_field(
    original: dict,
    generated: dict,
    field: str,
    errors: list[str],
    chunk_id: str,
) -> None:
    """Verify that a field survived embedding generation unchanged."""
    if original.get(field) != generated.get(field):
        errors.append(
            f"{chunk_id}: field '{field}' changed"
        )


def vector_statistics(vectors: list[list[float]]) -> dict:
    """Calculate basic sanity statistics across all vector values."""
    values = [
        value
        for vector in vectors
        for value in vector
    ]

    if not values:
        return {
            "min": None,
            "max": None,
            "mean": None,
            "nonzero": 0,
            "total_values": 0,
        }

    return {
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
        "nonzero": sum(value != 0 for value in values),
        "total_values": len(values),
    }


# ============================================================
# Dataset audit
# ============================================================

def audit_dataset(name: str, paths: dict) -> dict:
    print()
    print("=" * 78)
    print(f"AUDITING {name}")
    print("=" * 78)

    chunk_path = paths["chunks"]
    embedding_path = paths["embeddings"]

    if not chunk_path.exists():
        raise FileNotFoundError(
            f"Chunk file not found: {chunk_path}"
        )

    if not embedding_path.exists():
        raise FileNotFoundError(
            f"Embedding file not found: {embedding_path}"
        )

    print(f"Original chunks : {chunk_path}")
    print(f"Embeddings      : {embedding_path}")

    original = load_jsonl(chunk_path)
    generated = load_jsonl(embedding_path)

    print()
    print(f"Original records : {len(original)}")
    print(f"Embedding records: {len(generated)}")

    errors = []
    warnings = []

    # --------------------------------------------------------
    # 1. Record count
    # --------------------------------------------------------

    if len(original) != len(generated):
        errors.append(
            f"Record count mismatch: "
            f"{len(original)} original vs "
            f"{len(generated)} embeddings"
        )

    # --------------------------------------------------------
    # 2. Chunk IDs
    # --------------------------------------------------------

    original_ids = [record.get("chunk_id") for record in original]
    generated_ids = [record.get("chunk_id") for record in generated]

    original_id_set = set(original_ids)
    generated_id_set = set(generated_ids)

    # Duplicate IDs in source
    if len(original_ids) != len(original_id_set):
        errors.append("Duplicate chunk_id(s) in original chunks")

    # Duplicate IDs in embedding output
    if len(generated_ids) != len(generated_id_set):
        errors.append("Duplicate chunk_id(s) in embedding output")

    missing_ids = original_id_set - generated_id_set
    extra_ids = generated_id_set - original_id_set

    if missing_ids:
        errors.append(
            f"Missing embeddings for {len(missing_ids)} chunk(s): "
            f"{sorted(missing_ids)[:10]}"
        )

    if extra_ids:
        errors.append(
            f"Unexpected embedding chunk(s): "
            f"{sorted(extra_ids)[:10]}"
        )

    # --------------------------------------------------------
    # 3. Ordering
    # --------------------------------------------------------

    if original_ids != generated_ids:
        warnings.append(
            "Embedding record order differs from original chunk order"
        )

    # --------------------------------------------------------
    # 4. Metadata/content preservation
    # --------------------------------------------------------

    original_by_id = {
        record["chunk_id"]: record
        for record in original
    }

    generated_by_id = {
        record["chunk_id"]: record
        for record in generated
    }

    # These are the fields that existed before embedding.
    original_fields = [
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

    metadata_checks = 0

    for chunk_id in sorted(original_id_set & generated_id_set):
        source = original_by_id[chunk_id]
        result = generated_by_id[chunk_id]

        for field in original_fields:
            compare_field(
                source,
                result,
                field,
                errors,
                chunk_id,
            )

            metadata_checks += 1

    # --------------------------------------------------------
    # 5. Embedding metadata
    # --------------------------------------------------------

    all_vectors = []

    for chunk_id in sorted(generated_id_set):
        record = generated_by_id[chunk_id]

        model = record.get("embedding_model")
        dimensions = record.get("embedding_dimensions")
        embedding = record.get("embedding")

        if model != EXPECTED_MODEL:
            errors.append(
                f"{chunk_id}: incorrect embedding model: "
                f"{model!r}"
            )

        if dimensions != EXPECTED_DIMENSIONS:
            errors.append(
                f"{chunk_id}: incorrect embedding_dimensions: "
                f"{dimensions!r}"
            )

        if not isinstance(embedding, list):
            errors.append(
                f"{chunk_id}: embedding is not a list"
            )
            continue

        if len(embedding) != EXPECTED_DIMENSIONS:
            errors.append(
                f"{chunk_id}: expected "
                f"{EXPECTED_DIMENSIONS} dimensions, "
                f"got {len(embedding)}"
            )

        if len(embedding) == 0:
            errors.append(
                f"{chunk_id}: empty embedding"
            )
            continue

        # Validate values.
        for index, value in enumerate(embedding):
            if not isinstance(value, (int, float)):
                errors.append(
                    f"{chunk_id}: embedding[{index}] "
                    f"is not numeric"
                )
                break

            if not math.isfinite(value):
                errors.append(
                    f"{chunk_id}: embedding[{index}] "
                    f"is not finite"
                )
                break

        all_vectors.append(embedding)

    # --------------------------------------------------------
    # 6. Vector sanity statistics
    # --------------------------------------------------------

    stats = vector_statistics(all_vectors)

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    print()
    print("Checks")
    print("-" * 78)

    print(
        f"[{'PASS' if len(original) == len(generated) else 'FAIL'}] "
        f"Record count"
    )

    print(
        f"[{'PASS' if not missing_ids else 'FAIL'}] "
        f"No missing chunk IDs"
    )

    print(
        f"[{'PASS' if not extra_ids else 'FAIL'}] "
        f"No unexpected chunk IDs"
    )

    print(
        f"[{'PASS' if len(original_ids) == len(original_id_set) else 'FAIL'}] "
        f"No duplicate source chunk IDs"
    )

    print(
        f"[{'PASS' if len(generated_ids) == len(generated_id_set) else 'FAIL'}] "
        f"No duplicate embedding chunk IDs"
    )

    print(
        f"[{'PASS' if original_ids == generated_ids else 'WARN'}] "
        f"Record ordering preserved"
    )

    print(
        f"[{'PASS' if not any(
            error.endswith("changed")
            for error in errors
        ) else 'FAIL'}] "
        f"Original metadata/content preserved"
    )

    model_ok = all(
        record.get("embedding_model") == EXPECTED_MODEL
        for record in generated
    )

    print(
        f"[{'PASS' if model_ok else 'FAIL'}] "
        f"Embedding model = {EXPECTED_MODEL}"
    )

    dimensions_ok = all(
        record.get("embedding_dimensions") == EXPECTED_DIMENSIONS
        for record in generated
    )

    print(
        f"[{'PASS' if dimensions_ok else 'FAIL'}] "
        f"Embedding dimensions metadata = {EXPECTED_DIMENSIONS}"
    )

    vector_lengths_ok = all(
        isinstance(record.get("embedding"), list)
        and len(record["embedding"]) == EXPECTED_DIMENSIONS
        for record in generated
    )

    print(
        f"[{'PASS' if vector_lengths_ok else 'FAIL'}] "
        f"All vectors have {EXPECTED_DIMENSIONS} values"
    )

    numeric_ok = all(
        isinstance(value, (int, float))
        and math.isfinite(value)
        for record in generated
        if isinstance(record.get("embedding"), list)
        for value in record["embedding"]
    )

    print(
        f"[{'PASS' if numeric_ok else 'FAIL'}] "
        f"All vector values are finite numbers"
    )

    print()
    print("Vector statistics")
    print("-" * 78)
    print(f"Total vector values : {stats['total_values']:,}")
    print(f"Minimum value       : {stats['min']}")
    print(f"Maximum value       : {stats['max']}")
    print(f"Mean value          : {stats['mean']}")
    print(f"Non-zero values     : {stats['nonzero']:,}")

    print()
    print(f"Metadata fields checked: {metadata_checks}")

    if warnings:
        print()
        print("Warnings")
        for warning in warnings:
            print(f"  ! {warning}")

    if errors:
        print()
        print("ERRORS")
        for error in errors[:50]:
            print(f"  ✗ {error}")

        if len(errors) > 50:
            print(
                f"  ... and {len(errors) - 50} more errors"
            )

        print()
        print(f"{name} AUDIT: FAILED")

        return {
            "name": name,
            "passed": False,
            "records": len(generated),
            "errors": len(errors),
        }

    print()
    print(f"{name} AUDIT: PASSED")

    return {
        "name": name,
        "passed": True,
        "records": len(generated),
        "errors": 0,
    }


# ============================================================
# Main
# ============================================================

def main():
    print("Legal RAG — Local Embedding Output Audit")
    print()
    print("READ-ONLY AUDIT")
    print("No input or output files will be modified.")
    print()
    print(f"Expected model      : {EXPECTED_MODEL}")
    print(f"Expected dimensions : {EXPECTED_DIMENSIONS}")

    results = []

    for name, paths in DATASETS.items():
        result = audit_dataset(name, paths)
        results.append(result)

    print()
    print("=" * 78)
    print("FINAL AUDIT SUMMARY")
    print("=" * 78)

    total_records = 0
    all_passed = True

    for result in results:
        status = "PASS" if result["passed"] else "FAIL"

        print(
            f"{result['name']:>4}: "
            f"{status} — "
            f"{result['records']} embedding records"
        )

        total_records += result["records"]

        if not result["passed"]:
            all_passed = False

    print()
    print(f"Total embedding records: {total_records}")

    if all_passed and total_records == 642:
        print()
        print("✓ BNS: 447/447")
        print("✓ ICA: 195/195")
        print("✓ TOTAL: 642/642")
        print("✓ LOCAL EMBEDDING AUDIT PASSED")
    else:
        print()
        print("✗ LOCAL EMBEDDING AUDIT FAILED")
        raise SystemExit(1)


if __name__ == "__main__":
    main()