from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[2]

NORMALIZED_DIR = ROOT / "legal-rag-azure" / "data" / "processed" / "normalized"
CHUNKS_DIR = ROOT / "legal-rag-azure" / "data" / "processed" / "chunks"


DOCUMENTS = {
    "BNS": {
        "markdown": NORMALIZED_DIR / "bns.md",
        "json": CHUNKS_DIR / "bns_chunks.json",
        "jsonl": CHUNKS_DIR / "bns_chunks.jsonl",

        "expected_sections": 358,

        "expected_chapters": 20,

        "section_range": range(1, 359),
    },

    "ICA": {
        "markdown": NORMALIZED_DIR / "indian-contract-act.md",
        "json": CHUNKS_DIR / "ica_chunks.json",
        "jsonl": CHUNKS_DIR / "ica_chunks.jsonl",

        "expected_sections": 192,

        "expected_chapters": 11,

        "section_range": None,
    },
}


CHARS_PER_TOKEN = 4

SUSPICIOUSLY_TINY_SECTION_CHARS = 80

SUSPICIOUSLY_LARGE_SECTION_CHARS = 25_000

MAX_CHUNK_TOKENS = 650

SECTION_RE = re.compile(
    r"^##\s+Section\s+([0-9]{1,3}[A-Za-z]?(?:-[A-Za-z])?)\s*[-—.]?\s*(.*?)\s*$",
    re.IGNORECASE,
)

CHAPTER_RE = re.compile(
    r"^#\s+Chapter\s+(.+?)\s*$",
    re.IGNORECASE,
)

PAGE_NUMBER_RE = re.compile(
    r"^\s*(?:Page\s+)?\d+\s*$",
    re.IGNORECASE,
)

MARKDOWN_SEPARATOR_RE = re.compile(
    r"^\s*---+\s*$"
)

SUBSECTION_NESTED_RE = re.compile(
    r"\)\s*\("
)

OBVIOUS_GARBAGE_PATTERNS = [
    re.compile(r"\b[A-Za-z]{2,}(?:Act|Clause|Section)\w*\b"),
    re.compile(r"\b(?:ThisAct|thisAct|General ClausesAct)\b"),
    re.compile(r"\b(?:bynotification|intoforce)\b"),
    re.compile(r"\b[A-Za-z](?:and|is|person|writing)\b"),
]

HEADER_FOOTER_LIKE_RE = re.compile(
    r"^\s*(?:"
    r"Bharatiya Nyaya Sanhita,?\s*2023"
    r"|"
    r"The Indian Contract Act,?\s*1872"
    r"|"
    r"Page\s+\d+(?:\s+of\s+\d+)?"
    r")\s*$",
    re.IGNORECASE,
)


def approx_tokens(text: str) -> int:
    return (len(text) + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN


def print_status(label: str, passed: bool, detail: str = "") -> bool:
    symbol = "✓" if passed else "✗"

    if detail:
        print(f"  {symbol} {label}: {detail}")
    else:
        print(f"  {symbol} {label}")

    return passed


def normalize_section_number(value: str) -> str:
    return value.strip().upper()

def validate_markdown(name: str, config: dict) -> bool:
    path = config["markdown"]

    print()
    print("=" * 70)
    print(f"MARKDOWN VALIDATION: {name}")
    print("=" * 70)
    print(f"File: {path}")

    if not path.exists():
        print_status("File exists", False, "file not found")
        return False

    try:
        text = path.read_text(encoding="utf-8")
    except Exception as exc:
        print_status("File readable", False, str(exc))
        return False

    lines = text.splitlines()

    all_passed = True

    sections = []

    for i, line in enumerate(lines):
        match = SECTION_RE.match(line)

        if match:
            number = normalize_section_number(match.group(1))
            title = match.group(2).strip()

            sections.append({
                "number": number,
                "title": title,
                "line": i + 1,
            })

    section_numbers = [s["number"] for s in sections]

    expected_count = config["expected_sections"]
    actual_count = len(sections)

    passed = actual_count == expected_count

    all_passed &= print_status(
        "Expected section count",
        passed,
        f"expected={expected_count}, found={actual_count}",
    )

    missing_numbers = []

    expected_range = config.get("section_range")

    if expected_range is not None:
        actual_numeric = set()

        for number in section_numbers:
            match = re.fullmatch(r"(\d+)", number)
            if match:
                actual_numeric.add(int(match.group(1)))

        missing_numbers = [
            n for n in expected_range
            if n not in actual_numeric
        ]

    passed = len(missing_numbers) == 0

    detail = (
        "all expected section numbers present"
        if passed
        else f"missing={missing_numbers[:20]}"
    )

    all_passed &= print_status(
        "Section numbers are present",
        passed,
        detail,
    )

    counts = Counter(section_numbers)

    duplicates = {
        number: count
        for number, count in counts.items()
        if count > 1
    }

    passed = len(duplicates) == 0

    detail = (
        "none"
        if passed
        else str(duplicates)
    )

    all_passed &= print_status(
        "No duplicate section numbers",
        passed,
        detail,
    )

    chapters = []

    for i, line in enumerate(lines):
        if CHAPTER_RE.match(line):
            chapters.append({
                "line": i + 1,
                "text": line.strip(),
            })

    expected_chapters = config["expected_chapters"]
    actual_chapters = len(chapters)

    passed = actual_chapters == expected_chapters

    all_passed &= print_status(
        "Chapters are present",
        passed,
        f"expected={expected_chapters}, found={actual_chapters}",
    )

    empty_sections = []

    for index, section in enumerate(sections):

        start = section["line"]

        if index + 1 < len(sections):
            end = sections[index + 1]["line"] - 1
        else:
            end = len(lines)

        body = "\n".join(lines[start:end]).strip()

        if not body:
            empty_sections.append(section["number"])

    passed = len(empty_sections) == 0

    detail = (
        "none"
        if passed
        else str(empty_sections)
    )

    all_passed &= print_status(
        "No empty sections",
        passed,
        detail,
    )

    tiny_sections = []

    for index, section in enumerate(sections):

        start = section["line"]

        if index + 1 < len(sections):
            end = sections[index + 1]["line"] - 1
        else:
            end = len(lines)

        body = "\n".join(lines[start:end]).strip()

        if len(body) < SUSPICIOUSLY_TINY_SECTION_CHARS:
            tiny_sections.append(
                (section["number"], len(body))
            )

    if tiny_sections:
        print(
            f"  ! Suspiciously tiny sections: "
            f"{len(tiny_sections)} "
            f"(examples={tiny_sections[:10]})"
        )
    else:
        print(
            "  ✓ No suspiciously tiny sections"
        )

    giant_sections = []

    for index, section in enumerate(sections):

        start = section["line"]

        if index + 1 < len(sections):
            end = sections[index + 1]["line"] - 1
        else:
            end = len(lines)

        body = "\n".join(lines[start:end]).strip()

        if len(body) > SUSPICIOUSLY_LARGE_SECTION_CHARS:
            giant_sections.append(
                (
                    section["number"],
                    len(body),
                    approx_tokens(body),
                )
            )

    if giant_sections:
        print(
            f"  ! Giant sections requiring inspection: "
            f"{len(giant_sections)} "
            f"(examples={giant_sections[:5]})"
        )
    else:
        print(
            "  ✓ No giant sections unexpectedly truncated"
        )

    page_number_lines = []

    for i, line in enumerate(lines):
        if PAGE_NUMBER_RE.match(line):

            if not re.match(
                r"^\s*(?:Section|CHAPTER|Chapter)\b",
                line,
                re.IGNORECASE,
            ):
                page_number_lines.append(i + 1)

    passed = len(page_number_lines) == 0

    detail = (
        "none"
        if passed
        else f"possible page-number lines={page_number_lines[:20]}"
    )

    all_passed &= print_status(
        "No accidental page numbers",
        passed,
        detail,
    )

    header_footer_counts = Counter()

    for line in lines:
        stripped = line.strip()

        if HEADER_FOOTER_LIKE_RE.match(stripped):
            header_footer_counts[stripped] += 1

    repeated_headers = {
        text: count
        for text, count in header_footer_counts.items()
        if count > 2
    }

    passed = len(repeated_headers) == 0

    detail = (
        "none"
        if passed
        else str(repeated_headers)
    )

    all_passed &= print_status(
        "No repeated headers/footers",
        passed,
        detail,
    )

    garbage_hits = []

    for i, line in enumerate(lines, start=1):

        for pattern in OBVIOUS_GARBAGE_PATTERNS:

            match = pattern.search(line)

            if match:
                garbage_hits.append(
                    (i, match.group(0), line.strip()[:120])
                )

    if garbage_hits:
        print(
            f"  ! Possible extraction garbage: "
            f"{len(garbage_hits)} occurrence(s)"
        )

        for hit in garbage_hits[:10]:
            line_no, matched, content = hit
            print(
                f"      line {line_no}: "
                f"{matched!r} -> {content}"
            )
    else:
        print(
            "  ✓ No obvious extraction garbage"
        )

    print()

    return bool(all_passed)

def load_json_chunks(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        for key in ("chunks", "data", "documents"):
            if isinstance(data.get(key), list):
                return data[key]

        raise ValueError(
            "JSON contains an object but no chunks/data/documents list."
        )

    if not isinstance(data, list):
        raise ValueError(
            "Expected JSON array of chunk objects."
        )

    return data


def load_jsonl_chunks(path: Path) -> list[dict]:
    chunks = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):

            if not line.strip():
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {exc}"
                ) from exc

            chunks.append(obj)

    return chunks


def validate_json(name: str, config: dict) -> bool:

    print()
    print("=" * 70)
    print(f"JSON VALIDATION: {name}")
    print("=" * 70)

    json_path = config["json"]
    jsonl_path = config["jsonl"]

    if jsonl_path.exists():

        print(f"Using: {jsonl_path}")

        try:
            chunks = load_jsonl_chunks(jsonl_path)
        except Exception as exc:
            print_status(
                "JSON/JSONL readable",
                False,
                str(exc),
            )
            return False

    elif json_path.exists():

        print(f"Using: {json_path}")

        try:
            chunks = load_json_chunks(json_path)
        except Exception as exc:
            print_status(
                "JSON/JSONL readable",
                False,
                str(exc),
            )
            return False

    else:
        print_status(
            "JSON/JSONL exists",
            False,
            "neither JSON nor JSONL file found",
        )
        return False

    all_passed = True

    print(f"Chunks found: {len(chunks)}")
    print()

    missing_chunk_id = [
        i for i, chunk in enumerate(chunks, start=1)
        if not chunk.get("chunk_id")
    ]

    passed = len(missing_chunk_id) == 0

    detail = (
        "all chunks"
        if passed
        else f"missing at records={missing_chunk_id[:20]}"
    )

    all_passed &= print_status(
        "Every chunk has chunk_id",
        passed,
        detail,
    )

    missing_content = [
        i for i, chunk in enumerate(chunks, start=1)
        if "content" not in chunk
    ]

    passed = len(missing_content) == 0

    detail = (
        "all chunks"
        if passed
        else f"missing at records={missing_content[:20]}"
    )

    all_passed &= print_status(
        "Every chunk has content",
        passed,
        detail,
    )

    chunk_ids = [
        chunk.get("chunk_id")
        for chunk in chunks
        if chunk.get("chunk_id")
    ]

    counts = Counter(chunk_ids)

    duplicate_ids = {
        chunk_id: count
        for chunk_id, count in counts.items()
        if count > 1
    }

    passed = len(duplicate_ids) == 0

    detail = (
        "none"
        if passed
        else str(duplicate_ids)
    )

    all_passed &= print_status(
        "No duplicate chunk_id",
        passed,
        detail,
    )

    empty_content = []

    for i, chunk in enumerate(chunks, start=1):

        content = chunk.get("content")

        if not isinstance(content, str) or not content.strip():
            empty_content.append(i)

    passed = len(empty_content) == 0

    detail = (
        "none"
        if passed
        else f"records={empty_content[:20]}"
    )

    all_passed &= print_status(
        "No empty content",
        passed,
        detail,
    )

    missing_section_number = []

    special_types = {
        "preamble",
        "related_judgements",
        "schedule",
    }

    for i, chunk in enumerate(chunks, start=1):

        content_type = chunk.get("content_type")

        if content_type in special_types:
            continue

        if chunk.get("section_number") in (None, ""):
            missing_section_number.append(i)

    passed = len(missing_section_number) == 0

    detail = (
        "all statutory chunks"
        if passed
        else f"missing at records={missing_section_number[:20]}"
    )

    all_passed &= print_status(
        "Every statutory chunk has section_number",
        passed,
        detail,
    )

    missing_citation = []

    for i, chunk in enumerate(chunks, start=1):

        if not chunk.get("citation_text"):
            missing_citation.append(i)

    passed = len(missing_citation) == 0

    detail = (
        "all chunks"
        if passed
        else f"missing at records={missing_citation[:20]}"
    )

    all_passed &= print_status(
        "Every chunk has citation_text",
        passed,
        detail,
    )

    fake_nested = []

    for i, chunk in enumerate(chunks, start=1):

        subsection = chunk.get("subsection")

        if isinstance(subsection, str):

            if SUBSECTION_NESTED_RE.search(subsection):
                fake_nested.append(
                    (i, chunk.get("chunk_id"), subsection)
                )

    passed = len(fake_nested) == 0

    detail = (
        "none"
        if passed
        else str(fake_nested[:10])
    )

    all_passed &= print_status(
        "subsection does not contain fake nested paths",
        passed,
        detail,
    )

    contamination = []

    for i, chunk in enumerate(chunks, start=1):

        content = chunk.get("content", "")

        if re.search(
            r"^#\s+Chapter\s+",
            content,
            re.MULTILINE | re.IGNORECASE,
        ):
            contamination.append(
                (i, chunk.get("chunk_id"))
            )

    passed = len(contamination) == 0

    detail = (
        "none"
        if passed
        else str(contamination[:10])
    )

    all_passed &= print_status(
        "No chapter heading contamination",
        passed,
        detail,
    )

    duplicate_marginal_notes = []

    for i, chunk in enumerate(chunks, start=1):

        content = chunk.get("content", "")

        matches = re.findall(
            r"(?im)^Marginal note:\s*(.+?)\s*$",
            content,
        )

        if len(matches) > 1:
            duplicate_marginal_notes.append(
                (
                    i,
                    chunk.get("chunk_id"),
                    matches,
                )
            )

    passed = len(duplicate_marginal_notes) == 0

    detail = (
        "none"
        if passed
        else str(duplicate_marginal_notes[:10])
    )

    all_passed &= print_status(
        "No duplicate marginal notes",
        passed,
        detail,
    )

    separator_hits = []

    for i, chunk in enumerate(chunks, start=1):

        content = chunk.get("content", "")

        for line in content.splitlines():

            if MARKDOWN_SEPARATOR_RE.match(line):
                separator_hits.append(
                    (i, chunk.get("chunk_id"))
                )
                break

    passed = len(separator_hits) == 0

    detail = (
        "none"
        if passed
        else str(separator_hits[:10])
    )

    all_passed &= print_status(
        "No unexpected separators",
        passed,
        detail,
    )

    oversized = []

    max_seen = 0

    for i, chunk in enumerate(chunks, start=1):

        content = chunk.get("content", "")

        tokens = approx_tokens(content)

        max_seen = max(max_seen, tokens)

        if tokens > MAX_CHUNK_TOKENS:
            oversized.append(
                (
                    i,
                    chunk.get("chunk_id"),
                    tokens,
                )
            )

    passed = len(oversized) == 0

    detail = (
        f"max={max_seen} approx tokens"
        if passed
        else (
            f"max={max_seen}, "
            f"oversized={oversized[:10]}"
        )
    )

    all_passed &= print_status(
        "Chunk size within limits",
        passed,
        detail,
    )

    print()

    return bool(all_passed)

def main() -> int:

    print()
    print("=" * 70)
    print("LEGAL RAG READ-ONLY VALIDATOR")
    print("=" * 70)
    print()
    print("This validator does NOT modify any input files.")
    print("This validator does NOT create any output files.")
    print()

    overall_passed = True

    for name, config in DOCUMENTS.items():

        markdown_passed = validate_markdown(
            name,
            config,
        )

        json_passed = validate_json(
            name,
            config,
        )

        document_passed = (
            markdown_passed
            and json_passed
        )

        print()
        print(
            f"{name}: "
            f"{'PASSED' if document_passed else 'FAILED'}"
        )

        overall_passed &= document_passed

    print()
    print("=" * 70)

    if overall_passed:
        print("OVERALL VALIDATION: PASSED")
        print("=" * 70)
        return 0

    print("OVERALL VALIDATION: FAILED")
    print("=" * 70)

    return 1


if __name__ == "__main__":
    sys.exit(main())