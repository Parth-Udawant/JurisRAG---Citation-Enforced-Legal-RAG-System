from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path


# ============================================================
# Configuration
# ============================================================

TARGET_CHUNK_TOKENS = 450
MAX_CHUNK_TOKENS = 650

ROOT = Path(__file__).resolve().parents[2]

NORMALIZED_DIR = ROOT / "data" / "processed" / "normalized"
OUTPUT_DIR = ROOT / "data" / "processed" / "chunks"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


DOCUMENTS = [
    {
        "path": NORMALIZED_DIR / "bns.md",
        "act": "Bharatiya Nyaya Sanhita, 2023",
        "prefix": "BNS",
    },
    {
        "path": NORMALIZED_DIR / "indian-contract-act.md",
        "act": "Indian Contract Act, 1872",
        "prefix": "ICA",
    },
]


# ============================================================
# Markdown / legal-structure patterns
# ============================================================

CHAPTER_RE = re.compile(
    r"^#\s+Chapter\s+([IVXLCDM]+)\s*[—-]\s*(.+?)\s*$",
    re.IGNORECASE,
)

# Supports:
#   ## Section 1 — ...
#   ## Section 19-A — ...
#   ## Section 178A. ...
#   ## Section 174
SECTION_RE = re.compile(
    r"^##\s+Section\s+"
    r"([0-9]+(?:-[A-Za-z]+)?|[0-9]+[A-Za-z]+)"
    r"\s*(?:(?:—|-)\s*(.*?))?\s*$",
    re.IGNORECASE,
)

SPECIAL_RE = re.compile(
    r"^##\s+(Preamble|Related Judgements|Schedule)\s*$",
    re.IGNORECASE,
)

MARGINAL_RE = re.compile(
    r"^\s*\*\*Marginal note:\*\*\s*(.*?)\s*$",
    re.IGNORECASE,
)

# Explicit semantic blocks.
TYPE_RE = re.compile(
    r"^\s*\*\*(Explanation|Illustration|Exception)\s*:?\*\*"
    r"(?:\s+(.*))?\s*$",
    re.IGNORECASE,
)

# Explicit statutory markers.
#
# IMPORTANT:
# We only consider a marker structural when it occurs at the
# beginning of a Markdown line. We also reject cases where the
# text immediately following the marker begins with punctuation.
#
# This prevents a line such as:
#   **(3)** , (4) and (5) of section 310...
#
# from being incorrectly treated as a new statutory subsection.
MARKER_RE = re.compile(
    r"^\s*(?:[-*+]\s+)?\*\*\s*"
    r"(\([0-9]+\)|\([a-z]\)|\([ivxlcdm]+\))"
    r"\s*\*\*"
    r"(?:\s+(.*))?\s*$",
    re.IGNORECASE,
)


# ============================================================
# Utility functions
# ============================================================

def approx_tokens(text: str) -> int:
    """
    Conservative approximation used only for chunk-size control.

    It is intentionally not tied to a specific embedding tokenizer.
    Final Azure/OpenAI token counts can differ slightly.
    """
    return max(1, math.ceil(len(text) / 4))


def detect_marker(line: str) -> str | None:
    """
    Return an explicit legal marker such as '(1)' or '(a)'.

    Do NOT infer nested paths such as '(24)(a)(3)'.
    """
    match = MARKER_RE.match(line)
    if not match:
        return None

    remainder = (match.group(2) or "").strip()

    # Important OCR/legal-reference guard.
    # Example:
    #   **(3)** , (4) and (5) of section 8...
    #
    # This is a reference inside another provision, not a new
    # statutory subdivision.
    if remainder and remainder[0] in ",.;:)":
        return None

    return match.group(1)


def marker_kind(marker: str) -> str:
    value = marker[1:-1]

    if value.isdigit():
        return "numeric"

    if re.fullmatch(r"[a-z]", value, re.IGNORECASE):
        return "alpha"

    return "roman"


def parse_units(lines: list[str]) -> list[dict]:
    """
    Parse chapters, sections and Contract Act special blocks.

    Chapter headings are structural metadata and are NOT included
    in chunk content.
    """
    units = []
    current_chapter = None
    current_unit = None

    for raw_line in lines:
        line = raw_line.strip()

        chapter_match = CHAPTER_RE.match(line)
        if chapter_match:
            current_chapter = (
                chapter_match.group(1),
                chapter_match.group(2).strip(),
            )
            current_unit = None
            continue

        section_match = SECTION_RE.match(line)
        if section_match:
            current_unit = {
                "number": section_match.group(1),
                "title": (section_match.group(2) or "").strip(),
                "chapter": current_chapter,
                "lines": [],
                "special": None,
            }
            units.append(current_unit)
            continue

        special_match = SPECIAL_RE.match(line)
        if special_match:
            current_unit = {
                "number": None,
                "title": special_match.group(1),
                "chapter": current_chapter,
                "lines": [],
                "special": special_match.group(1)
                .lower()
                .replace(" ", "_"),
            }
            units.append(current_unit)
            continue

        if current_unit is not None:
            current_unit["lines"].append(raw_line)

    return units


def collect_chapter_headers(lines: list[str]) -> list[tuple[str, str]]:
    """
    Collect chapter identifiers/titles so obvious OCR/page-header
    contamination such as 'CHAPTERXX REPEAL AND SAVINGS' can be
    removed when it has been appended to legal text.
    """
    headers = []

    for line in lines:
        match = CHAPTER_RE.match(line.strip())
        if match:
            headers.append(
                (
                    match.group(1),
                    match.group(2).strip(),
                )
            )

    return headers


def remove_inline_chapter_headers(
    line: str,
    chapter_headers: list[tuple[str, str]],
) -> str:
    """
    Remove only chapter-header phrases that exactly correspond
    to chapter headings present in this source.

    This addresses page-layout/OCR contamination such as:
        ... legal text. CHAPTERXX REPEAL AND SAVINGS

    It does not perform general OCR correction.
    """
    cleaned = line

    for roman, title in chapter_headers:
        title_pattern = r"\s+".join(
            re.escape(part)
            for part in title.split()
        )

        pattern = re.compile(
            rf"\bCHAPTER\s*{re.escape(roman)}\s+"
            rf"{title_pattern}",
            re.IGNORECASE,
        )

        cleaned = pattern.sub("", cleaned)

    return re.sub(r"[ \t]{2,}", " ", cleaned).strip()


def make_context(act: str, unit: dict) -> str:
    """
    Parent context is included in every chunk.
    """
    parts = [f"Act: {act}"]

    if unit["chapter"]:
        parts.append(
            f"Chapter {unit['chapter'][0]} — {unit['chapter'][1]}"
        )

    if unit["number"] is not None:
        parts.append(
            f"Section {unit['number']} — {unit['title']}"
        )
    else:
        parts.append(unit["title"])

    # Marginal note is promoted into the parent context.
    for line in unit["lines"]:
        match = MARGINAL_RE.match(line)
        if match:
            parts.append(
                f"Marginal note: {match.group(1).strip()}"
            )
            break

    return "\n".join(parts)


# ============================================================
# Semantic block construction
# ============================================================

def build_blocks(unit: dict, chapter_headers: list[tuple[str, str]]) -> list[tuple]:
    """
    Convert a section into semantic blocks.

    Returned tuple:
        (marker, content_type, is_direct_subsection, raw_lines)

    Important design rule:
    --------------------------------
    We NEVER construct metadata such as:
        (24)(a)(3)(b)(25)

    A marker is stored exactly as it appears in the source:
        (24)
        (a)
        (i)

    Nested markers remain in content but are not promoted to
    section-level subsection metadata.

    The first explicit statutory marker in a section establishes
    the section's top-level marker stream.

    Examples:
        BNS §1:
            (1), (2), ... (6) -> direct

        BNS §4:
            (a), (b), (c), ... -> direct
            (1), (2) under (c) -> nested

        BNS §101:
            (a)-(d) -> direct
            (a)-(c) under "Provided that" -> nested
    """
    blocks = []

    current_lines = []
    current_marker = None
    current_type = "provision"
    current_direct = False

    top_kind = None
    nested_region = False

    def flush():
        nonlocal current_lines

        if current_lines and any(
            line.strip() for line in current_lines
        ):
            blocks.append(
                (
                    current_marker,
                    current_type,
                    current_direct,
                    current_lines,
                )
            )

        current_lines = []

    for raw_line in unit["lines"]:

        line = remove_inline_chapter_headers(
            raw_line,
            chapter_headers,
        )

        # Structural/non-content lines.
        if not line.strip():
            continue

        if line.strip() == "---":
            # Markdown separator between legal units/pages.
            # It is not legal content.
            continue

        if MARGINAL_RE.match(line):
            # Already promoted into parent context.
            continue

        marker = detect_marker(line)
        type_match = TYPE_RE.match(line)

        if marker:
            flush()

            kind = marker_kind(marker)

            if top_kind is None:
                # First explicit statutory marker establishes
                # the section's top-level stream.
                top_kind = kind
                nested_region = False
                direct = True

            elif nested_region:
                # For numeric top-level streams (very common in
                # BNS definitions), returning to a new numeric
                # marker closes an explanatory/nested region.
                #
                # Example:
                #   Explanation:
                #   (a) ...
                #   (b) ...
                #   (29) ...
                if kind == top_kind and top_kind == "numeric":
                    nested_region = False
                    direct = True
                else:
                    direct = False

            else:
                # Only markers belonging to the same top-level
                # stream are promoted to subsection metadata.
                direct = kind == top_kind

            current_lines = [line]
            current_marker = marker
            current_type = "provision"
            current_direct = direct

        elif type_match:
            flush()

            current_lines = [line]
            current_marker = None
            current_type = type_match.group(1).lower()
            current_direct = False

            # A marker after an Explanation belongs to the
            # explanation's nested structure rather than the
            # section-level subsection stream.
            if current_type == "explanation":
                nested_region = True

        else:
            current_lines.append(line)

            # "Provided that" introduces a nested statutory
            # condition in the normalized sources.
            stripped = re.sub(
                r"^\s*\*\*|\*\*\s*$",
                "",
                line,
            ).strip().lower()

            if stripped.startswith("provided that"):
                nested_region = True

    flush()
    return blocks


# ============================================================
# Size-aware splitting
# ============================================================

def split_large_block(
    marker: str | None,
    content_type: str,
    direct: bool,
    text: str,
) -> list[tuple]:
    """
    Split only when a semantic block is too large.

    Paragraph boundaries are preferred.
    Sentence fallback is used for unusually large paragraphs.
    """
    if approx_tokens(text) <= MAX_CHUNK_TOKENS:
        return [(marker, content_type, direct, text)]

    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n", text)
        if paragraph.strip()
    ]

    groups = []
    current = []

    for paragraph in paragraphs:
        trial = "\n\n".join(
            current + [paragraph]
        )

        if (
            current
            and approx_tokens(trial) > TARGET_CHUNK_TOKENS
        ):
            groups.append("\n\n".join(current))
            current = [paragraph]
        else:
            current.append(paragraph)

    if current:
        groups.append("\n\n".join(current))

    result = []

    for group in groups:
        if approx_tokens(group) <= MAX_CHUNK_TOKENS:
            result.append(
                (marker, content_type, direct, group)
            )
            continue

        # Sentence fallback.
        sentences = re.split(
            r"(?<=[.!?;])\s+",
            group,
        )

        current = []

        for sentence in sentences:
            trial = " ".join(
                current + [sentence]
            )

            if (
                current
                and approx_tokens(trial)
                > TARGET_CHUNK_TOKENS
            ):
                result.append(
                    (
                        marker,
                        content_type,
                        direct,
                        " ".join(current),
                    )
                )
                current = [sentence]
            else:
                current.append(sentence)

        if current:
            result.append(
                (
                    marker,
                    content_type,
                    direct,
                    " ".join(current),
                )
            )

    return result


# ============================================================
# Chunking
# ============================================================

def chunk_document(
    lines: list[str],
    act: str,
    source_file: str,
    prefix: str,
) -> tuple[list[dict], list[dict]]:
    """
    Return:
        units, chunks
    """
    units = parse_units(lines)
    chapter_headers = collect_chapter_headers(lines)

    all_chunks = []

    for unit in units:
        semantic_blocks = build_blocks(unit, chapter_headers)

        expanded_blocks = []

        for (
            marker,
            content_type,
            direct,
            raw_lines,
        ) in semantic_blocks:

            text = "\n".join(raw_lines).strip()

            if not text:
                continue

            expanded_blocks.extend(
                split_large_block(
                    marker,
                    content_type,
                    direct,
                    text,
                )
            )

        context = make_context(act, unit)

        groups = []
        current_group = []

        for block in expanded_blocks:
            candidate = (
                context
                + "\n\n"
                + "\n\n".join(
                    item[3]
                    for item in current_group + [block]
                )
            )

            if (
                current_group
                and approx_tokens(candidate)
                > TARGET_CHUNK_TOKENS
            ):
                groups.append(current_group)
                current_group = []

            current_group.append(block)

        if current_group:
            groups.append(current_group)

        for ordinal, items in enumerate(groups, start=1):

            body = "\n\n".join(
                item[3] for item in items
            )

            direct_markers = []

            for (
                marker,
                _content_type,
                direct,
                _text,
            ) in items:
                if (
                    marker
                    and direct
                    and marker not in direct_markers
                ):
                    direct_markers.append(marker)

            content_types = [
                item[1] for item in items
            ]

            if unit["special"]:
                content_type = unit["special"]
            elif len(set(content_types)) == 1:
                content_type = content_types[0]
            else:
                content_type = "mixed"

            subsection = (
                ", ".join(direct_markers)
                if direct_markers
                else None
            )

            if unit["special"]:
                chunk_id = (
                    f"{prefix}-special-"
                    f"{unit['special']}-c{ordinal}"
                )

                citation_text = (
                    f"{act}, {unit['title']}"
                )

            else:
                safe_section = re.sub(
                    r"[^A-Za-z0-9]+",
                    "_",
                    unit["number"],
                ).strip("_")

                chapter_number = (
                    unit["chapter"][0].lower()
                    if unit["chapter"]
                    else "na"
                )

                chunk_id = (
                    f"{prefix}-ch{chapter_number}"
                    f"-s{safe_section}-c{ordinal}"
                )

                citation_text = (
                    f"{prefix} §{unit['number']}"
                )

                # Specific citation only when the entire chunk
                # is one direct statutory provision.
                #
                # This deliberately prevents:
                #   BNS §101(a)
                #
                # when the (a) belongs to a nested Exception or
                # the chunk contains multiple semantic blocks.
                if (
                    len(items) == 1
                    and items[0][0] is not None
                    and items[0][2] is True
                    and items[0][1] == "provision"
                ):
                    citation_text += items[0][0]

            all_chunks.append(
                {
                    "chunk_id": chunk_id,
                    "act": act,
                    "chapter_number": (
                        unit["chapter"][0]
                        if unit["chapter"]
                        else None
                    ),
                    "chapter_title": (
                        unit["chapter"][1]
                        if unit["chapter"]
                        else None
                    ),
                    "section_number": unit["number"],
                    "section_title": unit["title"],
                    "subsection": subsection,
                    "content_type": content_type,
                    "content": (
                        context
                        + "\n\n"
                        + body
                    ),
                    "citation_id": chunk_id,
                    "citation_text": citation_text,
                    "source_file": source_file,
                }
            )

    return units, all_chunks


# ============================================================
# Validation
# ============================================================

REQUIRED_FIELDS = {
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
}


def validate_chunks(
    chunks: list[dict],
    units: list[dict],
    label: str,
) -> list[str]:

    errors = []

    ids = [chunk["chunk_id"] for chunk in chunks]

    duplicates = [
        chunk_id
        for chunk_id, count
        in Counter(ids).items()
        if count > 1
    ]

    if duplicates:
        errors.append(
            f"Duplicate chunk IDs: {duplicates[:10]}"
        )

    for chunk in chunks:

        missing = REQUIRED_FIELDS - set(chunk)

        if missing:
            errors.append(
                f"{chunk['chunk_id']}: "
                f"missing fields {sorted(missing)}"
            )

        if not chunk["content"].strip():
            errors.append(
                f"{chunk['chunk_id']}: empty content"
            )

        # We never want generated nested paths such as:
        # (24)(a)(3)(b)
        subsection = chunk["subsection"]

        if subsection and ")(" in subsection:
            errors.append(
                f"{chunk['chunk_id']}: "
                f"nested subsection metadata detected: "
                f"{subsection}"
            )

        # Chapter headings must not leak into chunk body.
        if re.search(
            r"^#\s+Chapter\b",
            chunk["content"],
            re.MULTILINE,
        ):
            errors.append(
                f"{chunk['chunk_id']}: "
                "chapter heading leaked into content"
            )

        # Horizontal separators are structural, not legal text.
        if re.search(
            r"^\s*---\s*$",
            chunk["content"],
            re.MULTILINE,
        ):
            errors.append(
                f"{chunk['chunk_id']}: "
                "horizontal separator leaked into content"
            )

        if chunk["content"].count("Marginal note:") > 1:
            errors.append(
                f"{chunk['chunk_id']}: "
                "duplicate marginal note"
            )

        if approx_tokens(chunk["content"]) > MAX_CHUNK_TOKENS:
            errors.append(
                f"{chunk['chunk_id']}: "
                f"approx token size "
                f"{approx_tokens(chunk['content'])} "
                f"> {MAX_CHUNK_TOKENS}"
            )

        # A subsection-specific citation must have a subsection.
        if (
            re.search(
                r"§[0-9]+(?:-[A-Za-z]+)?\([^)]+\)$",
                chunk["citation_text"],
            )
            and not chunk["subsection"]
        ):
            errors.append(
                f"{chunk['chunk_id']}: "
                "subsection citation without subsection metadata"
            )

    if not units:
        errors.append(
            f"{label}: no legal units detected"
        )

    return errors


# ============================================================
# Output
# ============================================================

def write_outputs(
    chunks: list[dict],
    prefix: str,
):
    jsonl_path = (
        OUTPUT_DIR / f"{prefix.lower()}_chunks.jsonl"
    )
    json_path = (
        OUTPUT_DIR / f"{prefix.lower()}_chunks.json"
    )

    with jsonl_path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as f:
        for chunk in chunks:
            f.write(
                json.dumps(
                    chunk,
                    ensure_ascii=False,
                )
                + "\n"
            )

    with json_path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as f:
        json.dump(
            chunks,
            f,
            ensure_ascii=False,
            indent=2,
        )

    return jsonl_path, json_path


# ============================================================
# Main
# ============================================================

def process_document(document: dict):

    input_path = document["path"]

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}"
        )

    print()
    print(f"Processing: {input_path}")

    lines = input_path.read_text(
        encoding="utf-8"
    ).splitlines()

    units, chunks = chunk_document(
        lines=lines,
        act=document["act"],
        source_file=input_path.name,
        prefix=document["prefix"],
    )

    errors = validate_chunks(
        chunks=chunks,
        units=units,
        label=document["prefix"],
    )

    jsonl_path, json_path = write_outputs(
        chunks=chunks,
        prefix=document["prefix"],
    )

    print(
        f"  Sections / blocks detected: {len(units)}"
    )
    print(f"  Chunks created: {len(chunks)}")
    print(
        "  Chunks with subsection: "
        f"{sum(c['subsection'] is not None for c in chunks)}"
    )
    print(
        "  Chunks without subsection: "
        f"{sum(c['subsection'] is None for c in chunks)}"
    )
    print(
        "  Content types: "
        f"{dict(Counter(c['content_type'] for c in chunks))}"
    )
    print(
        "  Approx max chunk tokens: "
        f"{max(approx_tokens(c['content']) for c in chunks)}"
    )
    print(f"  JSONL: {jsonl_path}")
    print(f"  JSON : {json_path}")

    if errors:
        print()
        print("  VALIDATION: FAILED")
        for error in errors:
            print(f"    - {error}")
    else:
        print("  VALIDATION: PASSED")

    return len(units), len(chunks), errors


def main():
    print("=" * 60)
    print("LEGAL RAG STRUCTURE-AWARE CHUNKER")
    print("=" * 60)

    total_chunks = 0
    all_errors = []

    for document in DOCUMENTS:
        try:
            units_count, chunks_count, errors = (
                process_document(document)
            )
            total_chunks += chunks_count
            all_errors.extend(errors)

        except Exception as exc:
            print(
                f"\nERROR processing "
                f"{document['path']}: {exc}"
            )
            all_errors.append(str(exc))

    print()
    print("=" * 60)

    if all_errors:
        print("CHUNKING COMPLETED WITH VALIDATION ERRORS")
        print("=" * 60)
        print(f"Total chunks: {total_chunks}")
        print(f"Validation errors: {len(all_errors)}")
    else:
        print("CHUNKING COMPLETE")
        print("=" * 60)
        print(f"Total chunks: {total_chunks}")
        print(f"Output directory: {OUTPUT_DIR}")
        print("Validation: PASSED")

    print("=" * 60)


if __name__ == "__main__":
    main()
