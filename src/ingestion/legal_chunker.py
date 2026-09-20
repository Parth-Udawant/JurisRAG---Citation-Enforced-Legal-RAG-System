"""
legal_chunker.py

Structure-aware hierarchical chunker for:
    - Bharatiya Nyaya Sanhita, 2023 (BNS)
    - Indian Contract Act, 1872 (ICA)

Input:
    data/processed/normalized/bns.md
    data/processed/normalized/indian-contract-act.md

Output:
    data/processed/chunks/bns_chunks.jsonl
    data/processed/chunks/bns_chunks.json

    data/processed/chunks/indian_contract_act_chunks.jsonl
    data/processed/chunks/indian_contract_act_chunks.json

Each chunk contains:

    chunk_id
    act
    chapter_number
    chapter_title
    section_number
    section_title
    subsection
    content_type
    content
    citation_id
    citation_text
    source_file

The chunker is intentionally conservative:
    - It does NOT rewrite legal wording.
    - It does NOT correct OCR.
    - It does NOT remove legal content.
    - It only identifies document hierarchy and creates chunks.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional


# ============================================================
# Configuration
# ============================================================

# Target size is measured approximately in tokens.
# We avoid depending on a tokenizer at this stage so that the
# chunker has no external model dependency.
TARGET_CHUNK_TOKENS = 450

# A chunk can exceed the target slightly when keeping a legal
# unit together.
MAX_CHUNK_TOKENS = 650

# If a single logical block is larger than this, it is split.
MAX_LOGICAL_BLOCK_TOKENS = 500

# Approximate token-to-character ratio.
# Legal English text is roughly 3.5-4.5 chars/token depending
# on punctuation. This is deliberately conservative.
CHARS_PER_TOKEN = 4


# ============================================================
# Data structures
# ============================================================

@dataclass
class SectionContext:
    number: str
    title: str
    marginal_note: Optional[str] = None


@dataclass
class ChapterContext:
    number: str
    title: str


@dataclass
class LegalBlock:
    """
    A semantically meaningful piece inside a section.

    Examples:
        subsection
        clause
        explanation
        exception
        illustration
        paragraph
    """

    text: str
    content_type: str = "provision"
    subsection: Optional[str] = None


@dataclass
class Section:
    number: str
    title: str
    chapter_number: Optional[str]
    chapter_title: Optional[str]

    marginal_note: Optional[str] = None

    blocks: list[LegalBlock] = field(default_factory=list)

    # Used for non-statutory/special sections such as:
    # Preamble, Related Judgements, Schedule.
    special_type: Optional[str] = None


# ============================================================
# Utility functions
# ============================================================

def normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def clean_heading_text(text: str) -> str:
    """
    Remove Markdown heading syntax and normalize whitespace.

    Does NOT change legal wording.
    """
    text = text.strip()

    # Remove trailing whitespace only.
    text = re.sub(r"[ \t]+$", "", text)

    return text


def approx_tokens(text: str) -> int:
    """
    Conservative approximate token count.

    We deliberately avoid loading a tokenizer because:
        1. Chunking should remain deterministic.
        2. Embedding model may change later.
        3. Exact tokenizer is better handled during embedding/indexing
           experiments.

    This is only used to control chunk size.
    """
    if not text:
        return 0

    return max(1, round(len(text) / CHARS_PER_TOKEN))


def slugify(value: str) -> str:
    value = value.lower().strip()

    value = re.sub(r"[^a-z0-9]+", "_", value)

    value = re.sub(r"_+", "_", value)

    return value.strip("_")


def clean_section_title(title: str) -> str:
    """
    Preserve the actual legal title while removing only the
    Markdown/heading delimiter.

    Supports:
        -
        –
        —
    """
    title = title.strip()

    # Remove one structural separator after the section number.
    title = re.sub(r"^\s*[-–—:]\s*", "", title)

    return title.strip()


def canonical_act_name(act: str) -> str:
    if act == "BNS":
        return "Bharatiya Nyaya Sanhita, 2023"

    if act == "ICA":
        return "Indian Contract Act, 1872"

    return act


def act_short_name(act: str) -> str:
    return {
        "BNS": "BNS",
        "ICA": "ICA",
    }.get(act, act)


# ============================================================
# Heading detection
# ============================================================

# Examples:
#
# # Chapter I — PRELIMINARY
# # Chapter II — ...
#
CHAPTER_RE = re.compile(
    r"^\s*#\s+Chapter\s+([A-Za-z0-9IVXLCDM]+)"
    r"\s*(?:[-–—:]\s*)?(.*?)\s*$",
    re.IGNORECASE,
)


# Examples:
#
# ## Section 1
# ## Section 1 — Short title
# ## Section 1 - Short title
# ## Section 19-A. ...
# ## Section 178A. ...
#
SECTION_RE = re.compile(
    r"^\s*##\s+Section\s+"
    r"([0-9]+(?:-[A-Za-z]+|[A-Za-z]+)?)"
    r"\s*(?:[-–—:.]\s*)?(.*?)\s*$",
    re.IGNORECASE,
)


# BNS:
#
# **Marginal note:** Definitions.
#
MARGINAL_NOTE_RE = re.compile(
    r"^\s*\*\*Marginal note:\*\*\s*(.*?)\s*$",
    re.IGNORECASE,
)


# Explanation / Exception / Illustration can appear as:
#
# **Explanation:**
# Explanation:
# **Exception:**
# Exception:
# **Illustration:**
# Illustration:
#
SPECIAL_MARKER_RE = re.compile(
    r"^\s*\**\s*(Explanation|Exception|Illustration)\s*:\s*\**\s*(.*?)\s*$",
    re.IGNORECASE,
)


# Examples:
#
# (1)
# (2)
# (a)
# (b)
# (i)
# (ii)
#
SUBSECTION_RE = re.compile(
    r"^\s*(\([0-9]+\))\s*(.*)$"
)


CLAUSE_RE = re.compile(
    r"^\s*(\([a-z]\))\s*(.*)$"
)


# ============================================================
# Markdown parser
# ============================================================

class LegalMarkdownParser:
    """
    Lightweight structural parser.

    It is intentionally NOT an AST framework.

    It only extracts enough structure to perform legal-aware
    chunking while preserving the source content.
    """

    def __init__(self, act: str):
        self.act = act

    def parse(self, markdown: str) -> list[Section]:
        markdown = normalize_newlines(markdown)

        lines = markdown.splitlines()

        chapters: list[ChapterContext] = []

        current_chapter: Optional[ChapterContext] = None
        current_section: Optional[Section] = None

        parsed_sections: list[Section] = []

        # Temporary content accumulated inside current section.
        current_blocks: list[LegalBlock] = []

        # Track current subsection so clauses can inherit it.
        current_subsection: Optional[str] = None

        def flush_section():
            nonlocal current_section
            nonlocal current_blocks
            nonlocal current_subsection

            if current_section is None:
                return

            current_section.blocks = current_blocks

            parsed_sections.append(current_section)

            current_section = None
            current_blocks = []
            current_subsection = None

        def add_text_to_current_block(
            text: str,
            content_type: str = "provision",
            subsection: Optional[str] = None,
        ):
            if not text.strip():
                return

            text = text.rstrip()

            # Merge with the previous block when it is the same
            # semantic type and same subsection.
            if current_blocks:
                previous = current_blocks[-1]

                if (
                    previous.content_type == content_type
                    and previous.subsection == subsection
                ):
                    previous.text += "\n\n" + text
                    return

            current_blocks.append(
                LegalBlock(
                    text=text,
                    content_type=content_type,
                    subsection=subsection,
                )
            )

        # ----------------------------------------------------
        # Main line-by-line parser
        # ----------------------------------------------------

        for raw_line in lines:

            line = raw_line.rstrip()

            # -----------------------------------------------
            # Chapter
            # -----------------------------------------------

            chapter_match = CHAPTER_RE.match(line)

            if chapter_match:
                flush_section()

                chapter_number = chapter_match.group(1).strip()
                chapter_title = chapter_match.group(2).strip()

                current_chapter = ChapterContext(
                    number=chapter_number,
                    title=chapter_title,
                )

                continue

            # -----------------------------------------------
            # Section
            # -----------------------------------------------

            section_match = SECTION_RE.match(line)

            if section_match:
                flush_section()

                section_number = section_match.group(1).strip()
                section_title = clean_section_title(
                    section_match.group(2)
                )

                current_section = Section(
                    number=section_number,
                    title=section_title,
                    chapter_number=(
                        current_chapter.number
                        if current_chapter
                        else None
                    ),
                    chapter_title=(
                        current_chapter.title
                        if current_chapter
                        else None
                    ),
                )

                continue

            # -----------------------------------------------
            # Preamble / Related Judgements / Schedule
            # -----------------------------------------------

            if re.match(
                r"^\s*##\s+(Preamble|Related Judgements|Schedule)\s*$",
                line,
                re.IGNORECASE,
            ):
                flush_section()

                special_name = re.sub(
                    r"^\s*##\s+",
                    "",
                    line,
                ).strip()

                current_section = Section(
                    number="",
                    title=special_name,
                    chapter_number=(
                        current_chapter.number
                        if current_chapter
                        else None
                    ),
                    chapter_title=(
                        current_chapter.title
                        if current_chapter
                        else None
                    ),
                    special_type=special_name.lower().replace(" ", "_"),
                )

                continue

            # -----------------------------------------------
            # Marginal note
            # -----------------------------------------------

            marginal_match = MARGINAL_NOTE_RE.match(line)

            if marginal_match and current_section:

                current_section.marginal_note = (
                    marginal_match.group(1).strip()
                )

                continue

            # -----------------------------------------------
            # Explanation / Exception / Illustration
            # -----------------------------------------------

            special_match = SPECIAL_MARKER_RE.match(line)

            if special_match and current_section:

                marker = special_match.group(1).lower()
                remainder = special_match.group(2).strip()

                content_type = marker

                if remainder:
                    add_text_to_current_block(
                        f"{special_match.group(1)}: {remainder}",
                        content_type=content_type,
                        subsection=current_subsection,
                    )
                else:
                    # Marker may be followed by content on later lines.
                    #
                    # Create a new semantic block with the marker.
                    add_text_to_current_block(
                        f"{special_match.group(1)}:",
                        content_type=content_type,
                        subsection=current_subsection,
                    )

                continue

            # -----------------------------------------------
            # Subsection
            # -----------------------------------------------

            subsection_match = SUBSECTION_RE.match(line)

            if subsection_match and current_section:

                current_subsection = subsection_match.group(1)

                remainder = subsection_match.group(2).strip()

                if remainder:
                    add_text_to_current_block(
                        f"{current_subsection} {remainder}",
                        content_type="provision",
                        subsection=current_subsection,
                    )
                else:
                    # Preserve subsection marker even when its text
                    # appears on subsequent lines.
                    add_text_to_current_block(
                        current_subsection,
                        content_type="provision",
                        subsection=current_subsection,
                    )

                continue

            # -----------------------------------------------
            # Clauses
            # -----------------------------------------------

            clause_match = CLAUSE_RE.match(line)

            if clause_match and current_section:

                clause = clause_match.group(1)
                remainder = clause_match.group(2).strip()

                if remainder:
                    add_text_to_current_block(
                        f"{clause} {remainder}",
                        content_type="provision",
                        subsection=current_subsection,
                    )
                else:
                    add_text_to_current_block(
                        clause,
                        content_type="provision",
                        subsection=current_subsection,
                    )

                continue

            # -----------------------------------------------
            # Ordinary content
            # -----------------------------------------------

            if current_section:

                if line.strip():
                    add_text_to_current_block(
                        line,
                        content_type="provision",
                        subsection=current_subsection,
                    )
                else:
                    # Preserve paragraph separation.
                    #
                    # We don't append empty blocks because the final
                    # renderer controls spacing.
                    continue

        flush_section()

        return parsed_sections


# ============================================================
# Semantic chunking
# ============================================================

class HierarchicalChunker:
    """
    Converts parsed legal sections into citation-ready chunks.

    Design principles:

    1. Preserve legal hierarchy.
    2. Preserve parent context.
    3. Keep semantic blocks together.
    4. Split only when necessary because of size.
    5. Never rewrite legal wording.
    6. Produce deterministic citation IDs.
    """

    def __init__(
        self,
        act: str,
        target_tokens: int = TARGET_CHUNK_TOKENS,
        max_tokens: int = MAX_CHUNK_TOKENS,
    ):
        self.act = act
        self.target_tokens = target_tokens
        self.max_tokens = max_tokens

    # --------------------------------------------------------
    # Citation
    # --------------------------------------------------------

    def build_citation(
        self,
        section_number: str,
        subsection: Optional[str],
    ) -> tuple[str, str]:

        short = act_short_name(self.act)

        normalized_section = section_number

        if subsection:
            subsection_clean = subsection.strip("()")

            citation_id = (
                f"{short}-{normalized_section}-{subsection_clean}"
            )

            citation_text = (
                f"{short} §{normalized_section}"
                f"{subsection}"
            )

        else:
            citation_id = (
                f"{short}-{normalized_section}"
            )

            citation_text = (
                f"{short} §{normalized_section}"
            )

        return citation_id, citation_text

    # --------------------------------------------------------
    # Parent context
    # --------------------------------------------------------

    def build_parent_context(
        self,
        section: Section,
    ) -> str:

        lines = [
            f"Act: {canonical_act_name(self.act)}",
        ]

        if section.chapter_number:
            lines.append(
                f"Chapter {section.chapter_number}"
                f" — {section.chapter_title}"
            )

        if section.number:
            lines.append(
                f"Section {section.number}"
                f" — {section.title}"
            )

        if section.marginal_note:
            lines.append(
                f"Marginal note: {section.marginal_note}"
            )

        return "\n".join(lines)

    # --------------------------------------------------------
    # Logical block splitting
    # --------------------------------------------------------

    def split_large_block(
        self,
        block: LegalBlock,
    ) -> list[LegalBlock]:

        if approx_tokens(block.text) <= self.max_tokens:
            return [block]

        # First attempt: paragraph-based splitting.
        paragraphs = re.split(
            r"\n\s*\n",
            block.text.strip(),
        )

        if len(paragraphs) <= 1:
            # No paragraph boundaries.
            # Fall back to sentence-aware splitting.
            return self.split_text_by_sentences(block)

        pieces: list[LegalBlock] = []

        current: list[str] = []
        current_tokens = 0

        for paragraph in paragraphs:

            paragraph = paragraph.strip()

            if not paragraph:
                continue

            paragraph_tokens = approx_tokens(paragraph)

            if (
                current
                and current_tokens + paragraph_tokens
                > self.max_tokens
            ):
                pieces.append(
                    LegalBlock(
                        text="\n\n".join(current),
                        content_type=block.content_type,
                        subsection=block.subsection,
                    )
                )

                current = []
                current_tokens = 0

            current.append(paragraph)
            current_tokens += paragraph_tokens

        if current:
            pieces.append(
                LegalBlock(
                    text="\n\n".join(current),
                    content_type=block.content_type,
                    subsection=block.subsection,
                )
            )

        return pieces

    def split_text_by_sentences(
        self,
        block: LegalBlock,
    ) -> list[LegalBlock]:

        text = block.text.strip()

        if approx_tokens(text) <= self.max_tokens:
            return [block]

        # Conservative sentence boundary.
        sentences = re.split(
            r"(?<=[.!?])\s+(?=[A-Z(])",
            text,
        )

        pieces: list[LegalBlock] = []

        current: list[str] = []
        current_tokens = 0

        for sentence in sentences:

            sentence = sentence.strip()

            if not sentence:
                continue

            sentence_tokens = approx_tokens(sentence)

            if (
                current
                and current_tokens + sentence_tokens
                > self.max_tokens
            ):
                pieces.append(
                    LegalBlock(
                        text=" ".join(current),
                        content_type=block.content_type,
                        subsection=block.subsection,
                    )
                )

                current = []
                current_tokens = 0

            current.append(sentence)
            current_tokens += sentence_tokens

        if current:
            pieces.append(
                LegalBlock(
                    text=" ".join(current),
                    content_type=block.content_type,
                    subsection=block.subsection,
                )
            )

        return pieces

    # --------------------------------------------------------
    # Build chunks
    # --------------------------------------------------------

    def chunk_section(
        self,
        section: Section,
    ) -> list[dict]:

        # ----------------------------------------------------
        # Special blocks
        # ----------------------------------------------------

        if section.special_type:

            content = "\n\n".join(
                block.text
                for block in section.blocks
                if block.text.strip()
            )

            if not content.strip():
                return []

            chunk_id = (
                f"{act_short_name(self.act)}-"
                f"{slugify(section.special_type)}"
            )

            return [
                {
                    "chunk_id": chunk_id,
                    "act": canonical_act_name(self.act),
                    "chapter_number": section.chapter_number,
                    "chapter_title": section.chapter_title,
                    "section_number": None,
                    "section_title": section.title,
                    "subsection": None,
                    "content_type": section.special_type,
                    "content": (
                        f"Act: {canonical_act_name(self.act)}\n"
                        f"{section.title}\n\n"
                        f"{content}"
                    ),
                    "citation_id": chunk_id,
                    "citation_text": (
                        f"{canonical_act_name(self.act)}, "
                        f"{section.title}"
                    ),
                    "source_file": None,
                }
            ]

        # ----------------------------------------------------
        # Expand oversized blocks
        # ----------------------------------------------------

        logical_blocks: list[LegalBlock] = []

        for block in section.blocks:
            logical_blocks.extend(
                self.split_large_block(block)
            )

        if not logical_blocks:
            return []

        # ----------------------------------------------------
        # Accumulate blocks into chunks
        # ----------------------------------------------------

        chunks: list[dict] = []

        current_blocks: list[LegalBlock] = []
        current_tokens = 0

        def emit_current():
            nonlocal current_blocks
            nonlocal current_tokens

            if not current_blocks:
                return

            # Determine dominant/first subsection.
            subsection = None

            for block in current_blocks:
                if block.subsection:
                    subsection = block.subsection
                    break

            citation_id, citation_text = self.build_citation(
                section.number,
                subsection,
            )

            parent_context = self.build_parent_context(
                section
            )

            body_parts = []

            for block in current_blocks:
                body_parts.append(
                    block.text.strip()
                )

            body = "\n\n".join(
                part
                for part in body_parts
                if part
            )

            content = (
                f"{parent_context}\n\n"
                f"{body}"
            )

            # Determine content type.
            types = {
                block.content_type
                for block in current_blocks
            }

            if len(types) == 1:
                content_type = next(iter(types))
            else:
                content_type = "mixed"

            chunk_index = len(chunks) + 1

            # Include chunk index only when a section has multiple
            # chunks. This makes IDs deterministic and unique.
            base_chunk_id = (
                f"{act_short_name(self.act)}-"
                f"ch{slugify(section.chapter_number or 'na')}-"
                f"s{slugify(section.number)}"
            )

            if len(logical_blocks) == 1:
                chunk_id = base_chunk_id
            else:
                chunk_id = (
                    f"{base_chunk_id}-"
                    f"c{chunk_index}"
                )

            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "act": canonical_act_name(self.act),
                    "chapter_number": section.chapter_number,
                    "chapter_title": section.chapter_title,
                    "section_number": section.number,
                    "section_title": section.title,
                    "subsection": subsection,
                    "content_type": content_type,
                    "content": content,
                    "citation_id": citation_id,
                    "citation_text": citation_text,
                    "source_file": None,
                }
            )

            current_blocks = []
            current_tokens = 0

        for block in logical_blocks:

            block_tokens = approx_tokens(block.text)

            # Never split a semantic block if it fits within
            # the maximum size.
            if (
                current_blocks
                and current_tokens + block_tokens
                > self.target_tokens
            ):
                emit_current()

            current_blocks.append(block)
            current_tokens += block_tokens

        emit_current()

        return chunks

    # --------------------------------------------------------
    # Whole document
    # --------------------------------------------------------

    def chunk_document(
        self,
        sections: list[Section],
        source_file: str,
    ) -> list[dict]:

        all_chunks: list[dict] = []

        for section in sections:

            section_chunks = self.chunk_section(
                section
            )

            for chunk in section_chunks:
                chunk["source_file"] = source_file

            all_chunks.extend(section_chunks)

        return all_chunks


# ============================================================
# Validation
# ============================================================

def validate_chunks(
    chunks: list[dict],
    source_file: str,
) -> None:

    required_fields = {
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

    for index, chunk in enumerate(chunks):

        missing = required_fields - set(chunk.keys())

        if missing:
            raise ValueError(
                f"{source_file}: chunk {index} is missing "
                f"fields: {sorted(missing)}"
            )

        if not chunk["chunk_id"]:
            raise ValueError(
                f"{source_file}: empty chunk_id "
                f"at chunk {index}"
            )

        if not chunk["content"].strip():
            raise ValueError(
                f"{source_file}: empty content "
                f"for chunk {index}"
            )

        if not chunk["citation_id"]:
            raise ValueError(
                f"{source_file}: empty citation_id "
                f"for chunk {index}"
            )

        if not chunk["citation_text"]:
            raise ValueError(
                f"{source_file}: empty citation_text "
                f"for chunk {index}"
            )

    # Check chunk IDs are unique.
    ids = [chunk["chunk_id"] for chunk in chunks]

    duplicates = {
        item
        for item in ids
        if ids.count(item) > 1
    }

    if duplicates:
        raise ValueError(
            f"{source_file}: duplicate chunk IDs: "
            f"{sorted(duplicates)}"
        )


# ============================================================
# I/O
# ============================================================

def write_jsonl(
    chunks: list[dict],
    output_path: Path,
) -> None:

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
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


def write_json(
    chunks: list[dict],
    output_path: Path,
) -> None:

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
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


# ============================================================
# File processing
# ============================================================

def process_file(
    input_path: Path,
    output_dir: Path,
    act: str,
    target_tokens: int,
    max_tokens: int,
) -> list[dict]:

    print(f"\nProcessing: {input_path}")

    markdown = input_path.read_text(
        encoding="utf-8"
    )

    parser = LegalMarkdownParser(
        act=act
    )

    sections = parser.parse(markdown)

    print(
        f"  Sections/blocks detected: "
        f"{len(sections)}"
    )

    chunker = HierarchicalChunker(
        act=act,
        target_tokens=target_tokens,
        max_tokens=max_tokens,
    )

    chunks = chunker.chunk_document(
        sections=sections,
        source_file=input_path.name,
    )

    validate_chunks(
        chunks,
        input_path.name,
    )

    stem = input_path.stem

    jsonl_path = (
        output_dir /
        f"{stem}_chunks.jsonl"
    )

    json_path = (
        output_dir /
        f"{stem}_chunks.json"
    )

    write_jsonl(
        chunks,
        jsonl_path,
    )

    write_json(
        chunks,
        json_path,
    )

    print(
        f"  Chunks created: {len(chunks)}"
    )

    print(
        f"  JSONL: {jsonl_path}"
    )

    print(
        f"  JSON : {json_path}"
    )

    return chunks


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Structure-aware hierarchical chunker "
            "for Indian legal Markdown."
        )
    )

    parser.add_argument(
        "--bns",
        type=Path,
        default=Path(
            r"D:\Programming\AI_Projects\legal-rag-azure\data\processed\normalized\bns.md"
        ),
        help="Path to BNS Markdown.",
    )

    parser.add_argument(
        "--contract",
        type=Path,
        default=Path(
            r"D:\Programming\AI_Projects\legal-rag-azure\data\processed\normalized\indian-contract-act.md"
        ),
        help="Path to Indian Contract Act Markdown.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            r"D:\Programming\AI_Projects\legal-rag-azure\data\processed\chunks"
        ),
        help=(
            "Output directory. "
            "Default: data/processed/chunks"
        ),
    )

    parser.add_argument(
        "--target-tokens",
        type=int,
        default=TARGET_CHUNK_TOKENS,
        help=(
            f"Target chunk size. "
            f"Default: {TARGET_CHUNK_TOKENS}"
        ),
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=MAX_CHUNK_TOKENS,
        help=(
            f"Maximum approximate chunk size. "
            f"Default: {MAX_CHUNK_TOKENS}"
        ),
    )

    args = parser.parse_args()

    if not args.bns.exists():
        raise FileNotFoundError(
            f"BNS Markdown not found: {args.bns}"
        )

    if not args.contract.exists():
        raise FileNotFoundError(
            f"Contract Act Markdown not found: "
            f"{args.contract}"
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    bns_chunks = process_file(
        input_path=args.bns,
        output_dir=args.output_dir,
        act="BNS",
        target_tokens=args.target_tokens,
        max_tokens=args.max_tokens,
    )

    contract_chunks = process_file(
        input_path=args.contract,
        output_dir=args.output_dir,
        act="ICA",
        target_tokens=args.target_tokens,
        max_tokens=args.max_tokens,
    )

    print("\n" + "=" * 60)
    print("CHUNKING COMPLETE")
    print("=" * 60)

    print(
        f"BNS chunks:              {len(bns_chunks)}"
    )

    print(
        f"Indian Contract Act:     "
        f"{len(contract_chunks)}"
    )

    print(
        f"Output directory:         "
        f"{args.output_dir.resolve()}"
    )


if __name__ == "__main__":
    main()