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

    data/processed/chunks/indian-contract-act_chunks.jsonl
    data/processed/chunks/indian-contract-act_chunks.json


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


Design goals:

    1. Preserve legal wording.
    2. Preserve parent context.
    3. Recognize BNS and Contract Act structures.
    4. Recognize bold Markdown subsection markers.
    5. Recognize nested legal subdivisions.
    6. Preserve marginal notes.
    7. Preserve explanations, exceptions and illustrations.
    8. Keep semantically related content together.
    9. Split only when chunks become too large.
    10. Produce citation-ready evidence objects.
    11. Produce deterministic, unique citation IDs.
"""

from __future__ import annotations

import argparse
import json
import re

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ============================================================
# Configuration
# ============================================================

# Approximate target size.
#
# This is deliberately not tied to a specific embedding model
# tokenizer yet. We can replace this with exact tokenizer
# counting later if required.
TARGET_CHUNK_TOKENS = 450

# A logical block can exceed the target slightly.
MAX_CHUNK_TOKENS = 650

# Approximate characters per token for legal English.
CHARS_PER_TOKEN = 4


# ============================================================
# Data structures
# ============================================================

@dataclass
class LegalBlock:
    """
    A semantically meaningful unit within a section.

    Examples:

        provision
        explanation
        exception
        illustration
    """

    text: str

    content_type: str = "provision"

    # Examples:
    #   (1)
    #   (8)
    #   (28)
    #   (28)(h)
    subsection: Optional[str] = None


@dataclass
class Section:
    """
    Represents a statutory section or a recognized special block.
    """

    number: str

    title: str

    chapter_number: Optional[str]

    chapter_title: Optional[str]

    marginal_note: Optional[str] = None

    blocks: list[LegalBlock] = field(
        default_factory=list
    )

    # Used for:
    #   Preamble
    #   Related Judgements
    #   Schedule
    special_type: Optional[str] = None


# ============================================================
# Regex patterns
# ============================================================

# Example:
#
# # Chapter I — PRELIMINARY
#
CHAPTER_RE = re.compile(
    r"^\s*#\s+Chapter\s+"
    r"([A-Za-z0-9IVXLCDM]+)"
    r"\s*(?:[-–—:]\s*)?"
    r"(.*?)\s*$",
    re.IGNORECASE,
)


# Supports:
#
# ## Section 1
# ## Section 1 - Short title
# ## Section 1 — Short title
# ## Section 19-A. ...
# ## Section 178A. ...
#
SECTION_RE = re.compile(
    r"^\s*##\s+Section\s+"
    r"([0-9]+(?:-[A-Za-z]+|[A-Za-z]+)?)"
    r"\s*(?:[-–—:.]\s*)?"
    r"(.*?)\s*$",
    re.IGNORECASE,
)


# Contract Act special blocks.
#
# ## Preamble
# ## Related Judgements
# ## Schedule
#
SPECIAL_SECTION_RE = re.compile(
    r"^\s*##\s+"
    r"(Preamble|Related Judgements|Schedule)"
    r"\s*$",
    re.IGNORECASE,
)


# BNS:
#
# **Marginal note:** Definitions.
#
MARGINAL_NOTE_RE = re.compile(
    r"^\s*\*\*Marginal note:\*\*\s*"
    r"(.*?)\s*$",
    re.IGNORECASE,
)


# Supports BOTH:
#
# Explanation:
# **Explanation:**
# Exception:
# **Exception:**
# Illustration:
# **Illustration:**
#
SPECIAL_MARKER_RE = re.compile(
    r"^\s*"
    r"(?:\*\*)?"
    r"(Explanation|Exception|Illustration)"
    r"\s*:\s*"
    r"(?:\*\*)?"
    r"(.*?)\s*$",
    re.IGNORECASE,
)


# IMPORTANT:
#
# Your actual Markdown uses:
#
# **(1)** ...
# **(a)** ...
# **(i)** ...
#
# The previous version did not account for **.
#
# This regex recognizes:
#
# (1)
# **(1)**
# (a)
# **(a)**
# (i)
# **(i)**
#
SUBDIVISION_RE = re.compile(
    r"^\s*"
    r"(?:\*\*)?"
    r"(\(\d+\)|\([a-z]\)|\([ivxlcdm]+\))"
    r"(?:\*\*)?"
    r"\s*(.*)$",
    re.IGNORECASE,
)


# ============================================================
# Utility functions
# ============================================================

def approx_tokens(text: str) -> int:
    """
    Approximate token count.

    This is only used for chunk-size control.

    Later, once the exact Azure OpenAI embedding model is selected,
    this can optionally be replaced with its tokenizer.
    """

    if not text:
        return 0

    return max(
        1,
        round(len(text) / CHARS_PER_TOKEN),
    )


def slugify(value: str) -> str:
    """
    Create a deterministic ID-safe representation.
    """

    value = value.lower().strip()

    value = re.sub(
        r"[^a-z0-9]+",
        "_",
        value,
    )

    value = re.sub(
        r"_+",
        "_",
        value,
    )

    return value.strip("_")


def canonical_act_name(act: str) -> str:

    if act == "BNS":
        return "Bharatiya Nyaya Sanhita, 2023"

    if act == "ICA":
        return "Indian Contract Act, 1872"

    return act


def act_short_name(act: str) -> str:

    if act == "BNS":
        return "BNS"

    if act == "ICA":
        return "ICA"

    return act


# ============================================================
# Legal subdivision hierarchy
# ============================================================

def marker_level(marker: str) -> str:
    """
    Determine the structural level of a legal marker.

    Examples:

        (1)  -> numeric
        (a)  -> alpha
        (i)  -> roman
    """

    value = marker[1:-1].lower()

    if value.isdigit():
        return "numeric"

    roman_values = {
        "i",
        "ii",
        "iii",
        "iv",
        "v",
        "vi",
        "vii",
        "viii",
        "ix",
        "x",
        "xi",
        "xii",
        "xiii",
        "xiv",
        "xv",
    }

    if value in roman_values:
        return "roman"

    return "alpha"


def update_subdivision_path(
    current_path: list[str],
    marker: str,
) -> list[str]:
    """
    Maintain a legal subdivision path.

    Examples:

        (1)
        (1)(a)
        (16)
        (16)(i)
        (28)
        (28)(a)

    The resulting path is represented as:

        ["(1)"]
        ["(1)", "(a)"]
        ["(16)", "(i)"]
        ["(28)", "(a)"]
    """

    level = marker_level(marker)

    # --------------------------------------------------------
    # Numeric
    # --------------------------------------------------------

    if level == "numeric":

        # If a numeric marker follows an alpha/roman marker,
        # it may be nested.
        if current_path:
            last_level = marker_level(
                current_path[-1]
            )

            if last_level in {
                "alpha",
                "roman",
            }:
                return current_path + [marker]

        # Otherwise this is a new top-level numeric subsection.
        return [marker]

    # --------------------------------------------------------
    # Alpha
    # --------------------------------------------------------

    if level == "alpha":

        if current_path:

            last_level = marker_level(
                current_path[-1]
            )

            # (1) -> (1)(a)
            if last_level == "numeric":
                return current_path + [marker]

            # (a) -> (b)
            if last_level == "alpha":
                return current_path[:-1] + [marker]

            # (i) -> (a)
            if last_level == "roman":
                return current_path[:-1] + [marker]

        return [marker]

    # --------------------------------------------------------
    # Roman
    # --------------------------------------------------------

    if level == "roman":

        if current_path:
            return current_path + [marker]

        return [marker]

    return [marker]


# ============================================================
# Markdown parser
# ============================================================

class LegalMarkdownParser:
    """
    Lightweight structure-aware parser.

    This is NOT intended to create a general-purpose AST.

    It extracts only the hierarchy required for legal chunking.
    """

    def parse(
        self,
        markdown: str,
    ) -> list[Section]:

        lines = markdown.replace(
            "\r\n",
            "\n",
        ).replace(
            "\r",
            "\n",
        ).splitlines()

        sections: list[Section] = []

        current_chapter_number: Optional[str] = None
        current_chapter_title: Optional[str] = None

        current_section: Optional[Section] = None

        current_blocks: list[LegalBlock] = []

        # Current legal subdivision.
        subdivision_path: list[str] = []

        # Current semantic block type.
        #
        # Important for:
        #
        # **Illustration:**
        #
        # paragraph 1
        #
        # paragraph 2
        #
        # Both paragraphs remain illustrations until another
        # structural marker appears.
        current_content_type = "provision"

        # ----------------------------------------------------
        # Helper: flush current section
        # ----------------------------------------------------

        def flush_section():

            nonlocal current_section
            nonlocal current_blocks
            nonlocal subdivision_path
            nonlocal current_content_type

            if current_section is not None:

                current_section.blocks = current_blocks

                sections.append(
                    current_section
                )

            current_section = None

            current_blocks = []

            subdivision_path = []

            current_content_type = "provision"

        # ----------------------------------------------------
        # Helper: append content
        # ----------------------------------------------------

        def add_block(
            text: str,
            content_type: str,
            subsection: Optional[str],
        ):

            if not text.strip():
                return

            text = text.rstrip()

            # Merge adjacent content belonging to the same
            # semantic/legal subdivision.
            if current_blocks:

                previous = current_blocks[-1]

                if (
                    previous.content_type
                    == content_type
                    and previous.subsection
                    == subsection
                ):

                    previous.text += (
                        "\n\n" + text
                    )

                    return

            current_blocks.append(
                LegalBlock(
                    text=text,
                    content_type=content_type,
                    subsection=subsection,
                )
            )

        # ----------------------------------------------------
        # Main parser
        # ----------------------------------------------------

        for raw_line in lines:

            line = raw_line.rstrip()

            # ================================================
            # Chapter
            # ================================================

            chapter_match = CHAPTER_RE.match(
                line
            )

            if chapter_match:

                flush_section()

                current_chapter_number = (
                    chapter_match.group(1).strip()
                )

                current_chapter_title = (
                    chapter_match.group(2).strip()
                )

                continue

            # ================================================
            # Section
            # ================================================

            section_match = SECTION_RE.match(
                line
            )

            if section_match:

                flush_section()

                section_number = (
                    section_match.group(1).strip()
                )

                section_title = (
                    section_match.group(2).strip()
                )

                current_section = Section(
                    number=section_number,
                    title=section_title,
                    chapter_number=(
                        current_chapter_number
                    ),
                    chapter_title=(
                        current_chapter_title
                    ),
                )

                continue

            # ================================================
            # Special Contract Act sections
            # ================================================

            special_section_match = (
                SPECIAL_SECTION_RE.match(line)
            )

            if special_section_match:

                flush_section()

                name = (
                    special_section_match
                    .group(1)
                    .strip()
                )

                current_section = Section(
                    number="",
                    title=name,
                    chapter_number=(
                        current_chapter_number
                    ),
                    chapter_title=(
                        current_chapter_title
                    ),
                    special_type=(
                        name.lower()
                        .replace(" ", "_")
                    ),
                )

                continue

            # ================================================
            # BNS marginal note
            # ================================================

            marginal_match = MARGINAL_NOTE_RE.match(
                line
            )

            if (
                marginal_match
                and current_section is not None
            ):

                current_section.marginal_note = (
                    marginal_match
                    .group(1)
                    .strip()
                )

                continue

            # ================================================
            # Explanation / Exception / Illustration
            # ================================================

            special_match = SPECIAL_MARKER_RE.match(
                line
            )

            if (
                special_match
                and current_section is not None
            ):

                current_content_type = (
                    special_match
                    .group(1)
                    .lower()
                )

                # IMPORTANT:
                #
                # Preserve the ORIGINAL Markdown line.
                #
                # We recognize **Illustration:** as an
                # illustration marker, but we do not rewrite
                # the legal source text.
                add_block(
                    text=line.strip(),
                    content_type=current_content_type,
                    subsection=(
                        "".join(subdivision_path)
                        if subdivision_path
                        else None
                    ),
                )

                continue

            # ================================================
            # Subsection / clause / nested subdivision
            # ================================================

            subdivision_match = (
                SUBDIVISION_RE.match(line)
            )

            if (
                subdivision_match
                and current_section is not None
            ):

                marker = (
                    subdivision_match
                    .group(1)
                    .strip()
                )

                subdivision_path = (
                    update_subdivision_path(
                        subdivision_path,
                        marker,
                    )
                )

                # A new legal subdivision starts a normal
                # provision block.
                current_content_type = "provision"

                # Preserve the exact source line.
                add_block(
                    text=line.strip(),
                    content_type="provision",
                    subsection=(
                        "".join(subdivision_path)
                    ),
                )

                continue

            # ================================================
            # Ordinary legal text
            # ================================================

            if (
                current_section is not None
                and line.strip()
            ):

                add_block(
                    text=line,
                    content_type=current_content_type,
                    subsection=(
                        "".join(subdivision_path)
                        if subdivision_path
                        else None
                    ),
                )

        # Flush final section.
        flush_section()

        return sections


# ============================================================
# Chunker
# ============================================================

class HierarchicalLegalChunker:
    """
    Creates citation-ready legal evidence chunks.
    """

    def __init__(
        self,
        act: str,
        target_tokens: int = TARGET_CHUNK_TOKENS,
        max_tokens: int = MAX_CHUNK_TOKENS,
    ):

        self.act = act

        self.target_tokens = (
            target_tokens
        )

        self.max_tokens = (
            max_tokens
        )

    # --------------------------------------------------------
    # Parent context
    # --------------------------------------------------------

    def parent_context(
        self,
        section: Section,
    ) -> str:

        lines = [
            f"Act: {canonical_act_name(self.act)}"
        ]

        if section.chapter_number:

            lines.append(
                f"Chapter "
                f"{section.chapter_number}"
                f" — "
                f"{section.chapter_title}"
            )

        if section.number:

            lines.append(
                f"Section "
                f"{section.number}"
                f" — "
                f"{section.title}"
            )

        if section.marginal_note:

            lines.append(
                "Marginal note: "
                f"{section.marginal_note}"
            )

        return "\n".join(lines)

    # --------------------------------------------------------
    # Split oversized block
    # --------------------------------------------------------

    def split_large_block(
        self,
        block: LegalBlock,
    ) -> list[LegalBlock]:

        if (
            approx_tokens(block.text)
            <= self.max_tokens
        ):

            return [block]

        # ----------------------------------------------------
        # First: paragraph boundaries
        # ----------------------------------------------------

        paragraphs = re.split(
            r"\n\s*\n",
            block.text.strip(),
        )

        if len(paragraphs) > 1:

            result: list[LegalBlock] = []

            current_parts: list[str] = []

            current_tokens = 0

            for paragraph in paragraphs:

                paragraph = paragraph.strip()

                if not paragraph:
                    continue

                tokens = approx_tokens(
                    paragraph
                )

                if (
                    current_parts
                    and current_tokens + tokens
                    > self.max_tokens
                ):

                    result.append(
                        LegalBlock(
                            text="\n\n".join(
                                current_parts
                            ),
                            content_type=(
                                block.content_type
                            ),
                            subsection=(
                                block.subsection
                            ),
                        )
                    )

                    current_parts = []

                    current_tokens = 0

                current_parts.append(
                    paragraph
                )

                current_tokens += tokens

            if current_parts:

                result.append(
                    LegalBlock(
                        text="\n\n".join(
                            current_parts
                        ),
                        content_type=(
                            block.content_type
                        ),
                        subsection=(
                            block.subsection
                        ),
                    )
                )

            return result

        # ----------------------------------------------------
        # Second: sentence boundaries
        # ----------------------------------------------------

        sentences = re.split(
            r"(?<=[.!?])\s+(?=[A-Z(])",
            block.text.strip(),
        )

        result: list[LegalBlock] = []

        current_parts: list[str] = []

        current_tokens = 0

        for sentence in sentences:

            sentence = sentence.strip()

            if not sentence:
                continue

            tokens = approx_tokens(
                sentence
            )

            if (
                current_parts
                and current_tokens + tokens
                > self.max_tokens
            ):

                result.append(
                    LegalBlock(
                        text=" ".join(
                            current_parts
                        ),
                        content_type=(
                            block.content_type
                        ),
                        subsection=(
                            block.subsection
                        ),
                    )
                )

                current_parts = []

                current_tokens = 0

            current_parts.append(
                sentence
            )

            current_tokens += tokens

        if current_parts:

            result.append(
                LegalBlock(
                    text=" ".join(
                        current_parts
                    ),
                    content_type=(
                        block.content_type
                    ),
                    subsection=(
                        block.subsection
                    ),
                )
            )

        return result

    # --------------------------------------------------------
    # Chunk one section
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

                    "act": canonical_act_name(
                        self.act
                    ),

                    "chapter_number": (
                        section.chapter_number
                    ),

                    "chapter_title": (
                        section.chapter_title
                    ),

                    "section_number": None,

                    "section_title": (
                        section.title
                    ),

                    "subsection": None,

                    "content_type": (
                        section.special_type
                    ),

                    "content": (
                        f"Act: "
                        f"{canonical_act_name(self.act)}"
                        "\n"
                        f"{section.title}"
                        "\n\n"
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
        # Expand blocks that are too large
        # ----------------------------------------------------

        logical_blocks: list[LegalBlock] = []

        for block in section.blocks:

            logical_blocks.extend(
                self.split_large_block(
                    block
                )
            )

        if not logical_blocks:
            return []

        # ----------------------------------------------------
        # Build chunks
        # ----------------------------------------------------

        chunk_groups: list[list[LegalBlock]] = []

        current_group: list[LegalBlock] = []

        current_tokens = 0

        for block in logical_blocks:

            block_tokens = approx_tokens(
                block.text
            )

            # Keep semantically coherent blocks together
            # until the target size is reached.
            if (
                current_group
                and current_tokens + block_tokens
                > self.target_tokens
            ):

                chunk_groups.append(
                    current_group
                )

                current_group = []

                current_tokens = 0

            current_group.append(
                block
            )

            current_tokens += block_tokens

        if current_group:

            chunk_groups.append(
                current_group
            )

        # ----------------------------------------------------
        # Convert groups into evidence objects
        # ----------------------------------------------------

        results: list[dict] = []

        parent = self.parent_context(
            section
        )

        base_id = (
            f"{act_short_name(self.act)}"
            f"-ch{slugify(section.chapter_number or 'na')}"
            f"-s{slugify(section.number or section.special_type or 'na')}"
        )

        for index, group in enumerate(
            chunk_groups,
            start=1,
        ):

            # -----------------------------------------------
            # Determine subsection metadata
            # -----------------------------------------------

            subsection_values = sorted(
                {
                    block.subsection
                    for block in group
                    if block.subsection
                }
            )

            if len(subsection_values) == 1:

                subsection = (
                    subsection_values[0]
                )

            elif subsection_values:

                # Multiple subdivisions occur in the same
                # semantic chunk.
                #
                # We preserve the range/context as a
                # comma-separated value rather than
                # fabricating a single subsection.
                subsection = ", ".join(
                    subsection_values
                )

            else:

                subsection = None

            # -----------------------------------------------
            # Content type
            # -----------------------------------------------

            content_types = {
                block.content_type
                for block in group
            }

            if len(content_types) == 1:

                content_type = next(
                    iter(content_types)
                )

            else:

                content_type = "mixed"

            # -----------------------------------------------
            # Content
            # -----------------------------------------------

            body = "\n\n".join(
                block.text.strip()
                for block in group
                if block.text.strip()
            )

            content = (
                f"{parent}\n\n"
                f"{body}"
            )

            # -----------------------------------------------
            # Citation identity
            # -----------------------------------------------

            # IMPORTANT:
            #
            # citation_id is unique per chunk.
            #
            # This is required for reliable citation
            # enforcement later.
            citation_id = (
                f"{base_id}-c{index}"
            )

            # Human-readable citation.
            #
            # If the entire chunk belongs to one subsection,
            # include it.
            #
            # Otherwise cite the parent section.
            if (
                section.number
                and len(subsection_values) == 1
            ):

                citation_text = (
                    f"{act_short_name(self.act)} "
                    f"§{section.number}"
                    f"{subsection_values[0]}"
                )

            elif section.number:

                citation_text = (
                    f"{act_short_name(self.act)} "
                    f"§{section.number}"
                )

            else:

                citation_text = (
                    f"{canonical_act_name(self.act)}, "
                    f"{section.title}"
                )

            # -----------------------------------------------
            # Final evidence object
            # -----------------------------------------------

            results.append(
                {
                    "chunk_id": citation_id,

                    "act": canonical_act_name(
                        self.act
                    ),

                    "chapter_number": (
                        section.chapter_number
                    ),

                    "chapter_title": (
                        section.chapter_title
                    ),

                    "section_number": (
                        section.number or None
                    ),

                    "section_title": (
                        section.title
                    ),

                    "subsection": subsection,

                    "content_type": content_type,

                    "content": content,

                    "citation_id": citation_id,

                    "citation_text": citation_text,

                    "source_file": None,
                }
            )

        return results

    # --------------------------------------------------------
    # Chunk document
    # --------------------------------------------------------

    def chunk_document(
        self,
        sections: list[Section],
        source_file: str,
    ) -> list[dict]:

        all_chunks: list[dict] = []

        for section in sections:

            section_chunks = (
                self.chunk_section(
                    section
                )
            )

            for chunk in section_chunks:

                chunk["source_file"] = (
                    source_file
                )

            all_chunks.extend(
                section_chunks
            )

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

    # --------------------------------------------------------
    # Field validation
    # --------------------------------------------------------

    for index, chunk in enumerate(chunks):

        missing = (
            required_fields
            - set(chunk.keys())
        )

        if missing:

            raise ValueError(
                f"{source_file}: "
                f"chunk {index} missing fields: "
                f"{sorted(missing)}"
            )

        if not chunk["chunk_id"]:

            raise ValueError(
                f"{source_file}: "
                f"empty chunk_id at "
                f"chunk {index}"
            )

        if not chunk["content"].strip():

            raise ValueError(
                f"{source_file}: "
                f"empty content at "
                f"chunk {index}"
            )

        if not chunk["citation_id"]:

            raise ValueError(
                f"{source_file}: "
                f"empty citation_id at "
                f"chunk {index}"
            )

        if not chunk["citation_text"]:

            raise ValueError(
                f"{source_file}: "
                f"empty citation_text at "
                f"chunk {index}"
            )

    # --------------------------------------------------------
    # Unique chunk IDs
    # --------------------------------------------------------

    chunk_ids = [
        chunk["chunk_id"]
        for chunk in chunks
    ]

    if len(chunk_ids) != len(
        set(chunk_ids)
    ):

        raise ValueError(
            f"{source_file}: duplicate "
            f"chunk_id detected."
        )

    # --------------------------------------------------------
    # Unique citation IDs
    # --------------------------------------------------------

    citation_ids = [
        chunk["citation_id"]
        for chunk in chunks
    ]

    if len(citation_ids) != len(
        set(citation_ids)
    ):

        raise ValueError(
            f"{source_file}: duplicate "
            f"citation_id detected."
        )


# ============================================================
# Output
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
    ) as file:

        for chunk in chunks:

            file.write(
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
    ) as file:

        json.dump(
            chunks,
            file,
            ensure_ascii=False,
            indent=2,
        )


# ============================================================
# Processing
# ============================================================

def process_file(
    input_path: Path,
    output_dir: Path,
    act: str,
    target_tokens: int,
    max_tokens: int,
) -> list[dict]:

    print()
    print(
        f"Processing: {input_path}"
    )

    markdown = input_path.read_text(
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # Parse
    # --------------------------------------------------------

    parser = LegalMarkdownParser()

    sections = parser.parse(
        markdown
    )

    # --------------------------------------------------------
    # Chunk
    # --------------------------------------------------------

    chunker = HierarchicalLegalChunker(
        act=act,
        target_tokens=target_tokens,
        max_tokens=max_tokens,
    )

    chunks = chunker.chunk_document(
        sections=sections,
        source_file=input_path.name,
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_chunks(
        chunks=chunks,
        source_file=input_path.name,
    )

    # --------------------------------------------------------
    # Output names
    # --------------------------------------------------------

    stem = input_path.stem

    jsonl_path = (
        output_dir
        / f"{stem}_chunks.jsonl"
    )

    json_path = (
        output_dir
        / f"{stem}_chunks.json"
    )

    write_jsonl(
        chunks,
        jsonl_path,
    )

    write_json(
        chunks,
        json_path,
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    subsection_count = sum(
        1
        for chunk in chunks
        if chunk["subsection"] is not None
    )

    content_types = {}

    for chunk in chunks:

        content_type = (
            chunk["content_type"]
        )

        content_types[
            content_type
        ] = (
            content_types.get(
                content_type,
                0,
            )
            + 1
        )

    print(
        f"  Sections / blocks detected: "
        f"{len(sections)}"
    )

    print(
        f"  Chunks created: "
        f"{len(chunks)}"
    )

    print(
        f"  Chunks with subsection: "
        f"{subsection_count}"
    )

    print(
        f"  Chunks without subsection: "
        f"{len(chunks) - subsection_count}"
    )

    print(
        f"  Content types: "
        f"{content_types}"
    )

    print(
        f"  JSONL: {jsonl_path}"
    )

    print(
        f"  JSON : {json_path}"
    )

    return chunks


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Structure-aware hierarchical "
            "legal Markdown chunker."
        )
    )

    parser.add_argument(
        "--bns",
        type=Path,
        default=Path(
            r"D:\Programming\AI_Projects\legal-rag-azure\data\processed\normalized\bns.md"
        ),
        help="Path to normalized BNS Markdown.",
    )

    parser.add_argument(
        "--contract",
        type=Path,
        default=Path(
            r"D:\Programming\AI_Projects\legal-rag-azure\data\processed\normalized\indian-contract-act.md"
        ),
        help=(
            "Path to normalized Indian Contract "
            "Act Markdown."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            r"D:\Programming\AI_Projects\legal-rag-azure\data\processed\chunks"
        ),
        help=(
            "Chunk output directory. "
            "Default: data/processed/chunks"
        ),
    )

    parser.add_argument(
        "--target-tokens",
        type=int,
        default=TARGET_CHUNK_TOKENS,
        help=(
            "Approximate target chunk size. "
            f"Default: {TARGET_CHUNK_TOKENS}"
        ),
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=MAX_CHUNK_TOKENS,
        help=(
            "Approximate maximum chunk size. "
            f"Default: {MAX_CHUNK_TOKENS}"
        ),
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Validate input files
    # --------------------------------------------------------

    if not args.bns.exists():

        raise FileNotFoundError(
            f"BNS Markdown not found: "
            f"{args.bns}"
        )

    if not args.contract.exists():

        raise FileNotFoundError(
            f"Indian Contract Act Markdown "
            f"not found: {args.contract}"
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # BNS
    # --------------------------------------------------------

    bns_chunks = process_file(
        input_path=args.bns,
        output_dir=args.output_dir,
        act="BNS",
        target_tokens=args.target_tokens,
        max_tokens=args.max_tokens,
    )

    # --------------------------------------------------------
    # Indian Contract Act
    # --------------------------------------------------------

    contract_chunks = process_file(
        input_path=args.contract,
        output_dir=args.output_dir,
        act="ICA",
        target_tokens=args.target_tokens,
        max_tokens=args.max_tokens,
    )

    # --------------------------------------------------------
    # Final statistics
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("CHUNKING COMPLETE")
    print("=" * 60)

    print(
        f"BNS chunks:              "
        f"{len(bns_chunks)}"
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