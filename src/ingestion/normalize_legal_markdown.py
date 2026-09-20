from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ============================================================
# Data model
# ============================================================

@dataclass
class Section:
    number: str
    title: str = ""
    marginal_note: Optional[str] = None
    content: list[str] = field(default_factory=list)
    source_pages: list[int] = field(default_factory=list)


@dataclass
class Chapter:
    number: str
    title: str
    pre_section_content: list[str] = field(default_factory=list)
    sections: list[Section] = field(default_factory=list)


@dataclass
class Document:
    title: str
    act_number: Optional[str] = None

    preamble: list[str] = field(default_factory=list)
    chapters: list[Chapter] = field(default_factory=list)

    other_blocks: list[tuple[str, list[str]]] = field(
        default_factory=list
    )


# ============================================================
# Regex patterns
# ============================================================

# Examples:
#
# ## Section 1
# ## Section 1 - Short title
# ## Section 19-A. Power to set aside...
# ## Section 178A. Pledge...
SECTION_RE = re.compile(
    r"^##\s+Section\s+"
    r"(?P<number>\d+[A-Za-z]?(?:-[A-Za-z]+)?)"
    r"(?:\s*(?:[-–—.:]\s*)?(?P<title>.*?))?\s*$",
    re.IGNORECASE,
)

# Examples:
#
# # Chapter I: PRELIMINARY
# # Chapter I Of the communication...
# # Chapter XI Of partnership
CHAPTER_RE = re.compile(
    r"^#\s+Chapter\s+"
    r"(?P<number>[IVXLCDM0-9]+)"
    r"\s*(?::|-|—)?\s*"
    r"(?P<title>.*)$",
    re.IGNORECASE,
)

# BNS:
#
# **Marginal note:** Short title...
MARGINAL_NOTE_RE = re.compile(
    r"^\s*\*\*Marginal note:\*\*\s*(?P<text>.*)$",
    re.IGNORECASE,
)

# Source page provenance in BNS:
#
# <!-- source_pages=1,2; section=1 -->
SOURCE_PAGE_RE = re.compile(
    r"<!--\s*source_pages=(?P<pages>[^;]+)"
    r"(?:;\s*section=(?P<section>[^ ]+))?\s*-->",
    re.IGNORECASE,
)

# Bold structural markers:
#
# **(1)**
# **(a)**
# **(i)**
#
# Also handles:
#
#  **(a)**
SUBSECTION_RE = re.compile(
    r"^\s*\*\*\s*"
    r"(?P<marker>"
    r"\(\d+\)|"
    r"\([a-z]\)|"
    r"\([ivxlcdm]+\)"
    r")"
    r"\s*\*\*\s*"
    r"(?P<text>.*)$",
    re.IGNORECASE,
)

# Explanation:
#
# **Explanation** — text
# **Explanation:** text
# Explanation : text
# Explanation 1 : text
EXPLANATION_RE = re.compile(
    r"^\s*(?:\*\*)?"
    r"Explanation(?:\s+\d+)?"
    r"(?:\*\*)?"
    r"\s*(?:[-–—:]|\s*:\s*)?\s*"
    r"(?P<text>.*)$",
    re.IGNORECASE,
)

# Exception:
#
# Exception : text
# **Exception** — text
EXCEPTION_RE = re.compile(
    r"^\s*(?:\*\*)?"
    r"Exception(?:\s+\d+)?"
    r"(?:\*\*)?"
    r"\s*(?:[-–—:]|\s*:\s*)?\s*"
    r"(?P<text>.*)$",
    re.IGNORECASE,
)

# Illustration:
#
# **Illustration** A...
# **Illustration**
# Illustration A...
ILLUSTRATION_RE = re.compile(
    r"^\s*(?:\*\*)?"
    r"Illustration"
    r"(?:\*\*)?"
    r"\s*(?P<text>.*)$",
    re.IGNORECASE,
)


# ============================================================
# Utility functions
# ============================================================

def clean_heading_title(title: str) -> str:
    """Normalize whitespace without changing legal wording."""
    title = re.sub(r"\s+", " ", title).strip()

    # Remove accidental trailing punctuation/spacing only where
    # it is clearly heading formatting.
    title = title.rstrip()

    return title


def normalize_structural_line(line: str) -> str:
    """
    Normalize subsection / explanation / exception / illustration
    markers while preserving the underlying statutory text.
    """

    line = line.strip()

    # -----------------------------------------
    # Subsections / clauses / subclauses
    # -----------------------------------------
    match = SUBSECTION_RE.match(line)

    if match:
        marker = match.group("marker")
        text = match.group("text").strip()

        if text:
            return f"**{marker}** {text}"

        return f"**{marker}**"

    # -----------------------------------------
    # Explanation
    # -----------------------------------------
    match = EXPLANATION_RE.match(line)

    if match:
        text = match.group("text").strip()

        if text:
            return f"**Explanation:** {text}"

        return "**Explanation:**"

    # -----------------------------------------
    # Exception
    # -----------------------------------------
    match = EXCEPTION_RE.match(line)

    if match:
        text = match.group("text").strip()

        if text:
            return f"**Exception:** {text}"

        return "**Exception:**"

    # -----------------------------------------
    # Illustration
    # -----------------------------------------
    match = ILLUSTRATION_RE.match(line)

    if match:
        text = match.group("text").strip()

        if text:
            return f"**Illustration:** {text}"

        return "**Illustration:**"

    return line


def clean_content(lines: list[str]) -> list[str]:
    """
    Clean structural formatting while preserving the actual
    legal text.
    """

    result = []

    for line in lines:

        # Remove BNS source-page HTML comments.
        if SOURCE_PAGE_RE.match(line.strip()):
            continue

        normalized = normalize_structural_line(line)

        result.append(normalized)

    # Collapse excessive blank lines.
    cleaned = []

    previous_blank = False

    for line in result:

        if not line.strip():

            if previous_blank:
                continue

            previous_blank = True
            cleaned.append("")

        else:

            previous_blank = False
            cleaned.append(line.rstrip())

    # Remove blank lines at boundaries.
    while cleaned and not cleaned[0].strip():
        cleaned.pop(0)

    while cleaned and not cleaned[-1].strip():
        cleaned.pop()

    return cleaned


def parse_source_pages(comment: str) -> list[int]:
    """
    Parse:
        <!-- source_pages=1,2; section=1 -->
    """

    match = SOURCE_PAGE_RE.match(comment.strip())

    if not match:
        return []

    pages_text = match.group("pages")

    pages = []

    for part in pages_text.split(","):
        part = part.strip()

        try:
            pages.append(int(part))
        except ValueError:
            pass

    return pages


# ============================================================
# Parser
# ============================================================

class LegalMarkdownParser:

    def __init__(self, path: str | Path):

        self.path = Path(path)

    def parse(self) -> Document:

        text = self.path.read_text(
            encoding="utf-8",
            errors="replace"
        )

        # Normalize line endings.
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        lines = text.splitlines()

        # ----------------------------------------------------
        # Document title
        # ----------------------------------------------------

        title = ""

        for line in lines:

            if line.startswith("# ") and not line.startswith("# Chapter"):
                title = line[2:].strip()
                break

        document = Document(title=title)

        # ----------------------------------------------------
        # State
        # ----------------------------------------------------

        current_chapter: Optional[Chapter] = None
        current_section: Optional[Section] = None

        current_special_block: Optional[str] = None
        special_buffer: list[str] = []

        pending_source_pages: list[int] = []

        # ----------------------------------------------------
        # Helpers
        # ----------------------------------------------------

        def flush_special_block():

            nonlocal current_special_block
            nonlocal special_buffer

            if current_special_block:

                content = clean_content(special_buffer)

                if content:
                    document.other_blocks.append(
                        (
                            current_special_block,
                            content
                        )
                    )

            current_special_block = None
            special_buffer = []

        def flush_section():

            nonlocal current_section
            nonlocal pending_source_pages

            if current_section is None:
                return

            current_section.content = clean_content(
                current_section.content
            )

            if pending_source_pages:
                current_section.source_pages = (
                    pending_source_pages
                )

            if current_chapter is not None:
                current_chapter.sections.append(
                    current_section
                )

            current_section = None
            pending_source_pages = []

        def flush_chapter():

            nonlocal current_chapter

            flush_section()

            if current_chapter is not None:

                current_chapter.pre_section_content = (
                    clean_content(
                        current_chapter.pre_section_content
                    )
                )

                document.chapters.append(
                    current_chapter
                )

            current_chapter = None

        # ----------------------------------------------------
        # Main parsing loop
        # ----------------------------------------------------

        for raw_line in lines:

            line = raw_line.rstrip()

            # -----------------------------------------------
            # Source page metadata
            # -----------------------------------------------

            source_match = SOURCE_PAGE_RE.match(
                line.strip()
            )

            if source_match:

                pending_source_pages = parse_source_pages(
                    line
                )

                continue

            # -----------------------------------------------
            # Document title
            # -----------------------------------------------

            if line.startswith("# ") and not line.startswith(
                "# Chapter"
            ):

                # Already captured as document title.
                continue

            # -----------------------------------------------
            # Chapter
            # -----------------------------------------------

            chapter_match = CHAPTER_RE.match(line)

            if chapter_match:

                flush_special_block()
                flush_chapter()

                chapter_number = (
                    chapter_match.group("number").strip()
                )

                chapter_title = clean_heading_title(
                    chapter_match.group("title")
                )

                current_chapter = Chapter(
                    number=chapter_number,
                    title=chapter_title
                )

                continue

            # -----------------------------------------------
            # Section
            # -----------------------------------------------

            section_match = SECTION_RE.match(line)

            if section_match:

                flush_special_block()
                flush_section()

                number = (
                    section_match.group("number")
                    .strip()
                )

                title = (
                    section_match.group("title") or ""
                )

                title = clean_heading_title(title)

                current_section = Section(
                    number=number,
                    title=title
                )

                continue

            # -----------------------------------------------
            # Marginal note
            # -----------------------------------------------

            marginal_match = MARGINAL_NOTE_RE.match(line)

            if marginal_match and current_section:

                marginal = marginal_match.group(
                    "text"
                ).strip()

                current_section.marginal_note = marginal

                # If BNS has no title in the section heading,
                # use marginal note as canonical section title.
                if not current_section.title:
                    current_section.title = marginal

                continue

            # -----------------------------------------------
            # Special Contract Act blocks
            # -----------------------------------------------

            stripped = line.strip()

            if stripped == "## PREAMBLE":

                flush_section()

                current_special_block = "Preamble"
                special_buffer = []

                continue

            if stripped == "## Related Judgements":

                flush_section()

                current_special_block = "Related Judgements"
                special_buffer = []

                continue

            if stripped == "## Schedule":

                flush_section()

                current_special_block = "Schedule"
                special_buffer = []

                continue

            if stripped == "## [9 OF 1872]":

                flush_section()

                document.act_number = "9 of 1872"

                continue

            # -----------------------------------------------
            # If inside a special block
            # -----------------------------------------------

            if current_special_block:

                special_buffer.append(line)
                continue

            # -----------------------------------------------
            # Normal content
            # -----------------------------------------------

            if current_section is not None:

                current_section.content.append(line)

            elif current_chapter is not None:

                current_chapter.pre_section_content.append(
                    line
                )

            else:

                # Document-level content.
                # For example Act number or preamble material.
                if line.strip():

                    document.preamble.append(line)

        # ----------------------------------------------------
        # Flush remaining state
        # ----------------------------------------------------

        flush_special_block()
        flush_chapter()

        return document


# ============================================================
# Markdown renderer
# ============================================================

class MarkdownRenderer:

    def render(self, document: Document) -> str:

        output = []

        # ----------------------------------------------------
        # Document title
        # ----------------------------------------------------

        output.append(f"# {document.title}")
        output.append("")

        # ----------------------------------------------------
        # Act number
        # ----------------------------------------------------

        if document.act_number:

            output.append(
                f"**Act number:** {document.act_number}"
            )

            output.append("")

        # ----------------------------------------------------
        # Document-level content
        # ----------------------------------------------------

        if document.preamble:

            output.extend(
                clean_content(document.preamble)
            )

            output.append("")

        # ----------------------------------------------------
        # Special blocks that occurred before chapters
        # ----------------------------------------------------

        # Preamble / related material is kept as explicit blocks.
        # These can be rendered before chapters.

        for name, content in document.other_blocks:

            if name in {"Preamble", "Related Judgements"}:

                output.append(f"## {name}")
                output.append("")
                output.extend(content)
                output.append("")

        # ----------------------------------------------------
        # Chapters
        # ----------------------------------------------------

        for chapter in document.chapters:

            output.append(
                f"# Chapter {chapter.number} — "
                f"{chapter.title}"
            )

            output.append("")

            if chapter.pre_section_content:

                output.extend(
                    chapter.pre_section_content
                )

                output.append("")

            # -----------------------------------------------
            # Sections
            # -----------------------------------------------

            for section in chapter.sections:

                title = section.title.strip()

                if title:

                    output.append(
                        f"## Section {section.number} "
                        f"— {title}"
                    )

                else:

                    output.append(
                        f"## Section {section.number}"
                    )

                output.append("")

                # Marginal note
                if section.marginal_note:

                    output.append(
                        f"**Marginal note:** "
                        f"{section.marginal_note}"
                    )

                    output.append("")

                # Statutory content
                output.extend(section.content)

                output.append("")

            # Chapter separator
            output.append("---")
            output.append("")

        # ----------------------------------------------------
        # Schedule
        # ----------------------------------------------------

        for name, content in document.other_blocks:

            if name == "Schedule":

                output.append("## Schedule")
                output.append("")
                output.extend(content)
                output.append("")

        # ----------------------------------------------------
        # Final cleanup
        # ----------------------------------------------------

        text = "\n".join(output)

        # Collapse >2 blank lines.
        text = re.sub(
            r"\n{3,}",
            "\n\n",
            text
        )

        return text.strip() + "\n"


# ============================================================
# File normalization
# ============================================================

def normalize_file(
    input_path: str,
    output_path: str
):

    parser = LegalMarkdownParser(input_path)

    document = parser.parse()

    renderer = MarkdownRenderer()

    normalized = renderer.render(document)

    Path(output_path).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    Path(output_path).write_text(
        normalized,
        encoding="utf-8"
    )

    print(f"Normalized: {input_path}")
    print(f"Output:     {output_path}")
    print()
    print(f"Title:      {document.title}")
    print(f"Chapters:   {len(document.chapters)}")
    print(
        f"Sections:   "
        f"{sum(len(c.sections) for c in document.chapters)}"
    )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    normalize_file(
        r"D:\Programming\AI_Projects\legal-rag-azure\data\processed\markdown\bns.md",
        r"D:\Programming\AI_Projects\legal-rag-azure\data\processed\normalized\bns.md"
    )

    normalize_file(
        r"D:\Programming\AI_Projects\legal-rag-azure\data\processed\markdown\indian-contract-act.md",
        r"D:\Programming\AI_Projects\legal-rag-azure\data\processed\normalized\indian-contract-act.md"
    )