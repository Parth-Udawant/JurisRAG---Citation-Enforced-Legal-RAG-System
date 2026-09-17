#!/usr/bin/env python3
"""
BNS PDF -> Canonical JSON + Markdown ingestion pipeline.

Architecture:
    PDF
      -> PyMuPDF
      -> coordinate-aware extraction
      -> BNS legal structure parser
      -> margin-note <-> section association
      -> canonical JSON
      -> Markdown

No embeddings/vector DB logic is included. JSON is the canonical output;
Markdown is rendered from the same parsed representation.

Designed for the Gazette of India BNS layout where:
  - odd-numbered PDF pages place marginal notes on the right
  - even-numbered PDF pages place marginal notes on the left

Usage:
    python bns_ingest.py Bns-1-15.pdf --output-dir output
    python bns_ingest.py BNS.pdf --output-dir output --include-front-matter
    python bns_ingest.py BNS.pdf --output-dir output --strict

Dependency:
    pip install pymupdf
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Optional

try:
    import pymupdf as fitz
except ImportError:
    # Backward-compatible fallback for older PyMuPDF installations.
    import fitz


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Config:
    # The BNS Gazette pages are approximately A4-sized. Ratios are preferable
    # to absolute x coordinates so the parser survives small PDF variations.
    left_margin_ratio: float = 0.20
    right_margin_ratio: float = 0.80

    # Ignore recurring Gazette header/footer material. These are intentionally
    # conservative; section 20 in the supplied sample starts around y=83 pt.
    top_ignore_pt: float = 70.0
    bottom_ignore_pt: float = 60.0

    # A margin-note group is associated with a section only when its first line
    # is no farther than this from the section's first line.
    margin_association_threshold_pt: float = 90.0

    # Used to decide whether adjacent margin lines belong to the same note.
    # The actual threshold is derived from the page's median line spacing.
    margin_group_multiplier: float = 1.25
    margin_group_min_pt: float = 11.0
    margin_group_max_pt: float = 15.0

    # Main-body x range used to recognize section starts. The section number
    # normally starts around 0.24 of page width in the supplied BNS PDF.
    section_x_min_ratio: float = 0.10
    section_x_max_ratio: float = 0.60
    subclause_x_min_ratio: float = 0.30

    # Chapter title recognition.
    chapter_title_center_tolerance_pt: float = 85.0
    chapter_title_max_gap_pt: float = 35.0


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class Line:
    page: int
    index: int
    text: str
    bbox: list[float]
    kind: str = "body"

    @property
    def x0(self) -> float:
        return self.bbox[0]

    @property
    def y0(self) -> float:
        return self.bbox[1]

    @property
    def x1(self) -> float:
        return self.bbox[2]

    @property
    def y1(self) -> float:
        return self.bbox[3]


@dataclass
class MarginNote:
    text: str
    page: int
    bbox: list[float]
    lines: list[str]
    anchor_y: float
    distance_to_section_pt: Optional[float] = None
    association_method: Optional[str] = None
    association_confidence: Optional[float] = None

@dataclass
class SectionCandidate:
    number: int
    line: Line
    page_width: float
    x_ratio: float
    y: float

@dataclass
class Chapter:
    number: str
    title: Optional[str]
    page: int
    line_index: int


@dataclass
class StructureBlock:
    type: str
    text: str
    marker: Optional[str]
    page_start: int
    page_end: int
    bbox: list[float]


@dataclass
class Section:
    number: str
    chapter_number: Optional[str]
    chapter_title: Optional[str]
    marginal_notes: list[MarginNote]
    blocks: list[StructureBlock]
    text: str
    source_pages: list[int]
    bbox: list[float]
    start_page: int
    start_y: float


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def clean_line_text(text: str) -> str:
    """Normalize whitespace without inventing/correcting legal wording."""
    return re.sub(r"\s+", " ", text).strip()


def normalized_for_matching(text: str) -> str:
    """Normalization used only for structural recognition, not output."""
    text = clean_line_text(text)
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    return text


def bbox_union(boxes: Iterable[list[float]]) -> list[float]:
    boxes = list(boxes)
    if not boxes:
        return [0.0, 0.0, 0.0, 0.0]
    return [
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    ]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def roman_or_number(value: str) -> str:
    return re.sub(r"\s+", "", value.upper())


def confidence_from_distance(distance: float, threshold: float) -> float:
    # This is a transparent geometry score, not an ML probability.
    return round(max(0.0, min(1.0, 1.0 - distance / threshold)), 4)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def is_horizontal_rule(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    return not stripped.replace("_", "").replace("-", "").replace("=", "").strip()


# ---------------------------------------------------------------------------
# Structural recognizers
# ---------------------------------------------------------------------------

SECTION_RE = re.compile(r"^\s*(\d{1,3})\.\s+", re.UNICODE)
CHAPTER_RE = re.compile(r"^\s*CHAPTER\s*([IVXLCDM]+)\s*$", re.IGNORECASE)
SUBSECTION_RE = re.compile(r"^\s*(\(\d+\))(?=\s)")
CLAUSE_RE = re.compile(r"^\s*(\([a-z]\))(?=\s|[A-Z“\"\'])", re.IGNORECASE)
SUBCLAUSE_RE = re.compile(r"^\s*(\([ivxlcdm]+\))(?=\s|[A-Z“\"\'])", re.IGNORECASE)
EXPLANATION_RE = re.compile(r"^\s*Explanation(?:\s+\d+)?\s*[.\u2014\u2013-]", re.IGNORECASE)
PROVISO_RE = re.compile(r"^\s*Provided\s+that\b", re.IGNORECASE)
ILLUSTRATIONS_RE = re.compile(r"^\s*Illustrations?\s*\.\s*$", re.IGNORECASE)


def is_section_start(line: Line, page_width: float, cfg: Config) -> Optional[str]:
    """Detect statutory section starts using text pattern + body-region geometry.

    The earlier implementation used a narrow x-range (18%-35% of page width).
    That is too brittle for the full BNS Gazette because some section headings
    are indented differently or have different line-breaking/layout. The PDF
    margin strips are already removed during extraction, so the section detector
    can safely use a much wider central-body range.
    """
    m = SECTION_RE.match(line.text)
    return m.group(1) if m else None


def is_chapter_start(line: Line) -> Optional[str]:
    m = CHAPTER_RE.match(normalized_for_matching(line.text))
    return roman_or_number(m.group(1)) if m else None


def is_probable_chapter_title(
    line: Line,
    page_width: float,
    previous: Line,
    cfg: Config,
) -> bool:
    if line.page != previous.page:
        return False
    if line.y0 < previous.y0:
        return False
    if line.y0 - previous.y0 > cfg.chapter_title_max_gap_pt:
        return False

    center = (line.x0 + line.x1) / 2.0
    if abs(center - page_width / 2.0) > cfg.chapter_title_center_tolerance_pt:
        return False

    text = clean_line_text(line.text)
    if not text or len(text) > 100:
        return False
    if SECTION_RE.match(text) or CHAPTER_RE.match(text):
        return False

    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    # Gazette chapter titles are normally uppercase.
    return sum(c.isupper() for c in letters) / len(letters) >= 0.75

def detect_section_candidates(body_lines, page_widths):

    candidates = []

    for line in body_lines:

        match = SECTION_RE.match(line.text)

        if not match:
            continue

        number = int(match.group(1))

        if not 1 <= number <= 358:
            continue

        width = page_widths[line.page]

        candidates.append(
            SectionCandidate(
                number=number,
                line=line,
                page_width=width,
                x_ratio=line.x0 / width,
                y=line.y0,
            )
        )

    return candidates

def resolve_section_sequence(candidates):

    by_number = {}

    for candidate in candidates:
        by_number.setdefault(
            candidate.number,
            []
        ).append(candidate)

    selected = []

    for expected_number in range(1, 359):

        candidates_for_number = by_number.get(
            expected_number,
            []
        )

        if not candidates_for_number:
            continue

        # If there are duplicate candidates for the same number,
        # choose the candidate most consistent with a legal section start.
        #
        # Prefer candidates whose x position is inside the broad
        # statutory body region.
        candidates_for_number.sort(
            key=lambda c: (
                0 if 0.10 <= c.x_ratio <= 0.70 else 1,
                c.page,
                c.y,
            )
        )

        selected.append(
            candidates_for_number[0]
        )

    return sorted(
        selected,
        key=lambda c: (
            c.line.page,
            c.line.y,
            c.line.x0,
        )
    )

# ---------------------------------------------------------------------------
# PDF extraction
# ---------------------------------------------------------------------------

class BNSExtractor:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def extract(self, pdf_path: Path) -> tuple[list[dict[str, Any]], list[Line], list[Line]]:
        pages: list[dict[str, Any]] = []
        all_body: list[Line] = []
        all_margin: list[Line] = []

        with fitz.open(pdf_path) as doc:
            for page_number, page in enumerate(doc, start=1):
                width = float(page.rect.width)
                height = float(page.rect.height)
                body: list[Line] = []
                margin: list[Line] = []

                data = page.get_text("dict", sort=False)
                local_index = 0

                for block in data.get("blocks", []):
                    if block.get("type") != 0:
                        continue

                    for raw_line in block.get("lines", []):
                        bbox = list(map(float, raw_line["bbox"]))
                        x0, y0, x1, y1 = bbox
                        text = clean_line_text(
                            "".join(span.get("text", "") for span in raw_line.get("spans", []))
                        )

                        if not text or is_horizontal_rule(text):
                            continue

                        if y0 < self.cfg.top_ignore_pt:
                            continue
                        if y1 > height - self.cfg.bottom_ignore_pt:
                            continue

                        line = Line(
                            page=page_number,
                            index=local_index,
                            text=text,
                            bbox=[round(v, 3) for v in bbox],
                        )
                        local_index += 1

                        # Treat both outer strips as peripheral text first.
                        # The expected marginal-note side is then selected by
                        # odd/even page parity. Opposite-side peripheral text
                        # (e.g. legislative footnote references such as
                        # "40 of 2019.") must not leak into statutory text.
                        on_left_edge = x1 <= width * self.cfg.left_margin_ratio
                        on_right_edge = x0 >= width * self.cfg.right_margin_ratio

                        expected_margin = (
                            (page_number % 2 == 1 and on_right_edge)
                            or (page_number % 2 == 0 and on_left_edge)
                        )
                        peripheral = on_left_edge or on_right_edge

                        if expected_margin:
                            line.kind = "margin"
                            margin.append(line)
                            all_margin.append(line)
                        elif peripheral:
                            line.kind = "peripheral"
                            # Preserve it in page diagnostics, but do not feed
                            # it into statutory section text.
                        else:
                            body.append(line)
                            all_body.append(line)

                body.sort(key=lambda l: (l.y0, l.x0))
                margin.sort(key=lambda l: (l.y0, l.x0))

                pages.append(
                    {
                        "page": page_number,
                        "width": round(width, 3),
                        "height": round(height, 3),
                        "margin_side": "right" if page_number % 2 else "left",
                        "body_line_count": len(body),
                        "margin_line_count": len(margin),
                        "body": body,
                        "margin": margin,
                    }
                )

        return pages, all_body, all_margin


# ---------------------------------------------------------------------------
# Margin-note grouping and association
# ---------------------------------------------------------------------------

class MarginAssociator:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def group_margin_lines(self, lines: list[Line]) -> list[list[Line]]:
        if not lines:
            return []

        lines = sorted(lines, key=lambda l: (l.page, l.y0, l.x0))
        groups: list[list[Line]] = []

        by_page: dict[int, list[Line]] = {}
        for line in lines:
            by_page.setdefault(line.page, []).append(line)

        for page, page_lines in by_page.items():
            page_lines.sort(key=lambda l: (l.y0, l.x0))
            gaps = [
                page_lines[i + 1].y0 - page_lines[i].y0
                for i in range(len(page_lines) - 1)
                if page_lines[i + 1].y0 > page_lines[i].y0
            ]
            base = median(gaps) if gaps else 9.6
            threshold = max(
                self.cfg.margin_group_min_pt,
                min(self.cfg.margin_group_max_pt, base * self.cfg.margin_group_multiplier),
            )

            current = [page_lines[0]]
            for prev, line in zip(page_lines, page_lines[1:]):
                gap = line.y0 - prev.y0
                if gap <= threshold:
                    current.append(line)
                else:
                    groups.append(current)
                    current = [line]
            groups.append(current)

        return groups

    def associate(
        self,
        pages: list[dict[str, Any]],
        section_starts: list[Line],
    ) -> tuple[dict[tuple[int, str], list[MarginNote]], list[str]]:
        """Associate marginal-note groups with statutory sections.

        Primary rule:
            same page + nearest section start by y-coordinate.

        Fallback rule:
            if a page contains margin text but no section starts, attach the
            note to the section that is active on that page (the latest section
            start before that page). This handles sections whose statutory text
            continues onto a following page.

        A large distance is retained as a warning rather than silently dropped.
        """
        warnings: list[str] = []
        associated: dict[tuple[int, str], list[MarginNote]] = {}

        starts = sorted(section_starts, key=lambda l: (l.page, l.y0, l.x0))
        sections_by_page: dict[int, list[tuple[Line, str]]] = {}
        for line in starts:
            m = SECTION_RE.match(line.text)
            if m:
                sections_by_page.setdefault(line.page, []).append((line, m.group(1)))

        # Latest section start before each page. Used only when the page has no
        # section start of its own.
        previous_start: Optional[tuple[Line, str]] = None

        for page in pages:
            page_number = page["page"]
            groups = self.group_margin_lines(page["margin"])
            starts_on_page = sorted(
                sections_by_page.get(page_number, []),
                key=lambda x: x[0].y0,
            )

            if not groups:
                # Advance active section state even when this page has no note.
                if starts_on_page:
                    previous_start = starts_on_page[-1]
                continue

            for group in groups:
                anchor_y = group[0].y0
                candidate: Optional[tuple[Line, str]] = None
                method = "same-page-coordinate-nearest-section-start"

                if starts_on_page:
                    candidate = min(
                        starts_on_page,
                        key=lambda item: abs(item[0].y0 - anchor_y),
                    )
                elif previous_start is not None:
                    candidate = previous_start
                    method = "continuation-page-active-section"

                if candidate is None:
                    warnings.append(
                        f"Page {page_number}: {len(groups)} margin-note group(s) found "
                        "but no section context was available."
                    )
                    continue

                best_line, best_number = candidate
                distance = abs(best_line.y0 - anchor_y)

                # Same-page notes should normally be close to their section.
                # Continuation-page notes can legitimately be farther away, so
                # use a separate, more conservative warning threshold there.
                if method == "same-page-coordinate-nearest-section-start":
                    confident = distance <= self.cfg.margin_association_threshold_pt
                else:
                    confident = True

                text = " ".join(line.text for line in group)
                bbox = bbox_union([line.bbox for line in group])
                note = MarginNote(
                    text=text,
                    page=page_number,
                    bbox=[round(v, 3) for v in bbox],
                    lines=[line.text for line in group],
                    anchor_y=round(anchor_y, 3),
                    distance_to_section_pt=round(distance, 3),
                    association_method=method,
                    association_confidence=(
                        confidence_from_distance(
                            distance,
                            self.cfg.margin_association_threshold_pt,
                        )
                        if method == "same-page-coordinate-nearest-section-start"
                        else None
                    ),
                )

                if confident:
                    associated.setdefault((best_line.page, best_number), []).append(note)
                else:
                    warnings.append(
                        f"Page {page_number}: margin note near y={anchor_y:.1f} could not be confidently associated "
                        f"with a section (nearest section {best_number}, distance {distance:.1f} pt)."
                    )

            if starts_on_page:
                previous_start = starts_on_page[-1]

        return associated, warnings


# ---------------------------------------------------------------------------
# Legal structure parser
# ---------------------------------------------------------------------------

class BNSStructureParser:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def detect_chapters(
        self,
        body_lines: list[Line],
        page_widths: dict[int, float],
    ) -> list[Chapter]:
        chapters: list[Chapter] = []
        by_page: dict[int, list[Line]] = {}
        for line in body_lines:
            by_page.setdefault(line.page, []).append(line)

        for page, lines in by_page.items():
            lines.sort(key=lambda l: (l.y0, l.x0))
            for i, line in enumerate(lines):
                number = is_chapter_start(line)
                if not number:
                    continue

                title = None
                if i + 1 < len(lines):
                    candidate = lines[i + 1]
                    if is_probable_chapter_title(
                        candidate,
                        page_widths[page],
                        line,
                        self.cfg,
                    ):
                        title = candidate.text

                chapters.append(
                    Chapter(
                        number=number,
                        title=title,
                        page=page,
                        line_index=line.index,
                    )
                )

        chapters.sort(key=lambda c: (c.page, c.line_index))
        return chapters

    def detect_section_starts(
        self,
        body_lines: list[Line],
        page_widths: dict[int, float],
    ) -> list[Line]:
        starts: list[Line] = []
        for line in body_lines:
            number = is_section_start(line, page_widths[line.page], self.cfg)
            if number is not None:
                starts.append(line)
        starts.sort(key=lambda l: (l.page, l.y0, l.x0))
        return starts

    def parse_blocks(self, lines: list[Line], page_widths: dict[int, float]) -> list[StructureBlock]:
        """Convert section lines into legal structure blocks without rewriting text."""
        if not lines:
            return []

        blocks: list[StructureBlock] = []
        current: Optional[dict[str, Any]] = None
        illustration_mode = False

        def flush() -> None:
            nonlocal current
            if not current:
                return
            text = " ".join(current["parts"]).strip()
            if text:
                blocks.append(
                    StructureBlock(
                        type=current["type"],
                        text=text,
                        marker=current.get("marker"),
                        page_start=min(current["pages"]),
                        page_end=max(current["pages"]),
                        bbox=bbox_union(current["boxes"]),
                    )
                )
            current = None

        for line in lines:
            text = line.text
            normalized = normalized_for_matching(text)

            # Section heading / first statutory line.
            section_match = SECTION_RE.match(text)
            if section_match and current is None:
                block_type = "section_text"
                marker = section_match.group(0).strip()
                flush()
                current = {
                    "type": block_type,
                    "marker": marker,
                    "parts": [text],
                    "pages": [line.page],
                    "boxes": [line.bbox],
                }
                illustration_mode = False
                continue

            if ILLUSTRATIONS_RE.match(normalized):
                flush()
                current = {
                    "type": "illustration_heading",
                    "marker": None,
                    "parts": [text],
                    "pages": [line.page],
                    "boxes": [line.bbox],
                }
                illustration_mode = True
                continue

            # Some Gazette illustrations put the word "Illustration." and
            # the complete illustration on the same physical line. Preserve
            # that as one illustration block rather than losing its text.
            if re.match(r"^\s*Illustration\.\s+", normalized, re.IGNORECASE):
                flush()
                current = {
                    "type": "illustration",
                    "marker": None,
                    "parts": [text],
                    "pages": [line.page],
                    "boxes": [line.bbox],
                }
                illustration_mode = False
                continue

            if EXPLANATION_RE.match(normalized):
                flush()
                current = {
                    "type": "explanation",
                    "marker": None,
                    "parts": [text],
                    "pages": [line.page],
                    "boxes": [line.bbox],
                }
                illustration_mode = False
                continue

            if PROVISO_RE.match(normalized):
                flush()
                current = {
                    "type": "proviso",
                    "marker": None,
                    "parts": [text],
                    "pages": [line.page],
                    "boxes": [line.bbox],
                }
                illustration_mode = False
                continue

            # Clause/subclause classification uses indentation as well as the
            # marker. In this Gazette layout, ordinary clauses start around
            # x=165 pt while nested subclauses start around x=190 pt. This is
            # necessary because (i) can technically be either a lettered
            # clause or a Roman-numeral subclause; the PDF indentation resolves
            # that ambiguity.
            clause = CLAUSE_RE.match(text)
            subclause = SUBCLAUSE_RE.match(text)

            if clause or subclause:
                marker = (clause or subclause).group(1)
                is_nested = bool(subclause) and (
                    len(marker.strip("()")) > 1
                    or line.x0 >= self.cfg.subclause_x_min_ratio * page_widths[line.page]
                )

                flush()
                current = {
                    "type": "illustration_item" if illustration_mode else (
                        "subclause" if is_nested else "clause"
                    ),
                    "marker": marker,
                    "parts": [text],
                    "pages": [line.page],
                    "boxes": [line.bbox],
                }
                continue

            subsection = SUBSECTION_RE.match(text)
            if subsection:
                flush()
                current = {
                    "type": "subsection",
                    "marker": subsection.group(1),
                    "parts": [text],
                    "pages": [line.page],
                    "boxes": [line.bbox],
                }
                illustration_mode = False
                continue

            if current is None:
                current = {
                    "type": "paragraph",
                    "marker": None,
                    "parts": [text],
                    "pages": [line.page],
                    "boxes": [line.bbox],
                }
            else:
                current["parts"].append(text)
                current["pages"].append(line.page)
                current["boxes"].append(line.bbox)

        flush()
        return blocks

    def build_sections(
        self,
        body_lines: list[Line],
        section_starts: list[Line],
        chapters: list[Chapter],
        margin_notes: dict[tuple[int, str], list[MarginNote]],
        page_widths: dict[int, float],
    ) -> list[Section]:
        # Global ordering is important for sections spanning pages.
        body_lines = sorted(body_lines, key=lambda l: (l.page, l.y0, l.x0))
        line_to_pos = {id(line): i for i, line in enumerate(body_lines)}

        starts = sorted(
            section_starts,
            key=lambda l: (l.page, l.y0, l.x0),
        )

        chapter_events = sorted(chapters, key=lambda c: (c.page, c.line_index))

        sections: list[Section] = []
        current_chapter_number: Optional[str] = None
        current_chapter_title: Optional[str] = None
        chapter_cursor = 0

        # Map global line positions to chapter state.
        chapter_by_pos: list[tuple[int, Chapter]] = []
        for chapter in chapter_events:
            positions = [
                i for i, line in enumerate(body_lines)
                if line.page == chapter.page and line.index == chapter.line_index
            ]
            if positions:
                chapter_by_pos.append((positions[0], chapter))
        chapter_by_pos.sort(key=lambda x: x[0])

        for start_i, start_line in enumerate(starts):
            start_pos = line_to_pos[id(start_line)]
            next_section_pos = (
                line_to_pos[id(starts[start_i + 1])]
                if start_i + 1 < len(starts)
                else len(body_lines)
            )

            # Apply chapter events before this section.
            while chapter_cursor < len(chapter_by_pos) and chapter_by_pos[chapter_cursor][0] <= start_pos:
                _, chapter = chapter_by_pos[chapter_cursor]
                current_chapter_number = chapter.number
                current_chapter_title = chapter.title
                chapter_cursor += 1

            # A section must not absorb a chapter heading/title occurring
            # between it and the next section.
            end_pos = next_section_pos
            for chapter_pos, _chapter in chapter_by_pos:
                if start_pos < chapter_pos < end_pos:
                    end_pos = chapter_pos
                    break

            section_lines = [
                line for line in body_lines[start_pos:end_pos]
                if line.kind == "body"
            ]
            if not section_lines:
                continue

            number_match = SECTION_RE.match(start_line.text)
            if not number_match:
                continue
            number = number_match.group(1)

            blocks = self.parse_blocks(section_lines, page_widths)
            text = "\n\n".join(block.text for block in blocks)
            source_pages = sorted({line.page for line in section_lines})
            section_bbox = bbox_union([line.bbox for line in section_lines])

            notes = margin_notes.get((start_line.page, number), [])
            notes = sorted(notes, key=lambda n: (n.anchor_y, n.bbox[0]))

            sections.append(
                Section(
                    number=number,
                    chapter_number=current_chapter_number,
                    chapter_title=current_chapter_title,
                    marginal_notes=notes,
                    blocks=blocks,
                    text=text,
                    source_pages=source_pages,
                    bbox=section_bbox,
                    start_page=start_line.page,
                    start_y=round(start_line.y0, 3),
                )
            )

        return sections


# ---------------------------------------------------------------------------
# Markdown renderer
# ---------------------------------------------------------------------------

class MarkdownRenderer:
    def render(
        self,
        canonical: dict[str, Any],
        include_front_matter: bool = False,
    ) -> str:
        doc_meta = canonical["document"]
        sections = canonical["sections"]
        chapters = canonical["chapters"]

        lines: list[str] = []
        title = doc_meta.get("title") or "THE BHARATIYA NYAYA SANHITA, 2023"
        lines.append(f"# {title}")
        lines.append("")
        lines.append("> Generated from the supplied PDF using coordinate-aware PyMuPDF extraction.")
        lines.append(f"> Source SHA-256: `{canonical['source']['sha256']}`")
        lines.append("")

        if include_front_matter:
            front = canonical.get("front_matter", [])
            if front:
                lines.append("## Front Matter")
                lines.append("")
                lines.extend(front)
                lines.append("")

        # Render in section order, inserting chapter headings when encountered.
        chapter_seen: set[str] = set()
        for section in sections:
            chapter_key = section.get("chapter_number")
            if chapter_key and chapter_key not in chapter_seen:
                title_part = section.get("chapter_title")
                heading = f"CHAPTER {chapter_key}"
                if title_part:
                    heading += f" — {title_part}"
                lines.append(f"## {heading}")
                lines.append("")
                chapter_seen.add(chapter_key)

            lines.append(f"### Section {section['section_number']}")
            lines.append("")

            for note in section.get("marginal_notes", []):
                lines.append(f"**Marginal Note:** {note['text']}")
                lines.append("")

            for block in section["structure_blocks"]:
                text = block["text"]
                if block["type"] in {"illustration_heading", "illustration"}:
                    lines.append(f"**{text}**")
                elif block["type"] == "explanation":
                    lines.append(text)
                elif block["type"] == "proviso":
                    lines.append(text)
                else:
                    lines.append(text)
                lines.append("")

            lines.append("---")
            lines.append("")

        return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Canonical serialization
# ---------------------------------------------------------------------------


def note_to_dict(note: MarginNote) -> dict[str, Any]:
    return asdict(note)


def block_to_dict(block: StructureBlock) -> dict[str, Any]:
    return asdict(block)


def section_to_dict(section: Section) -> dict[str, Any]:
    return {
        "section_number": section.number,
        "chapter_number": section.chapter_number,
        "chapter_title": section.chapter_title,
        "marginal_notes": [note_to_dict(n) for n in section.marginal_notes],
        "text": section.text,
        "structure_blocks": [block_to_dict(b) for b in section.blocks],
        "source_pages": section.source_pages,
        "source_bbox": [round(v, 3) for v in section.bbox],
        "start": {
            "page": section.start_page,
            "y": section.start_y,
        },
    }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class BNSIngestionPipeline:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.extractor = BNSExtractor(cfg)
        self.associator = MarginAssociator(cfg)
        self.parser = BNSStructureParser(cfg)
        self.renderer = MarkdownRenderer()

    def run(
        self,
        pdf_path: Path,
        output_dir: Path,
        include_front_matter: bool = False,
        strict: bool = False,
    ) -> tuple[Path, Path, dict[str, Any]]:
        logging.info("Opening PDF: %s", pdf_path)
        pages, body_lines, margin_lines = self.extractor.extract(pdf_path)

        page_widths = {p["page"]: p["width"] for p in pages}

        logging.info("Extracted %d body lines and %d margin lines", len(body_lines), len(margin_lines))

        section_starts = self.parser.detect_section_starts(body_lines, page_widths)
        chapters = self.parser.detect_chapters(body_lines, page_widths)

        logging.info("Detected %d section starts and %d chapters", len(section_starts), len(chapters))

        margin_map, warnings = self.associator.associate(
            pages,
            section_starts,
        )

        sections = self.parser.build_sections(
            body_lines,
            section_starts,
            chapters,
            margin_map,
            page_widths,
        )

        # Basic validation. Warnings are retained in JSON rather than silently
        # modifying source content.
        if not sections:
            warnings.append("No statutory sections were detected.")

        numbers = [int(s.number) for s in sections if s.number.isdigit()]
        if numbers:
            duplicates = sorted({n for n in numbers if numbers.count(n) > 1})
            if duplicates:
                warnings.append(f"Duplicate section numbers detected: {duplicates}")

            gaps = [
                f"{a}->{b}"
                for a, b in zip(numbers, numbers[1:])
                if b != a + 1
            ]
            if gaps:
                warnings.append(
                    "Non-consecutive section numbering detected. This can indicate missed section starts; "
                    f"observed transitions: {gaps[:20]}"
                )

            # The enacted BNS runs through section 358. If the supplied PDF is
            # the complete Act, explicitly report the missing section numbers.
            # This is diagnostic only; excerpts can legitimately have gaps.
            if max(numbers) >= 300:
                expected = set(range(1, max(numbers) + 1))
                missing = sorted(expected - set(numbers))
                if missing:
                    warnings.append(
                        f"Missing detected section numbers: {missing}. "
                        "Inspect section-start geometry before indexing this document."
                    )

        unassociated_groups = sum(
            1 for w in warnings if "could not be confidently associated" in w
        )
        if strict and (not sections or unassociated_groups):
            raise RuntimeError(
                "Strict mode failed: structural extraction or margin-note association produced warnings. "
                f"sections={len(sections)}, unassociated_margin_groups={unassociated_groups}"
            )

        # Front matter is represented separately and is deliberately not used
        # to build sections. This preserves the extracted source without
        # polluting statutory text with Gazette headers/footers.
        first_section_pos = None
        if section_starts:
            body_sorted = sorted(body_lines, key=lambda l: (l.page, l.y0, l.x0))
            first_id = id(section_starts[0])
            for i, line in enumerate(body_sorted):
                if id(line) == first_id:
                    first_section_pos = i
                    break

        front_matter: list[str] = []
        if first_section_pos is not None:
            front_matter = [
                line.text
                for line in sorted(body_lines, key=lambda l: (l.page, l.y0, l.x0))[:first_section_pos]
            ]

        # Extract a stable document title from front matter if available.
        title = next(
            (
                line for line in front_matter
                if "BHARATIYA NYAYA SANHITA" in line.upper()
            ),
            "THE BHARATIYA NYAYA SANHITA, 2023",
        )

        source = {
            "path": str(pdf_path.resolve()),
            "filename": pdf_path.name,
            "sha256": sha256_file(pdf_path),
            "pdf_page_count": len(pages),
        }

        canonical = {
            "schema_version": "bns-ingestion-1.0",
            "document": {
                "title": title,
                "document_type": "Indian legislation / Gazette of India",
                "law": "Bharatiya Nyaya Sanhita, 2023",
            },
            "source": source,
            "parser": {
                "engine": "PyMuPDF",
                "pymupdf_version": getattr(fitz, "VersionBind", None),
                "coordinate_aware": True,
                "margin_rule": {
                    "odd_pages": "right",
                    "even_pages": "left",
                    "left_margin_ratio": self.cfg.left_margin_ratio,
                    "right_margin_ratio": self.cfg.right_margin_ratio,
                    "association_threshold_pt": self.cfg.margin_association_threshold_pt,
                },
            },
            "statistics": {
                "pages_processed": len(pages),
                "sections_detected": len(sections),
                "chapters_detected": len(chapters),
                "margin_notes_associated": sum(len(s.marginal_notes) for s in sections),
                "warnings": len(warnings),
            },
            "front_matter": front_matter,
            "chapters": [asdict(c) for c in chapters],
            "sections": [section_to_dict(s) for s in sections],
            "page_metadata": [
                {
                    "page": p["page"],
                    "width": p["width"],
                    "height": p["height"],
                    "margin_side": p["margin_side"],
                    "body_line_count": p["body_line_count"],
                    "margin_line_count": p["margin_line_count"],
                }
                for p in pages
            ],
            "warnings": warnings,
        }

        markdown = self.renderer.render(
            canonical,
            include_front_matter=include_front_matter,
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        stem = pdf_path.stem
        json_path = output_dir / f"{stem}.json"
        md_path = output_dir / f"{stem}.md"

        json_text = json.dumps(
            canonical,
            ensure_ascii=False,
            indent=2,
        )
        atomic_write_text(json_path, json_text + "\n")
        atomic_write_text(md_path, markdown)

        logging.info("JSON written to: %s", json_path)
        logging.info("Markdown written to: %s", md_path)

        return json_path, md_path, canonical


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Parse BNS Gazette PDF into canonical JSON and Markdown."
    )
    parser.add_argument("pdf", type=Path, help="Input BNS PDF")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Directory for JSON/Markdown output (default: output)",
    )
    parser.add_argument(
        "--include-front-matter",
        action="store_true",
        help="Include raw extracted pre-section Gazette content in Markdown.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if no sections or any margin-note groups cannot be associated confidently.",
    )
    parser.add_argument(
        "--margin-threshold",
        type=float,
        default=90.0,
        help="Maximum section/margin-note anchor distance in PDF points (default: 90).",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s: %(message)s",
    )

    pdf_path = args.pdf.expanduser().resolve()
    if not pdf_path.exists():
        logging.error("Input PDF does not exist: %s", pdf_path)
        return 2
    if pdf_path.suffix.lower() != ".pdf":
        logging.error("Input must be a PDF: %s", pdf_path)
        return 2

    cfg = Config(
        margin_association_threshold_pt=args.margin_threshold,
    )

    try:
        pipeline = BNSIngestionPipeline(cfg)
        json_path, md_path, canonical = pipeline.run(
            pdf_path=pdf_path,
            output_dir=args.output_dir.expanduser().resolve(),
            include_front_matter=args.include_front_matter,
            strict=args.strict,
        )
    except (fitz.FileDataError, RuntimeError, OSError) as exc:
        logging.error("Ingestion failed: %s", exc)
        return 1

    print("\nBNS ingestion completed")
    print(f"  Pages:             {canonical['statistics']['pages_processed']}")
    print(f"  Chapters:          {canonical['statistics']['chapters_detected']}")
    print(f"  Sections:          {canonical['statistics']['sections_detected']}")
    print(f"  Margin notes:      {canonical['statistics']['margin_notes_associated']}")
    print(f"  Warnings:          {canonical['statistics']['warnings']}")
    print(f"  JSON:              {json_path}")
    print(f"  Markdown:          {md_path}")

    if canonical["warnings"]:
        print("\nWarnings:")
        for warning in canonical["warnings"]:
            print(f"  - {warning}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
