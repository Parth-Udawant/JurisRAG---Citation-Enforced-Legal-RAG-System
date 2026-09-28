from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Optional

try:
    import pymupdf  
except ImportError:
    import fitz as pymupdf  


PARSER_VERSION = "5.0.0"
EXPECTED_FIRST_SECTION = 1
EXPECTED_LAST_SECTION = 358

MARGIN_WIDTH_RATIO = 0.19
TOP_EXCLUSION_RATIO = 0.055
BOTTOM_EXCLUSION_RATIO = 0.055

MARGIN_ALIGNMENT_TOLERANCE_PT = 28.0

MIN_SECTION_HEADING_CHARS = 3

LEGISLATIVE_REFERENCE_RE = re.compile(
    r"^\s*\d+\s+of\s+\d{4}\.?\s*$", re.IGNORECASE
)
PAGE_NUMBER_RE = re.compile(r"^\s*\d{1,4}\s*$")

SECTION_START_RE = re.compile(r"^\s*(\d{1,3})\s*\.\s*(.*)$")

SECTION_NUMBER_ONLY_RE = re.compile(r"^\s*(\d{1,3})\s*\.\s*$")

CHAPTER_RE = re.compile(
    r"^\s*CHAPTER\s*([IVXLCDM]+)\s*$", re.IGNORECASE
)

SUBSECTION_RE = re.compile(r"^\s*\((\d+)\)\s*(.*)$")
CLAUSE_RE = re.compile(r"^\s*\(([a-z])\)\s*(.*)$", re.IGNORECASE)
SUBCLAUSE_RE = re.compile(r"^\s*\(([ivxlcdm]+)\)\s*(.*)$", re.IGNORECASE)
EXPLANATION_RE = re.compile(
    r"^\s*Explanation(?:\s+\d+)?\s*[\.:—–-]?\s*(.*)$",
    re.IGNORECASE,
)
ILLUSTRATION_RE = re.compile(
    r"^\s*Illustrations?\s*[\.:—–-]?\s*(.*)$",
    re.IGNORECASE,
)
PROVISO_RE = re.compile(
    r"^\s*Provided(?:\s+further)?\s+that\b\s*(.*)$",
    re.IGNORECASE,
)

WHITESPACE_RE = re.compile(r"[ \t]+")


@dataclass
class TextLine:
    page: int
    block_no: int
    line_no: int
    x0: float
    y0: float
    x1: float
    y1: float
    text: str
    font_size: float = 0.0
    flags: int = 0

    @property
    def height(self) -> float:
        return max(0.0, self.y1 - self.y0)

    @property
    def center_y(self) -> float:
        return (self.y0 + self.y1) / 2.0


@dataclass
class SectionCandidate:
    number: int
    page: int
    line_index: int
    y0: float
    y1: float
    x0: float
    x1: float
    text: str
    confidence: float
    reason: list[str] = field(default_factory=list)


@dataclass
class MarginGroup:
    page: int
    side: str
    x0: float
    y0: float
    x1: float
    y1: float
    text: str
    lines: list[TextLine] = field(default_factory=list)

    @property
    def center_y(self) -> float:
        return (self.y0 + self.y1) / 2.0


@dataclass
class SectionRecord:
    number: int
    chapter_number: Optional[str]
    chapter_title: Optional[str]
    marginal_notes: list[dict]
    text: str
    structure_blocks: list[dict]
    logical_blocks: list[dict]
    source_pages: list[int]
    start: dict
    end: dict


@dataclass
class ParseResult:
    document: dict
    chapters: list[dict]
    sections: list[SectionRecord]
    margin_notes: list[dict]
    warnings: list[str]
    diagnostics: dict


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00ad", "")  
    text = text.replace("\u200b", "")
    text = text.replace("\ufeff", "")
    return text.strip()


def clean_line_text(text: str) -> str:
    text = normalize_text(text)
    return WHITESPACE_RE.sub(" ", text)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_lines(doc: pymupdf.Document) -> dict[int, list[TextLine]]:
    pages: dict[int, list[TextLine]] = {}

    for page_no, page in enumerate(doc, start=1):
        page_dict = page.get_text("dict", sort=True)
        lines: list[TextLine] = []

        for block_no, block in enumerate(page_dict.get("blocks", [])):
            if block.get("type") != 0:
                continue

            for line_no, line in enumerate(block.get("lines", [])):
                spans = line.get("spans", [])
                if not spans:
                    continue

                pieces = []
                font_sizes = []
                flags = 0

                for span in spans:
                    pieces.append(span.get("text", ""))
                    if span.get("size"):
                        font_sizes.append(float(span["size"]))
                    flags |= int(span.get("flags", 0))

                text = clean_line_text("".join(pieces))
                if not text:
                    continue

                bbox = line.get("bbox", (0, 0, 0, 0))
                lines.append(
                    TextLine(
                        page=page_no,
                        block_no=block_no,
                        line_no=line_no,
                        x0=float(bbox[0]),
                        y0=float(bbox[1]),
                        x1=float(bbox[2]),
                        y1=float(bbox[3]),
                        text=text,
                        font_size=sum(font_sizes) / len(font_sizes)
                        if font_sizes
                        else 0.0,
                        flags=flags,
                    )
                )

        lines.sort(key=lambda x: (x.y0, x.x0, x.block_no, x.line_no))
        pages[page_no] = lines

    return pages


def is_in_margin_region(
    line: TextLine,
    page_width: float,
    page_height: float,
) -> tuple[bool, str]:
    
    if line.y1 < page_height * TOP_EXCLUSION_RATIO:
        return False, "header"
    if line.y0 > page_height * (1.0 - BOTTOM_EXCLUSION_RATIO):
        return False, "footer"

    width = page_width * MARGIN_WIDTH_RATIO

    if line.page % 2 == 1:
        return line.x0 >= page_width - width, "right"
    return line.x1 <= width, "left"


def is_obvious_noise(line: TextLine) -> bool:
    if PAGE_NUMBER_RE.fullmatch(line.text):
        return True
    if LEGISLATIVE_REFERENCE_RE.fullmatch(line.text):
        return True
    return False


def split_body_and_margin(
    pages: dict[int, list[TextLine]],
    doc: pymupdf.Document,
) -> tuple[dict[int, list[TextLine]], list[MarginGroup]]:
    
    body: dict[int, list[TextLine]] = defaultdict(list)
    margin_lines: dict[int, list[tuple[str, TextLine]]] = defaultdict(list)

    for page_no, lines in pages.items():
        page = doc[page_no - 1]
        page_width = float(page.rect.width)
        page_height = float(page.rect.height)

        for line in lines:
            if is_obvious_noise(line):
                continue

            in_margin, side = is_in_margin_region(
                line, page_width, page_height
            )

            if in_margin:
                margin_lines[page_no].append((side, line))
            else:
                body[page_no].append(line)

    groups: list[MarginGroup] = []

    for page_no, items in margin_lines.items():
       
        side = "right" if page_no % 2 == 1 else "left"
        side_lines = [
            line for item_side, line in items
            if item_side == side
        ]
        side_lines.sort(key=lambda x: (x.y0, x.x0))

       
        section_anchors: list[tuple[float, int]] = []

        for line in sorted(
            body.get(page_no, []),
            key=lambda x: (x.y0, x.x0, x.block_no, x.line_no),
        ):
            match = SECTION_START_RE.match(line.text)
            if not match:
                continue

            number = int(match.group(1))
            if not (
                EXPECTED_FIRST_SECTION
                <= number
                <= EXPECTED_LAST_SECTION
            ):
                continue

            section_anchors.append((line.y0, number))

        section_anchors.sort()

        
        anchor_margin_lines: list[tuple[float, int, TextLine]] = []

        for section_y, section_number in section_anchors:
            nearby = [
                line
                for line in side_lines
                if abs(line.y0 - section_y) <= 8.0
            ]

            if not nearby:
                continue

            anchor_line = min(
                nearby,
                key=lambda line: abs(line.y0 - section_y),
            )

            anchor_margin_lines.append(
                (anchor_line.y0, section_number, anchor_line)
            )

        anchor_margin_lines.sort(key=lambda x: x[0])

        
        current_number: Optional[int] = None
        current_lines: list[TextLine] = []

        anchor_lookup = {
            id(anchor_line): section_number
            for _, section_number, anchor_line in anchor_margin_lines
        }

        for line in side_lines:
            anchored_number = anchor_lookup.get(id(line))

            if anchored_number is not None:
                if current_lines:
                    groups.append(
                        make_margin_group(
                            page_no,
                            side,
                            current_lines,
                        )
                    )

                current_number = anchored_number
                current_lines = [line]
                continue

            if current_number is not None:
                
                current_lines.append(line)

        if current_lines:
            groups.append(
                make_margin_group(
                    page_no,
                    side,
                    current_lines,
                )
            )

        
        if side_lines and not anchor_margin_lines:
            groups.append(
                make_margin_group(
                    page_no,
                    side,
                    side_lines,
                )
            )

    return dict(body), groups

def make_margin_group(
    page: int,
    side: str,
    lines: list[TextLine],
) -> MarginGroup:
    return MarginGroup(
        page=page,
        side=side,
        x0=min(l.x0 for l in lines),
        y0=min(l.y0 for l in lines),
        x1=max(l.x1 for l in lines),
        y1=max(l.y1 for l in lines),
        text=" ".join(l.text for l in lines).strip(),
        lines=list(lines),
    )


def flatten_body_lines(body: dict[int, list[TextLine]]) -> list[TextLine]:
    lines: list[TextLine] = []
    for page_no in sorted(body):
        page_lines = sorted(
            body[page_no], key=lambda x: (x.y0, x.x0, x.block_no, x.line_no)
        )
        lines.extend(page_lines)
    return lines


def candidate_confidence(
    line: TextLine,
    number: int,
    remainder: str,
    page_width: float,
    page_height: float,
) -> tuple[float, list[str]]:
    
    score = 1.0
    reasons = ["strict-section-syntax"]

    if number < EXPECTED_FIRST_SECTION or number > EXPECTED_LAST_SECTION:
        return 0.0, ["outside-BNS-section-range"]

    if len(remainder.strip()) >= MIN_SECTION_HEADING_CHARS:
        score += 0.05
        reasons.append("has-section-text")

    
    x_ratio = line.x0 / max(page_width, 1.0)
    if 0.05 <= x_ratio <= 0.75:
        score += 0.05
        reasons.append("broad-body-x-position")

    if (
        page_height * TOP_EXCLUSION_RATIO
        <= line.y0
        <= page_height * (1 - BOTTOM_EXCLUSION_RATIO)
    ):
        score += 0.05
        reasons.append("body-y-position")

    return min(score, 1.0), reasons


def detect_section_candidates(
    body: dict[int, list[TextLine]],
    doc: pymupdf.Document,
) -> tuple[list[SectionCandidate], list[dict]]:
    
    candidates: list[SectionCandidate] = []
    diagnostics: list[dict] = []

    all_lines = flatten_body_lines(body)

    for i, line in enumerate(all_lines):
        match = SECTION_START_RE.match(line.text)
        if not match:
            match_number_only = SECTION_NUMBER_ONLY_RE.match(line.text)
            if not match_number_only:
                continue

            number = int(match_number_only.group(1))
            remainder = ""

            if number < 1 or number > EXPECTED_LAST_SECTION:
                continue

            if i + 1 < len(all_lines):
                next_line = all_lines[i + 1]
                
                if (
                    next_line.page == line.page
                    and next_line.y0 - line.y1 <= 18.0
                ):
                    remainder = next_line.text

            page = doc[line.page - 1]
            confidence, reasons = candidate_confidence(
                line,
                number,
                remainder,
                float(page.rect.width),
                float(page.rect.height),
            )

            candidates.append(
                SectionCandidate(
                    number=number,
                    page=line.page,
                    line_index=i,
                    y0=line.y0,
                    y1=line.y1,
                    x0=line.x0,
                    x1=line.x1,
                    text=(line.text + " " + remainder).strip(),
                    confidence=confidence,
                    reason=reasons + ["number-only-line"],
                )
            )
            diagnostics.append(candidate_to_dict(candidates[-1]))
            continue

        number = int(match.group(1))
        remainder = match.group(2)

        page = doc[line.page - 1]
        confidence, reasons = candidate_confidence(
            line,
            number,
            remainder,
            float(page.rect.width),
            float(page.rect.height),
        )

        if confidence <= 0:
            continue

        candidate = SectionCandidate(
            number=number,
            page=line.page,
            line_index=i,
            y0=line.y0,
            y1=line.y1,
            x0=line.x0,
            x1=line.x1,
            text=line.text,
            confidence=confidence,
            reason=reasons,
        )
        candidates.append(candidate)
        diagnostics.append(candidate_to_dict(candidate))

    candidates.sort(key=lambda c: (c.page, c.y0, c.x0))

    return candidates, diagnostics


def candidate_to_dict(candidate: SectionCandidate) -> dict:
    return {
        "number": candidate.number,
        "page": candidate.page,
        "line_index": candidate.line_index,
        "bbox": [
            round(candidate.x0, 2),
            round(candidate.y0, 2),
            round(candidate.x1, 2),
            round(candidate.y1, 2),
        ],
        "text": candidate.text,
        "confidence": round(candidate.confidence, 3),
        "reason": candidate.reason,
    }


def resolve_section_sequence(
    candidates: list[SectionCandidate],
) -> tuple[list[SectionCandidate], list[dict]]:
    
    selected: list[SectionCandidate] = []
    resolution_log: list[dict] = []

    by_number: dict[int, list[SectionCandidate]] = defaultdict(list)
    for c in candidates:
        by_number[c.number].append(c)

    
    position_map = {
        id(c): i for i, c in enumerate(candidates)
    }
    previous_position = -1

    for expected_number in range(
        EXPECTED_FIRST_SECTION, EXPECTED_LAST_SECTION + 1
    ):
        options = [
            c
            for c in by_number.get(expected_number, [])
            if position_map[id(c)] > previous_position
        ]

        if not options:
            resolution_log.append(
                {
                    "number": expected_number,
                    "status": "missing-candidate",
                }
            )
            continue

       
        options.sort(
            key=lambda c: (
                position_map[id(c)],
                -c.confidence,
            )
        )
        chosen = options[0]
        selected.append(chosen)
        previous_position = position_map[id(chosen)]

        if len(options) > 1:
            resolution_log.append(
                {
                    "number": expected_number,
                    "status": "duplicate-candidates",
                    "selected": candidate_to_dict(chosen),
                    "alternatives": [
                        candidate_to_dict(x) for x in options[1:]
                    ],
                }
            )
        else:
            resolution_log.append(
                {
                    "number": expected_number,
                    "status": "selected",
                    "candidate": candidate_to_dict(chosen),
                }
            )

    return selected, resolution_log


def validate_complete_bns(
    selected: list[SectionCandidate],
    candidates: list[SectionCandidate],
    strict: bool,
) -> dict:
    
    expected = set(
        range(EXPECTED_FIRST_SECTION, EXPECTED_LAST_SECTION + 1)
    )
    detected = {candidate.number for candidate in selected}

    missing = sorted(expected - detected)
    unexpected = sorted(detected - expected)

    selected_numbers = [candidate.number for candidate in selected]
    counts = Counter(selected_numbers)
    duplicates = sorted(
        number for number, count in counts.items() if count > 1
    )

    all_counts = Counter(c.number for c in candidates)
    candidate_duplicates = sorted(
        number for number, count in all_counts.items() if count > 1
    )

    is_complete = not missing and not unexpected and not duplicates

    result = {
        "expected_first_section": EXPECTED_FIRST_SECTION,
        "expected_last_section": EXPECTED_LAST_SECTION,
        "expected_count": len(expected),
        "detected_count": len(detected),
        "missing_sections": missing,
        "unexpected_sections": unexpected,
        "duplicate_sections_in_selected": duplicates,
        "numbers_with_multiple_raw_candidates": candidate_duplicates,
        "complete": is_complete,
        "strict_mode": strict,
    }

    if strict and not is_complete:
        parts = []
        if missing:
            parts.append(f"missing={missing}")
        if unexpected:
            parts.append(f"unexpected={unexpected}")
        if duplicates:
            parts.append(f"duplicates={duplicates}")

        raise RuntimeError(
            "BNS completeness validation failed: " + "; ".join(parts)
        )

    return result


def build_section_ranges(
    selected: list[SectionCandidate],
    body: dict[int, list[TextLine]],
) -> list[tuple[SectionCandidate, list[TextLine]]]:
    
    all_lines = flatten_body_lines(body)
    index_by_candidate = {
        (c.page, c.y0, c.x0, c.number): c.line_index
        for c in selected
    }

    ranges: list[tuple[SectionCandidate, list[TextLine]]] = []

    for i, candidate in enumerate(selected):
        start_idx = index_by_candidate[
            (candidate.page, candidate.y0, candidate.x0, candidate.number)
        ]

        if i + 1 < len(selected):
            next_candidate = selected[i + 1]
            end_idx = index_by_candidate[
                (
                    next_candidate.page,
                    next_candidate.y0,
                    next_candidate.x0,
                    next_candidate.number,
                )
            ]
        else:
            end_idx = len(all_lines)

        section_lines = all_lines[start_idx:end_idx]
        ranges.append((candidate, section_lines))

    return ranges


def associate_margin_notes(
    margin_groups: list[MarginGroup],
    selected: list[SectionCandidate],
    warnings: list[str],
    tolerance_pt: float,
) -> list[dict]:
    
    by_page: dict[int, list[SectionCandidate]] = defaultdict(list)
    for c in selected:
        by_page[c.page].append(c)

    active_by_page: dict[int, Optional[int]] = {}
    current: Optional[int] = None

    max_page = max(
        [g.page for g in margin_groups] + [c.page for c in selected],
        default=0,
    )

    for page in range(1, max_page + 1):
        active_by_page[page] = current
        for c in sorted(by_page.get(page, []), key=lambda x: x.y0):
            current = c.number

    associated: list[dict] = []

    for group in sorted(
        margin_groups, key=lambda g: (g.page, g.y0, g.x0)
    ):
        page_sections = sorted(
            by_page.get(group.page, []), key=lambda c: c.y0
        )
        active_before_page = active_by_page.get(group.page)

        chosen: Optional[SectionCandidate] = None
        method = ""
        distance: Optional[float] = None
        confidence = 0.0

        
        if page_sections:
            previous = active_before_page

            for idx, section in enumerate(page_sections):
                next_section = (
                    page_sections[idx + 1]
                    if idx + 1 < len(page_sections)
                    else None
                )

                if group.center_y >= section.y0 - tolerance_pt:
                    if (
                        next_section is None
                        or group.center_y < next_section.y0 - tolerance_pt
                    ):
                        chosen = section
                        method = "same-page-section-interval"
                        distance = abs(group.center_y - section.y0)
                        confidence = max(
                            0.0,
                            1.0 - distance / (tolerance_pt * 2.0),
                        )
                        break

            
            if chosen is None and group.center_y < page_sections[0].y0:
                if active_before_page is not None:
                    chosen = SectionCandidate(
                        number=active_before_page,
                        page=group.page,
                        line_index=-1,
                        y0=0,
                        y1=0,
                        x0=0,
                        x1=0,
                        text="",
                        confidence=0,
                        reason=[],
                    )
                    method = "previous-page-active-section"
                    distance = abs(group.center_y - page_sections[0].y0)
                    confidence = 0.70
        else:
           
            if active_before_page is not None:
                chosen = SectionCandidate(
                    number=active_before_page,
                    page=group.page,
                    line_index=-1,
                    y0=0,
                    y1=0,
                    x0=0,
                    x1=0,
                    text="",
                    confidence=0,
                    reason=[],
                )
                method = "continuation-page-active-section"
                confidence = 0.65

       
        if page_sections:
            nearest = min(
                page_sections,
                key=lambda c: abs(group.center_y - c.y0),
            )
            nearest_distance = abs(group.center_y - nearest.y0)

            if nearest_distance <= tolerance_pt:
                chosen = nearest
                method = "same-page-section-start-alignment"
                distance = nearest_distance
                confidence = max(
                    0.0,
                    1.0 - nearest_distance / tolerance_pt,
                )

        record = {
            "page": group.page,
            "side": group.side,
            "bbox": [
                round(group.x0, 2),
                round(group.y0, 2),
                round(group.x1, 2),
                round(group.y1, 2),
            ],
            "text": group.text,
            "associated_section_number": (
                chosen.number if chosen is not None else None
            ),
            "association_method": method or None,
            "distance_to_section_start_pt": (
                round(distance, 2) if distance is not None else None
            ),
            "confidence": round(confidence, 3),
        }

        if chosen is None:
            warnings.append(
                f"Page {group.page}: margin note could not be associated: "
                f"{group.text!r} (y={group.center_y:.1f})"
            )

        associated.append(record)

    return associated


def normalize_chapter_title(lines: list[str]) -> str:
    text = " ".join(x.strip() for x in lines if x.strip())
    text = re.sub(r"\s+", " ", text).strip()

    def collapse_spaced_word(match: re.Match) -> str:
        token = match.group(0)
        letters = re.findall(r"[A-Z]", token)
        return "".join(letters)

    text = re.sub(
        r"\b(?:[A-Z]\s+){1,}[A-Z]\b",
        collapse_spaced_word,
        text,
    )

    text = re.sub(r"\s+([,.;:])", r"\1", text)
    text = re.sub(r"\s+&\s+", " & ", text)

    return text.strip()


def is_chapter_title_line(text: str) -> bool:
    clean = clean_line_text(text)
    if not clean:
        return False

    letters = [c for c in clean if c.isalpha()]
    if not letters:
        return False

    uppercase_ratio = sum(c.isupper() for c in letters) / len(letters)

    if SECTION_START_RE.match(clean):
        return False
    if re.match(r"^\s*CHAPTER\s*[IVXLCDM]+\s*$", clean, re.I):
        return False

    return uppercase_ratio >= 0.80


def detect_chapters(
    body: dict[int, list[TextLine]],
) -> list[dict]:
    
    lines = flatten_body_lines(body)
    chapters: list[dict] = []

    for i, line in enumerate(lines):
        match = CHAPTER_RE.match(line.text)
        if not match:
            continue

        roman = match.group(1).upper()

        title_lines: list[str] = []

       
        for j in range(i + 1, min(i + 6, len(lines))):
            candidate = lines[j]

            if CHAPTER_RE.match(candidate.text):
                break

            if SECTION_START_RE.match(candidate.text):
                break

            if is_chapter_title_line(candidate.text):
                title_lines.append(candidate.text)
                continue

           
            if title_lines:
                break

        title = normalize_chapter_title(title_lines)

        chapters.append(
            {
                "number": roman,
                "title": title or None,
                "page": line.page,
                "y": round(line.y0, 2),
            }
        )

   
    unique: list[dict] = []
    seen: set[tuple] = set()

    for chapter in chapters:
        key = (
            chapter["number"],
            chapter["page"],
            chapter["y"],
        )

        if key not in seen:
            unique.append(chapter)
            seen.add(key)

    return unique

def chapter_for_section(
    section_page: int,
    chapters: list[dict],
) -> tuple[Optional[str], Optional[str]]:
    candidates = [
        c for c in chapters
        if c["page"] <= section_page
    ]
    if not candidates:
        return None, None

    chapter = candidates[-1]
    return chapter["number"], chapter.get("title")


def make_section_text(lines: list[TextLine]) -> str:
    
    cleaned = []
    for line in lines:
        text = clean_line_text(line.text)
        if text:
            cleaned.append(text)

    return "\n".join(cleaned).strip()


def classify_structure_line(
    text: str,
    section_number: int,
    first_line: bool,
    in_illustration: bool,
    current_subsection: Optional[str],
) -> tuple[str, Optional[str], str]:
    
    working = text

    if first_line:
        
        working = re.sub(
            rf"^\s*{section_number}\s*\.\s*",
            "",
            working,
            count=1,
        ).strip()

    
    if in_illustration:
        numeric = SUBSECTION_RE.match(working)
        if numeric:
            
            is_next_subsection = False
            if current_subsection:
                cm = re.fullmatch(r"\((\d+)\)", current_subsection)
                if cm:
                    is_next_subsection = int(numeric.group(1)) == int(cm.group(1)) + 1
            if not is_next_subsection:
                return "illustration_item", f"({numeric.group(1)})", numeric.group(2)
        alpha = CLAUSE_RE.match(working)
        if alpha:
            return "illustration_item", f"({alpha.group(1)})", alpha.group(2)
        roman = SUBCLAUSE_RE.match(working)
        if roman:
            return "illustration_item", f"({roman.group(1)})", roman.group(2)

    m = SUBSECTION_RE.match(working)
    if m:
        return "subsection", f"({m.group(1)})", m.group(2)

    m = SUBCLAUSE_RE.match(working)
    if m:
        return "subclause", f"({m.group(1)})", m.group(2)

    m = CLAUSE_RE.match(working)
    if m:
        return "clause", f"({m.group(1)})", m.group(2)

    m = EXPLANATION_RE.match(working)
    if m:
        return "explanation", "Explanation", m.group(1)

    m = ILLUSTRATION_RE.match(working)
    if m:
        return "illustration", "Illustration", m.group(1)

    m = PROVISO_RE.match(working)
    if m:
        return "proviso", "Provided that", m.group(1)

    return "text", None, working


def join_legal_lines(lines: list[str]) -> str:
    
    text = " ".join(x.strip() for x in lines if x and x.strip())
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"\s+([)—]?)", lambda m: m.group(0), text)
    return text.strip()


def parse_structure_blocks(lines: list[TextLine]) -> list[dict]:
    
    blocks: list[dict] = []

    for line in lines:
        text = clean_line_text(line.text)
        if not text:
            continue

        kind = "text"
        marker = None
        content = text

        m = SUBSECTION_RE.match(text)
        if m:
            kind = "subsection"
            marker = f"({m.group(1)})"
            content = m.group(2)

        m2 = SUBCLAUSE_RE.match(text)
        if m2 and kind == "text":
            kind = "subclause"
            marker = f"({m2.group(1)})"
            content = m2.group(2)

        m3 = CLAUSE_RE.match(text)
        if m3 and kind == "text":
            kind = "clause"
            marker = f"({m3.group(1)})"
            content = m3.group(2)

        if kind == "text":
            m4 = EXPLANATION_RE.match(text)
            if m4:
                kind = "explanation"
                marker = "Explanation"
                content = m4.group(1)

        if kind == "text":
            m5 = ILLUSTRATION_RE.match(text)
            if m5:
                kind = "illustration"
                marker = "Illustration"
                content = m5.group(1)

        if kind == "text":
            m6 = PROVISO_RE.match(text)
            if m6:
                kind = "proviso"
                marker = "Provided that"
                content = m6.group(1)

        blocks.append(
            {
                "page": line.page,
                "y": round(line.y0, 2),
                "kind": kind,
                "marker": marker,
                "text": content.strip() if content else "",
                "raw_text": text,
                "bbox": [
                    round(line.x0, 2),
                    round(line.y0, 2),
                    round(line.x1, 2),
                    round(line.y1, 2),
                ],
            }
        )

    return blocks


def parse_logical_blocks(
    lines: list[TextLine],
    section_number: int,
) -> list[dict]:
    
    logical: list[dict] = []
    current: Optional[dict] = None
    first_line = True
    in_illustration = False
    current_subsection: Optional[str] = None
    current_explanation: Optional[str] = None
    current_proviso: Optional[str] = None

    def flush() -> None:
        nonlocal current
        if current is None:
            return

        current["text"] = join_legal_lines(current.pop("_parts"))
        current["source_pages"] = sorted(
            {line["page"] for line in current["source_lines"]}
        )
        current["start"] = {
            "page": current["source_lines"][0]["page"],
            "y": current["source_lines"][0]["y"],
            "bbox": current["source_lines"][0]["bbox"],
        }
        current["end"] = {
            "page": current["source_lines"][-1]["page"],
            "y": current["source_lines"][-1]["y"],
            "bbox": current["source_lines"][-1]["bbox"],
        }
        logical.append(current)
        current = None

    for line in lines:
        text = clean_line_text(line.text)
        if not text:
            continue

        kind, marker, content = classify_structure_line(
            text=text,
            section_number=section_number,
            first_line=first_line,
            in_illustration=in_illustration,
            current_subsection=current_subsection,
        )
        first_line = False

    
        is_boundary = kind != "text"

        if is_boundary:
            flush()

            if kind == "subsection":
                current_subsection = marker
                current_explanation = None
                current_proviso = None
                in_illustration = False
            elif kind == "explanation":
                current_explanation = marker
                current_proviso = None
                in_illustration = False
            elif kind == "proviso":
                current_proviso = marker
                in_illustration = False
            elif kind == "illustration":
                in_illustration = True
            elif kind in {"clause", "subclause", "illustration_item"}:
                
                if kind != "illustration_item":
                    in_illustration = False

            parent_type = None
            parent_marker = None
            if kind in {"clause", "subclause"}:
                if current_subsection:
                    parent_type = "subsection"
                    parent_marker = current_subsection
                elif current_proviso:
                    parent_type = "proviso"
                    parent_marker = current_proviso
            elif kind == "illustration_item":
                parent_type = "illustration"
                parent_marker = "Illustration"

            current = {
                "kind": kind,
                "marker": marker,
                "parent_type": parent_type,
                "parent_marker": parent_marker,
                "text": "",
                "source_lines": [],
                "_parts": [],
            }

            
            if content:
                current["_parts"].append(content)
        else:
            if current is None:
                
                current = {
                    "kind": "section_text",
                    "marker": None,
                    "parent_type": None,
                    "parent_marker": None,
                    "text": "",
                    "source_lines": [],
                    "_parts": [],
                }
            current["_parts"].append(content)

        current["source_lines"].append(
            {
                "page": line.page,
                "y": round(line.y0, 2),
                "bbox": [
                    round(line.x0, 2),
                    round(line.y0, 2),
                    round(line.x1, 2),
                    round(line.y1, 2),
                ],
                "raw_text": text,
            }
        )

    flush()
    return logical

def section_record_from_range(
    candidate: SectionCandidate,
    lines: list[TextLine],
    chapters: list[dict],
    margin_notes: list[dict],
) -> SectionRecord:
    section_chapter_number, section_chapter_title = chapter_for_section(
        candidate.page,
        chapters,
    )

    relevant_notes = [
        note
        for note in margin_notes
        if note["associated_section_number"] == candidate.number
    ]

    text = make_section_text(lines)
    structure = parse_structure_blocks(lines)
    logical_blocks = parse_logical_blocks(lines, candidate.number)

    source_pages = sorted({line.page for line in lines})

    start_line = lines[0] if lines else None
    end_line = lines[-1] if lines else None

    start = {
        "page": candidate.page,
        "bbox": [
            round(candidate.x0, 2),
            round(candidate.y0, 2),
            round(candidate.x1, 2),
            round(candidate.y1, 2),
        ],
    }

    if end_line:
        end = {
            "page": end_line.page,
            "bbox": [
                round(end_line.x0, 2),
                round(end_line.y0, 2),
                round(end_line.x1, 2),
                round(end_line.y1, 2),
            ],
        }
    else:
        end = {
            "page": candidate.page,
            "bbox": [
                round(candidate.x0, 2),
                round(candidate.y0, 2),
                round(candidate.x1, 2),
                round(candidate.y1, 2),
            ],
        }

    return SectionRecord(
        number=candidate.number,
        chapter_number=section_chapter_number,
        chapter_title=section_chapter_title,
        marginal_notes=relevant_notes,
        text=text,
        structure_blocks=structure,
        logical_blocks=logical_blocks,
        source_pages=source_pages,
        start=start,
        end=end,
    )


def build_markdown(
    document: dict,
    chapters: list[dict],
    sections: list[SectionRecord],
) -> str:
    out: list[str] = []

    out.append("# Bharatiya Nyaya Sanhita, 2023")
    out.append("")
    out.append(
        f"> Source pages: {document['page_count']}  \n"
        f"> Parsed sections: {len(sections)}  \n"
        f"> Parser version: {document['parser_version']}"
    )
    out.append("")

    current_chapter = None

    for section in sections:
        chapter_key = (
            section.chapter_number,
            section.chapter_title,
        )

        if chapter_key != current_chapter and section.chapter_number:
            current_chapter = chapter_key
            title = section.chapter_title or ""
            out.append(
                f"# Chapter {section.chapter_number}"
                + (f": {title}" if title else "")
            )
            out.append("")

        out.append(f"## Section {section.number}")
        out.append("")

        if section.marginal_notes:
            for note in section.marginal_notes:
                out.append(
                    f"**Marginal note:** {note['text']}"
                )
                out.append("")

        if section.logical_blocks:
            for block in section.logical_blocks:
                label = block.get("marker")
                block_text = block.get("text", "").strip()
                if label and block.get("kind") != "illustration_item":
                    if block_text:
                        out.append(f"**{label}** {block_text}")
                    else:
                        out.append(f"**{label}**")
                elif block_text:
                    out.append(block_text)
                out.append("")
        elif section.text:
            out.append(section.text)

        out.append("")
        out.append(
            "<!-- "
            f"source_pages={','.join(map(str, section.source_pages))}; "
            f"section={section.number}"
            " -->"
        )
        out.append("")

    return "\n".join(out).rstrip() + "\n"


def build_json(
    pdf_path: Path,
    doc: pymupdf.Document,
    chapters: list[dict],
    sections: list[SectionRecord],
    margin_notes: list[dict],
    warnings: list[str],
    diagnostics: dict,
) -> dict:
    return {
        "document": {
            "title": "Bharatiya Nyaya Sanhita, 2023",
            "source_filename": pdf_path.name,
            "source_sha256": sha256_file(pdf_path),
            "page_count": len(doc),
            "parser_version": PARSER_VERSION,
            "ingestion_method": "PyMuPDF coordinate-aware extraction",
            "ocr_used": False,
            "llm_used": False,
        },
        "validation": diagnostics.get("validation", {}),
        "chapters": chapters,
        "sections": [asdict(section) for section in sections],
        "marginal_notes": margin_notes,
        "warnings": warnings,
        "diagnostics": diagnostics,
    }


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(content, encoding="utf-8")
    temp.replace(path)


def atomic_write_json(path: Path, data: dict) -> None:
    content = json.dumps(
        data,
        ensure_ascii=False,
        indent=2,
    )
    atomic_write_text(path, content + "\n")


def print_candidate_diagnostics(
    candidates: list[SectionCandidate],
    validation: dict,
) -> None:
    missing = validation.get("missing_sections", [])

    print("\n=== SECTION CANDIDATE DIAGNOSTICS ===")
    if missing:
        print("Missing sections:", ", ".join(map(str, missing)))
    else:
        print("No missing sections.")

    print("\nCandidates for missing/nearby numbers:")
    interesting = set(missing)

    for number in missing:
        interesting.update(
            n for n in range(number - 2, number + 3)
            if EXPECTED_FIRST_SECTION <= n <= EXPECTED_LAST_SECTION
        )

    for candidate in candidates:
        if candidate.number in interesting:
            print(
                f"page={candidate.page:3d} "
                f"num={candidate.number:3d} "
                f"x0={candidate.x0:7.1f} "
                f"x1={candidate.x1:7.1f} "
                f"y0={candidate.y0:7.1f} "
                f"y1={candidate.y1:7.1f} "
                f"conf={candidate.confidence:.2f} "
                f"text={candidate.text!r}"
            )


def ingest(
    pdf_path: Path,
    output_dir: Path,
    strict: bool,
    diagnose: bool,
    margin_tolerance: float,
) -> ParseResult:
    logging.info("Opening PDF: %s", pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file: {pdf_path}")

    doc = pymupdf.open(pdf_path)

    try:
        if len(doc) == 0:
            raise RuntimeError("PDF contains no pages.")

        pages = extract_lines(doc)
        body, margin_groups = split_body_and_margin(pages, doc)

        body_count = sum(len(lines) for lines in body.values())
        logging.info(
            "Extracted %d body lines and %d margin groups",
            body_count,
            len(margin_groups),
        )

        candidates, candidate_diagnostics = detect_section_candidates(
            body, doc
        )

        logging.info(
            "Detected %d raw section candidates",
            len(candidates),
        )

        selected, resolution_log = resolve_section_sequence(candidates)

        logging.info(
            "Resolved %d section starts",
            len(selected),
        )

       

        validation = validate_complete_bns(
            selected=selected,
            candidates=candidates,
            strict=strict,
        )

        warnings: list[str] = []

        if validation["missing_sections"]:
            warnings.append(
                "Missing BNS sections: "
                + ", ".join(
                    map(str, validation["missing_sections"])
                )
            )

        if validation["unexpected_sections"]:
            warnings.append(
                "Unexpected section numbers: "
                + ", ".join(
                    map(str, validation["unexpected_sections"])
                )
            )

        if validation["duplicate_sections_in_selected"]:
            warnings.append(
                "Duplicate selected section numbers: "
                + ", ".join(
                    map(str, validation["duplicate_sections_in_selected"])
                )
            )

        if not validation["complete"]:
            logging.warning(
                "BNS completeness validation failed: %s",
                validation,
            )

       
        section_ranges = build_section_ranges(selected, body)

        margin_notes = associate_margin_notes(
            margin_groups=margin_groups,
            selected=selected,
            warnings=warnings,
            tolerance_pt=margin_tolerance,
        )

        chapters = detect_chapters(body)

        sections = [
            section_record_from_range(
                candidate=candidate,
                lines=lines,
                chapters=chapters,
                margin_notes=margin_notes,
            )
            for candidate, lines in section_ranges
        ]

        diagnostics = {
            "validation": validation,
            "candidate_count": len(candidates),
            "selected_section_count": len(selected),
            "raw_margin_group_count": len(margin_groups),
            "associated_margin_note_count": sum(
                1
                for note in margin_notes
                if note["associated_section_number"] is not None
            ),
            "unassociated_margin_note_count": sum(
                1
                for note in margin_notes
                if note["associated_section_number"] is None
            ),
            "candidate_diagnostics": candidate_diagnostics,
            "resolution_log": resolution_log,
        }

        document = {
            "title": "Bharatiya Nyaya Sanhita, 2023",
            "source_filename": pdf_path.name,
            "source_sha256": sha256_file(pdf_path),
            "page_count": len(doc),
            "parser_version": PARSER_VERSION,
        }

        result = ParseResult(
            document=document,
            chapters=chapters,
            sections=sections,
            margin_notes=margin_notes,
            warnings=warnings,
            diagnostics=diagnostics,
        )

        output_dir.mkdir(parents=True, exist_ok=True)

        json_data = build_json(
            pdf_path=pdf_path,
            doc=doc,
            chapters=chapters,
            sections=sections,
            margin_notes=margin_notes,
            warnings=warnings,
            diagnostics=diagnostics,
        )

        markdown = build_markdown(
            document=document,
            chapters=chapters,
            sections=sections,
        )

        atomic_write_json(output_dir / "bns.json", json_data)
        atomic_write_text(output_dir / "bns.md", markdown)

        if diagnose:
            diagnostic_path = output_dir / "bns_diagnostics.json"
            atomic_write_json(
                diagnostic_path,
                {
                    "validation": validation,
                    "candidate_diagnostics": candidate_diagnostics,
                    "resolution_log": resolution_log,
                    "warnings": warnings,
                },
            )
            print_candidate_diagnostics(candidates, validation)
            logging.info(
                "Diagnostics written to: %s",
                diagnostic_path,
            )

        logging.info(
            "JSON written to: %s",
            output_dir / "bns.json",
        )
        logging.info(
            "Markdown written to: %s",
            output_dir / "bns.md",
        )

        return result

    finally:
        doc.close()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Parse Bharatiya Nyaya Sanhita PDF into canonical JSON "
            "and Markdown."
        )
    )

    parser.add_argument(
        "pdf",
        type=Path,
        help="Path to the BNS PDF.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/markdown"),
        help="Directory for bns.json and bns.md.",
    )

    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Require a complete BNS section sequence 1..358. "
            "Recommended for the complete official PDF."
        ),
    )

    parser.add_argument(
        "--diagnose",
        action="store_true",
        help=(
            "Write bns_diagnostics.json and print diagnostics for "
            "missing/nearby section candidates."
        ),
    )

    parser.add_argument(
        "--margin-tolerance",
        type=float,
        default=MARGIN_ALIGNMENT_TOLERANCE_PT,
        help=(
            "Vertical tolerance in points for aligning a marginal note "
            "to a section start. Default: 28."
        ),
    )

    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging level.",
    )

    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s: %(message)s",
    )

    try:
        result = ingest(
            pdf_path=args.pdf,
            output_dir=args.output_dir,
            strict=args.strict,
            diagnose=args.diagnose,
            margin_tolerance=args.margin_tolerance,
        )

        validation = result.diagnostics["validation"]

        print("\nBNS ingestion completed")
        print(f"  Pages:             {result.document['page_count']}")
        print(f"  Chapters:          {len(result.chapters)}")
        print(f"  Sections:          {len(result.sections)}")
        print(f"  Margin notes:      {len(result.margin_notes)}")
        print(f"  Warnings:          {len(result.warnings)}")
        print(f"  Validation passed: {validation['complete']}")
        print(f"  JSON:              {args.output_dir / 'bns.json'}")
        print(f"  Markdown:          {args.output_dir / 'bns.md'}")

        if result.warnings:
            print("\nWarnings:")
            for warning in result.warnings:
                print(f"  - {warning}")

        if not validation["complete"]:
            print("\nValidation details:")
            print(
                "  Missing:",
                validation["missing_sections"] or "None",
            )
            print(
                "  Unexpected:",
                validation["unexpected_sections"] or "None",
            )
            print(
                "  Duplicate selected:",
                validation["duplicate_sections_in_selected"] or "None",
            )
            print(
                "\nRun again with --diagnose to inspect candidate "
                "geometry around missing sections."
            )

            if args.strict:
                return 2

        return 0

    except Exception as exc:
        logging.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
