"""Report per-chapter word counts for the generated preliminary report.

The submission imposes strict per-chapter maximums (Introduction 1000,
Literature Review 2500, Design 2000, Feature Prototype 1500) and an overall
6000-word limit excluding the title page and reference list. Counting by hand
is error-prone, so this reads the generated .docx and reports the figures
directly.

Body text and table text are counted separately. Table content is usually
treated as excluded from prose word limits, but since that is a marker
judgement both totals are shown so the position is explicit either way.

Usage:
    python docs/word_count.py
"""

import os
import re

from docx import Document
from docx.document import Document as DocumentType
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

#: Per-chapter maximums from the assignment brief.
LIMITS = {
    "CHAPTER 1": 1000,
    "CHAPTER 2": 2500,
    "CHAPTER 3": 2000,
    "CHAPTER 4": 1500,
}

#: Sections excluded from the word count by the brief.
EXCLUDED = ("CHAPTER 5", "REFERENCES", "TITLE")


def iter_block_items(parent: DocumentType):
    """Yield paragraphs and tables in true document order.

    python-docx exposes paragraphs and tables as separate collections, which
    loses their relative ordering. Walking the underlying XML body preserves it,
    which is required to attribute a table to the chapter it sits in.

    Args:
        parent: The Document to walk.

    Yields:
        Paragraph or Table objects in document order.
    """
    body = parent.element.body
    for child in body.iterchildren():
        if child.tag == qn("w:p"):
            yield Paragraph(child, parent)
        elif child.tag == qn("w:tbl"):
            yield Table(child, parent)


def count_words(text: str) -> int:
    """Count whitespace-delimited tokens containing at least one letter or digit.

    Args:
        text: Raw text.

    Returns:
        Number of words. Bullet glyphs and stray punctuation are not counted.
    """
    return sum(1 for token in text.split() if re.search(r"[A-Za-z0-9]", token))


def main() -> None:
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "Preliminary_Project_Report_v4.docx")
    doc = Document(path)

    counts: dict[str, dict[str, int]] = {}
    subsections: list[tuple[str, int, int]] = []
    current = "TITLE"
    sub = None

    def add_sub(name: str) -> None:
        subsections.append((name, 0, 0))

    for block in iter_block_items(doc):
        if isinstance(block, Paragraph):
            text = block.text.strip()
            if not text:
                continue
            upper = text.upper()
            if block.style.name.startswith("Heading 1") or upper.startswith(
                    ("CHAPTER ", "REFERENCES")):
                current = upper.split(":")[0].strip()
                counts.setdefault(current, {"body": 0, "table": 0})
                sub = None
                continue
            if block.style.name.startswith("Heading 2"):
                sub = text
                add_sub(text)
                continue
            counts.setdefault(current, {"body": 0, "table": 0})
            counts[current]["body"] += count_words(text)
            if sub and subsections:
                name, body, tbl = subsections[-1]
                subsections[-1] = (name, body + count_words(text), tbl)
        else:
            counts.setdefault(current, {"body": 0, "table": 0})
            words = sum(count_words(cell.text)
                        for row in block.rows for cell in row.cells)
            counts[current]["table"] += words
            if sub and subsections:
                name, body, tbl = subsections[-1]
                subsections[-1] = (name, body, tbl + words)

    print(f"{'Section':14}{'Body':>8}{'Tables':>8}{'Total':>8}"
          f"{'Limit':>8}{'Status':>12}")
    print("-" * 58)

    counted_body = 0
    counted_total = 0
    for section, parts in counts.items():
        body, tbl = parts["body"], parts["table"]
        total = body + tbl
        limit = LIMITS.get(section)
        if section.startswith(EXCLUDED):
            status = "excluded"
        elif limit is None:
            status = ""
        else:
            counted_body += body
            counted_total += total
            over = body - limit
            status = "OK" if over <= 0 else f"OVER by {over}"
        limit_str = str(limit) if limit else "-"
        print(f"{section:14}{body:>8}{tbl:>8}{total:>8}{limit_str:>8}{status:>12}")

    print("-" * 58)
    print(f"{'Chapters 1-4':14}{counted_body:>8}{counted_total - counted_body:>8}"
          f"{counted_total:>8}{6000:>8}"
          f"{('OK' if counted_body <= 6000 else 'OVER'):>12}")
    print(f"\nBody text only, chapters 1-4: {counted_body} of 6000 "
          f"({6000 - counted_body} remaining)")
    print(f"Including table text        : {counted_total} of 6000 "
          f"({6000 - counted_total} remaining)")
    print("\nAppendices and references are excluded per the brief.")

    print("\nPer-subsection body words (largest first):")
    for name, body, tbl in sorted(subsections, key=lambda s: -s[1])[:16]:
        print(f"  {body:>5} body {tbl:>4} tbl   {name}")


if __name__ == "__main__":
    main()
