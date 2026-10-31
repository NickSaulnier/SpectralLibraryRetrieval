"""Sample descriptions from the splib07 HTML metadata release (HTMLmetadata.zip).

Each spectrum has a page named like its ASCII file without the prefix, e.g.
`Bytownite_HS106.3B_Plagio_BECKc_AREF.html`, holding sections such as SAMPLE_DESCRIPTION,
XRD_ANALYSIS, MICROSCOPIC_EXAMINATION and SPECTROSCOPIC_DISCUSSION. The pages are parsed as
plain text; nothing in them is executed.
"""

from __future__ import annotations

import html
import re
from pathlib import Path

SECTIONS = {
    "description": ("SAMPLE_DESCRIPTION:", "END_SAMPLE_DESCRIPTION."),
    "xrd": ("XRD_ANALYSIS:", "END_XRD_ANALYSIS."),
    "composition": ("COMPOSITION_DISCUSSION:", "END_COMPOSITION_DISCUSSION."),
    "microscope": ("MICROSCOPIC_EXAMINATION:", "END_MICROSCOPIC_EXAMINATION."),
    "spectroscopy": ("SPECTROSCOPIC_DISCUSSION:", "END_SPECTROSCOPIC_DISCUSSION."),
}
EMPTY = {"", "none", "not done yet", "no compositional analyses."}
NOTE_CHARS = 1200


def page_text(path: Path) -> str:
    data = path.read_bytes()
    try:
        raw = data.decode("utf-8")
    except UnicodeDecodeError:  # a few pages are Latin-1
        raw = data.decode("latin-1")
    raw = re.sub(r"<script.*?</script>|<style.*?</style>", "", raw, flags=re.S | re.I)
    text = html.unescape(re.sub(r"<[^>]+>", "\n", raw))
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())


def parse_page(path: str | Path) -> dict[str, str]:
    text = page_text(Path(path))
    out = {}
    for key, (start, end) in SECTIONS.items():
        i, j = text.find(start), text.find(end)
        if i >= 0 and j > i:
            body = " ".join(text[i + len(start) : j].split())
            body = re.sub(r"IMAGE_OF_SAMPLE:.*$", "", body).strip()
            if body.lower() not in EMPTY:
                out[key] = body
    return out


def purity_note(sections: dict[str, str]) -> str | None:
    """A compact note: description, then XRD findings, microscope and spectroscopy remarks."""
    parts = []
    for key, label in (
        ("description", ""),
        ("spectroscopy", "Spectroscopy: "),
        ("microscope", "Microscope: "),
        ("xrd", "XRD: "),
    ):
        if key in sections:
            parts.append(label + sections[key])
    if not parts:
        return None
    note = " | ".join(parts)
    return note if len(note) <= NOTE_CHARS else note[: NOTE_CHARS - 1] + "…"


def load_notes(pages_dir: str | Path) -> dict[str, str]:
    """source_record (page stem) -> note, for every page that has one."""
    notes = {}
    for p in Path(pages_dir).glob("*.html"):
        note = purity_note(parse_page(p))
        if note:
            notes[p.stem] = note
    return notes
