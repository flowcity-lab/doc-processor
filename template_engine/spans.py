"""Span-Extraktion und Diff-Logik fuer analyze_template.

PyMuPDF liefert pro Seite via page.get_text("dict") ein verschachteltes
Dict block -> line -> span. Ein Span ist die kleinste Einheit mit
einheitlichem Font/Size/Color.
"""

from __future__ import annotations

from typing import Iterable

import fitz  # PyMuPDF

# Toleranz beim Vergleich von Spans zwischen Briefpapier und Sample (in pt).
# Ein Span gilt als "im Briefpapier vorhanden", wenn Text identisch UND
# bbox-Mittelpunkt < BBOX_TOLERANCE_PT entfernt liegen.
BBOX_TOLERANCE_PT = 3.0

# Spans, deren Hoehe unter MIN_SPAN_HEIGHT_PT liegt (Trennlinien, Punkte,
# Designartefakte), werden ignoriert.
MIN_SPAN_HEIGHT_PT = 4.0


def _bbox_of_span(span: dict) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = span["bbox"]
    return (float(x0), float(y0), float(x1), float(y1))


def _center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    x0, y0, x1, y1 = bbox
    return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)


def _color_to_hex(color_int: int) -> str:
    """PyMuPDF kodiert Farbe als 24-bit-int."""
    return "#{:06x}".format(int(color_int) & 0xFFFFFF)


def _font_weight(font_name: str, flags: int) -> str:
    name = (font_name or "").lower()
    bold = bool(flags & 16) or "bold" in name or "black" in name
    return "bold" if bold else "normal"


def extract_page_spans(page: fitz.Page) -> list[dict]:
    """Liefert eine flache Liste aller Spans einer Seite.

    Jeder Eintrag: {text, bbox(x,y,w,h), style(font_family, font_size,
    font_weight, color), original_bbox(x0,y0,x1,y1)}.
    """
    out: list[dict] = []
    raw = page.get_text("dict")
    for block in raw.get("blocks", []):
        if block.get("type", 0) != 0:  # 0 = text, 1 = image
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = (span.get("text") or "").strip()
                if not text:
                    continue
                bbox = _bbox_of_span(span)
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                if h < MIN_SPAN_HEIGHT_PT or w <= 0:
                    continue
                out.append({
                    "text": text,
                    "bbox": {
                        "x": bbox[0],
                        "y": bbox[1],
                        "w": w,
                        "h": h,
                    },
                    "style": {
                        "font_family": span.get("font", "Helvetica"),
                        "font_size":   round(float(span.get("size", 10.0)), 2),
                        "font_weight": _font_weight(span.get("font", ""), int(span.get("flags", 0))),
                        "color":       _color_to_hex(span.get("color", 0)),
                    },
                    "_orig_bbox": bbox,
                })
    return out


def diff_sample_vs_letterhead(
    sample_spans: list[dict],
    letterhead_spans: list[dict],
) -> list[dict]:
    """Filtert aus sample_spans alles raus, was bereits im Briefpapier steht.

    Match-Kriterium: gleicher Text (case-insensitive, trim) UND Mittelpunkts-
    Distanz < BBOX_TOLERANCE_PT.
    """
    if not letterhead_spans:
        return list(sample_spans)
    static_index: list[tuple[str, tuple[float, float]]] = []
    for s in letterhead_spans:
        static_index.append((
            s["text"].lower().strip(),
            _center(s["_orig_bbox"]),
        ))

    variable: list[dict] = []
    for s in sample_spans:
        key = s["text"].lower().strip()
        cx, cy = _center(s["_orig_bbox"])
        is_static = False
        for st_key, (sx, sy) in static_index:
            if st_key != key:
                continue
            if abs(cx - sx) <= BBOX_TOLERANCE_PT and abs(cy - sy) <= BBOX_TOLERANCE_PT:
                is_static = True
                break
        if not is_static:
            variable.append(s)
    return variable


def merge_spans_to_blocks(spans: list[dict], y_tolerance_pt: float = 2.0) -> list[dict]:
    """Fasst horizontal benachbarte Spans der gleichen Zeile zu einem Block zusammen.

    Reduziert Field-Vorschlaege drastisch: 'Max', 'Mustermann' -> ein Feld
    'Max Mustermann'.
    """
    if not spans:
        return []
    sorted_spans = sorted(spans, key=lambda s: (s["bbox"]["y"], s["bbox"]["x"]))
    blocks: list[dict] = []
    current: list[dict] = [sorted_spans[0]]
    for sp in sorted_spans[1:]:
        last = current[-1]
        same_line = abs(sp["bbox"]["y"] - last["bbox"]["y"]) <= y_tolerance_pt
        gap = sp["bbox"]["x"] - (last["bbox"]["x"] + last["bbox"]["w"])
        if same_line and gap <= max(8.0, last["style"]["font_size"] * 0.6):
            current.append(sp)
        else:
            blocks.append(_combine(current))
            current = [sp]
    blocks.append(_combine(current))
    return blocks


def _combine(spans: Iterable[dict]) -> dict:
    spans_l = list(spans)
    text = " ".join(s["text"] for s in spans_l)
    x0 = min(s["bbox"]["x"] for s in spans_l)
    y0 = min(s["bbox"]["y"] for s in spans_l)
    x1 = max(s["bbox"]["x"] + s["bbox"]["w"] for s in spans_l)
    y1 = max(s["bbox"]["y"] + s["bbox"]["h"] for s in spans_l)
    first = spans_l[0]
    return {
        "text":  text,
        "bbox":  {"x": round(x0, 2), "y": round(y0, 2), "w": round(x1 - x0, 2), "h": round(y1 - y0, 2)},
        "style": dict(first["style"]),
    }
