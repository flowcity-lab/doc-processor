"""Analyze-Pipeline fuer das /template/analyze-Endpoint.

Verarbeitet Briefpapier + Sample-PDF und liefert die Field-Map-Vorschlaege
zurueck. Vision-LLM-Klassifikation (siehe docs/documents-feature-v2.md §10
Schritte 4-5) ist als Erweiterung vorgesehen; P1 deckt zuerst die deter-
ministische Heuristik-Pipeline aus heuristics.py ab. Nicht erkannte Felder
werden als kind=static mit dem erkannten Text vorgeschlagen, sodass der
Trainer im Drag-Editor zuordnen kann.
"""

from __future__ import annotations

import base64
import uuid
from typing import Any

import fitz  # PyMuPDF

from .heuristics import annotate_blocks
from .spans import (
    diff_sample_vs_letterhead,
    extract_page_spans,
    merge_spans_to_blocks,
)

# Maximal akzeptierte Brief­papier-Seitenzahl. Mehr macht keinen Sinn fuer
# Rechnungen/Angebote (first + rest), Rest wird verworfen + Warning.
MAX_LETTERHEAD_PAGES = 2

# Preview-PNG-Aufloesung beim Analyze (Drag-Editor-Hintergrund).
ANALYZE_PREVIEW_DPI = 150


def _render_page_png_b64(page: fitz.Page, dpi: int) -> str:
    mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    return base64.b64encode(pix.tobytes("png")).decode("ascii")


def _page_size_pt(page: fitz.Page) -> dict:
    rect = page.rect
    return {"w": round(float(rect.width), 2), "h": round(float(rect.height), 2)}


def _block_to_field(block: dict, page_role: str) -> dict:
    """Konvertiert einen gemergten Span-Block in einen FieldDefinition-Dict."""
    placeholder = block.get("suggested_placeholder") or ""
    has_placeholder = bool(placeholder)
    field_id = "f_" + uuid.uuid4().hex[:10]
    kind = "text" if has_placeholder else "static"
    out: dict[str, Any] = {
        "id": field_id,
        "kind": kind,
        "page_role": page_role,
        "bbox_pt": block["bbox"],
        "style": {
            "font_family":  block["style"]["font_family"],
            "font_size":    block["style"]["font_size"],
            "font_weight":  block["style"]["font_weight"],
            "color":        block["style"]["color"],
            "align":        "left",
            "line_height":  1.2,
        },
        "source":   "auto",
        "required": False,
    }
    if has_placeholder:
        out["placeholder"] = placeholder
    else:
        out["static_text"] = block["text"]
    out["_detected_text"] = block["text"]
    return out


def analyze_template(
    letterhead_bytes: bytes,
    sample_bytes: bytes,
    *,
    language: str = "de",
    doc_type: str = "invoice",
) -> dict:
    """Liefert den vollstaendigen Analyze-Response.

    Struktur siehe docs/documents-feature-v2.md §10 (POST /template/analyze).
    """
    warnings: list[str] = []
    letterhead = fitz.open(stream=letterhead_bytes, filetype="pdf")
    sample = fitz.open(stream=sample_bytes, filetype="pdf")
    try:
        if letterhead.page_count == 0:
            raise ValueError("Briefpapier-PDF enthaelt keine Seiten.")
        if sample.page_count == 0:
            raise ValueError("Sample-PDF enthaelt keine Seiten.")

        used_lh_pages = min(letterhead.page_count, MAX_LETTERHEAD_PAGES)
        if letterhead.page_count > MAX_LETTERHEAD_PAGES:
            warnings.append(
                f"Briefpapier hat {letterhead.page_count} Seiten - nur erste "
                f"{MAX_LETTERHEAD_PAGES} werden verwendet."
            )

        # Briefpapier-Spans pro Seitenrolle vorhalten (page1=first, page2=rest).
        lh_spans_first = extract_page_spans(letterhead.load_page(0))
        lh_spans_rest = (
            extract_page_spans(letterhead.load_page(1)) if used_lh_pages >= 2 else []
        )

        # Page-size aus Briefpapier-Seite 1 (Output-Standard).
        page_size = _page_size_pt(letterhead.load_page(0))

        # Preview-PNGs der genutzten Briefpapier-Seiten.
        preview_pngs = [
            _render_page_png_b64(letterhead.load_page(i), ANALYZE_PREVIEW_DPI)
            for i in range(used_lh_pages)
        ]

        # Felder aus Sample extrahieren: Sample-Seite 1 = page_role first,
        # Sample-Seiten >=2 = page_role rest. Wenn Briefpapier nur eine Seite
        # hat, kollabiert rest nach first (Output ist eh single-letterhead).
        all_fields: list[dict] = []
        for s_index in range(sample.page_count):
            s_page = sample.load_page(s_index)
            spans = extract_page_spans(s_page)
            if s_index == 0:
                static = lh_spans_first
                role = "first"
            else:
                if used_lh_pages >= 2:
                    static = lh_spans_rest
                    role = "rest"
                else:
                    static = lh_spans_first
                    role = "first"
            variable = diff_sample_vs_letterhead(spans, static)
            blocks = merge_spans_to_blocks(variable)
            blocks = annotate_blocks(blocks)
            for b in blocks:
                all_fields.append(_block_to_field(b, role))

        return {
            "letterhead": {
                "page_count":   used_lh_pages,
                "page_size_pt": page_size,
                "preview_pngs": preview_pngs,
            },
            "field_map": all_fields,
            "warnings": warnings,
            "meta": {"language": language, "doc_type": doc_type},
        }
    finally:
        letterhead.close()
        sample.close()
