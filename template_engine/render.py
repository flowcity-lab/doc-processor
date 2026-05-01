"""Render-Pipeline fuer das /template/render-Endpoint.

Briefpapier dient als pixelgenauer Hintergrund. Felder aus der Field-Map
werden via PyMuPDF auf die Output-Seiten gestempelt. Fuer P1 ist die
Logik bewusst minimal:

* `text` / `custom` -> insert_textbox mit Mustache-aufgeloestem Wert
* `static` -> insert_textbox mit static_text
* `items_table` -> einfache Tabelle aus context['items'] in die Bbox
* `tax_breakdown` -> Tabelle aus context['tax_breakdown']
* `qr` -> qrcode-Lib + insert_image

Auto-Pagination der Items-Tabelle und Vollausbau der Smart-Spalten kommen
in P9/P12. Hier rendert P1 alles auf einer Seite und schneidet bei
Ueberlauf ab (Warning wird ergaenzt).
"""

from __future__ import annotations

import base64
import io
from typing import Any

import fitz  # PyMuPDF

from .mustache import resolve as resolve_mustache
from .render_fields import draw_field, items_pagination

PREVIEW_DPI = 150


def _hex_to_rgb01(hex_color: str | None) -> tuple[float, float, float]:
    if not hex_color:
        return (0.0, 0.0, 0.0)
    h = hex_color.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    try:
        r = int(h[0:2], 16) / 255.0
        g = int(h[2:4], 16) / 255.0
        b = int(h[4:6], 16) / 255.0
        return (r, g, b)
    except Exception:
        return (0.0, 0.0, 0.0)


def _items_table_field(field_map: list[dict]) -> dict | None:
    """Erstes items_table-Feld auf page_role=first (Auto-Pagination-Anker)."""
    for f in field_map:
        if f.get("kind") == "items_table" and (f.get("page_role") or "first") == "first":
            return f
    return None


def _plan_pages(field_map: list[dict], context: dict, has_rest: bool) -> dict:
    """Plant Output-Seiten anhand items_table-Ueberlauf + rest-Felder.

    Rueckgabe:
      page_count          : Gesamtseiten
      items_per_page      : list[int]; Items pro Output-Seite (0 wenn dort kein items_table)
      items_field_id      : id des Auto-Pagination-Felds (oder None)
    """
    items_field = _items_table_field(field_map)
    items = list(context.get("items") or [])
    has_rest_fields = any((f.get("page_role") == "rest") for f in field_map)

    # Default: 1 Seite, 2 wenn rest-Felder vorhanden + Briefpapier hat rest.
    base_pages = 2 if (has_rest and has_rest_fields) else 1

    if not items_field or not items:
        return {"page_count": base_pages, "items_per_page": [0] * base_pages, "items_field_id": None}

    plan = items_pagination(items_field, items)
    cont = int(plan["continuation_pages"])
    rows_first = int(plan["rows_first"])
    rows_rest = int(plan["rows_rest"])
    total_pages = max(base_pages, 1 + cont)

    items_per_page = [0] * total_pages
    items_per_page[0] = rows_first
    remaining = len(items) - rows_first
    for p in range(1, total_pages):
        if remaining <= 0 or rows_rest <= 0:
            break
        take = min(rows_rest, remaining)
        items_per_page[p] = take
        remaining -= take

    return {
        "page_count": total_pages,
        "items_per_page": items_per_page,
        "items_field_id": items_field.get("id"),
    }


def render_template(
    letterhead_bytes: bytes,
    field_map: list[dict],
    context: dict,
    *,
    mode: str = "final",
) -> dict:
    """Rendert das finale Dokument oder Preview-PNGs."""
    warnings: list[str] = []
    src = fitz.open(stream=letterhead_bytes, filetype="pdf")
    try:
        if src.page_count == 0:
            raise ValueError("Briefpapier-PDF enthaelt keine Seiten.")
        lh_first_index = 0
        has_rest = src.page_count >= 2
        lh_rest_index = 1 if has_rest else 0

        plan = _plan_pages(field_map, context, has_rest)
        page_count = plan["page_count"]
        items_per_page = plan["items_per_page"]
        items_field_id = plan["items_field_id"]

        out = fitz.open()
        for i in range(page_count):
            # Seite 1 = first-Briefpapier; Folgeseiten = rest-Briefpapier (falls vorhanden,
            # sonst Fallback auf first damit das Layout konsistent bleibt).
            src_index = lh_first_index if i == 0 else lh_rest_index
            out.insert_pdf(src, from_page=src_index, to_page=src_index)

        # Felder pro Output-Page stempeln. page_role steuert die Verteilung:
        # - first -> Output-Seite 1
        # - rest  -> Output-Seiten 2..N
        # Spezialfall items_table (page_role=first): rendert auf JEDER Output-Seite den
        # entsprechenden Slice, mit Header-Repeat ab Seite 2 (skip_header=False).
        items_offset = 0
        for f in field_map:
            role = f.get("page_role", "first")
            is_paginated_items = (f.get("id") == items_field_id and items_field_id is not None)

            if is_paginated_items:
                # Rendert auf jeder Seite, an der ein Slice anfaellt.
                for ti, take in enumerate(items_per_page):
                    if take <= 0 or ti >= out.page_count:
                        continue
                    draw_field(
                        out.load_page(ti), f, context, _hex_to_rgb01, resolve_mustache, warnings,
                        items_offset=items_offset, items_max_rows=take, skip_items_header=False,
                    )
                    items_offset += take
                continue

            if role == "first":
                target_indices = [0]
            else:
                target_indices = list(range(1, page_count)) if page_count > 1 else []
            for ti in target_indices:
                if ti >= out.page_count:
                    continue
                draw_field(out.load_page(ti), f, context, _hex_to_rgb01, resolve_mustache, warnings)

        if mode == "preview":
            previews: list[str] = []
            mat = fitz.Matrix(PREVIEW_DPI / 72.0, PREVIEW_DPI / 72.0)
            for p in range(out.page_count):
                pix = out.load_page(p).get_pixmap(matrix=mat, alpha=False)
                previews.append(base64.b64encode(pix.tobytes("png")).decode("ascii"))
            out.close()
            return {"previews": previews, "warnings": warnings}

        buf = io.BytesIO()
        out.save(buf, deflate=True, garbage=4)
        out.close()
        return {"pdf_b64": base64.b64encode(buf.getvalue()).decode("ascii"), "warnings": warnings}
    finally:
        src.close()


def render_preview_pages(letterhead_bytes: bytes, *, dpi: int = 96) -> list[dict]:
    """Liefert pro Briefpapier-Seite ein PNG (base64) + Pt-Dimensionen.

    Wird vom Drag-Editor genutzt; das Frontend braucht beide Massstaebe (Pixel +
    Punkte), um Bbox-Drags korrekt umzurechnen.
    """
    src = fitz.open(stream=letterhead_bytes, filetype="pdf")
    try:
        out: list[dict] = []
        for i in range(src.page_count):
            page = src.load_page(i)
            mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            out.append({
                "index":   i,
                "role":    "first" if i == 0 else "rest",
                "width_pt":  round(float(page.rect.width), 2),
                "height_pt": round(float(page.rect.height), 2),
                "width_px":  pix.width,
                "height_px": pix.height,
                "png_b64":   base64.b64encode(pix.tobytes("png")).decode("ascii"),
            })
        return out
    finally:
        src.close()
