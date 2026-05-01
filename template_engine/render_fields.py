"""Per-Field-Render-Helfer fuer render.py.

Halten render.py klein und buendeln die Spezial-Logik je FieldKind.
"""

from __future__ import annotations

import io
from typing import Any, Callable

import fitz  # PyMuPDF


# PyMuPDF Base-14-Fonts (kein Embedding noetig). Mapping aus style.font_family.
def _fontname(style: dict) -> str:
    family = (style.get("font_family") or "").lower()
    weight = (style.get("font_weight") or "").lower()
    is_bold = weight in ("bold", "black", "700", "800", "900")
    if "times" in family or "serif" in family:
        return "tibo" if is_bold else "tiro"
    if "courier" in family or "mono" in family:
        return "cobo" if is_bold else "cour"
    return "hebo" if is_bold else "helv"


def _align_const(style: dict) -> int:
    a = (style.get("align") or "left").lower()
    return {
        "left":   fitz.TEXT_ALIGN_LEFT,
        "center": fitz.TEXT_ALIGN_CENTER,
        "right":  fitz.TEXT_ALIGN_RIGHT,
        "justify": fitz.TEXT_ALIGN_JUSTIFY,
    }.get(a, fitz.TEXT_ALIGN_LEFT)


def _bbox_to_rect(bbox: dict) -> fitz.Rect:
    return fitz.Rect(
        float(bbox["x"]),
        float(bbox["y"]),
        float(bbox["x"]) + float(bbox["w"]),
        float(bbox["y"]) + float(bbox["h"]),
    )


def _draw_text(
    page: fitz.Page,
    bbox: dict,
    text: str,
    style: dict,
    hex_to_rgb: Callable[[str | None], tuple[float, float, float]],
) -> int:
    """Stempelt Text in die Bbox. Single-Line via insert_text (zeichnet immer),
    Multi-Line via insert_textbox mit Auto-Expand bei Ueberlauf.
    Rueckgabe: 0 = ok, negativ = abgeschnitten.
    """
    text = text or ""
    rect = _bbox_to_rect(bbox)
    fontname = _fontname(style)
    fontsize = float(style.get("font_size") or 11)
    color = hex_to_rgb(style.get("color"))
    line_height = float(style.get("line_height") or 1.2)
    align = _align_const(style)

    if "\n" not in text:
        # PyMuPDF rechnet die x/y-Position als Baseline. Die Bbox ist top-left.
        # Wir setzen die Baseline so, dass der Glyph in der Bbox-Mitte sitzt.
        ascent = fontsize * 0.8
        baseline_y = rect.y0 + ascent
        # Horizontal-Alignment (rudimentaer; PyMuPDF liefert Textbreite via get_text_length).
        text_w = fitz.get_text_length(text, fontname=fontname, fontsize=fontsize)
        if align == fitz.TEXT_ALIGN_RIGHT:
            x = rect.x1 - text_w
        elif align == fitz.TEXT_ALIGN_CENTER:
            x = rect.x0 + (rect.width - text_w) / 2.0
        else:
            x = rect.x0
        page.insert_text((x, baseline_y), text, fontname=fontname, fontsize=fontsize, color=color)
        return 0

    # Multi-Line: insert_textbox; bei Ueberlauf Bbox nach unten expandieren.
    rc = page.insert_textbox(
        rect, text, fontname=fontname, fontsize=fontsize, color=color,
        align=align, lineheight=line_height,
    )
    if rc >= 0:
        return rc
    expanded = fitz.Rect(rect.x0, rect.y0, rect.x1, rect.y1 + abs(rc) + fontsize)
    return page.insert_textbox(
        expanded, text, fontname=fontname, fontsize=fontsize, color=color,
        align=align, lineheight=line_height,
    )


DEFAULT_ITEMS_COLUMNS: list[dict] = [
    {"key": "description", "label": "Bezeichnung", "width_pct": 50, "align": "left",  "mode": "always"},
    {"key": "quantity",    "label": "Menge",       "width_pct": 10, "align": "right", "mode": "always"},
    {"key": "unit",        "label": "Einheit",     "width_pct": 10, "align": "left",  "mode": "smart"},
    {"key": "unit_price",  "label": "Einzelpreis", "width_pct": 15, "align": "right", "mode": "always"},
    {"key": "tax_rate",    "label": "USt %",       "width_pct": 8,  "align": "right", "mode": "smart"},
    {"key": "line_total",  "label": "Summe",       "width_pct": 7,  "align": "right", "mode": "always"},
]

# Werte, die im Smart-Mode als "leer/null" gelten (fuer discount, tax_rate, ...).
_EMPTY_NUMERIC = {"", "0", "0,00", "0.00", "0,0", "0.0", "—", "-", None}


def _filter_smart_columns(columns: list[dict], items: list[dict]) -> list[dict]:
    """Smart-Mode: blendet Spalten aus, wenn alle Items denselben (oder leeren) Wert haben.
    - mode='never' -> immer raus
    - mode='always' -> immer drin
    - mode='smart' -> raus wenn (a) discount: kein Item hat positiven Wert,
                                (b) sonst: alle Werte identisch oder alle leer.
    """
    out: list[dict] = []
    for col in columns:
        mode = (col.get("mode") or "always").lower()
        if mode == "never":
            continue
        if mode == "smart":
            key = str(col.get("key") or "")
            values = [str(it.get(key, "") or "").strip() for it in items]
            if key == "discount":
                if all(v in _EMPTY_NUMERIC for v in values):
                    continue
            else:
                # Hide wenn alle leer ODER alle identisch (auch nicht-leer).
                unique = set(values)
                if len(unique) <= 1:
                    continue
        out.append(col)
    return out


def _row_height(style: dict, ref: dict) -> float:
    """Bevorzugt ref.row_height_pt, sonst style.font_size * line_height (default 1.4)."""
    explicit = ref.get("row_height_pt")
    if explicit is not None:
        try:
            v = float(explicit)
            if v > 0:
                return v
        except (TypeError, ValueError):
            pass
    fontsize = float(style.get("font_size") or 10)
    line_h = float(style.get("line_height") or 1.4)
    return fontsize * line_h


def items_pagination(field: dict, items: list[dict]) -> dict:
    """Berechnet pro Seite, wieviele Rows passen. Rueckgabe:
        { 'rows_first': int, 'rows_rest': int, 'continuation_pages': int }
    rows_first = Items auf der Original-Bbox (Seite 1, mit Header).
    rows_rest  = Items pro Folgeseite (gleiche Bbox, Header wiederholt).
    """
    bbox = field.get("bbox_pt") or {}
    style = field.get("style") or {}
    ref = field.get("ref") or {}
    columns = ref.get("columns") or DEFAULT_ITEMS_COLUMNS
    visible_cols = _filter_smart_columns(columns, items)

    row_h = _row_height(style, ref)
    header_h = row_h
    bbox_h = float(bbox.get("h") or 0)
    avail = max(0.0, bbox_h - header_h)
    rows_per_page = max(0, int(avail // row_h)) if row_h > 0 else 0
    n = len(items)
    if rows_per_page <= 0 or n == 0 or not visible_cols:
        return {"rows_first": n, "rows_rest": rows_per_page, "continuation_pages": 0}
    if n <= rows_per_page:
        return {"rows_first": n, "rows_rest": rows_per_page, "continuation_pages": 0}
    remaining = n - rows_per_page
    cont = (remaining + rows_per_page - 1) // rows_per_page
    return {"rows_first": rows_per_page, "rows_rest": rows_per_page, "continuation_pages": cont}


def _draw_items_table(
    page: fitz.Page,
    bbox: dict,
    items: list[dict],
    style: dict,
    ref: dict,
    hex_to_rgb: Callable[[str | None], tuple[float, float, float]],
    warnings: list[str],
    *,
    skip_header: bool = False,
    max_rows: int | None = None,
    item_offset: int = 0,
) -> int:
    """Zeichnet Header (optional) + Rows. Gibt Anzahl gerenderter Items zurueck.

    - columns aus ref.columns (oder Defaults) + Smart-Mode-Filter
    - max_rows begrenzt Rows; wenn None, fuellt verfuegbare Hoehe
    - item_offset: ab welchem Index der `items`-Liste gerendert wird (Continuation)
    """
    rect = _bbox_to_rect(bbox)
    if not items or rect.height <= 0:
        return 0
    columns = ref.get("columns") or DEFAULT_ITEMS_COLUMNS
    visible_cols = _filter_smart_columns(columns, items)
    if not visible_cols:
        return 0

    row_style = ref.get("row_style") or style
    header_style = ref.get("header_style") or {**style, "font_weight": "bold"}
    row_color = hex_to_rgb(row_style.get("color") or style.get("color"))
    head_color = hex_to_rgb(header_style.get("color") or style.get("color"))
    row_font = _fontname(row_style)
    head_font = _fontname(header_style)
    row_size = float(row_style.get("font_size") or style.get("font_size") or 10)
    head_size = float(header_style.get("font_size") or row_size)

    row_h = _row_height(style, ref)
    bbox_h = rect.height
    avail = bbox_h if skip_header else max(0.0, bbox_h - row_h)
    fit_rows = int(avail // row_h) if row_h > 0 else 0
    slice_items = items[item_offset:]
    if max_rows is not None:
        fit_rows = min(fit_rows, max_rows)
    fit_rows = min(fit_rows, len(slice_items))
    if fit_rows <= 0:
        return 0

    total_pct = sum(float(c.get("width_pct") or 0) for c in visible_cols) or 100.0
    col_widths = [rect.width * (float(c.get("width_pct") or 0) / total_pct) for c in visible_cols]
    x_offsets = [rect.x0]
    for w in col_widths[:-1]:
        x_offsets.append(x_offsets[-1] + w)

    align_map = {"left": fitz.TEXT_ALIGN_LEFT, "center": fitz.TEXT_ALIGN_CENTER, "right": fitz.TEXT_ALIGN_RIGHT}

    y = rect.y0
    if not skip_header:
        for i, col in enumerate(visible_cols):
            cell = fitz.Rect(x_offsets[i], y, x_offsets[i] + col_widths[i], y + row_h)
            page.insert_textbox(
                cell, str(col.get("label") or col.get("key") or ""),
                fontname=head_font, fontsize=head_size, color=head_color,
                align=align_map.get((col.get("align") or "left").lower(), fitz.TEXT_ALIGN_LEFT),
            )
        y += row_h

    for r in range(fit_rows):
        row = slice_items[r]
        for i, col in enumerate(visible_cols):
            val = row.get(col.get("key") or "", "")
            cell = fitz.Rect(x_offsets[i], y, x_offsets[i] + col_widths[i], y + row_h)
            page.insert_textbox(
                cell, str(val if val is not None else ""),
                fontname=row_font, fontsize=row_size, color=row_color,
                align=align_map.get((col.get("align") or "left").lower(), fitz.TEXT_ALIGN_LEFT),
            )
        y += row_h
    return fit_rows


def _draw_tax_breakdown(
    page: fitz.Page,
    bbox: dict,
    rows: list[dict],
    style: dict,
    hex_to_rgb: Callable[[str | None], tuple[float, float, float]],
    warnings: list[str],
) -> None:
    """Drei-Spalten-Tabelle (Satz | Netto | Steuer). Kein Pagebreak (max ~5 Saetze)."""
    rect = _bbox_to_rect(bbox)
    if not rows:
        return
    fontname = _fontname(style)
    fontsize = float(style.get("font_size") or 10)
    color = hex_to_rgb(style.get("color"))
    line_h = fontsize * float(style.get("line_height") or 1.3)

    col_widths = [rect.width * 0.40, rect.width * 0.30, rect.width * 0.30]
    headers = ["Satz", "Netto", "Steuer"]
    y = rect.y0
    x_offsets = [rect.x0]
    for w in col_widths[:-1]:
        x_offsets.append(x_offsets[-1] + w)

    # Header
    for i, h in enumerate(headers):
        cell = fitz.Rect(x_offsets[i], y, x_offsets[i] + col_widths[i], y + line_h)
        page.insert_textbox(cell, h, fontname="hebo", fontsize=fontsize, color=color,
                            align=(fitz.TEXT_ALIGN_RIGHT if i > 0 else fitz.TEXT_ALIGN_LEFT))
    y += line_h

    # Rows
    for row in rows:
        if y + line_h > rect.y1:
            warnings.append("tax_breakdown: Ueberlauf - Bbox zu klein.")
            break
        cells = [
            str(row.get("rate", "")) + "%" if row.get("rate") not in (None, "") else "",
            str(row.get("net", "")),
            str(row.get("tax", "")),
        ]
        for i, val in enumerate(cells):
            cell = fitz.Rect(x_offsets[i], y, x_offsets[i] + col_widths[i], y + line_h)
            page.insert_textbox(cell, val, fontname=fontname, fontsize=fontsize, color=color,
                                align=(fitz.TEXT_ALIGN_RIGHT if i > 0 else fitz.TEXT_ALIGN_LEFT))
        y += line_h


def _draw_qr(
    page: fitz.Page,
    bbox: dict,
    payload: str,
    error_correction: str = "M",
) -> None:
    """Stempelt einen QR-Code als PNG in die Bbox. Quadratisches Bild, daher
    nutzen wir die kleinere Bbox-Kante; Rest bleibt leer.
    """
    if not payload:
        return
    try:
        import qrcode
        from qrcode.constants import ERROR_CORRECT_L, ERROR_CORRECT_M, ERROR_CORRECT_Q, ERROR_CORRECT_H
    except ImportError:
        return
    ec_map = {"L": ERROR_CORRECT_L, "M": ERROR_CORRECT_M, "Q": ERROR_CORRECT_Q, "H": ERROR_CORRECT_H}
    qr = qrcode.QRCode(error_correction=ec_map.get(error_correction, ERROR_CORRECT_M), border=1)
    qr.add_data(payload)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    rect = _bbox_to_rect(bbox)
    side = min(rect.width, rect.height)
    square = fitz.Rect(rect.x0, rect.y0, rect.x0 + side, rect.y0 + side)
    page.insert_image(square, stream=buf.getvalue())


# EPC-QR / Girocode (EPC069-12). Maximal 331 Bytes UTF-8.
# Aufbau (Zeilen, durch \n getrennt):
#   1: BCD                (Service Tag)
#   2: 002                (Version - 002 = BIC optional)
#   3: 1                  (Encoding - 1 = UTF-8)
#   4: SCT                (Identification - SEPA Credit Transfer)
#   5: BIC                (8 oder 11 Zeichen, leer erlaubt in v002)
#   6: Beneficiary Name   (max 70, Pflicht)
#   7: IBAN               (max 34, Pflicht)
#   8: EUR<amount>        (z.B. EUR123.45, optional; leer = Empfaenger setzt Betrag)
#   9: Purpose            (max 4 Zeichen ISO20022, optional)
#  10: Structured Ref     (max 35, optional, gegenseitig exklusiv mit Zeile 11)
#  11: Unstructured Remit (max 140, optional)
#  12: B2O Info           (max 70, optional)
def _format_epc_amount(raw: str) -> str:
    """Normalisiert einen Betrag zu 'EUR<amount>' im Girocode-Format.
    Akzeptiert '12,34', '12.34', '12', '1.234,56'. Leere Eingabe -> '' (optional).
    """
    raw = (raw or "").strip()
    if not raw:
        return ""
    cleaned = raw.replace("EUR", "").replace("€", "").strip()
    # Wenn Komma als Dezimaltrenner: alle Punkte raus, Komma -> Punkt
    if "," in cleaned and (cleaned.rfind(",") > cleaned.rfind(".")):
        cleaned = cleaned.replace(".", "").replace(",", ".")
    else:
        cleaned = cleaned.replace(",", "")
    try:
        amount = float(cleaned)
    except ValueError:
        return ""
    if amount <= 0 or amount > 999999999.99:
        return ""
    return f"EUR{amount:.2f}"


def _build_epc_payload(ref: dict, context: dict, resolve_mustache: Callable[[str, dict], str], warnings: list[str]) -> str:
    """Baut den EPC069-12 Payload aus den ref-Templates. Gibt '' zurueck wenn
    Pflichtfelder (Name, IBAN) leer aufgeloest werden. Warnungen landen in warnings.
    """
    def _r(key: str) -> str:
        return resolve_mustache(str(ref.get(key) or ""), context).strip()

    bic       = _r("epc_bic").replace(" ", "")
    name      = _r("epc_name")[:70]
    iban      = _r("epc_iban").replace(" ", "")
    amount    = _format_epc_amount(_r("epc_amount"))
    purpose   = _r("epc_purpose")[:4]
    struct_ref = _r("epc_reference")[:35]
    remit     = _r("epc_remittance")[:140]

    if not name or not iban:
        warnings.append("epc_qr: name oder iban leer - QR wird uebersprungen.")
        return ""
    if struct_ref and remit:
        # Gegenseitig exklusiv - struct_ref hat Vorrang.
        warnings.append("epc_qr: epc_reference und epc_remittance gleichzeitig gesetzt - remittance ignoriert.")
        remit = ""

    lines = ["BCD", "002", "1", "SCT", bic, name, iban, amount, purpose, struct_ref, remit]
    # Trailing leere Zeilen droppen (EPC erlaubt es, aber kuerzer = kleinerer QR).
    while lines and lines[-1] == "":
        lines.pop()
    payload = "\n".join(lines)
    if len(payload.encode("utf-8")) > 331:
        warnings.append("epc_qr: Payload ueber 331 Bytes - QR koennte unscanbar sein.")
    return payload


def draw_field(
    page: fitz.Page,
    field: dict,
    context: dict,
    hex_to_rgb: Callable[[str | None], tuple[float, float, float]],
    resolve_mustache: Callable[[str, dict], str],
    warnings: list[str],
    *,
    items_offset: int = 0,
    items_max_rows: int | None = None,
    skip_items_header: bool = False,
) -> int:
    """Dispatcher: ruft den richtigen Renderer je FieldKind auf.

    items_offset / items_max_rows / skip_items_header: nur fuer kind=items_table relevant
    (vom Render-Orchestrator zur Auto-Pagination gesetzt). Rueckgabe = Anzahl gerenderter
    Items (nur bei items_table > 0; sonst 0).
    """
    kind = field.get("kind", "text")
    bbox = field.get("bbox_pt") or {}
    style = field.get("style") or {}
    ref = field.get("ref") or {}

    if kind == "text":
        value = resolve_mustache(field.get("placeholder") or "", context)
        _draw_text(page, bbox, value, style, hex_to_rgb)
    elif kind == "static":
        _draw_text(page, bbox, field.get("static_text") or "", style, hex_to_rgb)
    elif kind == "custom":
        key = (ref.get("custom_key") or "").strip()
        value = ""
        if key:
            value = resolve_mustache("{{custom." + key + "}}", context)
        _draw_text(page, bbox, value, style, hex_to_rgb)
    elif kind == "items_table":
        return _draw_items_table(
            page, bbox, list(context.get("items") or []), style, ref, hex_to_rgb, warnings,
            skip_header=skip_items_header, max_rows=items_max_rows, item_offset=items_offset,
        )
    elif kind == "tax_breakdown":
        _draw_tax_breakdown(page, bbox, list(context.get("tax_breakdown") or []), style, hex_to_rgb, warnings)
    elif kind == "qr":
        subtype = ref.get("qr_subtype", "text")
        if subtype == "epc_qr":
            payload = _build_epc_payload(ref, context, resolve_mustache, warnings)
            # EPC-Spec empfiehlt Error-Correction M; bei sehr langen Remit-Texten kann L sinnvoller sein.
            ec = "L" if len(payload.encode("utf-8")) > 250 else "M"
            _draw_qr(page, bbox, payload, error_correction=ec)
        else:
            payload_tpl = ref.get("qr_payload_template") or ""
            payload = resolve_mustache(payload_tpl, context)
            if subtype == "url" and payload and not payload.startswith(("http://", "https://")):
                payload = "https://" + payload
            _draw_qr(page, bbox, payload)
    else:
        warnings.append(f"Unbekannter FieldKind: {kind} - ignoriert.")
    return 0
