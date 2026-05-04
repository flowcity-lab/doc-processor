"""
DOCX-Rendering-Helfer fuer die VGSE-Invoice-Pipeline.

Verantwortlich fuer:
  * DOCX -> PDF Konvertierung (via headless LibreOffice / soffice)
  * PDF  -> PNGs pro Seite (via PyMuPDF/fitz)
  * Inspektion der in einer DOCX referenzierten Fonts (rFonts/Theme) sowie
    Abgleich gegen die im System verfuegbaren Fonts (fontconfig / fc-list).

Diese Funktionen werden von den Endpunkten `/docx/render-pages` und
`/docx/inspect-fonts` in main.py aufgerufen. Sie laufen synchron — Aufrufer
sollen sie via asyncio.to_thread(...) wegdelegieren, um den Event-Loop nicht
zu blockieren (LibreOffice braucht typischerweise 1-3 Sekunden).
"""

from __future__ import annotations

import base64
import io
import logging
import os
import re
import shutil
import subprocess
import tempfile
import zipfile
from typing import Iterable
from xml.etree import ElementTree as ET

import fitz  # PyMuPDF

log = logging.getLogger("dps.docx_render")

# Word-Namespace fuer XML-Suche (rFonts / theme).
_W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_RFONTS_TAG = f"{{{_W_NS}}}rFonts"
_RFONTS_ATTRS = ("ascii", "hAnsi", "cs", "eastAsia")

_SOFFICE_BIN = os.environ.get("SOFFICE_BIN", "soffice")


def _run_soffice_to_pdf(docx_bytes: bytes, out_dir: str) -> str:
    """Konvertiert DOCX-Bytes per LibreOffice in ein PDF und gibt den Pfad zurueck."""
    docx_path = os.path.join(out_dir, "input.docx")
    with open(docx_path, "wb") as f:
        f.write(docx_bytes)

    # `--headless` + `--convert-to pdf` schreibt input.pdf in den outdir.
    # Profile-Dir: separater Ordner, sonst kollidieren parallele soffice-Aufrufe.
    profile_dir = os.path.join(out_dir, "soffice-profile")
    cmd = [
        _SOFFICE_BIN,
        "--headless",
        f"-env:UserInstallation=file://{profile_dir}",
        "--convert-to", "pdf",
        "--outdir", out_dir,
        docx_path,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=90)
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"LibreOffice timeout nach 90s: {e}")
    if proc.returncode != 0:
        raise RuntimeError(
            f"LibreOffice exit={proc.returncode}: "
            f"stderr={proc.stderr.decode('utf-8', 'replace')[:400]}"
        )
    pdf_path = os.path.join(out_dir, "input.pdf")
    if not os.path.exists(pdf_path):
        raise RuntimeError("LibreOffice hat kein input.pdf erzeugt")
    return pdf_path


def render_docx_to_pages(
    docx_bytes: bytes,
    *,
    dpi: int = 110,
    max_pages: int = 12,
) -> list[dict]:
    """DOCX -> [{ page, width_px, height_px, png_b64 }, ...] (max_pages limitiert)."""
    if not docx_bytes:
        raise ValueError("docx_bytes ist leer")

    pages: list[dict] = []
    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = _run_soffice_to_pdf(docx_bytes, tmp)
        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        doc = fitz.open(pdf_path)
        try:
            n = min(doc.page_count, max_pages)
            for i in range(n):
                page = doc.load_page(i)
                pix = page.get_pixmap(matrix=matrix, alpha=False)
                buf = io.BytesIO()
                buf.write(pix.tobytes("png"))
                pages.append({
                    "page":      i + 1,
                    "width_px":  pix.width,
                    "height_px": pix.height,
                    "png_b64":   base64.b64encode(buf.getvalue()).decode("ascii"),
                })
        finally:
            doc.close()

    return pages


def _extract_referenced_fonts(docx_bytes: bytes) -> set[str]:
    """Liest alle in <w:rFonts ascii="..."/> referenzierten Font-Namen aus dem DOCX."""
    fonts: set[str] = set()
    with zipfile.ZipFile(io.BytesIO(docx_bytes)) as zf:
        for name in zf.namelist():
            if not (name.endswith(".xml") and ("word/" in name)):
                continue
            try:
                with zf.open(name) as f:
                    tree = ET.parse(f)
            except ET.ParseError:
                continue
            for el in tree.iter(_RFONTS_TAG):
                for attr in _RFONTS_ATTRS:
                    val = el.get(f"{{{_W_NS}}}{attr}")
                    if val:
                        fonts.add(val.strip())
    return fonts


def _list_system_fonts() -> set[str]:
    """fc-list aufrufen und alle Familien-Namen einsammeln (lowercase)."""
    try:
        proc = subprocess.run(
            ["fc-list", ":", "family"],
            capture_output=True,
            timeout=10,
            text=True,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return set()
    if proc.returncode != 0:
        return set()
    families: set[str] = set()
    for line in proc.stdout.splitlines():
        for part in line.split(","):
            name = part.strip().lower()
            if name:
                families.add(name)
    return families


def _docx_has_embedded_fonts(docx_bytes: bytes) -> bool:
    """`word/fonts/` im DOCX enthaelt eingebettete Font-Daten (.odttf/.ttf)."""
    with zipfile.ZipFile(io.BytesIO(docx_bytes)) as zf:
        for name in zf.namelist():
            if name.startswith("word/fonts/") and not name.endswith("/"):
                return True
    return False


def inspect_fonts(docx_bytes: bytes) -> dict:
    """Liefert {referenced, missing, embedded, system_has}."""
    referenced = sorted(_extract_referenced_fonts(docx_bytes))
    system = _list_system_fonts()
    embedded = _docx_has_embedded_fonts(docx_bytes)
    missing = sorted([f for f in referenced if f.lower() not in system]) if not embedded else []
    return {
        "referenced": referenced,
        "missing":    missing,
        "embedded":   embedded,
        "has_fc_list": bool(system),
    }
