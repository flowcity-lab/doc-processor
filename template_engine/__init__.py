"""Template-Engine fuer Document-Templates v2.

Architektur: das Briefpapier-PDF ist fixer Hintergrund; variable Daten
werden aus einer Field-Map (siehe Laravel docs/documents-feature-v2.md)
auf vordefinierten Bbox-Positionen ueberlagert.

Drei oeffentliche Funktionen:

* analyze_template(letterhead, sample) -> dict
    PyMuPDF extrahiert Text-Spans aus Sample-PDF und vergleicht mit
    Briefpapier. Was nur im Sample steht, sind Variable-Felder. Optional
    klassifiziert ein Vision-LLM die Spans (recipient.name, document.x, ...).

* render_template(letterhead, field_map, context) -> bytes
    Briefpapier wird Seite fuer Seite kopiert, dann werden die
    Field-Map-Felder mit Daten aus context per insert_textbox/insert_image
    drauf gestempelt. Items-Table und Tax-Breakdown werden als Tabellen
    in ihre Bbox gerendert.

* render_preview_pages(letterhead, dpi=96) -> list[dict]
    Rendert jede Briefpapier-Seite als PNG (base64) fuer den
    Drag-Editor im Frontend.
"""

from .analyze import analyze_template
from .render import render_template, render_preview_pages

__all__ = ["analyze_template", "render_template", "render_preview_pages"]
