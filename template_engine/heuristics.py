"""Regex-Heuristiken fuer initiale Placeholder-Vorschlaege.

Wird in analyze_template() angewandt, sobald die variablen Spans
zu Bloecken gemerged sind. Die Vorschlaege landen als
suggested_placeholder im Output und koennen vom Vision-LLM (optional)
oder vom Trainer im Drag-Editor ueberschrieben werden.
"""

from __future__ import annotations

import re

# Mappings: Pattern -> Placeholder. Reihenfolge = Prioritaet.
PATTERNS: list[tuple[re.Pattern, str]] = [
    # E-Mail
    (re.compile(r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$"), "{{recipient.email}}"),
    # IBAN (DE/AT) - nur als Fallback, normalerweise im Briefpapier
    (re.compile(r"^[A-Z]{2}\d{2}\s?(\d{4}\s?){3,7}\d{0,4}$"), "{{trainer.business_iban}}"),
    # Datum DD.MM.YYYY
    (re.compile(r"^\d{1,2}[./-]\d{1,2}[./-]\d{2,4}$"), "{{document.date}}"),
    # Datum YYYY-MM-DD
    (re.compile(r"^\d{4}-\d{1,2}-\d{1,2}$"), "{{document.date}}"),
    # Geld-Betrag (mit Waehrung)
    (re.compile(r"^[€$£¥]?\s?\d{1,3}([.,]\d{3})*[.,]\d{2}\s?(EUR|USD|CHF|€)?$", re.IGNORECASE), "{{document.total_gross}}"),
    # Telefon (locker)
    (re.compile(r"^\+?\d[\d\s/().\-]{7,}$"), "{{recipient.phone}}"),
]

# Keyword-Hints: wenn der Text auf einen typischen Label hindeutet, lese
# das Label und setze ein passendes Placeholder. Das LABEL selbst bleibt
# statisch (gehoert ins Briefpapier), aber der Wert dahinter ist variabel.
LABEL_HINTS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\brechnungs?(nr|nummer)\b", re.IGNORECASE),  "{{document.number}}"),
    (re.compile(r"\bangebots?(nr|nummer)\b", re.IGNORECASE),   "{{document.number}}"),
    (re.compile(r"\bkunden?(nr|nummer)\b", re.IGNORECASE),     "{{recipient.customer_number}}"),
    (re.compile(r"\b(datum|date)\b", re.IGNORECASE),           "{{document.date}}"),
    (re.compile(r"\bf[äa]llig\b", re.IGNORECASE),              "{{document.due_date}}"),
    (re.compile(r"\bleistungszeitraum\b", re.IGNORECASE),      "{{document.service_period}}"),
    (re.compile(r"\b(betreff|subject)\b", re.IGNORECASE),      "{{document.subject}}"),
    (re.compile(r"\b(summe|gesamt|total)\b", re.IGNORECASE),   "{{document.total_gross}}"),
    (re.compile(r"\bnetto\b", re.IGNORECASE),                  "{{document.total_net}}"),
    (re.compile(r"\b(ust|mwst|vat)\b", re.IGNORECASE),         "{{document.total_tax}}"),
]

# Recipient-Bloecke: Wenn der Text wie eine Empfaenger-Adresse
# strukturiert ist (mehrere Zeilen, Strasse + PLZ), schlagen wir
# {{recipient.address_block}} vor.
ADDRESS_HINT = re.compile(r"\d{4,5}\s+[A-Z]", re.IGNORECASE)


def suggest_placeholder(text: str, prev_block_text: str | None = None) -> str:
    """Heuristik: schlaegt einen Mustache-Placeholder fuer einen Text-Block vor.

    Strategie:
    1) Direktes PATTERNS-Match (z.B. Datum, Email, Geld).
    2) Vorheriger Block enthielt ein LABEL_HINT -> dieser Block ist der Wert.
    3) Adress-Heuristik (PLZ + Stadt).
    4) Fallback: leerer Placeholder, Trainer entscheidet im Editor.
    """
    text = (text or "").strip()
    if not text:
        return ""
    for pat, placeholder in PATTERNS:
        if pat.match(text):
            return placeholder
    if prev_block_text:
        for pat, placeholder in LABEL_HINTS:
            if pat.search(prev_block_text):
                return placeholder
    if ADDRESS_HINT.search(text) and "\n" in text:
        return "{{recipient.address_block}}"
    return ""


def annotate_blocks(blocks: list[dict]) -> list[dict]:
    """Fuegt jedem Block einen suggested_placeholder hinzu (in-place + Rueckgabe)."""
    prev_text: str | None = None
    for b in blocks:
        b["suggested_placeholder"] = suggest_placeholder(b["text"], prev_text)
        prev_text = b["text"]
    return blocks
