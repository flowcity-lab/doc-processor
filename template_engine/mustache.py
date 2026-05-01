"""Minimaler Mustache-Resolver fuer Field-Placeholder.

Nur Variablen mit Punkt-Notation werden unterstuetzt: {{recipient.name}},
{{document.total_gross}}, {{custom.projektnummer}}, {{trainer.business_iban}}.

Sections/Loops sind hier NICHT noetig - Items/Tax-Breakdown rendert die
Render-Logik selbst aus context['items']/context['tax_breakdown'].
"""

from __future__ import annotations

import re
from typing import Any

PLACEHOLDER_RE = re.compile(r"\{\{\s*([a-zA-Z0-9_.]+)\s*\}\}")


def _lookup(context: dict, dotted: str) -> Any:
    cur: Any = context
    for part in dotted.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def resolve(template: str, context: dict) -> str:
    """Ersetzt alle {{var.path}} im String. Unbekannte Variablen -> leerer String."""
    if not template:
        return ""

    def _sub(match: re.Match) -> str:
        value = _lookup(context, match.group(1))
        if value is None:
            return ""
        if isinstance(value, (list, dict)):
            return ""
        return str(value)

    return PLACEHOLDER_RE.sub(_sub, template)
