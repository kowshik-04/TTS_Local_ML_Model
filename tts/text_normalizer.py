from __future__ import annotations

import unicodedata


def normalize_text(text: str) -> str:
    """Normalize spacing and Unicode without translating or rewriting words."""
    normalized = unicodedata.normalize("NFC", text or "")
    paragraphs = [
        " ".join(line.split())
        for line in normalized.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    ]
    paragraphs = [paragraph for paragraph in paragraphs if paragraph]
    return "\n\n".join(paragraphs)
