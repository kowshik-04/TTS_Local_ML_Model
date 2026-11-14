"""Sentence-aware chunking for F5/IndicF5 and streaming TTS.

F5's own chunker uses UTF-8 byte counts.  The application keeps the same unit so
Telugu (three bytes per character) does not silently produce oversized chunks.
Long paragraphs are split at sentence boundaries first, then at word boundaries.
"""

from __future__ import annotations

import re


# Periods inside these tokens must not be treated as sentence endings.
_ABBREVIATIONS = {
    "dr", "mr", "mrs", "ms", "prof", "sr", "jr", "vs", "etc", "e.g", "i.e",
    "no", "fig", "figs", "st", "inc", "ltd", "govt", "approx", "dept", "univ",
    "డా", "శ్రీ", "శ్రీమతి", "కో", "అంటే", "ఉదా",
}
_ABBREVIATION_PATTERN = re.compile(
    r"(?<![A-Za-z0-9\u0C00-\u0C7F])(?:"
    + "|".join(re.escape(item) for item in sorted(_ABBREVIATIONS, key=len, reverse=True))
    + r")\.(?=\s|$)",
    re.IGNORECASE,
)
_PLACEHOLDER = "\uE000"
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?।！？…])\s+")
_PARAGRAPH_BOUNDARY = re.compile(r"\n\s*\n")
_TERMINAL = re.compile(r"[.!?।！？…]$")


def _protect_abbreviations(text: str) -> str:
    return _ABBREVIATION_PATTERN.sub(lambda match: match.group(0)[:-1] + _PLACEHOLDER, text)


def _restore_abbreviations(text: str) -> str:
    return text.replace(_PLACEHOLDER, ".")


def _pause_for_token(token: str, *, paragraph_end: bool, sentence_end: bool) -> float:
    if paragraph_end:
        return 0.55
    if sentence_end:
        if token.endswith(("?", "!", "？", "！")):
            return 0.38
        if token.endswith("…"):
            return 0.5
        return 0.3
    return 0.16


def _split_oversized(text: str, max_bytes: int) -> list[str]:
    words = text.split()
    chunks: list[str] = []
    current = ""
    for word in words:
        if len(word.encode("utf-8")) > max_bytes:
            raise ValueError(
                "A single word exceeds the synthesis chunk size. "
                "Shorten it or increase TTS_MAX_CHUNK_BYTES."
            )
        candidate = f"{current} {word}".strip()
        if current and len(candidate.encode("utf-8")) > max_bytes:
            chunks.append(current)
            current = word
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def split_text(text: str, max_bytes: int = 420) -> list[tuple[str, float]]:
    """Split at paragraphs/sentences, then words, returning (chunk, pause) pairs."""
    if max_bytes < 32:
        raise ValueError("max_bytes must be at least 32.")
    text = (text or "").strip()
    if not text:
        return []
    protected = _protect_abbreviations(text)
    chunks: list[tuple[str, float]] = []
    paragraphs = _PARAGRAPH_BOUNDARY.split(protected)
    for paragraph_index, paragraph in enumerate(paragraphs):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        sentences = [part.strip() for part in _SENTENCE_BOUNDARY.split(paragraph) if part.strip()]
        paragraph_chunks: list[str] = []
        current = ""
        for sentence in sentences:
            candidate = f"{current} {sentence}".strip()
            if current and len(candidate.encode("utf-8")) > max_bytes:
                paragraph_chunks.append(current)
                current = sentence
            else:
                current = candidate
            if len(current.encode("utf-8")) > max_bytes:
                paragraph_chunks.extend(_split_oversized(current, max_bytes))
                current = ""
        if current:
            paragraph_chunks.append(current)
        for chunk_index, chunk in enumerate(paragraph_chunks):
            is_last_in_paragraph = chunk_index == len(paragraph_chunks) - 1
            is_last_paragraph = paragraph_index == len(paragraphs) - 1
            sentence_end = bool(_TERMINAL.search(chunk))
            pause = 0.0 if is_last_paragraph and is_last_in_paragraph else _pause_for_token(
                chunk,
                paragraph_end=is_last_in_paragraph,
                sentence_end=sentence_end,
            )
            chunks.append((_restore_abbreviations(chunk).strip(), pause))
    return chunks


def indicf5_max_bytes(reference_text: str, reference_seconds: float, speed: float = 1.0) -> int:
    """Reproduce F5's internal ``max_chars`` estimate in UTF-8 bytes.

    F5 computes a character budget from the reference duration and transcript
    length, then may split an input chunk again inside ``infer_process``.  Sizing
    our chunks below that boundary avoids an internal thread pool and keeps MPS
    memory predictable.  The implementation is conservative for short refs.
    """
    reference_bytes = max(1, len((reference_text or "").strip().encode("utf-8")))
    duration = max(0.5, float(reference_seconds or 0.0))
    speed = max(0.5, float(speed))
    estimate = int(reference_bytes / duration * max(22.0 - duration, 2.0) * speed)
    return max(64, min(420, estimate))


def split_for_indicf5(
    text: str,
    reference_text: str,
    reference_seconds: float,
    *,
    speed: float = 1.0,
    max_bytes: int = 420,
) -> list[tuple[str, float]]:
    limit = min(max_bytes, indicf5_max_bytes(reference_text, reference_seconds, speed))
    return split_text(text, max_bytes=limit)
