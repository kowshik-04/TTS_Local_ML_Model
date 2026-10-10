from __future__ import annotations

import re


def _split_oversized(text: str, max_bytes: int) -> list[str]:
    words = text.split()
    chunks: list[str] = []
    current = ""
    for word in words:
        if len(word.encode("utf-8")) > max_bytes:
            raise ValueError("A single word exceeds the synthesis chunk size.")
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
    """Split at paragraphs/sentence boundaries, then at word boundaries if needed."""
    if max_bytes < 32:
        raise ValueError("max_bytes must be at least 32.")
    chunks: list[tuple[str, float]] = []
    paragraphs = re.split(r"\n\s*\n", text.strip())
    for paragraph_index, paragraph in enumerate(paragraphs):
        sentences = re.split(r"(?<=[.!?।！？])\s+", paragraph.strip())
        paragraph_chunks: list[str] = []
        current = ""
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
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
            pause = 0.0 if is_last_paragraph and is_last_in_paragraph else (
                0.45 if is_last_in_paragraph else 0.24
            )
            chunks.append((chunk, pause))
    return chunks
