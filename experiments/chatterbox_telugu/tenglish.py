"""Offline Romanized Telugu preview helpers for the isolated Chatterbox POC."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Callable, Sequence


_TELUGU = re.compile(r"[\u0c00-\u0c7f]")
_TOKEN = re.compile(
    r"[A-Za-z]+[A-Za-z0-9]*(?:['’][A-Za-z]+)?|\d+(?:[.,:/-]\d+)*|"
    r"[\u0c00-\u0c7f]+|[^\w\s]+|\s+|[^\s]"
)
_ACRONYM = re.compile(r"^[A-Z][A-Z0-9._+-]*$")
_VERSION = re.compile(r"^[A-Za-z]+[0-9]+[A-Za-z0-9._+-]*$")
_ENGLISH_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "hello", "i", "in", "is", "it", "my", "of", "on", "or", "the",
    "this", "to", "we", "with", "you",
}
_PROTECTED_WORDS = {
    "ai", "api", "cloud", "consultant", "gpu", "hat", "http", "https",
    "java", "javascript", "junior", "kubernetes", "linux", "macos", "ml",
    "mps", "openai", "openshift", "python", "red", "technologies",
    "typescript",
}
_PROPER_NAMES = {
    "chennai", "hyderabad", "india", "kowshik", "telangana",
}
_TRANSLITERATOR_CODE = """
import contextlib
import json
import sys

words = json.load(sys.stdin)
with contextlib.redirect_stdout(sys.stderr):
    from tts.transliteration import _get_engine, _indicxlit_model_dirs, _models_ready
    if not any(_models_ready(path) for path in _indicxlit_model_dirs()):
        raise RuntimeError("IndicXlit Telugu model is not cached; offline conversion stopped.")
    engine = _get_engine()
    candidates = [engine.translit_word(word, "te", topk=1) for word in words]
print(json.dumps(candidates, ensure_ascii=False))
"""


@dataclass(frozen=True)
class TenglishPreview:
    source_text: str
    converted_text: str


def _transliteration_python() -> Path:
    override = os.environ.get("TENG_XLIT_PYTHON")
    if override:
        interpreter = Path(override).expanduser()
    else:
        project_root = Path(__file__).resolve().parents[2]
        interpreter = project_root / ".venv" / "bin" / "python"
        if not interpreter.is_file():
            interpreter = project_root / ".venv" / "Scripts" / "python.exe"
    if not interpreter.is_file():
        raise RuntimeError(
            "Offline Tenglish conversion needs the existing AI4Bharat IndicXlit "
            "environment. Set TENG_XLIT_PYTHON to its Python executable; no "
            "transliteration dependency or model was installed by this experiment."
        )
    return interpreter


def _transliterate_words(words: Sequence[str]) -> list[str]:
    if not words:
        return []
    process = subprocess.run(
        [str(_transliteration_python()), "-c", _TRANSLITERATOR_CODE],
        input=json.dumps(list(words), ensure_ascii=False),
        text=True,
        capture_output=True,
        check=False,
        env=os.environ.copy(),
        cwd=Path(__file__).resolve().parents[2],
    )
    if process.returncode:
        details = process.stderr.strip() or process.stdout.strip()
        raise RuntimeError(f"IndicXlit failed ({process.returncode}): {details}")
    try:
        candidates = json.loads(process.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError("IndicXlit returned invalid transliteration data.") from error
    if not isinstance(candidates, list) or len(candidates) != len(words):
        raise RuntimeError("IndicXlit did not return one result for each input word.")
    converted: list[str] = []
    for source, options in zip(words, candidates):
        if (
            not isinstance(options, list)
            or not options
            or not isinstance(options[0], str)
            or not options[0].strip()
            or not _TELUGU.search(options[0])
        ):
            raise RuntimeError(f"IndicXlit returned no Telugu text for {source!r}.")
        converted.append(options[0].strip())
    return converted


def _preserve_latin(token: str, sentence_start: bool) -> bool:
    lowered = token.lower()
    return bool(
        _ACRONYM.fullmatch(token)
        or _VERSION.fullmatch(token)
        or lowered in _ENGLISH_WORDS
        or lowered in _PROTECTED_WORDS
        or lowered in _PROPER_NAMES
        or (token.istitle() and not sentence_start)
    )


def preview_tenglish(
    text: str,
    transliterate_words: Callable[[Sequence[str]], list[str]] | None = None,
) -> TenglishPreview:
    """Build an editable script preview while preserving non-Telugu content."""
    if not text.strip():
        raise ValueError("Enter Romanized Telugu text before creating a preview.")

    translator = transliterate_words or _transliterate_words
    tokens = [match.group(0) for match in _TOKEN.finditer(text)]
    translatable: list[str] = []
    cache: dict[str, str] = {}
    sentence_start = True
    for token in tokens:
        is_latin = token.isascii() and token.isalpha()
        if not is_latin:
            if any(character in token for character in ".!?।！？"):
                sentence_start = True
            continue
        if not _preserve_latin(token, sentence_start):
            if token.lower() not in cache:
                translatable.append(token)
        sentence_start = False

    translated = translator(translatable)
    if len(translated) != len(translatable):
        raise RuntimeError("Transliterator omitted one or more input words.")
    for source, result in zip(translatable, translated):
        if not result or not _TELUGU.search(result):
            raise RuntimeError(f"Transliterator returned no Telugu text for {source!r}.")
        cache[source.lower()] = result

    output: list[str] = []
    sentence_start = True
    for token in tokens:
        if token.isascii() and token.isalpha():
            if _preserve_latin(token, sentence_start):
                output.append(token)
            else:
                try:
                    output.append(cache[token.lower()])
                except KeyError as error:
                    raise RuntimeError(
                        f"Transliterator did not produce text for {token!r}."
                    ) from error
            sentence_start = False
        else:
            output.append(token)
            if any(character in token for character in ".!?।！？"):
                sentence_start = True

    return TenglishPreview(source_text=text, converted_text="".join(output))


def text_for_synthesis(source_text: str, converted_text: str | None) -> str:
    """Choose exactly the unedited source or editable converted preview."""
    return source_text if converted_text is None else converted_text
