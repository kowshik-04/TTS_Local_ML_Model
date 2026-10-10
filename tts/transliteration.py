from __future__ import annotations

import re
import importlib.util
import os
from functools import lru_cache
from pathlib import Path

from tts.text_normalizer import normalize_text


_PROTECTED = re.compile(
    r"\b(?:Red\s+Hat|Junior\s+Consultant|Linux|Kubernetes|OpenShift|"
    r"Cloud\s+technologies|Cloud|API|HTTP|HTTPS|AI|ML|GPU|CPU|"
    r"Python|JavaScript|TypeScript|macOS|MPS)\b",
    re.IGNORECASE,
)
_PROTECTED_WORDS = {
    "red", "hat", "junior", "consultant", "technologies",
}
_TOKEN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:[.,]\d+)*|[^\w\s]+|\s+")
_UPPERCASE = re.compile(r"^[A-Z0-9][A-Z0-9._+-]*$")
_COMMON_ENGLISH = {
    "a", "an", "and", "are", "as", "at", "by", "for", "from", "i", "in",
    "is", "it", "my", "of", "on", "or", "the", "to", "we", "with",
}


def _indicxlit_model_dirs() -> tuple[Path, Path]:
    spec = importlib.util.find_spec("ai4bharat.transliteration")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError(
            "Romanized Telugu conversion requires ai4bharat-transliteration. "
            "Install requirements.txt and run `python -m tts.setup_models`."
        )
    package_dir = Path(next(iter(spec.submodule_search_locations))) / "rnn"
    package_models = package_dir / "models" / "v0.5"
    user_models = Path.home() / ".AI4Bharat_Xlit_Models" / "v0.5"
    return package_models, user_models


def _models_ready(models_dir: Path) -> bool:
    return all(
        (
            (models_dir / "telugu" / "te_101_model.pth").is_file(),
            (models_dir / "telugu" / "te_scripts.json").is_file(),
        )
    )


@lru_cache(maxsize=2)
def _get_engine(allow_download: bool = False):
    package_models, user_models = _indicxlit_model_dirs()
    package_dir = package_models.parents[1]
    models_dir = package_models if os.access(package_dir, os.W_OK | os.X_OK) else user_models
    if not allow_download and not any(_models_ready(path) for path in (package_models, user_models)):
        raise RuntimeError(
            "IndicXlit's Telugu model files are not cached. Run "
            "`python -m tts.setup_models` once while online, then retry offline."
        )

    try:
        from ai4bharat.transliteration import XlitEngine
    except ImportError as error:
        raise RuntimeError(
            "Romanized Telugu conversion needs the offline IndicXlit package. "
            "Install project dependencies and run `python -m tts.setup_models` once while online."
        ) from error
    return XlitEngine("te", beam_width=6, rescore=False, model_type="rnn")


def _transliterate_segment(engine, segment: str) -> str:
    leading = segment[: len(segment) - len(segment.lstrip())]
    trailing = segment[len(segment.rstrip()) :]
    result = engine.translit_sentence(segment.strip())
    if isinstance(result, dict):
        translated = result.get("te")
    else:
        translated = result
    if not isinstance(translated, str) or not translated.strip():
        raise RuntimeError("IndicXlit did not return Telugu text for the Romanized input.")
    return f"{leading}{translated.strip()}{trailing}"


def romanized_telugu_to_script(text: str) -> str:
    """Use AI4Bharat's offline neural transliterator, preserving known English tokens."""
    text = normalize_text(text)
    if not text:
        raise ValueError("Enter Romanized Telugu text before creating a preview.")
    engine = _get_engine()
    output: list[str] = []
    roman_run: list[str] = []
    sentence_start = True

    def flush() -> None:
        if roman_run:
            output.append(_transliterate_segment(engine, "".join(roman_run)))
            roman_run.clear()

    for match in _TOKEN.finditer(text):
        token = match.group(0)
        if token.isspace():
            if roman_run:
                roman_run.append(token)
            else:
                output.append(token)
            continue
        if token.isascii() and token.isalpha():
            keep = bool(
                _UPPERCASE.fullmatch(token)
                or token.lower() in _COMMON_ENGLISH
                or token.lower() in _PROTECTED_WORDS
                or _PROTECTED.fullmatch(token)
                or (token.istitle() and not sentence_start)
            )
            if keep:
                flush()
                output.append(token)
            else:
                roman_run.append(token)
            sentence_start = False
            continue
        if token[0].isdigit():
            flush()
            output.append(token)
            sentence_start = False
            continue
        flush()
        output.append(token)
        if any(character in token for character in ".!?।！？"):
            sentence_start = True
    flush()
    return normalize_text("".join(output))


def prepare_transliterator() -> None:
    """Load IndicXlit during the explicit, network-enabled setup step."""
    _get_engine(allow_download=True)
