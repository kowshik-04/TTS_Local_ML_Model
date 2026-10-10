from __future__ import annotations

from tts.indicf5_engine import IndicF5Engine


LANG_TELUGU = "Telugu"
LANG_ENGLISH = "English"
LANG_MIXED = "Telugu + English"
LANG_ROMANIZED = "Romanized Telugu"


class EngineRouter:
    """Explicitly route only to engines whose behavior is enabled and documented."""

    def __init__(self, indicf5: IndicF5Engine | None = None) -> None:
        self.indicf5 = indicf5 or IndicF5Engine()

    def generate(self, language: str, reference_path: str, reference_text: str,
                 text: str, progress=None):
        if language == LANG_ENGLISH:
            raise ValueError(
                "English-only custom-voice synthesis is not enabled: no compatible local "
                "English engine with arbitrary reference-voice conditioning has been validated. "
                "No preset-speaker or cloud fallback is used."
            )
        if language == LANG_ROMANIZED:
            raise ValueError(
                "Create and review the Telugu-script preview before generating Romanized Telugu."
            )
        if language not in {LANG_TELUGU, LANG_MIXED}:
            raise ValueError(f"Unsupported language mode: {language}")
        return self.indicf5.generate(reference_path, reference_text, text, progress=progress)
