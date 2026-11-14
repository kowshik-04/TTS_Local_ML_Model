from __future__ import annotations

import inspect
import logging
import threading
from typing import Iterator

from tts.config import Settings
from tts.indicf5_engine import IndicF5Engine
from tts.languages import (
    LANG_ENGLISH,
    LANG_MIXED,
    LANG_ROMANIZED,
    LANG_TELUGU,
    canonical_language,
)


LOGGER = logging.getLogger("local-voice-studio.router")

# Re-exported for existing imports.
__all__ = [
    "EngineRouter",
    "LANG_ENGLISH",
    "LANG_MIXED",
    "LANG_ROMANIZED",
    "LANG_TELUGU",
]


def _call_with_supported(function, *args, **kwargs):
    try:
        parameters = inspect.signature(function).parameters
    except (TypeError, ValueError):
        return function(*args, **kwargs)
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return function(*args, **kwargs)
    filtered = {key: value for key, value in kwargs.items() if key in parameters}
    return function(*args, **filtered)


class EngineRouter:
    """Route only to engines whose behavior is enabled and documented.

    ``engine=auto`` keeps IndicF5 as the validated default for Telugu and
    Telugu-English text.  The optional Chatterbox-Telugu engine is used for
    English-only text when enabled, and can be selected explicitly for any
    language with ``engine=chatterbox``.  It is never silently substituted after
    an engine or model failure, and no cloud/preset-speaker fallback exists.
    """

    def __init__(
        self,
        indicf5: IndicF5Engine | None = None,
        *,
        settings: Settings | None = None,
        chatterbox=None,
    ) -> None:
        self.settings = settings or Settings.from_env()
        self.indicf5 = indicf5 or IndicF5Engine(
            device=self.settings.device,
            options=self.settings.options,
            reference_cache_dir=self.settings.reference_cache_dir,
            max_reference_seconds=self.settings.max_aligned_reference_seconds,
            max_chunk_bytes=self.settings.max_chunk_bytes,
            inference_threads=self.settings.inference_threads,
        )
        if chatterbox is not None:
            self.chatterbox = chatterbox
        elif self.settings.enable_chatterbox:
            from tts.chatterbox_engine import ChatterboxEngine

            self.chatterbox = ChatterboxEngine(
                model_dir=self.settings.chatterbox_dir,
                device=self.settings.device,
                options=self.settings.chatterbox_options,
                reference_cache_dir=self.settings.reference_cache_dir,
                max_reference_seconds=self.settings.max_aligned_reference_seconds,
            )
        else:
            self.chatterbox = None

    @property
    def chatterbox_enabled(self) -> bool:
        return self.chatterbox is not None

    def resolve_engine(self, language: str, engine: str = "auto") -> str:
        language = canonical_language(language, self.settings.default_language)
        engine = (engine or "auto").lower()
        if engine not in {"auto", "indicf5", "chatterbox"}:
            raise ValueError(f"Unsupported engine {engine!r}; use auto, indicf5 or chatterbox.")
        if engine == "chatterbox":
            if self.chatterbox is None:
                raise ValueError(
                    "The experimental Chatterbox engine is not enabled. Set "
                    "TTS_ENABLE_CHATTERBOX=1 and download the model first."
                )
            return "chatterbox"
        if engine == "indicf5":
            if language == LANG_ENGLISH:
                raise ValueError(
                    "English-only custom-voice synthesis is not enabled with IndicF5. "
                    "Use the optional experimental Chatterbox-Telugu engine or provide a "
                    "validated bilingual engine."
                )
            return "indicf5"
        # auto: IndicF5 remains the documented default for every language it
        # supports.  Chatterbox is not a hidden fallback; it is selected only
        # for English (where IndicF5 is unsupported) or by an explicit request.
        if language == LANG_ENGLISH:
            if self.chatterbox is not None:
                return "chatterbox"
            raise ValueError(
                "English-only custom-voice synthesis is not enabled. No local English "
                "engine with arbitrary reference-voice conditioning has been validated. "
                "Enable the experimental Chatterbox-Telugu engine after listening tests; "
                "no preset-speaker or cloud fallback is used."
            )
        return "indicf5"

    def _engine(self, name: str):
        if name == "chatterbox":
            if self.chatterbox is None:
                raise ValueError("Chatterbox is not enabled.")
            return self.chatterbox
        return self.indicf5

    def ensure_ready(self, engine: str) -> None:
        """Load the selected engine before a response starts.

        This converts missing dependencies, gated weights and initialization
        failures into a normal pre-stream error instead of a 200 response with
        no audio.  Test doubles without a loader are treated as ready.
        """
        backend = self._engine(engine)
        loader = getattr(backend, "load", None)
        if loader is None:
            return
        try:
            loader()
        except Exception:
            LOGGER.exception("Engine %s failed to initialize", engine)
            raise
        LOGGER.info("Engine %s is ready", engine)

    def generate(
        self,
        language: str,
        reference_path: str,
        reference_text: str,
        text: str,
        progress=None,
        *,
        engine: str = "auto",
        options=None,
        cancel_event: threading.Event | None = None,
    ):
        language = canonical_language(language, self.settings.default_language)
        if language == LANG_ROMANIZED:
            raise ValueError(
                "Create and review the Telugu-script preview before generating Romanized Telugu."
            )
        chosen = self.resolve_engine(language, engine)
        backend = self._engine(chosen)
        kwargs = {"progress": progress, "cancel_event": cancel_event}
        if chosen == "indicf5":
            kwargs["options"] = options or self.settings.options
            return _call_with_supported(
                backend.generate, reference_path, reference_text, text, **kwargs
            )
        return _call_with_supported(
            backend.generate,
            reference_path,
            reference_text,
            text,
            language=language,
            **kwargs,
        )

    def generate_stream(
        self,
        language: str,
        reference_path: str,
        reference_text: str,
        text: str,
        progress=None,
        *,
        engine: str = "auto",
        options=None,
        cancel_event: threading.Event | None = None,
    ) -> Iterator[tuple]:
        language = canonical_language(language, self.settings.default_language)
        if language == LANG_ROMANIZED:
            raise ValueError(
                "Create and review the Telugu-script preview before generating Romanized Telugu."
            )
        chosen = self.resolve_engine(language, engine)
        backend = self._engine(chosen)
        kwargs = {"progress": progress, "cancel_event": cancel_event}
        if chosen == "indicf5":
            kwargs["options"] = options or self.settings.options
        else:
            kwargs["language"] = language
        if hasattr(backend, "generate_stream"):
            yield from _call_with_supported(
                backend.generate_stream,
                reference_path,
                reference_text,
                text,
                **kwargs,
            )
            return
        audio, sample_rate = _call_with_supported(
            backend.generate, reference_path, reference_text, text, **kwargs
        )
        yield audio, sample_rate, 1, 1

    def warmup(self, engine: str = "auto") -> dict:
        engine = (engine or "auto").lower()
        if engine not in {"auto", "indicf5", "chatterbox"}:
            raise ValueError(f"Unsupported engine {engine!r}; use auto, indicf5 or chatterbox.")
        results = {}
        if engine in {"auto", "indicf5"}:
            results["indicf5"] = self.indicf5.warmup()
        if engine in {"auto", "chatterbox"}:
            if self.chatterbox is None:
                if engine == "chatterbox":
                    raise RuntimeError(
                        "The configured TTS_ENGINE is chatterbox, but the engine is not enabled. "
                        "Set TTS_ENABLE_CHATTERBOX=1 and download the model first."
                    )
            else:
                results["chatterbox"] = self.chatterbox.warmup()
        return results

    def info(self) -> dict:
        return {
            "indicf5": self.indicf5.info(),
            "chatterbox": self.chatterbox.info() if self.chatterbox is not None else {
                "engine": "chatterbox",
                "enabled": False,
                "experimental": True,
            },
            "profile": self.settings.profile,
            "options": self.settings.options.__dict__,
        }

    def capabilities(self) -> list[dict]:
        chatterbox_ready = self.chatterbox is not None
        return [
            {
                "language": LANG_TELUGU,
                "engine": "indicf5",
                "status": "enabled",
                "certified": "local Telugu baseline; listening review required",
                "optional_engine": "chatterbox" if chatterbox_ready else None,
            },
            {
                "language": LANG_MIXED,
                "engine": "indicf5",
                "status": "experimental",
                "certified": (
                    "IndicF5 is the default; Tenglish pronunciation is not "
                    "quality-certified. Select engine=chatterbox explicitly to A/B test the fine-tune."
                ),
                "optional_engine": "chatterbox" if chatterbox_ready else None,
            },
            {
                "language": LANG_ENGLISH,
                "engine": "chatterbox" if chatterbox_ready else None,
                "status": "experimental" if chatterbox_ready else "not enabled",
                "certified": "no validated local English custom-voice engine"
                if not chatterbox_ready
                else "community fine-tune; not certified",
            },
            {
                "language": LANG_ROMANIZED,
                "engine": "indicf5",
                "status": "preview_required",
                "certified": "transliteration is ambiguous; review the Telugu-script preview",
            },
        ]
