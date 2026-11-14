"""Central configuration for the inference core, HTTP API and voice agent.

All values are environment-driven so the same code runs on a Mac demo box and a
Linux GPU worker.  Nothing here imports torch; configuration must stay cheap and
importable by tests and CLI tools.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path

from tts.languages import LANG_ENGLISH, LANG_MIXED, LANG_ROMANIZED, LANG_TELUGU


ROOT = Path(__file__).resolve().parent.parent
PROFILE_NAMES = ("quality", "balanced", "realtime", "custom")


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _bool(name: str, default: bool) -> bool:
    value = _env(name)
    if not value:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int, minimum: int | None = None) -> int:
    value = _env(name)
    if not value:
        return default
    try:
        parsed = int(value)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer, got {value!r}.") from error
    if minimum is not None and parsed < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {parsed}.")
    return parsed


def _float(name: str, default: float, minimum: float | None = None) -> float:
    value = _env(name)
    if not value:
        return default
    try:
        parsed = float(value)
    except ValueError as error:
        raise ValueError(f"{name} must be a number, got {value!r}.") from error
    if minimum is not None and parsed < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {parsed}.")
    return parsed


def _path(name: str, default: Path) -> Path:
    return Path(_env(name) or default).expanduser().resolve()


@dataclass(frozen=True)
class InferenceOptions:
    """Sampler settings passed to the F5/IndicF5 inference loop."""

    nfe_step: int = 32
    cfg_strength: float = 2.0
    sway_sampling_coef: float = -1.0
    speed: float = 1.0
    target_rms: float = 0.1
    cross_fade_duration: float = 0.15

    def validate(self) -> None:
        if not 4 <= self.nfe_step <= 128:
            raise ValueError("nfe_step must be between 4 and 128.")
        if not 0.0 <= self.cfg_strength <= 8.0:
            raise ValueError("cfg_strength must be between 0 and 8.")
        if not -1.0 <= self.sway_sampling_coef <= 1.0:
            raise ValueError("sway_sampling_coef must be between -1 and 1.")
        if not 0.5 <= self.speed <= 2.0:
            raise ValueError("speed must be between 0.5 and 2.0.")
        if not 0.0 < self.target_rms <= 0.5:
            raise ValueError("target_rms must be between 0 and 0.5.")
        if not 0.0 <= self.cross_fade_duration <= 1.0:
            raise ValueError("cross_fade_duration must be between 0 and 1.")


PROFILES: dict[str, InferenceOptions] = {
    # F5 defaults. This is the reference-quality path and the safe default.
    "quality": InferenceOptions(nfe_step=32, cfg_strength=2.0, speed=1.0),
    # A measured trade-off candidate. Validate with eval/benchmark.py before
    # moving production traffic to it.
    "balanced": InferenceOptions(nfe_step=24, cfg_strength=2.0, speed=1.0),
    # First-token latency profile for conversational agents. Quality is not
    # quality-certified; always A/B this against the quality profile.
    "realtime": InferenceOptions(nfe_step=16, cfg_strength=2.0, speed=1.04),
}


@dataclass(frozen=True)
class ChatterboxOptions:
    exaggeration: float = 0.5
    cfg_weight: float = 0.5
    temperature: float = 0.8
    repetition_penalty: float = 1.2
    min_p: float = 0.05
    top_p: float = 1.0


@dataclass(frozen=True)
class Settings:
    # Paths
    root: Path = ROOT
    output_dir: Path = ROOT / "outputs"
    voices_dir: Path = ROOT / "voices"
    cache_dir: Path = ROOT / "cache"
    reference_cache_dir: Path = ROOT / "cache" / "references"

    # Engine
    device: str = "auto"
    engine: str = "auto"
    profile: str = "quality"
    options: InferenceOptions = field(default_factory=lambda: PROFILES["quality"])
    chatterbox_options: ChatterboxOptions = field(default_factory=ChatterboxOptions)
    enable_chatterbox: bool = False
    chatterbox_dir: Path = ROOT / "experiments" / "chatterbox_telugu" / "models" / "chatterbox-telugu"
    chatterbox_model_id: str = "shankarpandala/chatterbox-telugu"
    warmup: bool = True
    inference_threads: int = 0

    # Text / reference limits
    default_language: str = LANG_TELUGU
    max_text_chars: int = 2000
    max_chunk_bytes: int = 420
    min_reference_seconds: float = 1.0
    max_reference_seconds: float = 60.0
    max_aligned_reference_seconds: float = 12.0
    min_voice_seconds: float = 3.0
    recommended_reference_seconds: float = 10.0

    # Output processing
    target_rms_dbfs: float = -20.0
    peak_dbfs: float = -1.5
    highpass_hz: float = 55.0
    fade_ms: float = 8.0
    tail_ms: float = 80.0

    # Exact-output cache
    cache_enabled: bool = True
    cache_max_bytes: int = 10 * 1024 * 1024 * 1024

    # API
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    api_db: Path = ROOT / "data" / "accounts.sqlite3"
    api_static_key: str = ""
    api_require_auth: bool = True
    api_default_monthly_chars: int = 100_000
    api_default_rate_limit: int = 60
    api_max_request_bytes: int = 30 * 1024 * 1024
    api_cors_origins: tuple[str, ...] = ()
    api_trusted_proxies: tuple[str, ...] = ()
    api_metrics_enabled: bool = True
    api_queue_timeout_seconds: float = 120.0
    api_concurrency: int = 1

    # Agent
    enable_agent: bool = True
    agent_host: str = "127.0.0.1"
    agent_port: int = 8001
    agent_api_key: str = ""
    agent_stt_provider: str = "local"
    agent_stt_model: str = "large-v3"
    agent_stt_device: str = "auto"
    agent_stt_compute_type: str = "auto"
    agent_stt_language: str = "te"
    agent_stt_base_url: str = "https://api.openai.com/v1"
    agent_stt_api_key: str = ""
    agent_llm_base_url: str = "http://127.0.0.1:11434/v1"
    agent_llm_model: str = "qwen2.5:7b"
    agent_llm_api_key: str = ""
    agent_llm_temperature: float = 0.4
    agent_llm_max_tokens: int = 300
    agent_system_prompt: str = ""
    agent_vad_frame_ms: int = 20
    agent_vad_speech_dbfs: float = -42.0
    agent_vad_silence_ms: int = 650
    agent_vad_min_speech_ms: int = 250
    agent_vad_max_utterance_seconds: float = 25.0
    agent_vad_pre_roll_ms: int = 200
    agent_input_sample_rate: int = 16_000
    agent_max_history_turns: int = 8
    agent_tts_url: str = ""
    agent_tts_api_key: str = ""
    agent_tts_engine: str = "auto"
    agent_voice_id: str = ""
    agent_output_sample_rate: int = 24000

    def validate(self) -> None:
        self.options.validate()
        if self.profile not in PROFILE_NAMES:
            raise ValueError(f"Unknown inference profile: {self.profile}")
        if self.device not in {"auto", "cpu", "mps", "cuda", "cuda:0", "cuda:1", "cuda:2", "cuda:3"}:
            raise ValueError(f"Unsupported device: {self.device}")
        if self.engine not in {"auto", "indicf5", "chatterbox"}:
            raise ValueError(f"Unsupported engine: {self.engine}")
        if not 1 <= self.max_text_chars <= 100_000:
            raise ValueError("max_text_chars must be between 1 and 100000.")
        if not 0.5 <= self.min_reference_seconds < self.max_aligned_reference_seconds:
            raise ValueError("Reference length limits are inconsistent.")
        if self.api_concurrency < 1:
            raise ValueError("api_concurrency must be >= 1.")
        if self.agent_tts_engine not in {"auto", "indicf5", "chatterbox"}:
            raise ValueError(
                "AGENT_TTS_ENGINE must be auto, indicf5 or chatterbox, got "
                f"{self.agent_tts_engine!r}."
            )
        if self.agent_stt_provider not in {"local", "openai"}:
            raise ValueError("AGENT_STT_PROVIDER must be 'local' or 'openai'.")
        if not 8000 <= self.agent_input_sample_rate <= 48_000:
            raise ValueError("agent_input_sample_rate must be between 8000 and 48000.")
        if self.agent_output_sample_rate != 24_000:
            raise ValueError("agent_output_sample_rate must be 24000 for the current TTS stream.")
        if self.agent_vad_silence_ms <= 0 or self.agent_vad_min_speech_ms <= 0:
            raise ValueError("VAD durations must be positive.")

    @classmethod
    def from_env(cls) -> "Settings":
        profile = _env("TTS_PROFILE", "quality").lower()
        if profile not in PROFILE_NAMES:
            raise ValueError(f"TTS_PROFILE must be one of {PROFILE_NAMES}, got {profile!r}.")
        if profile == "custom":
            base = InferenceOptions(
                nfe_step=_int("TTS_NFE_STEP", 32, 4),
                cfg_strength=_float("TTS_CFG_STRENGTH", 2.0),
                sway_sampling_coef=_float("TTS_SWAY_COEF", -1.0),
                speed=_float("TTS_SPEED", 1.0),
                target_rms=_float("TTS_TARGET_RMS", 0.1),
                cross_fade_duration=_float("TTS_CROSSFADE", 0.15),
            )
        else:
            base = PROFILES[profile]
            base = replace(
                base,
                nfe_step=_int("TTS_NFE_STEP", base.nfe_step, 4),
                cfg_strength=_float("TTS_CFG_STRENGTH", base.cfg_strength),
                sway_sampling_coef=_float("TTS_SWAY_COEF", base.sway_sampling_coef),
                speed=_float("TTS_SPEED", base.speed),
                target_rms=_float("TTS_TARGET_RMS", base.target_rms),
                cross_fade_duration=_float("TTS_CROSSFADE", base.cross_fade_duration),
            )
        chatterbox = ChatterboxOptions(
            exaggeration=_float("TTS_CHATTERBOX_EXAGGERATION", 0.5),
            cfg_weight=_float("TTS_CHATTERBOX_CFG_WEIGHT", 0.5),
            temperature=_float("TTS_CHATTERBOX_TEMPERATURE", 0.8),
            repetition_penalty=_float("TTS_CHATTERBOX_REPETITION_PENALTY", 1.2),
            min_p=_float("TTS_CHATTERBOX_MIN_P", 0.05),
            top_p=_float("TTS_CHATTERBOX_TOP_P", 1.0),
        )
        settings = cls(
            root=_path("TTS_ROOT", ROOT),
            output_dir=_path("TTS_OUTPUT_DIR", ROOT / "outputs"),
            voices_dir=_path("TTS_VOICES_DIR", ROOT / "voices"),
            cache_dir=_path("TTS_CACHE_DIR", ROOT / "cache"),
            reference_cache_dir=_path("TTS_REFERENCE_CACHE_DIR", ROOT / "cache" / "references"),
            device=_env("TTS_DEVICE", "auto").lower(),
            engine=_env("TTS_ENGINE", "auto").lower(),
            profile=profile,
            options=base,
            chatterbox_options=chatterbox,
            enable_chatterbox=_bool("TTS_ENABLE_CHATTERBOX", False),
            chatterbox_dir=_path("TTS_CHATTERBOX_DIR", ROOT / "experiments" / "chatterbox_telugu" / "models" / "chatterbox-telugu"),
            chatterbox_model_id=_env("TTS_CHATTERBOX_MODEL_ID", "shankarpandala/chatterbox-telugu"),
            warmup=_bool("TTS_WARMUP", True),
            inference_threads=_int("TTS_INFERENCE_THREADS", 0, 0),
            default_language=_env("TTS_DEFAULT_LANGUAGE", LANG_TELUGU),
            max_text_chars=_int("TTS_MAX_TEXT_CHARS", 2000, 1),
            max_chunk_bytes=_int("TTS_MAX_CHUNK_BYTES", 420, 32),
            min_reference_seconds=_float("TTS_MIN_REFERENCE_SECONDS", 1.0),
            max_reference_seconds=_float("TTS_MAX_REFERENCE_SECONDS", 60.0),
            max_aligned_reference_seconds=_float("TTS_MAX_ALIGNED_REFERENCE_SECONDS", 12.0),
            min_voice_seconds=_float("TTS_MIN_VOICE_SECONDS", 3.0),
            recommended_reference_seconds=_float("TTS_RECOMMENDED_REFERENCE_SECONDS", 10.0),
            target_rms_dbfs=_float("TTS_TARGET_RMS_DBFS", -20.0),
            peak_dbfs=_float("TTS_PEAK_DBFS", -1.5),
            highpass_hz=_float("TTS_HIGHPASS_HZ", 55.0, 0.0),
            fade_ms=_float("TTS_FADE_MS", 8.0, 0.0),
            tail_ms=_float("TTS_TAIL_MS", 80.0, 0.0),
            cache_enabled=_bool("TTS_CACHE_ENABLED", True),
            cache_max_bytes=_int("TTS_CACHE_MAX_BYTES", 10 * 1024 * 1024 * 1024, 0),
            api_host=_env("TTS_API_HOST", "127.0.0.1"),
            api_port=_int("TTS_API_PORT", 8000, 1),
            api_db=_path("TTS_API_DB", ROOT / "data" / "accounts.sqlite3"),
            api_static_key=_env("TTS_API_KEY"),
            api_require_auth=_bool("TTS_API_REQUIRE_AUTH", True),
            api_default_monthly_chars=_int("TTS_API_DEFAULT_MONTHLY_CHARS", 100_000, 0),
            api_default_rate_limit=_int("TTS_API_DEFAULT_RATE_LIMIT", 60, 0),
            api_max_request_bytes=_int("TTS_API_MAX_REQUEST_BYTES", 30 * 1024 * 1024, 1024),
            api_cors_origins=tuple(
                origin.strip()
                for origin in _env("TTS_API_CORS_ORIGINS").split(",")
                if origin.strip()
            ),
            api_trusted_proxies=tuple(
                proxy.strip()
                for proxy in _env("TTS_API_TRUSTED_PROXIES").split(",")
                if proxy.strip()
            ),
            api_metrics_enabled=_bool("TTS_API_METRICS", True),
            api_queue_timeout_seconds=_float("TTS_API_QUEUE_TIMEOUT", 120.0, 1.0),
            api_concurrency=_int("TTS_API_CONCURRENCY", 1, 1),
            enable_agent=_bool("TTS_ENABLE_AGENT", True),
            agent_host=_env("AGENT_HOST", "127.0.0.1"),
            agent_port=_int("AGENT_PORT", 8001, 1),
            agent_api_key=_env("AGENT_API_KEY"),
            agent_stt_provider=_env("AGENT_STT_PROVIDER", "local").lower(),
            agent_stt_model=_env("AGENT_STT_MODEL", "large-v3"),
            agent_stt_device=_env("AGENT_STT_DEVICE", "auto"),
            agent_stt_compute_type=_env("AGENT_STT_COMPUTE_TYPE", "auto"),
            agent_stt_language=_env("AGENT_STT_LANGUAGE", "te"),
            agent_stt_base_url=_env("AGENT_STT_BASE_URL", "https://api.openai.com/v1"),
            agent_stt_api_key=_env("AGENT_STT_API_KEY"),
            agent_llm_base_url=_env("AGENT_LLM_BASE_URL", "http://127.0.0.1:11434/v1"),
            agent_llm_model=_env("AGENT_LLM_MODEL", "qwen2.5:7b"),
            agent_llm_api_key=_env("AGENT_LLM_API_KEY"),
            agent_llm_temperature=_float("AGENT_LLM_TEMPERATURE", 0.4),
            agent_llm_max_tokens=_int("AGENT_LLM_MAX_TOKENS", 300, 1),
            agent_system_prompt=_env("AGENT_SYSTEM_PROMPT"),
            agent_vad_frame_ms=_int("AGENT_VAD_FRAME_MS", 20, 10),
            agent_vad_speech_dbfs=_float("AGENT_VAD_SPEECH_DBFS", -42.0),
            agent_vad_silence_ms=_int("AGENT_VAD_SILENCE_MS", 650, 100),
            agent_vad_min_speech_ms=_int("AGENT_VAD_MIN_SPEECH_MS", 250, 50),
            agent_vad_max_utterance_seconds=_float("AGENT_VAD_MAX_UTTERANCE_SECONDS", 25.0, 1.0),
            agent_vad_pre_roll_ms=_int("AGENT_VAD_PRE_ROLL_MS", 200, 0),
            agent_input_sample_rate=_int("AGENT_INPUT_SAMPLE_RATE", 16_000, 8_000),
            agent_max_history_turns=_int("AGENT_MAX_HISTORY_TURNS", 8, 1),
            agent_tts_url=_env("AGENT_TTS_URL"),
            agent_tts_api_key=_env("AGENT_TTS_API_KEY"),
            agent_tts_engine=_env("AGENT_TTS_ENGINE", "auto").lower(),
            agent_voice_id=_env("AGENT_VOICE_ID"),
            agent_output_sample_rate=_int("AGENT_OUTPUT_SAMPLE_RATE", 24000, 8000),
        )
        settings.validate()
        return settings


def get_settings() -> Settings:
    """Read once per process unless the caller explicitly asks for a refresh."""
    global _SETTINGS
    if _SETTINGS is None:
        _SETTINGS = Settings.from_env()
    return _SETTINGS


_SETTINGS: Settings | None = None


def reset_settings() -> None:
    """Test helper: clear the cached settings."""
    global _SETTINGS
    _SETTINGS = None


LANGUAGES = (LANG_TELUGU, LANG_ENGLISH, LANG_MIXED, LANG_ROMANIZED)
