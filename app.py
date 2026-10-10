from __future__ import annotations

import logging
import os
import time
from pathlib import Path

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr

from tts.audio_utils import (
    AudioValidationError,
    clear_generated_outputs,
    prepared_reference,
    save_outputs,
)
from tts.engine_router import (
    LANG_ENGLISH,
    LANG_MIXED,
    LANG_ROMANIZED,
    LANG_TELUGU,
    EngineRouter,
)
from tts.text_normalizer import normalize_text
from tts.transliteration import romanized_telugu_to_script
from tts.voice_styles import STYLE_DESCRIPTION


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger("local-voice-studio")
ROUTER = EngineRouter()


def _format_audio_info(info) -> str:
    warning_text = " ".join(info.warnings)
    return (
        f"Valid reference: {info.duration_seconds:.1f}s, mono, "
        f"{info.sample_rate:,} Hz."
        + (f" {warning_text}" if warning_text else "")
    )


def validate_reference(audio_path: str | None, transcript: str):
    if not audio_path:
        return "Choose a recording or record one with the microphone first."
    if not transcript or not transcript.strip():
        return "Enter the words actually spoken in the reference recording."
    try:
        with prepared_reference(audio_path) as (_working_copy, info):
            return _format_audio_info(info)
    except (AudioValidationError, OSError) as error:
        return f"Reference audio is not ready: {error}"


def make_text_preview(language: str, text: str):
    normalized = normalize_text(text)
    if not normalized:
        raise gr.Error("Enter text before creating a preview.")
    try:
        if language == LANG_ROMANIZED:
            normalized = romanized_telugu_to_script(normalized)
    except (ValueError, RuntimeError) as error:
        raise gr.Error(str(error)) from error
    return normalized


def _clear_preview():
    return ""


def _count_text(text: str):
    text = text or ""
    words = len(text.split())
    return f"{len(text):,} characters · {words:,} words"


def generate_speech(
    language: str,
    source_text: str,
    preview_text: str,
    reference_audio: str | None,
    reference_transcript: str,
    progress=gr.Progress(track_tqdm=False),
):
    if not reference_audio:
        raise gr.Error("Upload or record your custom reference voice first.")
    if not reference_transcript or not reference_transcript.strip():
        raise gr.Error("Enter the transcript that matches the reference recording.")
    if language == LANG_ROMANIZED:
        target_text = normalize_text(preview_text)
        if not target_text:
            raise gr.Error("Create and review the Telugu-script preview before generation.")
    else:
        target_text = normalize_text(source_text)
    if not target_text:
        raise gr.Error("Enter the text to synthesize.")

    started_at = time.perf_counter()

    def update_progress(current: int, total: int):
        progress(current / total, desc=f"Synthesizing segment {current} of {total}")

    try:
        progress(0, desc="Loading the local speech model")
        engine_language = LANG_TELUGU if language == LANG_ROMANIZED else language
        audio, sample_rate = ROUTER.generate(
            engine_language,
            reference_audio,
            reference_transcript.strip(),
            target_text,
            progress=update_progress,
        )
        wav_path, mp3_path = save_outputs(audio, sample_rate, OUTPUT_DIR)
    except (AudioValidationError, ValueError, RuntimeError, OSError) as error:
        LOGGER.exception("Local speech generation failed")
        raise gr.Error(str(error)) from error

    duration = len(audio) / sample_rate
    elapsed = time.perf_counter() - started_at
    status = (
        f"Generated {duration:.1f}s of audio in {elapsed:.1f}s at {sample_rate:,} Hz. "
        "WAV is PCM 24-bit; MP3 is 192 kbps (lossy). "
        "Mixed-language pronunciation is experimental and has not been quality-certified."
    )
    return str(wav_path), str(mp3_path), str(wav_path), status


def delete_generated_files():
    try:
        count = clear_generated_outputs(OUTPUT_DIR)
    except OSError as error:
        LOGGER.exception("Could not clear generated audio")
        return f"Could not delete generated output files: {error}"
    return f"Deleted {count} generated audio file(s). Reference recordings were not touched."


DESCRIPTION = """
Use **your own voice recording** as the only speaker reference. Audio and text are
processed on this Mac; no preset speakers, cloud TTS, or remote fallback are used.
Only clone a voice you own or have permission to use.

**Validated engine:** AI4Bharat IndicF5 for Telugu with custom reference audio.
English-only custom-voice synthesis is not enabled because a suitable local engine
has not been validated. Telugu-English code-switching can be attempted, but English
pronunciation and cross-language speaker similarity are experimental. Romanized
Telugu uses an offline IndicXlit preview; informal spellings can be ambiguous.
"""


with gr.Blocks(title="Local Multilingual Voice Studio") as demo:
    gr.Markdown("# Local Voice Studio")
    gr.Markdown(DESCRIPTION)

    with gr.Group():
        gr.Markdown("## 1. Your custom reference voice")
        with gr.Row():
            reference_audio_input = gr.Audio(
                sources=["upload", "microphone"],
                type="filepath",
                label="Upload or record your reference (WAV, MP3, M4A, FLAC, OGG)",
                format="wav",
            )
            reference_transcript = gr.Textbox(
                label="Exact transcript of the recording",
                placeholder="Type the words that are actually spoken in this recording.",
                lines=5,
            )
        gr.Markdown(
            "**Recording tips:** Use a quiet room, speak clearly and naturally, avoid music, "
            "echo and clipping, and read the transcript exactly as spoken. A clear 10–15 second "
            "sample is recommended. The model may trim longer references."
        )
        gr.Markdown(
            "**Suggested Telugu script (edit the transcript to match what you said):**\n\n"
            "నమస్కారం! నా పేరు కౌశిక్. నేను తెలుగు మరియు ఇంగ్లీష్‌లో మాట్లాడగలను. "
            "నా రోజువారీ పనిలో కొత్త విషయాలు నేర్చుకోవడం, సమస్యలను పరిష్కరించడం నాకు ఇష్టం. "
            "నేను Linux, Kubernetes, OpenShift మరియు Cloud technologies గురించి మాట్లాడుతాను."
        )
        validate_reference_button = gr.Button("Validate reference and transcript")
        reference_status = gr.Markdown("Upload or record audio, then enter its exact transcript.")

    with gr.Group():
        gr.Markdown("## 2. Text and language")
        language = gr.Dropdown(
            choices=[LANG_TELUGU, LANG_ENGLISH, LANG_MIXED, LANG_ROMANIZED],
            value=LANG_TELUGU,
            label="Input language",
        )
        source_text = gr.Textbox(
            label="Text to speak",
            placeholder="తెలుగు, English, Telugu-English mixed text, or Romanized Telugu",
            lines=8,
        )
        text_count = gr.Markdown("0 characters · 0 words")
        preview_button = gr.Button("Prepare / refresh normalized-text preview")
        normalized_preview = gr.Textbox(
            label="Normalized text preview (editable for Romanized Telugu)",
            lines=6,
            interactive=True,
        )
        gr.Markdown(
            "Romanized Telugu conversion uses a local IndicXlit model. Review and correct "
            "the script preview before synthesis; pronunciation-based spelling is ambiguous."
        )

    with gr.Group():
        gr.Markdown("## 3. Delivery")
        gr.Markdown(STYLE_DESCRIPTION)
        gr.Markdown(
            "Long text is split at sentence/word boundaries and synthesized in order. "
            "Natural pauses are inserted between segments. Generation is sequential to "
            "avoid exhausting unified memory."
        )
        generate_button = gr.Button("Generate speech", variant="primary", size="lg")

    with gr.Group():
        gr.Markdown("## 4. Listen and download")
        with gr.Row():
            audio_output = gr.Audio(label="Generated audio", type="filepath")
            with gr.Column():
                mp3_output = gr.File(label="Download MP3 (192 kbps)")
                wav_output = gr.File(label="Download WAV (PCM 24-bit)")
        generation_status = gr.Markdown("Ready.")
        clear_button = gr.Button("Delete generated files", variant="secondary")
        clear_status = gr.Markdown()

    validate_reference_button.click(
        fn=validate_reference,
        inputs=[reference_audio_input, reference_transcript],
        outputs=reference_status,
    )
    source_text.input(fn=_count_text, inputs=source_text, outputs=text_count)
    preview_button.click(
        fn=make_text_preview,
        inputs=[language, source_text],
        outputs=normalized_preview,
    )
    source_text.change(fn=_clear_preview, outputs=normalized_preview)
    language.change(fn=_clear_preview, outputs=normalized_preview)
    generate_button.click(
        fn=generate_speech,
        inputs=[language, source_text, normalized_preview, reference_audio_input, reference_transcript],
        outputs=[audio_output, mp3_output, wav_output, generation_status],
    )
    clear_button.click(fn=delete_generated_files, outputs=clear_status)


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(
        server_name="127.0.0.1",
        share=False,
        inbrowser=True,
        theme=gr.themes.Soft(),
    )
