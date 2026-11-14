"""Telugu Voice Lab: local-first UI on top of the existing IndicF5 engine."""

from __future__ import annotations

import html
import json
import os
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import gradio as gr

from tts.audio_utils import AudioValidationError, save_outputs
from tts.config import Settings
from tts.engine_router import LANG_MIXED, LANG_ROMANIZED, LANG_TELUGU, EngineRouter
from tts.health import collect_health
from tts.history import delete_item, discover_history, write_metadata
from tts.library import ReferenceError, discover_references, file_sha256, save_reference
from tts.service import SpeechRequest, TTSError, TTSService
from tts.transliteration import romanized_telugu_to_script

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
REFERENCE_DIR = BASE_DIR / "references"
CSS = (BASE_DIR / "static" / "voice_lab.css").read_text(encoding="utf-8")
MODES = [LANG_TELUGU, LANG_MIXED, LANG_ROMANIZED]
MAX_CHARS = 2000
SETTINGS = Settings.from_env()
ROUTER = EngineRouter(settings=SETTINGS)
SERVICE = TTSService(SETTINGS, router=ROUTER)
esc = html.escape


def _refs():
    return {r.stem: r for r in discover_references(REFERENCE_DIR)}


def _ref_choices():
    return [(f"{r.label}" + ("" if r.valid else " (unusable)"), r.stem) for r in _refs().values()]


def _empty(title: str, body: str) -> str:
    return f'<div class="vl-empty"><b>{esc(title)}</b>{esc(body)}</div>'


def _badge(text: str, level: str) -> str:
    return f'<span class="vl-badge {level}">{esc(text)}</span>'


def header_html() -> str:
    loaded = getattr(ROUTER.indicf5, "_bundle", None) is not None
    model = _badge("Model loaded", "ok") if loaded else _badge("Model loads on first generation", "warn")
    return (
        '<div class="vl-header"><div><h1>Telugu Voice Lab</h1>'
        "<p>Local voice-cloned Telugu speech with IndicF5. Nothing leaves this machine.</p></div>"
        f'<div class="vl-badges">{_badge("LOCAL PROCESSING", "ok")}{model}</div></div>'
    )


# ---------- Studio ----------
def count_text(text: str) -> str:
    text = text or ""
    return f'<span class="vl-note">{len(text)} / {MAX_CHARS} characters · {len(text.split())} words</span>'


def mode_note(mode: str):
    if mode == LANG_ROMANIZED:
        return (
            gr.update(visible=True),
            '<span class="vl-note">Convert Latin-script Telugu to Telugu script, review and edit it below. '
            "The edited preview is exactly what is spoken. Spellings are ambiguous, so always review.</span>",
        )
    if mode == LANG_MIXED:
        return gr.update(visible=False), (
            '<span class="vl-note warn">Telugu + English is experimental: IndicF5 is trained for Telugu, '
            "English words may be mispronounced.</span>"
        )
    return gr.update(visible=False), '<span class="vl-note">Pure Telugu script is the best-supported mode.</span>'


def convert_romanized(text: str):
    if not (text or "").strip():
        raise gr.Error("Enter Romanized Telugu text first.")
    try:
        return romanized_telugu_to_script(text)
    except Exception as exc:
        raise gr.Error(f"Transliteration unavailable offline: {exc}") from exc


def pick_reference(stem: str):
    ref = _refs().get(stem)
    if ref is None:
        return None, "", '<span class="vl-note">No reference selected.</span>'
    if not ref.valid:
        return None, ref.transcript, f'<span class="vl-note warn">{esc(ref.error or "Unusable reference")}</span>'
    warn = "".join(f" ⚠ {esc(w)}" for w in ref.warnings)
    info = (f'<span class="vl-note">{esc(ref.label)} · {ref.duration_seconds:.1f}s · '
            f'{ref.sample_rate} Hz{warn}</span>')
    return str(ref.wav_path), ref.transcript, info


def _result_html(meta: dict) -> str:
    stats = "".join(
        f'<div class="vl-stat">{k}<b>{esc(str(v))}</b></div>'
        for k, v in [("Duration", f"{meta['duration_seconds']:.2f}s"), ("Sample rate", f"{meta['sample_rate']} Hz"),
                     ("Generated in", f"{meta['generation_seconds']:.1f}s"), ("Engine", meta.get("engine", "IndicF5")),
                     ("Cache", "hit" if meta.get("cache_hit") else "miss"), ("Mode", meta["mode"])]
    )
    return (f'<div class="vl-stats">{stats}</div><div class="vl-note">Spoken text:</div>'
            f'<div class="vl-item telugu-text">{esc(meta["text"])}</div>'
            '<div class="vl-note">Find it later in the History tab.</div>')


NO_RESULT = _empty("No audio yet", "Write some text, choose a reference voice, then press Generate.")


def generate(mode, source_text, preview_text, ref_stem, ref_audio, ref_text, progress=gr.Progress()):
    source_text = (source_text or "").strip()
    if not source_text:
        raise gr.Error("Enter some text to speak.")
    if len(source_text) > MAX_CHARS:
        raise gr.Error(f"Text is too long ({len(source_text)} > {MAX_CHARS} characters).")
    if not ref_audio:
        raise gr.Error("Choose or upload a reference recording.")
    if not (ref_text or "").strip():
        raise gr.Error("The reference transcript is required.")
    if mode == LANG_ROMANIZED:
        if not (preview_text or "").strip():
            raise gr.Error("Convert and review the Telugu-script preview before generating.")
        spoken, route_mode = preview_text, LANG_TELUGU
    else:
        spoken, route_mode = source_text, mode
    ref_info = _refs().get(ref_stem)
    ref_path = Path(ref_audio)
    from_library = ref_info is not None and ref_info.wav_path.resolve() == ref_path.resolve()
    try:
        started = time.perf_counter()
        result = SERVICE.synthesize(
            SpeechRequest(
                text=spoken,
                language=route_mode,
                reference_path=str(ref_path),
                reference_transcript=ref_text.strip(),
                reference_name=ref_info.label if from_library else "Custom upload",
            ),
            progress=lambda c, t: progress(c / max(t, 1), desc=f"Chunk {c}/{t}"),
        )
        elapsed = time.perf_counter() - started
        wav, mp3 = save_outputs(result.audio, result.sample_rate, OUTPUT_DIR)
    except (AudioValidationError, TTSError, ValueError, RuntimeError) as exc:
        raise gr.Error(str(exc)) from exc
    meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"), "mode": mode, "text": result.normalized_text,
        "source_text": source_text, "reference_name": ref_info.label if from_library else "Custom upload",
        "reference_sha256": file_sha256(ref_path), "reference_transcript": ref_text.strip(),
        "engine": result.engine, "sample_rate": result.sample_rate,
        "duration_seconds": len(result.audio) / result.sample_rate,
        "generation_seconds": elapsed, "cache_hit": result.cache_hit,
    }
    write_metadata(wav, meta)
    return str(wav), str(wav), str(mp3), _result_html(meta), header_html()


# ---------- References ----------
def references_html() -> str:
    refs = list(_refs().values())
    if not refs:
        return _empty("No references yet", "Add NN_name.wav and NN_name.txt files, or save one below.")
    cards = []
    for r in refs:
        status = _badge("Ready", "ok") if r.valid else _badge("Unusable", "error")
        extra = "".join(f"<div class='meta'>⚠ {esc(w)}</div>" for w in r.warnings)
        if r.error:
            extra += f"<div class='meta'>⚠ {esc(r.error)}</div>"
        dur = f"{r.duration_seconds:.1f}s · {r.sample_rate} Hz" if r.duration_seconds else "unreadable"
        cards.append(f'<div class="vl-item"><h4>{esc(r.label)}</h4>{status}'
                     f'<div class="meta">{dur}</div>{extra}'
                     f'<div class="snip">{esc(r.transcript[:140])}</div></div>')
    return f'<div class="vl-grid">{"".join(cards)}</div>'


def preview_reference(stem: str):
    r = _refs().get(stem)
    return (str(r.wav_path) if r and r.valid else None), (r.transcript if r else "")


def add_reference(name, audio, transcript):
    try:
        info = save_reference(REFERENCE_DIR, name, audio, transcript)
    except (ReferenceError, AudioValidationError) as exc:
        raise gr.Error(str(exc)) from exc
    return (f"Saved as {info.wav_path.name}", references_html(), gr.update(choices=_ref_choices()),
            gr.update(choices=_ref_choices()))


# ---------- History ----------
def _history_choices(query=""):
    return [(f"{i.created} · {(i.text[:40] + '…') if i.text else i.stem}" + (" [legacy]" if i.legacy else ""),
             i.stem) for i in discover_history(OUTPUT_DIR, query)]


def refresh_history(query=""):
    choices = _history_choices(query)
    note = "" if choices else _empty("No generations found", "Generated speech will appear here.")
    return gr.update(choices=choices, value=None), note


def show_history(stem):
    item = next((i for i in discover_history(OUTPUT_DIR) if i.stem == stem), None)
    if item is None:
        return None, None, None, _empty("Select an item", "Choose a generation to see its details.")
    rows = [("Created", item.created), ("Duration", f"{item.duration_seconds:.2f}s" if item.duration_seconds else "?"),
            ("Size", f"{item.size_bytes / 1024:.0f} KB")]
    if item.legacy:
        rows.append(("Metadata", "None (generated before Telugu Voice Lab — text not recorded)"))
    else:
        m = item.metadata
        rows += [("Mode", m.get("mode", "?")), ("Reference", m.get("reference_name", "?")),
                 ("Generation time", f"{m.get('generation_seconds', 0):.1f}s"), ("Text", item.text)]
    if item.malformed_metadata:
        rows.append(("Metadata", "Sidecar file is malformed and was ignored"))
    rows.append(("Deletable", "Yes (created by this app)" if item.owned_by_app else "No (not created by this app)"))
    table = "".join(f"<tr><td>{esc(k)}</td><td>{esc(str(v))}</td></tr>" for k, v in rows)
    return (str(item.wav_path or item.mp3_path),
            str(item.wav_path) if item.wav_path else None,
            str(item.mp3_path) if item.mp3_path else None,
            f'<table class="vl-table">{table}</table>')


def remove_history(stem, confirmed):
    if not stem:
        raise gr.Error("Select an item first.")
    if not confirmed:
        raise gr.Error("Tick the confirmation box to delete.")
    try:
        removed = delete_item(OUTPUT_DIR, stem)
    except (PermissionError, ValueError) as exc:
        raise gr.Error(str(exc)) from exc
    gr.Info(f"Deleted {', '.join(removed)}")
    return refresh_history()[0], False


# ---------- Health ----------
def health_html():
    rows = collect_health(OUTPUT_DIR, ROUTER.indicf5)
    table = "".join(
        f"<tr><td>{esc(k)}</td><td>{esc(v)} {_badge(lvl, lvl) if lvl in ('ok', 'warn', 'error') else ''}</td></tr>"
        for k, v, lvl in rows)
    return f'<table class="vl-table">{table}</table>'


ABOUT = """
**Telugu Voice Lab** wraps the local IndicF5 model. Delivery and timbre follow your reference recording;
there is no emotion control. Quality is not guaranteed — judge by listening.

* English-only synthesis is **not enabled** unless the optional, experimental Chatterbox-Telugu engine is installed.
* Telugu + English is experimental. Romanized Telugu relies on offline transliteration that may be ambiguous.
* Reference clips are limited to 12 seconds of aligned speech because IndicF5 clips longer audio while keeping the transcript.
* Legacy files in `outputs/` and all reference files are never modified; only items created by this app can be deleted.
"""


def build() -> gr.Blocks:
    theme = gr.themes.Base(primary_hue="violet", neutral_hue="slate").set(
        body_background_fill="#0e1015", body_text_color="#eceef5"
    )
    with gr.Blocks(title="Telugu Voice Lab", css=CSS, theme=theme) as demo:
        header = gr.HTML(header_html())
        with gr.Tabs():
            with gr.Tab("Studio"):
                with gr.Row():
                    with gr.Column(scale=3, elem_classes="vl-card"):
                        mode = gr.Radio(MODES, value=LANG_TELUGU, label="Language mode")
                        note = gr.HTML(mode_note(LANG_TELUGU)[1])
                        text = gr.Textbox(label="Text", lines=7, max_lines=20, elem_classes="telugu-text",
                                          placeholder="తెలుగు వచనం ఇక్కడ రాయండి…")
                        counter = gr.HTML(count_text(""))
                        with gr.Group(visible=False) as romanized_box:
                            convert = gr.Button("Convert to Telugu script")
                            preview = gr.Textbox(label="Telugu-script preview (editable — this exact text is spoken)",
                                                 lines=5, elem_classes="telugu-text")
                        ref_choice = gr.Dropdown(_ref_choices(), label="Reference voice", value=None)
                        ref_info = gr.HTML('<span class="vl-note">No reference selected.</span>')
                        with gr.Accordion("Reference audio and transcript", open=False):
                            ref_audio = gr.Audio(label="Reference audio (select above or upload / record)",
                                                 type="filepath", sources=["upload", "microphone"])
                            ref_text = gr.Textbox(label="Reference transcript (must match the audio)", lines=3,
                                                  elem_classes="telugu-text")
                        go = gr.Button("Generate speech", variant="primary", elem_classes="vl-generate")
                    with gr.Column(scale=2, elem_classes="vl-card"):
                        gr.Markdown("### Result")
                        result_info = gr.HTML(NO_RESULT)
                        out_audio = gr.Audio(label="Generated speech", interactive=False)
                        with gr.Row():
                            wav_file = gr.DownloadButton("WAV", visible=True)
                            mp3_file = gr.DownloadButton("MP3", visible=True)
            with gr.Tab("Voice Library"):
                with gr.Column(elem_classes="vl-card"):
                    refs_html = gr.HTML(references_html())
                    ref_pick = gr.Dropdown(_ref_choices(), label="Preview a reference")
                    with gr.Row():
                        ref_prev_audio = gr.Audio(label="Reference audio", interactive=False)
                        ref_prev_text = gr.Textbox(label="Transcript", interactive=False, lines=3)
                with gr.Accordion("Save a new reference", open=False, elem_classes="vl-card"):
                    new_name = gr.Textbox(label="Name")
                    new_audio = gr.Audio(label="Clean recording (one speaker, 3–12 s)", type="filepath",
                                         sources=["upload", "microphone"])
                    new_text = gr.Textbox(label="Exact transcript", lines=3)
                    save_btn = gr.Button("Save reference (never overwrites existing ones)")
                    save_status = gr.Markdown()
            with gr.Tab("Generation History"):
                with gr.Column(elem_classes="vl-card"):
                    with gr.Row():
                        search = gr.Textbox(label="Search", placeholder="text, reference, or file name", scale=4)
                        refresh = gr.Button("Refresh", scale=1)
                    hist_empty = gr.HTML()
                    hist_pick = gr.Dropdown(_history_choices(), label="Generation", value=None)
                    hist_audio = gr.Audio(label="Playback", interactive=False)
                    with gr.Row():
                        hist_wav = gr.DownloadButton("Download WAV")
                        hist_mp3 = gr.DownloadButton("Download MP3")
                    hist_meta = gr.HTML(_empty("Select an item", "Choose a generation to see its details."))
            with gr.Tab("System Health"):
                with gr.Column(elem_classes="vl-card"):
                    gr.Markdown(ABOUT)
                    health = gr.HTML(health_html())
                    gr.Button("Re-check").click(health_html, outputs=health)

        gr.HTML('<div class="vl-footer" style="text-align:center;opacity:.6;padding:18px;font-size:.85rem">Telugu Voice Lab · speech generated by local IndicF5 inference · no cloud, no API keys</div>')

        mode.change(mode_note, mode, [romanized_box, note])
        text.input(count_text, text, counter)
        convert.click(convert_romanized, text, preview)
        ref_choice.change(pick_reference, ref_choice, [ref_audio, ref_text, ref_info])
        go.click(generate, [mode, text, preview, ref_choice, ref_audio, ref_text],
                 [out_audio, wav_file, mp3_file, result_info, header]).then(
            lambda: gr.update(choices=_history_choices()), outputs=hist_pick)
        ref_pick.change(preview_reference, ref_pick, [ref_prev_audio, ref_prev_text])
        save_btn.click(add_reference, [new_name, new_audio, new_text],
                       [save_status, refs_html, ref_choice, ref_pick])
        search.change(refresh_history, search, [hist_pick, hist_empty])
        refresh.click(refresh_history, search, [hist_pick, hist_empty])
        hist_pick.change(show_history, hist_pick, [hist_audio, hist_wav, hist_mp3, hist_meta])
    return demo


if __name__ == "__main__":
    build().queue(default_concurrency_limit=1).launch(
        server_name="127.0.0.1",
        server_port=int(os.environ.get("VOICE_LAB_PORT", "7860")),
        share=False,
        inbrowser=os.environ.get("VOICE_LAB_NO_BROWSER") is None,
        allowed_paths=[str(OUTPUT_DIR), str(REFERENCE_DIR)],
    )
