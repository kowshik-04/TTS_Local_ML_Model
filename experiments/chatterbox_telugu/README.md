# Chatterbox Telugu POC

This isolated experiment evaluates `shankarpandala/chatterbox-telugu` without
changing the production IndicF5 engine or UI.

It performs a one-time local Chatterbox model download only when invoked with
`--download`. Normal runs use `ChatterboxMultilingualTTS.from_local()` and the
local model directory. It produces English, Telugu-script, mixed-language, and
raw/converted Romanized Telugu samples using project reference recordings.
Chatterbox accepts reference audio but does not accept a matching reference
transcript, so the transcripts remain project metadata rather than model input.

Run with the experiment virtual environment:

```bash
.venv/bin/python run_poc.py --download
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python run_poc.py
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python run_poc.py --all
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python run_poc.py \
  --tenglish "Nenu chalaa baagunnaanu. Nenu Red Hat lo Linux vadutanu."
```

The default command runs Telugu only. `--all` runs the existing Telugu,
English, and Telugu-English mixed cases plus a same-reference comparison of
raw Tenglish and its Telugu-script preview. A custom `--tenglish` run prints
the original and converted text, then offers an editable CLI prompt when run
interactively. To supply a correction non-interactively, use
`--tenglish-preview "నేను చాలా బాగున్నాను."`; the exact corrected text is
recorded and passed to `model.generate()`. The results JSON records source,
converted (when applicable), and synthesis text. Each case records a failure
and continues, so a failed case does not remove earlier output; the command
exits nonzero if any case fails. Output and result filenames include a unique
run ID and do not overwrite prior results.

Tenglish conversion reuses the repository's already-installed AI4Bharat
IndicXlit RNN and cached Telugu model through the sibling project's Python
environment. It does not add dependencies, download transliteration files, or
change production files. By default it looks for `../../.venv/bin/python`
relative to this experiment; set `TENG_XLIT_PYTHON` if the existing offline
IndicXlit environment is elsewhere. The converter fails instead of attempting
an online model download when its Telugu files are not cached. It preserves
Unicode Telugu, punctuation, numbers, and recognized English terms; its
transliterations are suggestions, not uniquely correct spellings. Review and
edit the preview before synthesis when accuracy matters.
Run the compatibility checks before generation:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

The
experiment includes a process-scoped compatibility shim for this checkpoint's
documented 2521-token Telugu vocabulary; it never edits Chatterbox or project
production files.

The isolated environment pins `setuptools<81` because Perth 1.0.1 imports
`pkg_resources`; with newer setuptools, Perth suppresses that import failure and
Chatterbox later crashes while constructing its watermark component.

## Repeatable Telugu evaluation

`evaluation_corpus.py` defines 30 pure Telugu prompts and 8 separate
Telugu-English prompts, categorized for conversation/formality, vowels and
aspiration, conjuncts, similar-sounding words, punctuation, numbers, names,
and long sentences. `evaluate_suite.py` synthesizes the same target text with
the locally cached Chatterbox Telugu model and the existing IndicF5 engine.
IndicF5 receives its matching transcript; Chatterbox receives the same
reference audio but does not accept that transcript.

Run both engines offline from this directory:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python evaluate_suite.py
```

Use `--limit 1` for a quick model-loading/synthesis check, `--engines
chatterbox` or `--engines indicf5` for a single engine, and `--device mps` or
`--device cpu` to override Chatterbox's automatic device selection. IndicF5
uses the project's existing `.venv`; set `INDICF5_PYTHON` or
`CHATTERBOX_PYTHON` only if an interpreter is installed elsewhere.

Each invocation creates a new `outputs/evaluations/<run-id>/` directory with
the exact corpus manifest, per-engine case records and WAV files, reference
signal measurements, an aggregate `results.json`, worker logs, and an
editable `human_review_template.json`. Existing POC and evaluation outputs
are not overwritten. The records distinguish source text, the exact string
passed to each engine, and the text/normalization visible immediately before
tokenization or synthesis. WAV validity, duration, and signal measurements
are technical checks only; fill in pronunciation errors, missing words,
prosody, and speaker similarity after listening.
