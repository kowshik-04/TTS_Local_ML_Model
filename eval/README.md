# Evaluation and benchmark harness

This directory measures the deployed HTTP API. It is intentionally separate
from the model code so the same harness works on a Mac demo box, an NVIDIA GPU
worker, or a remote deployment.

## What it measures

| Metric | Meaning |
|---|---|
| `wall_ms` | Client-observed request duration. |
| `ttfb_ms` | Time until the first audio byte for `/v1/tts/stream`; equal to wall time for sync requests. |
| `generation_ms` | Server-reported generation time from the `X-Generation-Ms` header when available. |
| `rtf` | Generation time divided by audio duration. Lower is faster than real time. |
| `audio_seconds_per_wall_second` | Speedup relative to the client-observed wall clock. |
| Cache hits | Exact-output cache reuse (`X-Cache: hit`). |
| Failures | HTTP/transport errors with the API's stable error code. |

It does **not** measure pronunciation, naturalness, emotion or noise robustness.
Those require human listening. The harness writes `human_review_template.json`
for that purpose and saves the generated audio by default. `eval/quality.py`
adds automated speaker-similarity and ASR error-rate measurements; see
[Quality, speaker similarity and WER](#quality-speaker-similarity-and-wer).

## Prerequisites

1. Backend running with a voice profile enrolled:

   ```bash
   uvicorn api.main:app --host 127.0.0.1 --port 8000
   python -m api.admin create-key --name benchmark --quota 1000000 --rate 10000
   ```

2. A stored voice ID from `POST /v1/voices`. The benchmark never invents a
   speaker or falls back to a preset voice.

## Run

```bash
export TTS_API_KEY="vls_..."
python eval/benchmark.py --voice-id voice_ab12cd34
```

Useful variants:

```bash
# Only the validated Telugu path, streamed, with four in-flight requests.
python eval/benchmark.py --voice-id voice_ab12cd34 --group telugu \
  --transport stream --concurrency 4 --runs 2

# Compare inference profiles by running each server with a different env.
TTS_PROFILE=realtime python -m uvicorn api.main:app --port 8000
python eval/benchmark.py --voice-id voice_ab12cd34 --profile realtime

# Inspect the corpus without spending quota.
python eval/benchmark.py --dry-run
```

Output is written to `eval/results/<timestamp>-<id>/`:

- `manifest.json` — exact settings, cases and environment. The API key is never written.
- `results.json` — one record per trial.
- `summary.json` / `summary.md` — aggregates by transport, language and category.
- `audio/` — generated WAV files for listening.
- `human_review_template.json` — blank 1–5 scores and notes fields.

The process exits non-zero if any trial failed. The directory is still written
so partial failures can be diagnosed.

## Reading the numbers honestly

- Run at least 3 repetitions after a warmup. First-request model loading is
  excluded from the recorded trials by default (`--no-warmup` disables this).
- The sync endpoint includes complete generation plus encoding. The stream
  endpoint's TTFB is the meaningful conversational metric.
- `--concurrency` sends parallel requests; if the server's
  `TTS_API_CONCURRENCY=1`, requests queue. Queue time is part of `wall_ms`, which
  is the point: it shows the deployed behaviour, not an idealised single-request
  microbenchmark.
- Character quotas apply to benchmark requests, including cache hits.
- Do not claim "production ready", "natural", or "speaker-identical" from these
  numbers. Publish listening scores, the exact corpus, the reference voice's
  consent basis, and the hardware/software manifest with any quality claim.

## Quality, speaker similarity and WER

For consented human audio, use the reproducible
[human-reference protocol](HUMAN-REFERENCE-PROTOCOL.md). It defines consent and
retention, recording requirements, the fixed 30 Telugu / 8 Tenglish / 6 English
corpus, speaker and ASR models, sample counts, listening scales, and reporting
rules. The protocol is preparation only; it contains no human results.

`eval/quality.py` evaluates generated files against one reference recording and
a JSON manifest. It is separate from the serving path and uses optional public
models:

```bash
pip install -r requirements-quality.txt
python eval/quality.py --manifest eval/quality-manifest.json --dry-run
python eval/quality.py --manifest eval/quality-manifest.json \
  --asr-model large-v3-turbo --device cpu
```

`--device cpu` is the validated portable speaker-model path. The command
downloads the public SpeechBrain and faster-whisper weights into
`--model-cache` (default `.cache/quality-models`) on first use. Use
`--no-speaker` or `--no-asr` to isolate one half of the report.

It reports:

- Speaker-embedding cosine similarity using SpeechBrain
  `spkrec-ecapa-voxceleb`.
- ASR word and character error rates using faster-whisper, after documented
  Unicode/punctuation normalization.
- Duration, sample rate, channel count, peak, RMS, clipping and near-silence
  fractions for every generated file.
- Per-sample model identifiers, expected text, ASR hypothesis, failures, file
  hashes, package versions and a Markdown/JSON report.
- Dynamic warnings for synthetic references and for samples shorter than 2.0 s.

The manifest requires a single reference and a list of generated samples. Human
references must set `consent_confirmed: true`; non-human test signals must set
`synthetic: true`, which marks every similarity number as pipeline-only evidence.
`reference.notes` is copied into the Markdown report so a synthetic voice's
origin is visible next to the results. See the module docstring in
`eval/quality.py` for the exact schema.

These are proxies, not a quality certificate. Embedding similarity is
model-dependent and does not prove perceptual identity. Telugu and Tenglish ASR
can be unreliable, especially for very short clips, code-switching, names and
numbers. WER/CER can exceed 1.0 when the recognizer inserts or invents text;
that is an ASR failure signature, not a pronunciation score. Short clips
(< 2.0 s) are flagged as outliers because endpointing, ASR hallucination and
speaker-embedding instability can dominate them. A small ASR model may
transliterate Telugu into another script; the report records the exact model ID
and raw hypothesis so that failure is visible. No universal pass threshold is
defined. Use the report alongside the benchmark's `human_review_template.json`,
multiple consented speakers and human listening.

## Corpus

The corpus contains 30 Telugu prompts, 8 Telugu-English prompts and 6 optional
English prompts. It covers conversation, formal speech, aspiration, conjuncts,
similar-sounding words, punctuation, numbers, names and long sentences. The
English cases are only meaningful when the experimental Chatterbox-Telugu
engine is enabled; otherwise the API correctly returns `engine_unavailable`.

The prompts are fixed so two builds can be compared on identical text. They are
not a substitute for a larger, consented, domain-specific evaluation set.
