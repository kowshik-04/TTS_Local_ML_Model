# Local Voice Studio

A production-oriented, local-first speech stack for **Telugu and Telugu-English
(Tenglish)**:

- a custom-reference TTS HTTP API with API keys, quotas, rate limits, exact-output
  caching, voice profiles and PCM streaming;
- a real-time voice agent that combines server-side VAD, local or hosted STT, any
  OpenAI-compatible LLM, and the TTS API;
- a bundled browser agent demo at `/` with microphone capture, 24 kHz playback,
  interruption/barge-in and text test turns;
- the original Gradio studios, kept working on the same inference core.

The validated engine remains **AI4Bharat IndicF5** for Telugu and Telugu-English
(Tenglish) text. Optional Chatterbox-Telugu support exists for explicit
English/Tenglish experiments, but it is disabled by default and is not
quality-certified. Requests fail explicitly when the selected engine is
unavailable; there is no silent fallback.

No preset speakers, cloud TTS fallback or arbitrary server-side file access are
used. A voice must be explicitly enrolled with consent.

---

## One-command setup and launch

On macOS (Apple Silicon) or Linux, the root [`run.sh`](run.sh) script performs
the installation, caches the model weights and starts the Telugu Voice Lab:

```bash
export HF_TOKEN=hf_...          # one-time gated-model access; see below
./run.sh
```

The script:

1. creates or reuses `.venv`;
2. installs the pinned PyTorch and Python dependencies, selecting CPU PyTorch
   on machines without an NVIDIA GPU and CUDA PyTorch when `nvidia-smi` works;
3. downloads and caches IndicF5 + Vocos (about 1.5 GB on the first run);
4. starts the Voice Lab at <http://127.0.0.1:7860>.

Useful variants:

```bash
./run.sh --setup-only          # install and cache, but do not start the UI
./run.sh --no-browser          # headless/SSH; print the local URL instead
./run.sh --skip-install        # reuse an existing .venv
./run.sh --skip-models         # reuse already-cached weights
./run.sh --port 7861           # use another port
./run.sh --device cpu          # force CPU; default is auto (MPS/CUDA/CPU)
```

### Required one-time model access

The `ai4bharat/IndicF5` repository is gated. The script does not bypass that
gate:

1. Sign in at <https://huggingface.co/ai4bharat/IndicF5>.
2. Accept the model terms.
3. Export `HF_TOKEN`, or log in once with `hf auth login` /
   `huggingface-cli login`.

After the weights are cached, the app runs offline and the token is not needed
for inference. `run.sh --skip-models` starts from an existing cache.

The script expects Python 3.11 or 3.12, Git and FFmpeg. It can install FFmpeg
with Homebrew on macOS, and on Debian/Ubuntu it installs it automatically when
run as root or with passwordless `sudo`; otherwise it prints the exact command.
Windows is not natively supported; run the script inside WSL2.

This gets the application running without manual installation steps, but voice
cloning still requires a 3–12 second reference recording and its **exact**
transcript. The repository does not ship a preset or synthetic human voice.
Optional Romanized-Telugu input needs the separate `make install-transliteration`
step; core Telugu/Tenglish synthesis does not.

---

## What is actually validated

This repository is honest about the boundary between engineering and quality:

| Area | Status |
|---|---|
| IndicF5 Telugu with a custom reference | The project’s validated baseline. Re-run the benchmark and listening review on your hardware. |
| Telugu-English mixed text | **Experimental.** IndicF5 is the `engine=auto` default; pronunciation and cross-language speaker similarity are not certified. |
| English-only with `engine=auto` | Uses Chatterbox only when `TTS_ENABLE_CHATTERBOX=1`. Otherwise **rejected** with `engine_unavailable`; IndicF5 is not treated as an English engine. |
| Optional Chatterbox-Telugu | **Experimental community fine-tune**, CC-BY-4.0, disabled by default. Explicit `engine=chatterbox`; not certified for English, Tenglish or speaker similarity. |
| Romanized Telugu | Offline IndicXlit preview, always shown for review; spelling is ambiguous. |
| Speed/latency claims | Only what `eval/benchmark.py` measures on your machine. No claim is made from this checkout because model weights are intentionally not bundled. |
| Speaker similarity / naturalness | Requires human listening. The benchmark writes a review template; it does not invent scores. |

The original IndicF5 baseline was previously exercised on Apple Silicon with MPS.
That is not a guarantee for a different model revision, reference recording,
operating system or GPU.

---

## Architecture

```
Browser demo (agent/static/index.html)
  │  WebSocket: JSON control + PCM16 binary
  ▼
Voice agent (agent/server.py, default :8001)
  ├─ server-side energy VAD + endpointing (agent/vad.py)
  ├─ STT: local faster-whisper or OpenAI-compatible (agent/stt.py)
  ├─ LLM: any OpenAI-compatible streaming chat endpoint (agent/llm.py)
  └─ TTS client: /v1/tts/stream PCM16 @ 24 kHz (agent/tts_client.py)
         │
         ▼
TTS API (api/main.py, default :8000)
  ├─ auth / quotas / rate limits (SQLite)
  ├─ TTSService: text frontend, voice resolution, exact cache, post-processing
  ├─ EngineRouter: IndicF5 (default), optional Chatterbox (experimental)
  └─ VoiceStore: consent, transcripts, 12 s aligned-reference cap
         │
         ▼
Local IndicF5 + Vocos weights (HF cache, never downloaded by the API on demand
unless the model loader needs them)
```

The inference engine is serialized per process. The API adds a bounded queue
(`TTS_API_CONCURRENCY`, default 1) so a GPU is not oversubscribed. Run multiple
processes/workers only after reading the concurrency caveats below.

---

## Quick start — Apple Silicon

### 1. Prerequisites

- Python 3.11
- FFmpeg (`brew install ffmpeg`)
- `git` (the pinned AI4Bharat F5-TTS/IndicF5 source package is installed from Git)
- Xcode command-line tools for the PyTorch/MPS build
- Hugging Face account with access to the gated
  [ai4bharat/IndicF5](https://huggingface.co/ai4bharat/IndicF5) model and
  `HF_TOKEN` exported

### 2. Install

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch==2.11.0 torchaudio==2.11.0
pip install -r requirements.txt
```

For repeatable setup, `make install-mac` does the same. Use the existing `.venv`
if one is already present instead of recreating it.

`requirements-core.txt` installs the `f5_tts` package from the pinned AI4Bharat
IndicF5 commit, not from PyPI. PyPI has no `f5-tts==0.1.0`, and newer PyPI
releases use a different checkpoint loader. The source install requires `git`
and network access at install time; the Docker image includes `git`.

Romanized-Telugu support is optional. AI4Bharat's PyPI metadata declares
`fairseq`, but the RNN transliterator used here does not import it, and
`fairseq 0.12.1` cannot build on Python 3.11/current pip. Install it with the
targeted no-deps step:

```bash
python -m pip install --no-deps -r requirements-transliteration.txt
python -m pip install pydload pandas
# or: make install-transliteration
```

Without that step, TTS and the agent work normally; `POST /v1/transliterate`
returns an explicit `transliteration_unavailable` error. The no-deps install
intentionally leaves `fairseq` out, so `pip check` reports it (and other unused
Flask/gevent metadata) from AI4Bharat's package. The RNN transliteration path
was verified without `fairseq`; TTS and the agent do not import it.

### 3. Cache model assets once

```bash
export HF_TOKEN=hf_...
python -m tts.setup_models
```

This caches IndicF5, Vocos and, when the optional transliteration package is
installed, the local IndicXlit Telugu transliterator. It fails explicitly
instead of silently falling back to a cloud TTS service. If transliteration was
not installed, add `--skip-transliterator`.

Optional experimental English/Tenglish engine:

```bash
python -m pip install -r experiments/chatterbox_telugu/requirements.txt
python -m tts.setup_models --chatterbox
export TTS_ENABLE_CHATTERBOX=1
```

The base Docker image does not include the optional Chatterbox runtime. Build a
derived image that installs `experiments/chatterbox_telugu/requirements.txt`
before enabling `TTS_ENABLE_CHATTERBOX` in a container.

### 4. Configure

```bash
cp .env.example .env
set -a; source .env; set +a
```

For the simplest local setup, set one static key in `.env`:

```dotenv
TTS_API_KEY=replace-with-a-long-random-secret
AGENT_TTS_API_KEY=replace-with-the-same-secret
AGENT_VOICE_ID=voice_...   # after enrolling a voice below
```

The static key is an admin-scoped local key and is unlimited. For anything
shared, create scoped SQLite keys instead:

```bash
python -m api.admin create-key --name "Acme production" --quota 500000 --rate 120
```

The secret is printed once. Store it in your secret manager; only its SHA-256
hash is written to SQLite.

### 5. Start the API

```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Health: <http://127.0.0.1:8000/healthz> · OpenAPI: <http://127.0.0.1:8000/docs>

### 6. Enroll a voice (with consent)

Record **8–12 seconds** of clean speech and copy its exact transcript. Longer
clips are rejected: IndicF5 can clip long references while retaining the full
transcript, which silently misaligns the voice.

```bash
curl -sS http://127.0.0.1:8000/v1/voices \
  -H "Authorization: Bearer $TTS_API_KEY" \
  -F name="My Telugu voice" \
  -F transcript="నమస్కారం. ఇది నా స్వంత వాయిస్ నమూనా." \
  -F consent=true \
  -F language=Telugu \
  -F engine_hint=indicf5 \
  -F audio=@reference.wav | tee voice.json
```

Copy `voice_id` from the response. Only clone voices you own or have documented
permission to use. The API accepts uploads, never arbitrary server paths.

### 7. Synthesize

```bash
curl -sS http://127.0.0.1:8000/v1/tts \
  -H "Authorization: Bearer $TTS_API_KEY" \
  -H "Content-Type: application/json" \
  -d "{\"text\":\"నమస్కారం. ఈ రోజు ఎలా ఉన్నారు?\",\"language\":\"Telugu\",\"voice_id\":\"$VOICE_ID\",\"format\":\"wav\"}" \
  --output speech.wav
```

### 8. Start the agent and open the demo

The agent needs an LLM. With [Ollama](https://ollama.com):

```bash
ollama pull qwen2.5:7b
export AGENT_LLM_BASE_URL=http://127.0.0.1:11434/v1
export AGENT_LLM_MODEL=qwen2.5:7b
export AGENT_TTS_URL=http://127.0.0.1:8000
export AGENT_TTS_API_KEY=$TTS_API_KEY
export AGENT_VOICE_ID=$VOICE_ID
export AGENT_API_KEY=optional-local-gatekeeper

uvicorn agent.server:create_app_from_env --factory --host 127.0.0.1 --port 8001
```

Open <http://127.0.0.1:8001/>. The page has no CDN/external JS dependencies.
Grant microphone permission, press **Connect**, then **Talk**. The demo downsamples
microphone audio to 16 kHz PCM16, streams it to the agent, schedules 24 kHz PCM16
replies, and interrupts the active answer when you start speaking again.

The first local STT request downloads the selected faster-whisper model into the
HF cache. To use a hosted STT instead:

```bash
export AGENT_STT_PROVIDER=openai
export AGENT_STT_BASE_URL=https://api.openai.com/v1
export AGENT_STT_API_KEY=...
export AGENT_STT_MODEL=whisper-1
```

The LLM can be Ollama, vLLM, LM Studio, or any hosted OpenAI-compatible
`/chat/completions` endpoint.

### 9. Optional Gradio apps

```bash
python app.py          # simpler studio, uses the production service
python voice_lab_app.py  # Voice Lab with library, history, health
```

They use the same text frontend, reference preparation, cache and post-processing
as the API.

---

## Quick start — NVIDIA Linux

```bash
sudo apt-get install ffmpeg libsndfile1
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.11.0 torchaudio==2.11.0
pip install -r requirements.txt
export HF_TOKEN=hf_...
python -m tts.setup_models
export TTS_DEVICE=cuda
```

Confirm the driver and PyTorch see the GPU (`nvidia-smi`, then
`python -c "import torch; print(torch.cuda.is_available())"`). If CUDA is not
available, use `TTS_DEVICE=cpu` and expect substantially slower generation.

Native services are provided in `deploy/systemd/`; adjust user/path/venv first.

---

## Docker

```bash
cp .env.example .env
# Set TTS_API_KEY and AGENT_TTS_API_KEY to the same strong secret.
# Set AGENT_VOICE_ID after enrolling a voice, or leave blank and enter it in the demo.

docker compose run --rm api python -m tts.setup_models
docker compose up -d --build
```

- API: <http://127.0.0.1:8000>
- Agent/demo: <http://127.0.0.1:8001>

For NVIDIA acceleration on Linux:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```

The image installs CUDA PyTorch by default. Build a CPU image with
`--build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu`. The image
installs `git` for the pinned IndicF5/F5-TTS source and validates the F5-TTS
import at build time; see the validation report for the ARM64 build evidence.

Model weights are not baked into the image. The `hf_models` volume is shared by
the API and agent; `data/`, `voices/`, `cache/` and `outputs/` are bind mounts.
Back up `data/` (accounts) and `voices/` (consented profiles).

The optional Chatterbox-Telugu runtime is not part of the base image. If you
want it in containers, derive an image that installs
`experiments/chatterbox_telugu/requirements.txt`; the default image fails
explicitly with `engine_initialization_failed` if you enable it without those
dependencies.

Docker Desktop on Apple Silicon does not expose MPS; use the native macOS path
for Apple GPU acceleration.

---

## HTTP API

All `/v1` routes require a key with the relevant scope unless
`TTS_API_REQUIRE_AUTH=0` (not recommended). Send either:

```http
Authorization: Bearer vls_...
```

or `X-API-Key: vls_...`. Errors use stable codes:

```json
{"error":{"code":"voice_not_found","message":"...","request_id":"req_..."}}
```

### Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/healthz` | Public liveness. |
| `GET` | `/readyz` | Model/warmup readiness; 503 when not ready. |
| `POST` | `/v1/tts` | Synchronous synthesis: `wav`, `mp3`, `opus`, `pcm`. |
| `POST` | `/v1/tts/stream` | Raw PCM16 mono 24 kHz streaming response. |
| `WS` | `/v1/tts/ws` | Text-to-PCM WebSocket with cancel. |
| `POST` | `/v1/preview` | Normalized text, chunks and warnings; no synthesis. |
| `POST` | `/v1/transliterate` | Romanized Telugu → editable Telugu script. |
| `GET/POST` | `/v1/voices` | List / enroll consented voice profiles. |
| `GET/DELETE` | `/v1/voices/{voice_id}` | Inspect or delete an owned profile. |
| `GET` | `/v1/me` | Key scopes, quota and usage. |
| `GET` | `/v1/capabilities` | Language/engine capability matrix and limits. |
| `GET` | `/v1/health` | Authenticated service/engine/cache health. |
| `GET` | `/metrics` | Prometheus text format; admin scope only. |

### Request options

```jsonc
{
  "text": "నమస్కారం. 100 Mbps మరియు 5G.",
  "language": "Telugu",          // Telugu | English | Telugu + English | Romanized Telugu
  "voice_id": "voice_ab12cd34",
  "engine": "auto",              // omitted = TTS_ENGINE; auto | indicf5 | chatterbox
  "profile": "quality",          // quality | balanced | realtime | custom
  "speed": 1.0,                  // 0.5–2.0
  "use_cache": true,
  "format": "wav"                // sync endpoint only: wav | mp3 | opus | pcm
}
```

Successful audio responses include `X-Request-Id`, `X-Audio-Format`,
`X-Sample-Rate`, `X-Audio-Duration-Ms`, `X-Generation-Ms`, `X-Rtf`, `X-Engine`,
`X-Voice-Id`, `X-Cache`, warning headers, and `X-Experimental: true` where
relevant. The server never returns silent audio as success.

### Engine routing

- If `engine` is omitted, the deployment default `TTS_ENGINE` is used. It
  defaults to `auto`.
- `engine=auto` sends Telugu and Telugu-English to IndicF5. It sends
  English-only text to Chatterbox only when `TTS_ENABLE_CHATTERBOX=1`;
  otherwise English is rejected.
- `engine=chatterbox` explicitly selects the optional fine-tune for any
  language, including Telugu/Tenglish. It must be enabled and downloaded.
- Missing weights, initialization errors and generation errors return stable
  error codes (`engine_initialization_failed`, `generation_failed`). The router
  never switches engines or presets after a failure. Streaming responses prime
  the first audio chunk before sending headers, so a generation failure produces
  a JSON error rather than an empty 200 response.

### Streaming example

```bash
curl -N http://127.0.0.1:8000/v1/tts/stream \
  -H "Authorization: Bearer $TTS_API_KEY" \
  -H "Content-Type: application/json" \
  -d "{\"text\":\"నమస్కారం. ఇది స్ట్రీమింగ్ పరీక్ష.\",\"language\":\"Telugu\",\"voice_id\":\"$VOICE_ID\"}" \
  --output speech.pcm
ffmpeg -f s16le -ar 24000 -ac 1 -i speech.pcm speech.wav
```

### TTS WebSocket

Connect to `ws://127.0.0.1:8000/v1/tts/ws`, then send:

```json
{"type":"start","api_key":"vls_...","text":"నమస్కారం","voice_id":"voice_...","language":"Telugu"}
```

The server replies with a `ready` JSON event, binary PCM16 frames at 24 kHz, and a
`done` JSON event. Send `{"type":"cancel"}` to stop generation. The API key may
also be supplied in the `Authorization`/`X-API-Key` header instead of the first
message.

### Quotas and rate limits

- Characters are reserved before synthesis and refunded if no audio is sent.
- Monthly quotas and per-key rates are configured per SQLite key.
- Cache hits still consume request rate and character quota.
- `TTS_API_CONCURRENCY` bounds simultaneous GPU work; excess streaming requests
  queue up to `TTS_API_QUEUE_TIMEOUT` seconds.

---

## Voice agent protocol

Endpoint: `ws(s)://<agent-host>/v1/agent/ws`. Text events are JSON; audio is binary
PCM16 mono.

Client → server:

```jsonc
{"type":"start","voice_id":"voice_...","language":"Telugu","api_key":"...","sample_rate":16000}
// binary PCM16 mono frames at exactly AGENT_INPUT_SAMPLE_RATE
{"type":"end_utterance"}   // force endpointing; otherwise server VAD ends the turn
{"type":"interrupt"}       // barge in on the active answer
{"type":"say","text":"..."} // text-only test turn, no microphone
{"type":"reset"}           // clear conversation history
{"type":"ping"}
{"type":"stop"}
```

Server → client:

- `ready` — negotiated rates, voice, language
- `speech_start` (`interrupted`: bool), `vad_end` — VAD/endpointing
- `transcript` — STT result
- `assistant_delta`, `phrase` — streamed LLM text and speech phrases
- binary PCM16 at 24 kHz
- `audio_end`, `turn_end` — per-phrase and per-turn completion
- `interrupted`, `listening`, `reset`, `pong`, `error`

The agent streams each phrase to TTS as soon as a sentence boundary is available.
Starting to speak cancels the active response task and sends `interrupted`; an
answer is only committed to conversation history after it is fully produced.

### Agent configuration

| Variable | Meaning |
|---|---|
| `AGENT_TTS_URL` | TTS API base URL, e.g. `http://127.0.0.1:8000`. |
| `AGENT_TTS_API_KEY` | Key with `tts` scope. |
| `AGENT_TTS_ENGINE` | `auto` (default), `indicf5`, or `chatterbox`. `auto` keeps IndicF5 for Telugu/Tenglish; it never falls back after a failure. |
| `AGENT_VOICE_ID` | Default enrolled voice; the browser may override it. |
| `AGENT_API_KEY` | Optional local gatekeeper for the agent WebSocket. |
| `AGENT_STT_PROVIDER` | `local` (faster-whisper) or `openai`. |
| `AGENT_STT_MODEL` | e.g. `large-v3`, `whisper-1`. |
| `AGENT_LLM_BASE_URL` | OpenAI-compatible base URL, e.g. Ollama `http://127.0.0.1:11434/v1`. |
| `AGENT_LLM_MODEL` | e.g. `qwen2.5:7b`. |
| `AGENT_LLM_MAX_TOKENS` | Token budget per assistant turn (default 300). Reasoning models may need more so they can emit answer text after their reasoning. |
| `AGENT_VAD_SILENCE_MS` | End-of-turn silence; lower is snappier but cuts slow speakers. |
| `AGENT_INPUT_SAMPLE_RATE` | Client must resample to this rate (default 16000). |
| `AGENT_OUTPUT_SAMPLE_RATE` | Fixed at 24000 for the current TTS stream. |

The browser demo is a protocol reference, not an authentication boundary. If you
expose it remotely, use HTTPS (microphone capture requires a secure context except
on localhost), set `AGENT_API_KEY`, and put a TLS reverse proxy in front. Never
embed the TTS or agent secret in a public page; the demo asks the user for the
agent key at connect time.

### Latency tuning

- `TTS_PROFILE=realtime` uses fewer sampler steps; it is a candidate that must be
  A/B-tested for pronunciation and speaker similarity.
- The agent sends the first phrase at a comma and later phrases at sentence
  boundaries. Keep LLM answers to 1–3 short sentences via
  `AGENT_SYSTEM_PROMPT`.
- Local LLM/STT latency often dominates RTF. Measure the whole loop, not just TTS.
- `AGENT_VAD_SILENCE_MS` trades responsiveness against interrupting slow speakers.

---

## Voices, consent and privacy

- A stored profile requires `consent=true`, a non-empty exact transcript, and
  1–12 seconds of aligned reference speech.
- The API has no endpoint for synthesizing from an arbitrary file path.
- `SpeechRequest.reference_path` exists only in the internal Python service used
  by the Gradio apps; it is never exposed by the HTTP API.
- Voice profiles are owned by the creating API key; admins can access all.
- Images, model files, disabled cloud fallbacks and preset speakers are not used.
- Uploaded voice references are stored as prepared 24 kHz WAV plus a JSON
  sidecar. Delete a profile through `DELETE /v1/voices/{id}`.
- The configured STT and LLM may be local or hosted. The demo cannot know which
  backend the operator chose; disclose that in your own privacy notice.

Reference quality matters more than clone “strength”. Use one speaker, a quiet
room, no music/echo/clipping, natural pace, and a transcript that matches the
audio exactly. A 12-second cap prevents F5’s long-reference/transcript
misalignment failure mode; it is not an arbitrary file-size limit.

---

## Evaluation

```bash
export TTS_API_KEY=vls_...
export TTS_VOICE_ID=voice_...
python eval/benchmark.py --voice-id "$TTS_VOICE_ID" --transport both --concurrency 1
```

This writes `eval/results/<run>/` with manifest, per-trial JSON, summary,
generated audio and `human_review_template.json`. It measures wall time, TTFB,
server generation time, RTF, cache hits and concurrency behaviour. It does not
score quality. Read [`eval/README.md`](eval/README.md) before publishing numbers.

For automated speaker-similarity and ASR error-rate measurements, install the
optional quality dependencies and run `eval/quality.py` against a consented
reference plus generated files:

```bash
pip install -r requirements-quality.txt
python eval/quality.py --manifest eval/quality-manifest.json --dry-run
python eval/quality.py --manifest eval/quality-manifest.json \
  --asr-model large-v3-turbo --device cpu
```

The manifest requires explicit consent (or marks a synthetic test signal), and
the report labels similarity as a proxy rather than proof of perceptual identity.
See [`eval/README.md`](eval/README.md#quality-speaker-similarity-and-wer) and the
[consented human-reference protocol](eval/HUMAN-REFERENCE-PROTOCOL.md) for the
sample counts, listening procedure, consent rules, and model versions.

Run the unit suite:

```bash
python -m unittest discover -s tests -v
```

The suite covers config/frontend, chunking, post-processing, cache, voice store,
service, API auth/quotas/voices/streaming/WebSocket, VAD, sentence buffering,
agent session cancellation, agent WebSocket protocol, the eval harness, the
offline quality manifest/metrics, and the Chatterbox loader compatibility shim.
It uses fakes and does not need model weights.

---

## Operations and security

- Default bind addresses are `127.0.0.1`. Put TLS in front of any remote access;
  see `deploy/nginx.conf.example`.
- Use scoped SQLite keys for tenants. The static `TTS_API_KEY` is an admin key.
- Revoke keys with `python -m api.admin revoke --key-id key_...`.
- Back up `data/accounts.sqlite3`, `voices/` and the consented reference source
  material. Passwords/secrets are not stored in plaintext; API keys are hashed.
- The exact-audio cache is content-addressed and bounded by
  `TTS_CACHE_MAX_BYTES`. It never acts as a voice or account directory.
- `/metrics` is admin-only. Metrics are in-process; multi-process deployments need
  a shared metrics backend.
- Rate limiting and quotas are in-process/SQLite-backed. If you run multiple API
  workers or hosts, use a shared rate limiter and database of record; otherwise
  per-key limits are per process.
- The inference engine serializes generation per process. Do not raise
  `TTS_API_CONCURRENCY` beyond what your GPU memory can support.
- Set `TTS_API_CORS_ORIGINS` only for trusted browser origins. The demo is
  same-origin with the agent and does not need CORS.
- Container health checks cover liveness, not model quality.

### Deploying with Docker Compose

```bash
cp .env.example .env
# edit secrets
docker compose run --rm api python -m tts.setup_models
docker compose up -d --build
curl -fsS http://127.0.0.1:8000/healthz
```

The compose file binds to loopback by default. Change `ports:` only behind a
firewall/reverse proxy. The agent shares the HF model cache volume for optional
local STT.

---

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `401 GatedRepo` / model download fails | Accept the IndicF5 terms and set `HF_TOKEN`. |
| `ffmpeg is required` | Install FFmpeg and restart the process. |
| `English-only custom-voice synthesis is not enabled` | Expected unless `TTS_ENABLE_CHATTERBOX=1`; enabling Chatterbox makes `engine=auto` use it for English. |
| `engine_initialization_failed` | The selected engine could not load: missing/gated weights, a missing dependency, or an invalid model path. No fallback is attempted. |
| `engine_unavailable` for mixed text | The selected engine is disabled or does not support mixed text. Under `auto`, mixed text now uses IndicF5; explicitly request `engine=chatterbox` to A/B test it. |
| `Reference audio is ... safe maximum is 12s` | Trim the reference and re-enable its exact transcript. |
| `voice_not_found` | The voice belongs to another key, was deleted, or the ID is wrong. |
| Agent connects then closes 4401 | Set the correct `AGENT_API_KEY` in the demo; set `AGENT_TTS_API_KEY` for synthesis. |
| Agent hears nothing | Microphone permission, wrong input device, HTTPS requirement, or the browser is sending a rate other than `AGENT_INPUT_SAMPLE_RATE`. |
| No audio after a valid transcript | Check `AGENT_TTS_URL`, the TTS key/voice ID, and the API logs. Silent output is rejected rather than returned. |
| Agent reports `empty_llm_response` | The LLM produced no speakable content (for example, a reasoning model used its whole token budget on reasoning). Increase `AGENT_LLM_MAX_TOKENS` or use a non-reasoning model; the turn fails explicitly and no empty assistant message is committed. |
| `429` on `/v1/me` | The key’s per-minute limit is low; create a higher-rate key for benchmarks. |
| `readyz` returns 503 | Warmup failed. Check the model/device logs; `TTS_WARMUP=0` skips warmup but does not fix a broken model. |

---

## Repository layout

```
api/                 FastAPI HTTP/WebSocket API, accounts, auth, metrics
agent/               VAD, STT, LLM, TTS client, session pipeline, WS server
agent/static/        Bundled no-build browser demo
tts/                 Inference core, service, voice store, cache, frontends
eval/                Fixed corpus and technical benchmark harness
deploy/              nginx and systemd examples
tests/               Dependency-light unit/integration tests with fakes
run.sh               One-command install, model cache and Voice Lab launcher
app.py               Simple Gradio studio
voice_lab_app.py     Voice Lab Gradio studio
Dockerfile           API + agent image
docker-compose.yml   Single-host API + agent deployment
experiments/         Isolated Chatterbox-Telugu POC/evaluation reference
```

## Attribution

IndicF5, F5-TTS, Vocos, IndicXlit, Chatterbox-Telugu and faster-whisper are
third-party projects. See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) and
review the upstream model cards and licenses before commercial use. The optional
Chatterbox-Telugu fine-tune is CC-BY-4.0 and requires attribution.
