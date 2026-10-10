# Local Voice Studio

A local Gradio interface for custom-reference Telugu speech using the existing
AI4Bharat IndicF5 direct `DiT` checkpoint loader. The application has no preset
speaker list and does not send recordings or text to a hosted service.

## Existing environment and baseline

The existing `.venv` was inspected rather than recreated. It uses Python 3.11.17,
PyTorch 2.11.0, torchaudio 2.11.0, Transformers 4.46.3, F5-TTS 0.1.0, Vocos 0.1.0,
Gradio 6.29.1, NumPy 1.26.4, and SciPy 1.17.1. FFmpeg 9.0.2 is installed.

Before edits, `test_tts.py` completed on MPS, loaded the IndicF5 checkpoint with
zero missing and unexpected keys, and generated `telugu_output.wav` at 24 kHz.
PyTorch/Vocos emitted deprecated STFT/ISTFT output-buffer resize warnings during
that successful run. The warnings are non-fatal in this baseline; this project
does not modify Vocos or PyTorch internals, and the warnings alone do not establish
whether audio quality is affected.

`references/processed/` contains copies prepared by the prior script using linear
interpolation to 24 kHz. The new workflow does not use those development assets:
user recordings are decoded into a temporary working copy and resampled with
SciPy's polyphase anti-aliased resampler. Original recordings are not overwritten.

## Setup

Use the existing environment:

```bash
cd ~/TTS
source .venv/bin/activate
brew install ffmpeg
pip install -r requirements.txt
```

For a new Apple Silicon environment, install a PyTorch build with MPS support
before installing the remaining packages; follow the current PyTorch macOS
installation instructions if the pinned release is unavailable for that Python
version. Do not replace the validated IndicF5 loader with `AutoModel`.

Download/cache the model files and transliterator once while online:

```bash
python -m tts.setup_models
```

This downloads IndicF5 from `ai4bharat/IndicF5`, Vocos from
`charactr/vocos-mel-24khz`, and warms AI4Bharat IndicXlit for Telugu transliteration.
IndicXlit's large all-language rescore dictionaries are not downloaded; it uses its
neural model with beam search for Telugu.
The setup command fails explicitly if any required model is unavailable.

## Run

```bash
cd ~/TTS
source .venv/bin/activate
python app.py
```

The server binds to `127.0.0.1` only, does not create a share link, and disables
Gradio analytics. The first inference loads local weights into MPS when available,
otherwise CPU. Inference is serialized to limit memory usage.

## Offline check

After model setup, disconnect from the network and launch with Hugging Face offline
flags:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python app.py
```

Generate a short Telugu sample and a Romanized Telugu preview. If the assets are
missing, the application reports that setup must be completed; it does not fall
back to a remote service. Upload decoding, resampling, speech generation, and MP3
encoding are local. Model setup is the only step expected to require network access.

## Use

1. Upload a WAV/MP3/M4A/FLAC/OGG recording or record through the browser microphone.
2. Play it in the page, enter the transcript exactly as spoken, then validate it.
3. Choose a text mode and enter the text. For Romanized Telugu, prepare the Telugu
   script preview, correct it if needed, then generate.
4. Listen to the output and download WAV or MP3. WAV is PCM 24-bit; MP3 is lossy
   192 kbps.

Audio validation rejects empty, silent, invalid, very short, excessively long, or
heavily clipped recordings. A clear 10–15 second reference is recommended. IndicF5's
reference preprocessing may trim a longer recording to around 15 seconds. Basic
signal checks cannot reliably detect all noise, echo, or transcript mismatches.
Use only voices you own or have permission to clone.

## Capabilities and limits

| Feature | Current status |
|---|---|
| Telugu script | Enabled with the validated local IndicF5 reference-conditioned engine. |
| Romanized Telugu | IndicXlit local conversion with an editable Telugu-script preview; informal spelling is ambiguous. |
| Telugu-English mixed text | Routed to IndicF5 as experimental; mixed-language pronunciation and cross-language speaker similarity are not quality-certified. |
| English-only custom-voice synthesis | Not enabled. No local English engine with arbitrary reference-voice conditioning was validated. The app does not substitute a preset speaker or cloud service. |
| Style/emotion controls | No dropdown. The model has no validated style control; prosody is conditioned by the user's reference recording. |
| Long text | Sentence/paragraph-aware, sequential chunks with pauses and visible segment progress. |
| Audio formats | WAV and MP3 outputs; input decoding handled locally by FFmpeg. |
| Automatic network access during synthesis | IndicF5/Vocos use cache-only Hugging Face lookups; transliteration must be warmed during setup. No cloud inference is used. |

IndicF5's published model card identifies 11 Indic languages including Telugu, not
English. A successful mixed-input inference is not by itself evidence of fluent
English or equivalent speaker identity. Kokoro and other preset-speaker engines are
not used because they would not satisfy the custom-reference-voice requirement.

## Tests and known-good baseline

Run unit tests:

```bash
python -m unittest discover -s tests -v
```

`test_tts.py` uses the validated direct loader with cache-only model access and
performs real inference:

```bash
python test_tts.py
```

The current baseline is a successful Telugu generation on MPS. This change has not
yet demonstrated quality parity for English-only speech, code-switching, style
transfer, or voice similarity across languages. The STFT/ISTFT warnings remain
unmodified pending a controlled audio-quality comparison.

Generated files are stored in `outputs/` and named `voice_*.wav` / `voice_*.mp3`.
The app's “Delete generated files” button only removes those named outputs; it never
deletes reference recordings or the existing `telugu_*` samples.
