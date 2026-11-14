# Consented human-reference quality protocol

This is a procedure, not a quality result. Do not fill in scores until real
consented speakers and listeners have completed the runs. The synthetic Hindi
reference used for the pipeline smoke test must never be used as evidence of
human voice cloning.

## 1. Consent and data handling

- Use adult speakers who have given written, specific consent for local voice
  cloning, evaluation, and the intended deployment. Record the consent form and
  the allowed retention period; the form must cover generated audio as well as
  the reference recording.
- Do not collect minors, bystanders, or speech that contains private third-party
  information. Do not commit reference or generated audio to the repository.
- Keep the reference and generated files in an access-controlled directory.
  Publish only aggregated measurements, the prompt list, and the model/runtime
  manifest, unless each speaker separately consents to publishing samples.
- The speaker may withdraw and request deletion. Record the deletion procedure
  before collecting data.

## 2. Reference recording procedure

For each speaker, collect:

| Item | Requirement |
|---|---|
| Speaker count | Minimum 3 consented speakers for generalization; a single speaker is a case study only |
| Takes | 3 takes of 6–12 s each; retain one held-out take and one reference take |
| Format | 24 kHz or 48 kHz mono PCM16 WAV; no clipping, noise reduction, or pitch processing |
| Content | Natural conversational Telugu, with an exact human-checked transcript |
| Environment | Quiet room, consistent microphone and distance, no music or second speaker |
| Metadata | Speaker ID, language background, transcript, consent ID, recording date, microphone, sample rate |
| Privacy | No names, phone numbers, addresses, or other personal data in the prompt transcripts |

Use one enrolled reference per speaker for all comparisons. Keep the reference
audio and generated audio separate from the serving `voices/` directory after
the experiment; the serving profile should be created only through the normal
consent-gated voice enrollment flow.

## 3. Test corpus and sample counts

Use the fixed prompts in `eval/corpus.py` without editing them between builds:

| Language | Prompts | Per speaker | 3 speakers |
|---|---:|---:|---:|
| Telugu | 30 | 30 | 90 |
| Telugu + English (Tenglish) | 8 | 8 | 24 |
| English (optional, Chatterbox only) | 6 | 6 | 18 |
| **Total** | **44** | **44** | **132** |

For the first, smaller run, use one speaker and all 44 prompts. That is a
single-speaker case study and must be labeled as such. Do not compare a
single-speaker result to a multi-speaker claim.

Do not use the reference transcript as a test prompt. Use the committed corpus
prompts so two runs are comparable. Generate each prompt once per speaker per
engine; do not average repeated stochastic samples unless the repetition scheme
is declared in the manifest.

## 4. Generation procedure

1. Accept the IndicF5 gating terms and cache the model as documented in the
   main README. Keep IndicF5 as the default Telugu/Tenglish engine.
2. Select the engine explicitly: `engine=indicf5` for Telugu/Tenglish;
   Chatterbox is optional and experimental and may be used only for the
   explicitly labeled English/Tenglish comparison.
3. Warm the model with a separate request, then disable cache reuse
   (`TTS_CACHE_ENABLED=0`) for the measured run. Ensure every sample has a
   distinct prompt/voice/engine combination.
4. Record the exact model identifiers and revisions, inference profile
   (`TTS_PROFILE`), device, Python/torch versions, and package versions.
5. Save the generated WAVs and construct a `quality.py` manifest with
   `reference.consent_confirmed: true`, `reference.synthetic: false`, and a
   `reference.notes` value that points to the consent record—not to private
   identifying details.

## 5. Automated measurements

Use the existing quality harness, not a serving-path shortcut:

```bash
python eval/quality.py \
  --manifest /path/to/human-reference-manifest.json \
  --device cpu \
  --asr-model large-v3-turbo \
  --asr-device cpu \
  --asr-compute-type int8 \
  --model-cache /path/to/quality-model-cache
```

Record:

- **Speaker similarity:** SpeechBrain `spkrec-ecapa-voxceleb` cosine similarity
  for every generated sample against the enrolled reference. Report per-sample
  values plus median, IQR, and range. This is a model-dependent proxy, not proof
  of perceptual identity.
- **ASR quality:** faster-whisper `large-v3-turbo`, with `te` for
  Telugu/Tenglish and `en` for English, after the normalization in
  `eval/quality.py`. Report WER and CER per sample and aggregate median/range.
  Run `large-v3` as a sensitivity check when compute permits; do not treat one
  recognizer as ground truth.
- **Audio health:** duration, sample rate, channels, peak, RMS, clipping and
  near-silence fractions. Exclude failed or empty audio from quality claims.
- **Short clips:** flag samples under 2.0 s as unstable; do not use them for a
  headline score.
- **WER/CER above 1.0:** report as an ASR insertion/hallucination signal, not as
  a pronunciation score.
- Do not introduce pass/fail thresholds. There is no validated universal
  threshold for this corpus.

Model/version record to include with every report:

| Component | Validated/current value |
|---|---|
| IndicF5 | AI4Bharat source pin `13f7c4d627cc10111aea8fe9c0039462cacacdc7` |
| Vocos | `0.1.0` |
| Speaker model | `speechbrain/spkrec-ecapa-voxceleb`, SpeechBrain 1.1.2 |
| ASR | `Systran/faster-whisper-large-v3-turbo`, faster-whisper 1.2.1 |
| Audio I/O | PyAV 18.1.0, soundfile 0.14.0 |
| Optional comparison | Chatterbox-Telugu `shankarpandala/chatterbox-telugu`, revision `d4341468c1738ea669ef096baccecc68cc8ba9f3`, Chatterbox-TTS 0.1.7 |

The quality script records the actual Python, package, model, file hash, and
device versions in `quality.json`; use those recorded values, not this table, in
the final report.

## 6. Human listening procedure

- Use 3 listeners per sample minimum. Do not let the speaker rate their own
  generated samples unless that is explicitly reported as a separate condition.
- Randomize sample order and blind the listener to engine, model, speaker, and
  whether the reference or generated audio is presented.
- Use headphones in a quiet room. Present each sample once; allow replay only
  under a predeclared rule.
- Rate three separate 5-point scales: naturalness, speaker similarity to the
  consented reference, and intelligibility. Add a free-text field for
  pronunciation, names, numbers, code-switching, and artifacts.
- Include an attention/quality check (for example, a known-bad sample) and
  report exclusions. Report inter-rater agreement with a named statistic; do not
  invent agreement values.
- The speaker-similarity question must be identity similarity, not
  "does this sound pleasant" or "is this the same language."
- Store raw ratings in a blank template derived from
  `human_review_template.json`; do not overwrite or fabricate ratings.

## 7. Reporting rules

A report is complete only when it contains:

1. speaker count, consent basis, and recording metadata;
2. the exact 30/8/6 corpus and per-speaker sample counts;
3. model IDs, revisions, package versions, device, and inference profile;
4. per-sample and aggregate speaker similarity, WER/CER, audio-health fields;
5. listening sample counts, listener count, rating scales, exclusions, and
   agreement statistic;
6. known limitations: synthesized-reference warnings (if any), short clips,
   Tenglish/English ASR coverage, embedding-model dependence, and hardware.

If any of these are missing, mark the area **NOT TESTED** or **BLOCKED**, not
PASS. The current repository has no consented human reference or human
listening data; this protocol is preparation only.
