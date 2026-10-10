# English and Telugu voice-cloning research

Research date: 2026-10-10. This document separates published claims from the
local proof of concept; no model is considered integrated or quality-validated
until it produces inspected local audio in this repository.

## Ranked candidates

1. `shankarpandala/chatterbox-telugu` is the closest apparent fit. Its model
   card claims retained English, Telugu, Telugu-English code-switching, and
   zero-shot cloning from a 6-15 second reference clip. It documents `mps` and
   CPU execution, but accepts audio only, not a reference transcript. The
   published snapshot is 3.21 GB: `t3_mtl_te.safetensors` is 2.14 GB and
   `s3gen.pt` is 1.06 GB. License: CC-BY-4.0 for the fine-tune (attribution
   required), with MIT-licensed base acoustic components. It is not suitable
   for production integration yet: the installed `chatterbox-tts` 0.1.7
   `from_local` loader accepts the Telugu T3 checkpoint, but this fine-tune
   also needs an experiment-only compatibility shim for its expanded token
   vocabulary and Telugu language mapping. Local generation now succeeds with
   that shim. The model card provides no independent MOS, intelligibility,
   code-switch, or speaker-similarity evaluation.

2. AI4Bharat IndicF5 remains the verified Telugu baseline. It is a 0.4B
   F5-TTS-derived model with a three-input interface: text, reference audio,
   and matching reference transcript. It supports Telugu but its published
   supported-language list does not include English. License: MIT. It is the
   only candidate in this project with real local Telugu output verified.

3. Official Chatterbox Multilingual V3 is a 500M zero-shot voice-cloning
   model and documents CPU/MPS use. It supports English and cross-language
   cloning, but its official language list does not include Telugu. License:
   MIT. It cannot satisfy the target workflow alone.

4. `reenigne314/chatterbox-indic-lora` adds Telugu while retaining English on
   paper, but documents preset language/gender conditioning, explicitly says
   Telugu-English code-mix has not been trained, and reports Telugu CER 0.2853
   while lacking formal speaker-similarity or MOS evaluation. The demonstrated
   path is CUDA-first. License: MIT. Reject for the complete requirement.

5. Praxy Voice claims Telugu, English, custom cloning, and code-mix, but its
   published router uses a Haiku-driven transliteration preprocessor for
   code-mixed Telugu and a Modal-backed serving path. That violates the
   project requirement of no cloud or hidden remote fallback. Its code is MIT
   and its R6 LoRA is Apache-2.0, but it is rejected as-is.

6. Coqui XTTS v2 supports local cross-language cloning from a reference clip,
   but its official 16-language list does not include Telugu. Its Coqui Public
   Model License is also more restrictive than MIT/Apache-2.0. Reject for this
   bilingual requirement.

7. Indic-Speak supports English and Telugu, but its model card explicitly
   states that it does not support voice cloning. It uses a 3.36B stack and an
   Indic Open Model License. Reject for custom-reference synthesis.

## Local POC status

The POC harness is `run_poc.py`, isolated from production. Offline five-case
run `20261010_015527_f39dd629` completed on MPS with the local checkpoint.
Model construction and checkpoint loading succeeded with the experiment-only
compatibility shim; the shim temporarily supplies the Telugu
vocabulary/language mapping and does not modify installed Chatterbox source or
production files. The strict checkpoint loads completed as part of model
construction.

All five cases recorded `succeeded` in
`outputs/results_20261010_015527_f39dd629.json`:

| Case | Reference | WAV duration | Generation time |
|---|---|---:|---:|
| Telugu | `references/01_conversation.wav` | 6.56 s | 14.017 s |
| English | `references/01_conversation.wav` | 8.40 s | 24.223 s |
| Telugu-English mixed | `references/08_telugu_english.wav` | 7.72 s | 65.172 s |
| Raw Tenglish | `references/01_conversation.wav` | 10.56 s | 88.665 s |
| Converted Telugu-script Tenglish | `references/01_conversation.wav` | 9.08 s | 101.753 s |

The raw and converted Tenglish cases use the same source sentence and
reference audio. The JSON records both input forms and confirms that the
converted text is the exact string passed to `model.generate()`. All five
files decode as mono 24 kHz PCM WAV with finite samples; the run's peak process
memory was 4940.8 MB. These measurements establish technical generation and
file output only; they do not establish intelligibility, Telugu pronunciation,
code-switch quality, naturalness, or speaker similarity.

Romanized Telugu preview uses the already-installed offline AI4Bharat
IndicXlit RNN from the sibling project's environment; no dependency or
transliteration model was added to the experiment environment. IndicXlit's
best candidate can still be ambiguous (for example, `Nenu` may produce `నేనూ`
instead of the intended `నేను`), so the custom CLI flow provides an editable
preview. A separate custom run confirmed the manually corrected preview was
the text synthesized and recorded in the results JSON. This preprocessing
does not make the experiment a production replacement; human listening and
quality assessment remain necessary.

Chatterbox accepts reference audio but not a matching reference transcript, so
it cannot use or validate transcript conditioning as IndicF5 does. The
experiment remains a candidate for human listening and evaluation, not a
production replacement. The original `outputs/telugu.wav` from the earlier
Telugu-only run was preserved.

## Repeatable evaluation baseline

`evaluation_corpus.py` now defines 30 pure Telugu prompts and 8 separate
Telugu-English prompts. The categorized cases exercise conversation and
formal text, vowels/aspiration, conjuncts, similar-sounding words,
punctuation, numbers, names, technical code-switching, and long sentences.
`evaluate_suite.py` runs each selected target through Chatterbox and
IndicF5 in separate local interpreters, with offline Hugging Face flags,
unique run directories, exact engine-boundary text records, WAV validity and
duration measurements, and an editable listening-review template.

The one-case offline smoke run `20261010_021643_63dcb09e` completed on MPS:
both engines succeeded (2/2 outputs) on the identical Telugu target,
`ఈ రోజు ఉదయం వాతావరణం చాలా ఆహ్లాదంగా ఉంది.` Chatterbox recorded 6.920
seconds and IndicF5 3.531 seconds; both WAVs were mono 24 kHz with finite
samples and valid peak/RMS values. These durations differ, but duration is
not an intelligibility or pronunciation score. The exact input and recorded
engine text agree for both outputs.

The complete 38-case run `20261010_021813_8c19ecf5` also completed offline
on MPS: 76/76 engine-case generations succeeded (38 per engine), with no
recorded failures. A round-trip check reopened all 76 WAVs and verified file
existence, mono channel count, sample rate, sample count, duration metadata,
and finite samples. Chatterbox output duration ranged from 4.800 to 14.720
seconds (median 6.360); IndicF5 ranged from 2.624 to 14.485 seconds (median
4.235). These are technical measurements only. The results and
`human_review_template.json` are under
`outputs/evaluations/20261010_021813_8c19ecf5/`; listening-based comparison
and error categorization are still pending.

The selected reference `01_conversation.wav` was read from the existing
processed reference path: mono, 24 kHz, 12.240 seconds, peak 0.180359, RMS
0.023959, no clipped samples, and 5.7482% near-silent samples under the
runner's `1e-4` threshold. Its transcript record matches
`references/01_conversation.txt`. These file/signal facts do not verify
single-speaker content, audible noise, or transcript accuracy; those fields
remain pending listening. No output has yet been scored for pronunciation,
missing words, prosody, or speaker similarity.

The second reference `08_telugu_english.wav` is mono, 24 kHz, 14.064 seconds,
peak 0.354706, RMS 0.041556, with no clipped samples and 4.0751% near-silent
samples by the same threshold. Its recorded transcript also matches the
on-disk transcript. Listening checks for both references are still pending.

Reproduce the one-case check with:

```bash
cd experiments/chatterbox_telugu
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python evaluate_suite.py --limit 1 --device auto
```

## References

- IndicF5 model card: https://huggingface.co/ai4bharat/IndicF5
- Chatterbox official repository: https://github.com/resemble-ai/chatterbox
- Telugu Chatterbox model card: https://huggingface.co/shankarpandala/chatterbox-telugu
- Indic LoRA model card: https://huggingface.co/reenigne314/chatterbox-indic-lora
- Praxy Voice repository: https://github.com/praxelhq/praxy
- XTTS v2 documentation: https://docs.coqui.ai/en/latest/models/xtts.html
- Indic-Speak model card: https://huggingface.co/bodhan-ai/indic-speak
