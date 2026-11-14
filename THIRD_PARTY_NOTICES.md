# Third-party notices

This project builds on the following upstream work. Verify the exact license
text and model-card terms before redistribution; model licenses can change.

## AI4Bharat IndicF5

- Model: <https://huggingface.co/ai4bharat/IndicF5>
- Repository: <https://github.com/AI4Bharat/IndicF5>
- Used as the validated Telugu reference-conditioned engine. The model is gated
  on Hugging Face and requires accepting its terms.

## F5-TTS and Vocos

- F5-TTS: <https://github.com/SWivid/F5-TTS>
- IndicF5's source-pinned F5-TTS package:
  <https://github.com/AI4Bharat/IndicF5/tree/13f7c4d627cc10111aea8fe9c0039462cacacdc7>
- Vocos: <https://github.com/charactr-platform/vocos>
- Used by the local inference loader and the 24 kHz vocoder. The `f5_tts`
  package is installed from AI4Bharat's IndicF5 repository at the pinned commit
  because no compatible 0.1.0 release exists on PyPI. `datasets`, `pyarrow` and
  `fsspec` are pinned to versions that still support the required NumPy 1.26
  runtime.

## AI4Bharat IndicXlit

- <https://github.com/AI4Bharat/IndicXlit>
- Used for optional offline Romanized-Telugu transliteration previews.
  Transliteration is ambiguous and is always shown for review.
- Installed with `--no-deps` from the upstream wheel because its declared
  `fairseq` dependency is only used by the transformer mode; this project uses
  the RNN model. The small `pydload`/`pandas` runtime deps are installed
  explicitly. See `requirements-transliteration.txt`.

## Chatterbox-Telugu (optional, experimental)

- Model: <https://huggingface.co/shankarpandala/chatterbox-telugu>
- Base project: <https://github.com/resemble-ai/chatterbox>
- The fine-tune is distributed under **CC-BY-4.0**. If you enable it, retain
  attribution to the model author and the upstream Chatterbox project, and
  review the model card for additional terms.
- This project does not bundle the weights. It is disabled by default and is not
  quality-certified for English, Telugu-English, or speaker similarity.

## faster-whisper / CTranslate2 (optional agent STT)

- <https://github.com/SYSTRAN/faster-whisper>
- Whisper model weights carry their own upstream licenses. Review the model you
  select for commercial use.

## Fonts and browser demo

The bundled demo uses system fonts only and has no external JavaScript or CDN
dependencies.
