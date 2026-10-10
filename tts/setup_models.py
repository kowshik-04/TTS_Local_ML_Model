"""Download model assets once; the application only reads cached model files."""

from __future__ import annotations

from huggingface_hub import hf_hub_download

from tts.indicf5_engine import REPO_ID, VOCODER_REPO_ID
from tts.transliteration import prepare_transliterator


def main() -> None:
    print("Downloading/caching IndicF5 model files...")
    hf_hub_download(REPO_ID, filename="checkpoints/vocab.txt")
    hf_hub_download(REPO_ID, filename="model.safetensors")
    print("Downloading/caching Vocos model files...")
    hf_hub_download(VOCODER_REPO_ID, filename="config.yaml")
    hf_hub_download(VOCODER_REPO_ID, filename="pytorch_model.bin")
    print("Warming AI4Bharat IndicXlit's local Telugu RNN transliterator...")
    try:
        prepare_transliterator()
    except (ImportError, RuntimeError) as error:
        raise SystemExit(
            f"Could not prepare IndicXlit: {error}"
        ) from error
    print("Model assets are cached. You can now disconnect from the network and run the app.")


if __name__ == "__main__":
    main()
