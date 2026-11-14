"""Download model assets once; the application only reads cached model files.

IndicF5 is required.  The Chatterbox-Telugu engine is optional, experimental and
CC-BY-4.0 licensed; download it only if you intend to evaluate it.
"""

from __future__ import annotations

import argparse

from huggingface_hub import hf_hub_download, snapshot_download

from tts.config import Settings
from tts.indicf5_engine import REPO_ID, VOCODER_REPO_ID
from tts.transliteration import prepare_transliterator


def download_indicf5() -> None:
    print("Downloading/caching IndicF5 model files...")
    hf_hub_download(REPO_ID, filename="checkpoints/vocab.txt")
    hf_hub_download(REPO_ID, filename="model.safetensors")
    print("Downloading/caching Vocos model files...")
    hf_hub_download(VOCODER_REPO_ID, filename="config.yaml")
    hf_hub_download(VOCODER_REPO_ID, filename="pytorch_model.bin")


def download_chatterbox(settings: Settings) -> None:
    print(
        f"Downloading optional Chatterbox-Telugu model {settings.chatterbox_model_id!r} "
        f"into {settings.chatterbox_dir} ..."
    )
    try:
        snapshot_download(
            settings.chatterbox_model_id,
            local_dir=settings.chatterbox_dir,
        )
    except Exception as error:
        raise SystemExit(
            "Could not download Chatterbox-Telugu. Check your network, Hugging Face "
            f"access/token and the model id. Details: {error}"
        ) from error
    print(
        "Chatterbox-Telugu is an optional community fine-tune (CC-BY-4.0). "
        "It is experimental; run listening tests and keep attribution before "
        "using it in a product."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--chatterbox",
        action="store_true",
        help="Also download the optional experimental Chatterbox-Telugu model.",
    )
    parser.add_argument(
        "--skip-transliterator",
        action="store_true",
        help="Skip warming the local IndicXlit Telugu transliterator.",
    )
    parser.add_argument(
        "--chatterbox-model-id",
        default=None,
        help="Override TTS_CHATTERBOX_MODEL_ID for this download.",
    )
    args = parser.parse_args(argv)

    settings = Settings.from_env()
    if args.chatterbox_model_id:
        from dataclasses import replace

        settings = replace(settings, chatterbox_model_id=args.chatterbox_model_id)

    download_indicf5()
    if not args.skip_transliterator:
        print("Warming AI4Bharat IndicXlit's local Telugu RNN transliterator...")
        try:
            prepare_transliterator()
        except (ImportError, RuntimeError) as error:
            raise SystemExit(f"Could not prepare IndicXlit: {error}") from error
    if args.chatterbox:
        download_chatterbox(settings)

    print("Required model assets are cached. You can now disconnect from the network.")
    if args.chatterbox:
        print(
            "Optional Chatterbox assets are cached. Enable it with "
            "TTS_ENABLE_CHATTERBOX=1 and set TTS_CHATTERBOX_DIR if needed."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
