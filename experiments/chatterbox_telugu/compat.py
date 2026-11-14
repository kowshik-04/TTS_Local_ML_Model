"""Compatibility checks for the Telugu Chatterbox checkpoint.

This module does not modify Chatterbox files.  It temporarily adapts the
in-memory upstream configuration while loading a checkpoint that explicitly
extends the upstream multilingual tokenizer from 2454 to 2521 entries.
"""

from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
from tempfile import mkdtemp
from typing import Iterator

from safetensors import safe_open
from tokenizers import Tokenizer


TELUGU_LANGUAGE_ID = "te"
TELUGU_LANGUAGE_NAME = "Telugu"
EXPECTED_VOCAB_SIZE = 2521
EXPECTED_T3_MODEL = "t3_mtl_te.safetensors"
_EXPECTED_TENSOR_SHAPES = {
    "text_emb.weight": (EXPECTED_VOCAB_SIZE, 1024),
    "text_head.weight": (EXPECTED_VOCAB_SIZE, 1024),
    "text_pos_emb.emb.weight": (2050, 1024),
    "speech_emb.weight": (8194, 1024),
    "speech_head.weight": (8194, 1024),
    "cond_enc.spkr_enc.weight": (1024, 256),
    "cond_enc.emotion_adv_fc.weight": (1024, 1),
}


@contextmanager
def loader_model_dir(
    model_dir: str | Path,
    *,
    alias_filename: str = "t3_mtl23ls_v2.safetensors",
) -> Iterator[Path]:
    """Expose the Telugu checkpoint under the filename stock 0.1.7 expects.

    ``ChatterboxMultilingualTTS.from_local`` in chatterbox-tts 0.1.7 accepts no
    ``t3_model`` argument and hardcodes ``t3_mtl23ls_v2.safetensors``.  Newer or
    patched builds may accept the filename directly.  A temporary directory of
    symlinks keeps the downloaded model tree untouched while supporting both
    interfaces.  The caller keeps this context open only for the load call;
    the loaded model retains no file handles.
    """
    model_dir = Path(model_dir)
    telugu_checkpoint = model_dir / EXPECTED_T3_MODEL
    if not telugu_checkpoint.is_file():
        yield model_dir
        return
    temporary = Path(mkdtemp(prefix="chatterbox-loader-"))
    try:
        for entry in model_dir.iterdir():
            if not entry.is_file():
                continue
            target_name = (
                alias_filename if entry.name == EXPECTED_T3_MODEL else entry.name
            )
            target = temporary / target_name
            try:
                os.symlink(entry, target)
            except OSError:
                shutil.copy2(entry, target)
        yield temporary
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


class CompatibilityError(RuntimeError):
    """The local checkpoint and its documented compatibility contract differ."""


def preflight(model_dir: str | Path) -> dict[str, object]:
    """Validate tokenizer and checkpoint metadata without allocating the T3 model."""
    model_dir = Path(model_dir)
    config_path = model_dir / "config.json"
    tokenizer_path = model_dir / "grapheme_mtl_merged_expanded_v1.json"
    checkpoint_path = model_dir / EXPECTED_T3_MODEL
    for path in (config_path, tokenizer_path, checkpoint_path):
        if not path.is_file():
            raise CompatibilityError(f"Missing required model file: {path.name}")

    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CompatibilityError(f"Could not read model config: {error}") from error
    if config.get("language") != TELUGU_LANGUAGE_ID:
        raise CompatibilityError(f"Expected language '{TELUGU_LANGUAGE_ID}', got {config.get('language')!r}.")
    if config.get("vocab_size") != EXPECTED_VOCAB_SIZE:
        raise CompatibilityError(
            f"Expected config vocab_size {EXPECTED_VOCAB_SIZE}, got {config.get('vocab_size')!r}."
        )
    if config.get("t3_model") != EXPECTED_T3_MODEL:
        raise CompatibilityError(
            f"Expected config t3_model '{EXPECTED_T3_MODEL}', got {config.get('t3_model')!r}."
        )

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    vocabulary = tokenizer.get_vocab(with_added_tokens=True)
    tokenizer_size = tokenizer.get_vocab_size(with_added_tokens=True)
    if tokenizer_size != EXPECTED_VOCAB_SIZE or len(vocabulary) != EXPECTED_VOCAB_SIZE:
        raise CompatibilityError(
            f"Expected tokenizer vocabulary size {EXPECTED_VOCAB_SIZE}, got {tokenizer_size}."
        )
    if max(vocabulary.values(), default=-1) != EXPECTED_VOCAB_SIZE - 1:
        raise CompatibilityError("Tokenizer IDs are not contiguous through the expected final ID.")
    if vocabulary.get(f"[{TELUGU_LANGUAGE_ID}]") != EXPECTED_VOCAB_SIZE - 1:
        raise CompatibilityError("Tokenizer does not map [te] to the trained Telugu language-token ID.")

    with safe_open(checkpoint_path, framework="pt", device="cpu") as checkpoint:
        for key, expected_shape in _EXPECTED_TENSOR_SHAPES.items():
            if key not in checkpoint.keys():
                raise CompatibilityError(f"Checkpoint does not contain {key}.")
            actual_shape = tuple(checkpoint.get_slice(key).get_shape())
            if actual_shape != expected_shape:
                raise CompatibilityError(
                    f"Checkpoint {key} has shape {actual_shape}; expected {expected_shape}."
                )

    return {
        "vocab_size": EXPECTED_VOCAB_SIZE,
        "telugu_language_token_id": vocabulary[f"[{TELUGU_LANGUAGE_ID}]"],
        "checkpoint": str(checkpoint_path),
    }


@contextmanager
def compatibility_scope() -> Iterator[None]:
    """Temporarily adapt stock Chatterbox's in-memory Telugu configuration."""
    from chatterbox import mtl_tts
    from chatterbox.models.t3.modules.t3_config import T3Config

    original_multilingual = T3Config.__dict__["multilingual"]
    had_telugu = TELUGU_LANGUAGE_ID in mtl_tts.SUPPORTED_LANGUAGES
    original_telugu_name = mtl_tts.SUPPORTED_LANGUAGES.get(TELUGU_LANGUAGE_ID)

    @classmethod
    def extended_multilingual(cls):
        return cls(text_tokens_dict_size=EXPECTED_VOCAB_SIZE)

    T3Config.multilingual = extended_multilingual
    mtl_tts.SUPPORTED_LANGUAGES[TELUGU_LANGUAGE_ID] = TELUGU_LANGUAGE_NAME
    try:
        yield
    finally:
        T3Config.multilingual = original_multilingual
        if had_telugu:
            mtl_tts.SUPPORTED_LANGUAGES[TELUGU_LANGUAGE_ID] = original_telugu_name
        else:
            mtl_tts.SUPPORTED_LANGUAGES.pop(TELUGU_LANGUAGE_ID, None)
