from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from f5_tts.model import DiT
from f5_tts.infer.utils_infer import (
    load_model,
    load_vocoder,
    infer_process,
    preprocess_ref_audio_text,
)

# --------------------------------------------------
# Configuration
# --------------------------------------------------

REPO_ID = "ai4bharat/IndicF5"
VOCODER_REPO_ID = "charactr/vocos-mel-24khz"

REF_AUDIO = "reference.wav"

REF_TEXT = (
    "నమస్కారం! నా పేరు కౌశిక్. "
    "నేను హైదరాబాద్‌లో పనిచేస్తున్నాను. "
    "నాకు తెలుగు భాష చాలా ఇష్టం."
)

GEN_TEXT = (
    "అందరికీ నమస్కారం! "
    "ఇది నా మొదటి తెలుగు టెక్స్ట్ టు స్పీచ్ పరీక్ష. "
    "ఈ వాయిస్ ఎలా ఉందో నాకు చాలా ఆసక్తిగా ఉంది."
)

OUTPUT = "telugu_output.wav"

# --------------------------------------------------
# Device
# --------------------------------------------------

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(f"Using device: {device}")

# --------------------------------------------------
# Read cached model files
# --------------------------------------------------

print("Checking cached IndicF5 files...")

vocab_path = hf_hub_download(
    REPO_ID,
    filename="checkpoints/vocab.txt",
    local_files_only=True,
)

ckpt_path = hf_hub_download(
    REPO_ID,
    filename="model.safetensors",
    local_files_only=True,
)
vocoder_config = hf_hub_download(
    VOCODER_REPO_ID,
    filename="config.yaml",
    local_files_only=True,
)
hf_hub_download(
    VOCODER_REPO_ID,
    filename="pytorch_model.bin",
    local_files_only=True,
)

print(f"Vocab: {vocab_path}")
print(f"Checkpoint: {ckpt_path}")

# --------------------------------------------------
# Load Vocos
# --------------------------------------------------

print("\nLoading Vocos...")

vocoder = load_vocoder(
    vocoder_name="vocos",
    is_local=True,
    local_path=str(Path(vocoder_config).parent),
    device=device,
)

print("Vocos loaded.")

# --------------------------------------------------
# Build IndicF5 model
# --------------------------------------------------

print("\nBuilding IndicF5 model...")

model = load_model(
    DiT,
    dict(
        dim=1024,
        depth=22,
        heads=16,
        ff_mult=2,
        text_dim=512,
        conv_layers=4,
    ),
    mel_spec_type="vocos",
    vocab_file=vocab_path,
    device=device,
)

print("Model architecture created.")

# --------------------------------------------------
# Load ONLY the EMA model weights
# --------------------------------------------------

print("\nLoading IndicF5 checkpoint...")

state_dict = load_file(
    ckpt_path,
    device=device
)

state_dict = {
    key.replace("ema_model._orig_mod.", ""): value
    for key, value in state_dict.items()
    if key.startswith("ema_model.")
}

missing, unexpected = model.load_state_dict(
    state_dict,
    strict=False
)

print(f"Missing keys: {len(missing)}")
print(f"Unexpected keys: {len(unexpected)}")

if missing:
    print("Missing:", missing[:10])

if unexpected:
    print("Unexpected:", unexpected[:10])

model.eval()

print("IndicF5 checkpoint loaded.")

# --------------------------------------------------
# Prepare reference audio
# --------------------------------------------------

print("\nPreparing reference audio...")

ref_audio, ref_text = preprocess_ref_audio_text(
    REF_AUDIO,
    REF_TEXT,
    device=device,
)

print("Reference audio ready.")

# --------------------------------------------------
# Generate Telugu
# --------------------------------------------------

print("\nGenerating Telugu speech...")
print(f"Text: {GEN_TEXT}")

audio, sample_rate, _ = infer_process(
    ref_audio,
    ref_text,
    GEN_TEXT,
    model,
    vocoder,
    mel_spec_type="vocos",
    device=device,
)

# --------------------------------------------------
# Save
# --------------------------------------------------

if audio.dtype == np.int16:
    audio = audio.astype(np.float32) / 32768.0

audio = np.asarray(audio, dtype=np.float32)

sf.write(
    OUTPUT,
    audio,
    samplerate=sample_rate,
)

print("\n===================================")
print("SUCCESS!")
print(f"Audio saved to: {OUTPUT}")
print(f"Sample rate: {sample_rate}")
print("===================================")
