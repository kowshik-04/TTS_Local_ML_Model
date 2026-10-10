"""Offline Chatterbox Telugu experiment; does not import or alter the production app."""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
import traceback
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))

import numpy as np
import soundfile as sf
import torch
from huggingface_hub import snapshot_download

from compat import EXPECTED_T3_MODEL, compatibility_scope, preflight
from tenglish import preview_tenglish, text_for_synthesis


MODEL_ID = "shankarpandala/chatterbox-telugu"
MODEL_DIR = ROOT / "models" / "chatterbox-telugu"
OUTPUT_DIR = ROOT / "outputs"
CASES = (
    (
        "telugu",
        "నమస్కారం! ఇది స్థానిక కస్టమ్ వాయిస్ పరీక్ష.",
        ROOT.parent.parent / "references" / "01_conversation.wav",
        "te",
        "direct",
    ),
    (
        "english",
        "Hello! My name is Kowshik. I am testing a local custom voice generator.",
        ROOT.parent.parent / "references" / "01_conversation.wav",
        "en",
        "direct",
    ),
    (
        "telugu_english_mixed",
        "నేను Red Hatలో Junior Consultantగా పనిచేస్తున్నాను. Linux మరియు Kubernetes ఉపయోగిస్తాను.",
        ROOT.parent.parent / "references" / "08_telugu_english.wav",
        "te",
        "direct",
    ),
    (
        "tenglish_raw",
        "Nenu chalaa baagunnaanu. Nenu Red Hat lo Linux mariyu Kubernetes vadutanu.",
        ROOT.parent.parent / "references" / "01_conversation.wav",
        "te",
        "raw",
    ),
    (
        "tenglish_converted",
        "Nenu chalaa baagunnaanu. Nenu Red Hat lo Linux mariyu Kubernetes vadutanu.",
        ROOT.parent.parent / "references" / "01_conversation.wav",
        "te",
        "converted",
    ),
)


def select_device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def peak_memory_megabytes() -> float:
    # macOS reports ru_maxrss in bytes; Linux uses KiB.
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return usage / (1024 * 1024) if usage > 10_000_000 else usage / 1024


def download_model() -> Path:
    return Path(snapshot_download(MODEL_ID, local_dir=MODEL_DIR))


def persist_results(results: dict, results_path: Path) -> None:
    temporary_path = results_path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(results_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--download",
        action="store_true",
        help="Allow the one-time model download before offline generation.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run every case, including raw and converted Tenglish comparisons.",
    )
    parser.add_argument(
        "--tenglish",
        metavar="TEXT",
        help="Preview and synthesize custom Romanized Telugu in an editable CLI stage.",
    )
    parser.add_argument(
        "--tenglish-preview",
        metavar="TELUGU_SCRIPT",
        help="Use this corrected preview with --tenglish instead of prompting.",
    )
    args = parser.parse_args()
    if args.all and args.tenglish is not None:
        parser.error("--all and --tenglish cannot be used together.")
    if args.tenglish_preview is not None and args.tenglish is None:
        parser.error("--tenglish-preview requires --tenglish.")
    if args.download:
        download_model()
    if not MODEL_DIR.is_dir():
        raise SystemExit("Model is absent. Run once with --download while online.")

    from chatterbox.mtl_tts import ChatterboxMultilingualTTS

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = select_device()
    run_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
    results_path = OUTPUT_DIR / f"results_{run_id}.json"
    results = {
        "model": MODEL_ID,
        "device": device,
        "preflight": preflight(MODEL_DIR),
        "model_load": "pending",
        "cases": [],
    }
    if args.tenglish is not None:
        selected_cases = (
            (
                "tenglish_custom",
                args.tenglish,
                ROOT.parent.parent / "references" / "01_conversation.wav",
                "te",
                "custom",
            ),
        )
    else:
        selected_cases = CASES if args.all else CASES[:1]
    persist_results(results, results_path)
    with compatibility_scope():
        try:
            model = ChatterboxMultilingualTTS.from_local(
                MODEL_DIR,
                device=device,
                t3_model=EXPECTED_T3_MODEL,
            )
        except Exception as error:
            traceback.print_exc()
            results["model_load"] = f"failed: {type(error).__name__}: {error}"
            persist_results(results, results_path)
            raise
        results["model_load"] = "succeeded"
        persist_results(results, results_path)

        for name, text, reference, language_id, text_mode in selected_cases:
            started = time.perf_counter()
            output = OUTPUT_DIR / f"{name}_{run_id}.wav"
            case_result = {
                "name": name,
                "language_id": language_id,
                "reference": str(reference),
                "output": str(output),
                "source_text": text,
            }
            try:
                if not reference.is_file():
                    raise RuntimeError(f"Missing reference audio: {reference}")
                converted_text = None
                if text_mode in {"converted", "custom"}:
                    preview = preview_tenglish(text).converted_text
                    if text_mode == "custom":
                        print(f"Original Tenglish: {text}")
                        print(f"Telugu-script preview: {preview}")
                        if args.tenglish_preview is not None:
                            preview = args.tenglish_preview
                        elif sys.stdin.isatty():
                            edited = input(
                                "Edit the Telugu preview (press Enter to accept it): "
                            )
                            if edited.strip():
                                preview = edited
                    if not preview.strip():
                        raise ValueError("The Telugu-script preview cannot be empty.")
                    case_result["converted_text"] = preview
                    converted_text = preview
                synthesis_text = text_for_synthesis(text, converted_text)
                case_result["synthesis_text"] = synthesis_text
                audio = model.generate(
                    synthesis_text,
                    language_id=language_id,
                    audio_prompt_path=str(reference),
                )
                samples = np.asarray(audio.detach().cpu()).squeeze()
                if samples.ndim != 1 or samples.size == 0:
                    raise RuntimeError("Synthesis returned empty or non-mono audio.")
                if not np.isfinite(samples).all():
                    raise RuntimeError("Synthesis returned non-finite samples.")
                peak = float(np.abs(samples).max())
                rms = float(np.sqrt(np.mean(samples**2)))
                if rms < 1e-5 or peak > 1.05:
                    raise RuntimeError(
                        f"Synthesis returned unusable audio (peak={peak:.6f}, rms={rms:.6f})."
                    )
                with TemporaryDirectory(prefix=".poc-staging-", dir=OUTPUT_DIR) as temp_dir:
                    staged_output = Path(temp_dir) / output.name
                    sf.write(staged_output, samples, model.sr, subtype="PCM_24")
                    staged_output.replace(output)
                case_result.update(
                    {
                        "status": "succeeded",
                        "duration_seconds": round(len(samples) / model.sr, 3),
                        "elapsed_seconds": round(time.perf_counter() - started, 3),
                        "peak_memory_mb": round(peak_memory_megabytes(), 1),
                        "finite": True,
                        "peak": round(peak, 6),
                        "rms": round(rms, 6),
                    }
                )
            except Exception as error:
                traceback.print_exc()
                case_result.update(
                    {
                        "status": "failed",
                        "error": f"{type(error).__name__}: {error}",
                        "elapsed_seconds": round(time.perf_counter() - started, 3),
                        "peak_memory_mb": round(peak_memory_megabytes(), 1),
                    }
                )
            results["cases"].append(case_result)
            persist_results(results, results_path)
    results["results_file"] = str(results_path)
    print(json.dumps(results, indent=2))
    if any(case["status"] == "failed" for case in results["cases"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
