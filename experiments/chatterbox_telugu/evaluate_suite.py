"""Reproducible, offline audio evaluation for Chatterbox and IndicF5."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from uuid import uuid4

import numpy as np
import soundfile as sf

from evaluation_corpus import CASES, REFERENCE_ROOT, REFERENCE_TRANSCRIPTS, cases_as_json


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[1]
OUTPUT_ROOT = ROOT / "outputs" / "evaluations"
HUMAN_REVIEW_TEMPLATE = {
    "pronunciation_errors": [],
    "missing_words": [],
    "substitutions": [],
    "additions": [],
    "prosody_notes": None,
    "speaker_similarity_0_to_5": None,
    "reference_quality_notes": None,
}


def write_json(path: Path, value: object) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def inspect_reference(reference_id: str) -> dict[str, object]:
    audio_path = REFERENCE_ROOT / "processed" / f"{reference_id}.wav"
    transcript_path = REFERENCE_ROOT / f"{reference_id}.txt"
    samples, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
    mono = samples.mean(axis=1)
    peak = float(np.max(np.abs(mono))) if mono.size else 0.0
    rms = float(np.sqrt(np.mean(mono.astype(np.float64) ** 2))) if mono.size else 0.0
    return {
        "audio_path": str(audio_path),
        "transcript_path": str(transcript_path),
        "transcript": transcript_path.read_text(encoding="utf-8").strip(),
        "sample_rate": int(sample_rate),
        "channels": int(samples.shape[1]),
        "duration_seconds": round(len(samples) / sample_rate, 3),
        "peak": round(peak, 6),
        "rms": round(rms, 6),
        "clipped_sample_fraction": round(float(np.mean(np.abs(mono) >= 0.999)), 8),
        "near_silent_sample_fraction": round(float(np.mean(np.abs(mono) < 1e-4)), 6),
        "speaker_count_verified_by_listening": None,
        "background_noise_assessed_by_listening": None,
        "transcript_accuracy_verified_by_listening": None,
    }


def _audio_record(samples: object, sample_rate: int) -> dict[str, object]:
    audio = np.asarray(samples, dtype=np.float32).reshape(-1)
    if not audio.size:
        raise RuntimeError("Synthesis returned empty audio.")
    if not np.isfinite(audio).all():
        raise RuntimeError("Synthesis returned non-finite audio.")
    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    if rms < 1e-5 or peak > 1.05:
        raise RuntimeError(f"Invalid audio signal (peak={peak:.6f}, rms={rms:.6f}).")
    return {
        "sample_rate": int(sample_rate),
        "duration_seconds": round(len(audio) / sample_rate, 3),
        "sample_count": int(len(audio)),
        "finite": True,
        "peak": round(peak, 6),
        "rms": round(rms, 6),
    }


def _result_base(case: dict, engine_name: str, output_path: Path) -> dict:
    reference = REFERENCE_ROOT / "processed" / f"{case['reference_id']}.wav"
    return {
        "case_id": case["case_id"],
        "category": case["category"],
        "text_class": "mixed_telugu_english" if case["case_id"].startswith("mix") else "pure_telugu",
        "engine": engine_name,
        "input_text": case["text"],
        "language_id": case["language_id"],
        "reference_audio": str(reference),
        "reference_transcript": REFERENCE_TRANSCRIPTS[case["reference_id"]],
        "reference_transcript_used_by_engine": engine_name == "indicf5",
        "output_path": str(output_path),
        "human_review": dict(HUMAN_REVIEW_TEMPLATE),
    }


def run_worker(engine_name: str, manifest_path: Path, run_dir: Path, device: str) -> int:
    cases = json.loads(manifest_path.read_text(encoding="utf-8"))
    engine_dir = run_dir / engine_name
    engine_dir.mkdir(parents=True, exist_ok=True)
    result_path = engine_dir / "cases.jsonl"
    if engine_name == "chatterbox":
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS, punc_norm

        from compat import EXPECTED_T3_MODEL, compatibility_scope

        if device == "auto":
            torch = __import__("torch")
            actual_device = "mps" if torch.backends.mps.is_available() else "cpu"
        else:
            actual_device = device
        model_context = compatibility_scope()
    else:
        from tts.chunking import split_text
        from tts.indicf5_engine import IndicF5Engine, format_device
        from tts.text_normalizer import normalize_text

        actual_device = format_device()
        model_context = None

    model_load_started = time.perf_counter()
    if engine_name == "chatterbox":
        with model_context:
            model = ChatterboxMultilingualTTS.from_local(
                ROOT / "models" / "chatterbox-telugu",
                device=actual_device,
                t3_model=EXPECTED_T3_MODEL,
            )
            _run_cases(
                cases,
                engine_name,
                engine_dir,
                result_path,
                model,
                actual_device,
                punc_norm=punc_norm,
            )
    else:
        model = IndicF5Engine()
        model._load()
        _run_cases(
            cases,
            engine_name,
            engine_dir,
            result_path,
            model,
            actual_device,
            normalize_text=normalize_text,
            split_text=split_text,
        )
    (engine_dir / "worker.json").write_text(
        json.dumps(
            {
                "engine": engine_name,
                "device": actual_device,
                "model_load_seconds": round(time.perf_counter() - model_load_started, 3),
                "return_status": "completed",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


def _run_cases(
    cases: list[dict],
    engine_name: str,
    engine_dir: Path,
    result_path: Path,
    model,
    device: str,
    *,
    punc_norm=None,
    normalize_text=None,
    split_text=None,
) -> None:
    with result_path.open("a", encoding="utf-8") as output_file:
        for case in cases:
            output_path = engine_dir / f"{case['case_id']}.wav"
            result = _result_base(case, engine_name, output_path)
            started = time.perf_counter()
            try:
                if engine_name == "chatterbox":
                    cleaned_text = punc_norm(case["text"])
                    tokenizer_text = model.tokenizer.preprocess_text(
                        cleaned_text, language_id=case["language_id"].lower()
                    )
                    settings = {
                        "language_id": case["language_id"],
                        "audio_prompt_path": result["reference_audio"],
                        "exaggeration": 0.5,
                        "cfg_weight": 0.5,
                        "temperature": 0.8,
                        "repetition_penalty": 1.2,
                        "min_p": 0.05,
                        "top_p": 1.0,
                    }
                    audio = model.generate(
                        case["text"],
                        language_id=settings["language_id"],
                        audio_prompt_path=settings["audio_prompt_path"],
                        exaggeration=settings["exaggeration"],
                        cfg_weight=settings["cfg_weight"],
                        temperature=settings["temperature"],
                        repetition_penalty=settings["repetition_penalty"],
                        min_p=settings["min_p"],
                        top_p=settings["top_p"],
                    )
                    if hasattr(audio, "detach"):
                        audio = audio.detach().cpu().numpy()
                    audio = np.asarray(audio).squeeze()
                    result["text_passed_to_generate"] = case["text"]
                    result["text_after_punctuation_normalization"] = cleaned_text
                    result["model_preprocessed_text"] = tokenizer_text
                    result["generation_settings"] = settings
                    sample_rate = model.sr
                else:
                    effective_text = normalize_text(case["text"])
                    chunks = split_text(effective_text)
                    settings = {
                        "reference_transcript": result["reference_transcript"],
                        "reference_transcript_used": True,
                        "text_normalization": "NFC and whitespace only",
                        "chunk_max_bytes": 420,
                        "chunks": [chunk for chunk, _ in chunks],
                        "inter_chunk_pauses_seconds": [pause for _, pause in chunks],
                    }
                    audio, sample_rate = model.generate(
                        result["reference_audio"],
                        result["reference_transcript"],
                        case["text"],
                    )
                    result["text_passed_to_generate"] = case["text"]
                    result["model_preprocessed_text"] = effective_text
                    result["generation_settings"] = settings

                metrics = _audio_record(audio, sample_rate)
                sf.write(output_path, audio, sample_rate, subtype="PCM_24")
                result.update(metrics)
                result["status"] = "succeeded"
                result["device"] = device
            except Exception as error:
                result.update(
                    {
                        "status": "failed",
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
            result["elapsed_seconds"] = round(time.perf_counter() - started, 3)
            output_file.write(json.dumps(result, ensure_ascii=False) + "\n")
            output_file.flush()
            os.fsync(output_file.fileno())


def _load_worker_results(run_dir: Path) -> list[dict]:
    results: list[dict] = []
    for engine in ("chatterbox", "indicf5"):
        result_path = run_dir / engine / "cases.jsonl"
        if result_path.is_file():
            results.extend(
                json.loads(line)
                for line in result_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
    return results


def _engine_python(engine_name: str) -> Path:
    variable = "INDICF5_PYTHON" if engine_name == "indicf5" else "CHATTERBOX_PYTHON"
    default = (
        PROJECT_ROOT / ".venv" / "bin" / "python"
        if engine_name == "indicf5"
        else Path(sys.executable)
    )
    path = Path(os.environ.get(variable, str(default))).expanduser()
    if not path.is_file():
        raise RuntimeError(
            f"{engine_name} Python interpreter not found at {path}; set {variable}."
        )
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, help="Evaluate only the first N corpus cases.")
    parser.add_argument(
        "--engines",
        choices=("both", "chatterbox", "indicf5"),
        default="both",
    )
    parser.add_argument("--device", help="Override the Chatterbox device (default: auto).")
    parser.add_argument("--worker-engine", choices=("chatterbox", "indicf5"), help=argparse.SUPPRESS)
    parser.add_argument("--manifest", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--run-dir", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_engine:
        if args.manifest is None or args.run_dir is None:
            parser.error("Worker mode requires --manifest and --run-dir.")
        raise SystemExit(run_worker(args.worker_engine, args.manifest, args.run_dir, args.device or "auto"))

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be a positive integer.")
    selected_cases = list(CASES[:args.limit] if args.limit else CASES)
    run_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
    run_dir = OUTPUT_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = run_dir / "manifest.json"
    write_json(manifest_path, cases_as_json()[: len(selected_cases)])
    references = {
        reference_id: inspect_reference(reference_id)
        for reference_id in sorted({case.reference_id for case in selected_cases})
    }
    results_path = run_dir / "results.json"
    engines = (
        ["chatterbox", "indicf5"]
        if args.engines == "both"
        else [args.engines]
    )
    results: dict[str, object] = {
        "run_id": run_id,
        "created_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "offline": True,
        "suite_version": 1,
        "case_count": len(selected_cases),
        "engines": engines,
        "references": references,
        "cases": cases_as_json()[: len(selected_cases)],
        "results": [],
        "engine_runs": [],
        "human_review_status": "pending; listening required",
        "interpretation_note": (
            "Generation, duration, and signal validity are not pronunciation or "
            "speaker-similarity scores. Human listening is required."
        ),
    }
    write_json(results_path, results)
    base_environment = os.environ.copy()
    base_environment["HF_HUB_OFFLINE"] = "1"
    base_environment["TRANSFORMERS_OFFLINE"] = "1"

    for engine_name in engines:
        engine_dir = run_dir / engine_name
        engine_dir.mkdir()
        log_path = engine_dir / "worker.log"
        environment = base_environment.copy()
        if engine_name == "indicf5":
            previous_pythonpath = environment.get("PYTHONPATH", "")
            environment["PYTHONPATH"] = os.pathsep.join(
                part for part in (str(PROJECT_ROOT), previous_pythonpath) if part
            )
        command = [
            str(_engine_python(engine_name)),
            str(Path(__file__).resolve()),
            "--worker-engine",
            engine_name,
            "--manifest",
            str(manifest_path),
            "--run-dir",
            str(run_dir),
            "--device",
            args.device or "auto",
        ]
        started = time.perf_counter()
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        log_path.write_text(
            completed.stdout + completed.stderr,
            encoding="utf-8",
        )
        worker_result = {
            "engine": engine_name,
            "command": command,
            "log_path": str(log_path),
            "exit_code": completed.returncode,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }
        results["engine_runs"].append(worker_result)
        results["results"] = _load_worker_results(run_dir)
        write_json(results_path, results)

    expected = len(selected_cases) * len(engines)
    completed_results = results["results"]
    succeeded = sum(result["status"] == "succeeded" for result in completed_results)
    results["summary"] = {
        "expected_engine_cases": expected,
        "recorded_engine_cases": len(completed_results),
        "succeeded": succeeded,
        "failed": expected - succeeded,
        "unrecorded_engine_cases": expected - len(completed_results),
    }
    results["status"] = "completed" if succeeded == expected else "partial"
    review_cases = []
    for result in completed_results:
        if result["status"] != "succeeded":
            continue
        review_cases.append(
            {
                "case_id": result["case_id"],
                "engine": result["engine"],
                "text": result["input_text"],
                "audio_path": result["output_path"],
                **HUMAN_REVIEW_TEMPLATE,
            }
        )
    write_json(run_dir / "human_review_template.json", review_cases)
    write_json(results_path, results)
    print(
        json.dumps(
            {
                "run_id": run_id,
                "status": results["status"],
                "results_path": str(results_path),
                "human_review_template": str(run_dir / "human_review_template.json"),
                "summary": results["summary"],
                "engine_runs": results["engine_runs"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if succeeded != expected:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
