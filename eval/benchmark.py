#!/usr/bin/env python3
"""Repeatable technical benchmark for the Local Voice Studio HTTP API.

The benchmark measures wall latency, time to first audio byte, generation time
reported by the server, real-time factor and concurrency behaviour.  It does not
measure pronunciation, naturalness or speaker similarity; use the generated
audio and ``human_review_template.json`` for that.

Examples:
    TTS_API_KEY=... python eval/benchmark.py --voice-id voice_ab12cd34
    TTS_API_KEY=... python eval/benchmark.py --voice-id voice_ab12cd34 \
        --transport stream --concurrency 4 --runs 3 --group telugu
    python eval/benchmark.py --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import platform
import statistics
import sys
import time
import uuid
import wave
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.corpus import CASES, ENGLISH, MIXED, TELUGU, Case, select_cases  # noqa: E402


BENCHMARK_VERSION = "1.0"
TRANSPORTS = ("sync", "stream")


@dataclass
class Trial:
    case_id: str
    category: str
    language: str
    transport: str
    run: int
    ok: bool
    status_code: int = 0
    error: str = ""
    error_code: str = ""
    wall_ms: float = 0.0
    ttfb_ms: float = 0.0
    generation_ms: float | None = None
    audio_ms: float = 0.0
    bytes: int = 0
    rtf: float | None = None
    audio_per_wall: float | None = None
    cache_hit: bool = False
    engine: str = ""
    experimental: bool = False
    warning_count: int = 0
    chars: int = 0
    audio_file: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


def _mean(values: list[float]) -> float | None:
    return round(statistics.fmean(values), 2) if values else None


def _stats(trials: list[Trial]) -> dict[str, Any]:
    if not trials:
        return {"trials": 0}
    successful = [trial for trial in trials if trial.ok]
    walls = [trial.wall_ms for trial in successful]
    ttfbs = [trial.ttfb_ms for trial in successful]
    rtfs = [trial.rtf for trial in successful if trial.rtf is not None]
    audio_ms = sum(trial.audio_ms for trial in successful)
    wall_ms = sum(trial.wall_ms for trial in successful)
    return {
        "trials": len(trials),
        "successful": len(successful),
        "failed": len(trials) - len(successful),
        "success_rate": round(len(successful) / len(trials), 4),
        "wall_ms_p50": _percentile(walls, 0.50),
        "wall_ms_p95": _percentile(walls, 0.95),
        "wall_ms_mean": _mean(walls),
        "ttfb_ms_p50": _percentile(ttfbs, 0.50),
        "ttfb_ms_p95": _percentile(ttfbs, 0.95),
        "rtf_p50": _percentile(rtfs, 0.50),
        "rtf_p95": _percentile(rtfs, 0.95),
        "audio_seconds": round(audio_ms / 1000.0, 2),
        "wall_seconds": round(wall_ms / 1000.0, 2),
        "audio_seconds_per_wall_second": round(audio_ms / wall_ms, 3) if wall_ms else None,
        "cache_hits": sum(1 for trial in successful if trial.cache_hit),
        "chars": sum(trial.chars for trial in successful),
    }


def _group_stats(trials: list[Trial], key: str) -> dict[str, Any]:
    groups: dict[str, list[Trial]] = {}
    for trial in trials:
        groups.setdefault(str(getattr(trial, key)), []).append(trial)
    return {name: _stats(items) for name, items in sorted(groups.items())}


def _error_details(status_code: int, body: bytes) -> tuple[str, str]:
    text = body.decode("utf-8", errors="replace").strip()
    try:
        payload = json.loads(text)
        error = payload.get("error") or {}
        return str(error.get("code", "")), str(error.get("message", text))[:800]
    except (ValueError, AttributeError):
        return "", text[:800] or f"HTTP {status_code}"


def _write_wav(path: Path, pcm: bytes, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    even = pcm[: len(pcm) - (len(pcm) % 2)]
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(even)


def _save_audio(
    audio_dir: Path,
    *,
    case: Case,
    transport: str,
    run: int,
    audio: bytes,
    audio_format: str,
) -> str:
    audio_dir.mkdir(parents=True, exist_ok=True)
    if transport == "stream" or audio_format == "pcm":
        path = audio_dir / f"{case.case_id}-r{run}-{transport}.wav"
        _write_wav(path, audio, 24_000)
    elif audio_format == "wav":
        path = audio_dir / f"{case.case_id}-r{run}-{transport}.wav"
        path.write_bytes(audio)
    else:
        suffix = "mp3" if audio_format == "mp3" else "opus"
        path = audio_dir / f"{case.case_id}-r{run}-{transport}.{suffix}"
        path.write_bytes(audio)
    return str(path)


async def _run_trial(
    client: httpx.AsyncClient,
    *,
    args: argparse.Namespace,
    case: Case,
    transport: str,
    run: int,
    audio_dir: Path | None,
) -> Trial:
    payload = {
        "text": case.text,
        "language": case.language,
        "voice_id": args.voice_id,
        "engine": "auto",
        "use_cache": not args.no_cache,
    }
    if args.profile:
        payload["profile"] = args.profile
    trial = Trial(
        case_id=case.case_id,
        category=case.category,
        language=case.language,
        transport=transport,
        run=run,
        ok=False,
        chars=len(case.text),
    )
    started = time.perf_counter()
    try:
        if transport == "stream":
            async with client.stream("POST", "/v1/tts/stream", json=payload) as response:
                status = response.status_code
                trial.status_code = status
                headers = response.headers
                trial.engine = headers.get("x-engine", "")
                trial.cache_hit = headers.get("x-cache", "").lower() == "hit"
                trial.experimental = headers.get("x-experimental", "").lower() == "true"
                trial.warning_count = int(headers.get("x-warning-count", "0") or 0)
                if status >= 400:
                    body = await response.aread()
                    trial.error_code, trial.error = _error_details(status, body)
                    return trial
                chunks: list[bytes] = []
                first_byte_at: float | None = None
                async for chunk in response.aiter_bytes():
                    if not chunk:
                        continue
                    if first_byte_at is None:
                        first_byte_at = time.perf_counter()
                    chunks.append(chunk)
            finished = time.perf_counter()
            audio = b"".join(chunks)
            trial.bytes = len(audio)
            trial.audio_ms = len(audio) / 2 / 24_000 * 1000
            trial.wall_ms = (finished - started) * 1000
            trial.ttfb_ms = ((first_byte_at or finished) - started) * 1000
            if audio_dir is not None:
                trial.audio_file = _save_audio(
                    audio_dir, case=case, transport=transport, run=run,
                    audio=audio, audio_format="pcm",
                )
        else:
            response = await client.post(
                "/v1/tts", json={**payload, "format": args.format}
            )
            finished = time.perf_counter()
            trial.status_code = response.status_code
            trial.engine = response.headers.get("x-engine", "")
            trial.cache_hit = response.headers.get("x-cache", "").lower() == "hit"
            trial.experimental = response.headers.get("x-experimental", "").lower() == "true"
            trial.warning_count = int(response.headers.get("x-warning-count", "0") or 0)
            if response.status_code >= 400:
                trial.error_code, trial.error = _error_details(
                    response.status_code, response.content
                )
                return trial
            audio = response.content
            trial.bytes = len(audio)
            trial.audio_ms = float(response.headers.get("x-audio-duration-ms", "0") or 0)
            trial.wall_ms = (finished - started) * 1000
            trial.ttfb_ms = trial.wall_ms
            if audio_dir is not None:
                trial.audio_file = _save_audio(
                    audio_dir, case=case, transport=transport, run=run,
                    audio=audio, audio_format=args.format,
                )

        generation = headers.get("x-generation-ms") if transport == "stream" else response.headers.get("x-generation-ms")
        if generation:
            trial.generation_ms = float(generation)
        if trial.generation_ms is not None and trial.audio_ms:
            trial.rtf = trial.generation_ms / trial.audio_ms
        if trial.wall_ms and trial.audio_ms:
            trial.audio_per_wall = trial.audio_ms / trial.wall_ms
        trial.ok = True
    except httpx.TimeoutException as error:
        trial.wall_ms = (time.perf_counter() - started) * 1000
        trial.error = f"timeout: {error}"
    except httpx.HTTPError as error:
        trial.wall_ms = (time.perf_counter() - started) * 1000
        trial.error = f"transport error: {error}"
    except Exception as error:  # keep the benchmark running so one case cannot abort a run
        trial.wall_ms = (time.perf_counter() - started) * 1000
        trial.error = f"{type(error).__name__}: {error}"
    return trial


async def _run_all(
    args: argparse.Namespace,
    cases: list[Case],
    transports: tuple[str, ...],
    audio_dir: Path | None,
) -> list[Trial]:
    base_url = args.base_url.rstrip("/")
    if base_url.endswith("/v1"):
        base_url = base_url[:-3]
    headers = {"Authorization": f"Bearer {args.api_key}"}
    timeout = httpx.Timeout(args.timeout, connect=15.0)
    semaphore = asyncio.Semaphore(args.concurrency)
    trials: list[Trial] = []

    async with httpx.AsyncClient(base_url=base_url, headers=headers, timeout=timeout) as client:
        if not args.no_warmup:
            print("Warming up with the first case (not recorded)…", file=sys.stderr)
            await _run_trial(
                client, args=args, case=cases[0], transport=transports[0],
                run=0, audio_dir=None,
            )

        async def one(case: Case, transport: str, run: int) -> Trial:
            async with semaphore:
                trial = await _run_trial(
                    client, args=args, case=case, transport=transport, run=run,
                    audio_dir=audio_dir,
                )
            status = "ok" if trial.ok else f"FAIL {trial.error_code or trial.error}"
            print(
                f"{case.case_id} {transport} run {run}: {status} "
                f"wall={trial.wall_ms:.0f}ms ttfb={trial.ttfb_ms:.0f}ms",
                file=sys.stderr,
            )
            return trial

        tasks = [
            one(case, transport, run)
            for run in range(1, args.runs + 1)
            for case in cases
            for transport in transports
        ]
        trials = list(await asyncio.gather(*tasks))
    trials.sort(key=lambda item: (item.case_id, item.run, item.transport))
    return trials


def _summary_markdown(manifest: dict[str, Any], summary: dict[str, Any]) -> str:
    lines = [
        "# Local Voice Studio benchmark",
        "",
        f"- Generated: `{manifest['generated_at']}`",
        f"- API: `{manifest['base_url']}`",
        f"- Voice: `{manifest['voice_id']}`",
        f"- Profile: `{manifest['profile'] or 'server default'}`",
        f"- Cases: {manifest['case_count']} · runs: {manifest['runs']} · concurrency: {manifest['concurrency']}",
        f"- Transports: {', '.join(manifest['transports'])}",
        "",
        "## Overall",
        "",
        "| Metric | Value |",
        "|---|---|",
    ]
    overall = summary["overall"]
    for key in (
        "trials", "successful", "failed", "success_rate", "wall_ms_p50",
        "wall_ms_p95", "ttfb_ms_p50", "ttfb_ms_p95", "rtf_p50", "rtf_p95",
        "audio_seconds", "wall_seconds", "audio_seconds_per_wall_second",
        "cache_hits", "chars",
    ):
        value = overall.get(key)
        lines.append(f"| {key} | {value} |")
    lines.extend(["", "## By transport", ""])
    lines.append("| Transport | Trials | Success | TTFR p50 | Wall p50 | RTF p50 | Audio/wall |")
    lines.append("|---|---|---|---|---|---|---|")
    for name, stats in summary["by_transport"].items():
        lines.append(
            f"| {name} | {stats['trials']} | {stats['success_rate']:.1%} | "
            f"{_fmt(stats.get('ttfb_ms_p50'))} | {_fmt(stats.get('wall_ms_p50'))} | "
            f"{_fmt(stats.get('rtf_p50'))} | {_fmt(stats.get('audio_seconds_per_wall_second'))} |"
        )
    lines.extend(["", "## By language", ""])
    lines.append("| Language | Trials | Success | Wall p50 | RTF p50 |")
    lines.append("|---|---|---|---|---|")
    for name, stats in summary["by_language"].items():
        lines.append(
            f"| {name} | {stats['trials']} | {stats.get('success_rate', 0):.1%} | "
            f"{_fmt(stats.get('wall_ms_p50'))} | {_fmt(stats.get('rtf_p50'))} |"
        )
    failures = [
        trial for trial in summary.pop("_trials", []) if not trial.ok
    ]
    if failures:
        lines.extend(["", "## Failures", ""])
        for trial in failures:
            lines.append(
                f"- `{trial.case_id}` {trial.transport} run {trial.run}: "
                f"`{trial.error_code or trial.status_code}` {trial.error}"
            )
    lines.extend(
        [
            "",
            "> These are technical measurements only. They do not establish pronunciation,",
            "> naturalness, speaker similarity or production readiness. Listen to the saved",
            "> audio and complete `human_review_template.json` before making quality claims.",
            "",
        ]
    )
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _review_template(trials: list[Trial], output_dir: Path) -> dict[str, Any]:
    items = []
    for trial in trials:
        if not trial.ok:
            continue
        audio_path = Path(trial.audio_file) if trial.audio_file else None
        items.append(
            {
                "case_id": trial.case_id,
                "language": trial.language,
                "category": trial.category,
                "transport": trial.transport,
                "run": trial.run,
                "engine": trial.engine,
                "text": next(
                    case.text for case in CASES if case.case_id == trial.case_id
                ),
                "audio_file": (
                    str(audio_path.relative_to(output_dir)) if audio_path else None
                ),
                "intelligibility_1_5": None,
                "speaker_similarity_1_5": None,
                "prosody_1_5": None,
                "artifacts": "",
                "notes": "",
            }
        )
    return {
        "instructions": (
            "Listen to every successful case on headphones. Score 1 (bad) to 5 "
            "(excellent). Compare speaker similarity with the enrolled reference "
            "voice. Do not report automated metrics as quality evidence."
        ),
        "items": items,
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default=os.environ.get("TTS_API_BASE_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--api-key", default=os.environ.get("TTS_API_KEY") or os.environ.get("VOICE_API_KEY"))
    parser.add_argument("--voice-id", default=os.environ.get("TTS_VOICE_ID"))
    parser.add_argument("--profile", choices=("quality", "balanced", "realtime", "custom"))
    parser.add_argument("--transport", choices=("sync", "stream", "both"), default="both")
    parser.add_argument("--format", choices=("wav", "mp3", "pcm"), default="wav", help="Sync response format.")
    parser.add_argument("--group", choices=("all", "telugu", "mixed", "english"), default="all")
    parser.add_argument("--case-id", action="append", dest="case_ids", help="Run only this case id; repeatable.")
    parser.add_argument("--limit", type=int, help="Run the first N selected cases.")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "eval" / "results")
    parser.add_argument("--no-cache", action="store_true", help="Send use_cache=false.")
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--no-save-audio", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _select_group(group: str, cases: list[Case]) -> list[Case]:
    if group == "telugu":
        return [case for case in cases if case.language == TELUGU]
    if group == "mixed":
        return [case for case in cases if case.language == MIXED]
    if group == "english":
        return [case for case in cases if case.language == ENGLISH]
    return cases


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cases = select_cases(args.case_ids)
        cases = _select_group(args.group, cases)
        if args.limit is not None:
            if args.limit < 1:
                raise ValueError("limit must be >= 1")
            cases = cases[: args.limit]
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    if not cases:
        print("error: no cases selected", file=sys.stderr)
        return 2
    if args.runs < 1 or args.concurrency < 1:
        print("error: --runs and --concurrency must be >= 1", file=sys.stderr)
        return 2
    if args.dry_run:
        for case in cases:
            print(f"{case.case_id}\t{case.language}\t{case.category}\t{case.text}")
        print(f"\n{len(cases)} case(s); use --limit/--group/--case-id to narrow.", file=sys.stderr)
        return 0
    if not args.api_key:
        print("error: provide --api-key or set TTS_API_KEY", file=sys.stderr)
        return 2
    if not args.voice_id:
        print("error: provide --voice-id or set TTS_VOICE_ID", file=sys.stderr)
        return 2

    transports = TRANSPORTS if args.transport == "both" else (args.transport,)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:4]
    output_dir = args.output_dir / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = None if args.no_save_audio else output_dir / "audio"

    print(
        f"Running {len(cases) * args.runs * len(transports)} trial(s) against "
        f"{args.base_url} with concurrency={args.concurrency}",
        file=sys.stderr,
    )
    started = time.perf_counter()
    trials = asyncio.run(_run_all(args, cases, transports, audio_dir))
    elapsed = time.perf_counter() - started

    summary = {
        "overall": _stats(trials),
        "by_transport": _group_stats(trials, "transport"),
        "by_language": _group_stats(trials, "language"),
        "by_category": _group_stats(trials, "category"),
        "_trials": trials,
    }
    manifest = {
        "benchmark_version": BENCHMARK_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "base_url": args.base_url,
        "voice_id": args.voice_id,
        "profile": args.profile,
        "transports": list(transports),
        "format": args.format,
        "cache_enabled": not args.no_cache,
        "warmup": not args.no_warmup,
        "concurrency": args.concurrency,
        "runs": args.runs,
        "case_count": len(cases),
        "cases": [case.as_dict() for case in cases],
        "elapsed_seconds": round(elapsed, 2),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    public_summary = {key: value for key, value in summary.items() if key != "_trials"}
    _write_json(output_dir / "manifest.json", manifest)
    _write_json(output_dir / "results.json", [trial.as_dict() for trial in trials])
    _write_json(output_dir / "summary.json", public_summary)
    _write_json(output_dir / "human_review_template.json", _review_template(trials, output_dir))
    (output_dir / "summary.md").write_text(_summary_markdown(manifest, summary), encoding="utf-8")

    overall = summary["overall"]
    print(
        f"Done: {overall['successful']}/{overall['trials']} successful · "
        f"wall p50={_fmt(overall.get('wall_ms_p50'))} ms · "
        f"TTFB p50={_fmt(overall.get('ttfb_ms_p50'))} ms · "
        f"audio/wall={_fmt(overall.get('audio_seconds_per_wall_second'))}",
        file=sys.stderr,
    )
    print(str(output_dir))
    return 0 if overall["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
