#!/usr/bin/env python3
"""Direct Modal smoke/comparison test for Qwen3-ASR GPU deployments.

This intentionally bypasses the CPU orchestrator and product app. It downloads a
real Apple Podcast sample, cuts deterministic clips, calls the original GPU app
and the isolated Qwen3 L4/vLLM GPU app, then writes comparison artifacts under
/tmp.
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List


DEFAULT_APPLE_URL = (
    "https://podcasts.apple.com/cn/podcast/%E5%B0%8Flin%E8%AF%B4/"
    "id1752639453?i=1000767368971"
)
DEFAULT_OUTPUT_DIR = Path("/tmp/qwen3-asr-smoke/apple-1000767368971")
DEFAULT_OLD_APP = "transcribe-modal-gpu"
DEFAULT_NEW_APP = "transcribe-modal-gpu-qwen3-l4-vllm"
FUNCTION_NAME = "transcribe_and_diarization_audio_function"
QWEN_PROTECTED_TERMS = ("阿莫西林", "阿司匹林", "布洛芬")
PUNCTUATION_CHARS = set("，。！？；：、,.!?;:")

MODAL_GPU_PRICE_PER_SECOND = {
    "T4": 0.000164,
    "L4": 0.000222,
    "A10": 0.000306,
    "L40S": 0.000542,
    "A100-40GB": 0.000583,
    "A100-80GB": 0.000694,
    "H100": 0.001097,
}
MODAL_CPU_PRICE_PER_SECOND = 0.0000131
MODAL_MEMORY_GIB_PRICE_PER_SECOND = 0.00000222


def _log(message: str) -> None:
    print(message, flush=True)


def _urlopen_bytes(url: str, *, timeout: int = 60) -> bytes:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
            )
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def find_apple_audio_url(apple_url: str) -> str:
    _log(f"Fetching Apple Podcast page: {apple_url}")
    page = _urlopen_bytes(apple_url, timeout=30).decode("utf-8", errors="replace")
    urls = re.findall(r'https://[^\s^"]+(?:\.mp3|\.m4a)', page)
    if not urls:
        raise RuntimeError(
            "Could not find an .mp3/.m4a URL in the Apple page. "
            "Download the audio manually and pass --audio-file."
        )
    candidate = html.unescape(urls[-1])
    nested = re.findall(r'=https?://[^\s^"]+(?:\.mp3|\.m4a)', candidate)
    if nested:
        candidate = nested[-1][1:]
    _log(f"Resolved audio URL: {candidate[:180]}")
    return candidate


def download_audio(audio_url: str, destination: Path) -> None:
    if destination.exists() and destination.stat().st_size > 0:
        _log(f"Reusing downloaded audio: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    _log(f"Downloading audio to {destination}")
    request = urllib.request.Request(audio_url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=120) as response, destination.open("wb") as out:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
    _log(f"Downloaded {destination.stat().st_size / (1024 * 1024):.2f} MB")


def run_checked(cmd: List[str]) -> None:
    _log("+ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def cut_clip(source: Path, destination: Path, seconds: int) -> None:
    if destination.exists() and destination.stat().st_size > 0:
        _log(f"Reusing clip: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    run_checked(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(source),
            "-t",
            str(seconds),
            "-vn",
            "-acodec",
            "libmp3lame",
            "-ar",
            "16000",
            "-ac",
            "1",
            str(destination),
        ]
    )


def format_srt_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    if millis >= 1000:
        secs += 1
        millis -= 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def write_srt(result: Dict[str, Any], path: Path) -> None:
    segments = result.get("segments") or []
    with path.open("w", encoding="utf-8") as out:
        index = 1
        for segment in segments:
            text = str(segment.get("text") or "").strip()
            if not text:
                continue
            out.write(f"{index}\n")
            out.write(
                f"{format_srt_time(segment.get('start', 0))} --> "
                f"{format_srt_time(segment.get('end', 0))}\n"
            )
            out.write(text + "\n\n")
            index += 1


def modal_second_price(gpu_type: str, cpu_count: float, memory_gib: float) -> float:
    normalized_gpu = str(gpu_type or "").strip().upper()
    if normalized_gpu not in MODAL_GPU_PRICE_PER_SECOND:
        raise RuntimeError(
            f"Unknown GPU type for cost estimate: {gpu_type!r}. "
            f"Known: {', '.join(sorted(MODAL_GPU_PRICE_PER_SECOND))}"
        )
    return (
        MODAL_GPU_PRICE_PER_SECOND[normalized_gpu]
        + float(cpu_count) * MODAL_CPU_PRICE_PER_SECOND
        + float(memory_gib) * MODAL_MEMORY_GIB_PRICE_PER_SECOND
    )


def attach_benchmark_metrics(
    result: Dict[str, Any],
    *,
    clip_seconds: int,
    gpu_type: str,
    cpu_count: float,
    memory_gib: float,
    region_multiplier: float,
) -> None:
    elapsed = float(result.get("_elapsed_seconds") or 0.0)
    per_second = modal_second_price(gpu_type, cpu_count, memory_gib) * region_multiplier
    result["_benchmark"] = {
        "audio_seconds": clip_seconds,
        "elapsed_seconds": elapsed,
        "audio_seconds_per_processing_second": (
            clip_seconds / elapsed if elapsed > 0 else 0.0
        ),
        "modal_cost_usd": elapsed * per_second,
        "modal_price_per_second_usd": per_second,
        "gpu_type": str(gpu_type).strip().upper(),
        "cpu_count": cpu_count,
        "memory_gib": memory_gib,
        "region_multiplier": region_multiplier,
        "oom_detected": "out of memory" in str(result.get("error_message") or "").lower()
        or "cuda oom" in str(result.get("error_message") or "").lower()
        or "oom" in str(result.get("error_message") or "").lower(),
    }


def build_payload(clip_path: Path, clip_seconds: int, case: Dict[str, Any]) -> Dict[str, Any]:
    audio_b64 = base64.b64encode(clip_path.read_bytes()).decode("ascii")
    payload: Dict[str, Any] = {
        "request_type": "transcribe",
        "audio_file_data": audio_b64,
        "audio_file_name": clip_path.name,
        "model_size": case.get("model_size", "large-v3"),
        "language": case.get("language", "auto"),
        "enable_speaker_diarization": False,
        "chunk_start_time": 0,
        "chunk_end_time": clip_seconds,
    }
    if case.get("asr_backend"):
        payload["asr_backend"] = case["asr_backend"]
    if case.get("asr_model_id"):
        payload["asr_model_id"] = case["asr_model_id"]
    if case.get("qwen_context"):
        payload["qwen_context"] = case["qwen_context"]
    return payload


def call_modal(app_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    import modal

    fn = modal.Function.from_name(app_name, FUNCTION_NAME)
    started = time.time()
    if hasattr(fn, "remote"):
        result = fn.remote(payload)
    else:
        result = fn.call(payload)
    elapsed = time.time() - started
    if not isinstance(result, dict):
        raise RuntimeError(f"Modal function returned non-dict result: {type(result)!r}")
    result["_elapsed_seconds"] = elapsed
    return result


def validate_result(name: str, clip_seconds: int, result: Dict[str, Any]) -> List[str]:
    issues: List[str] = []
    status = result.get("processing_status")
    if status != "success":
        issues.append(f"{name}: processing_status={status!r} error={result.get('error_message')!r}")
        return issues

    segments = result.get("segments") or []
    if not segments:
        issues.append(f"{name}: no segments")
        return issues

    previous_start = -1.0
    for index, segment in enumerate(segments):
        start = float(segment.get("start", 0))
        end = float(segment.get("end", 0))
        if start < -0.001 or end < -0.001:
            issues.append(f"{name}: negative timestamp at segment {index}")
        if end < start:
            issues.append(f"{name}: end before start at segment {index}")
        if start < previous_start - 0.001:
            issues.append(f"{name}: non-monotonic timestamp at segment {index}")
        previous_start = start

    max_end = max(float(segment.get("end", 0)) for segment in segments)
    if max_end > clip_seconds + 2.0:
        issues.append(f"{name}: max end {max_end:.2f}s exceeds clip {clip_seconds}s")

    model_used = str(result.get("model_used") or "")
    if name.endswith("new_qwen3") and not model_used.startswith("qwen3_asr:"):
        issues.append(f"{name}: expected qwen3_asr model_used, got {model_used!r}")
    if name.endswith("fallback") and not model_used.startswith("qwen3_asr_fallback_whisper:"):
        issues.append(f"{name}: expected qwen fallback model_used, got {model_used!r}")

    text_len = len(str(result.get("text") or "").strip())
    density = len(segments) / max(clip_seconds, 1)
    result["_validation"] = {
        "text_len": text_len,
        "segments_per_second": density,
        "max_end": max_end,
    }
    if name.endswith("new_qwen3") and density > 2.0:
        issues.append(
            f"{name}: segment density {density:.2f}/s is too high; "
            "Qwen timestamps likely need subtitle-level aggregation"
        )
    if name.endswith("new_qwen3") and density < 0.3:
        issues.append(
            f"{name}: segment density {density:.2f}/s is too low; "
            "Qwen subtitle aggregation is likely too coarse"
        )
    if name.endswith("new_qwen3"):
        segment_texts = [str(segment.get("text") or "") for segment in segments]
        joined_segments = "".join(segment_texts)
        for term in QWEN_PROTECTED_TERMS:
            if term in joined_segments and not any(term in text for text in segment_texts):
                issues.append(f"{name}: protected term split across subtitles: {term}")

        full_text = str(result.get("text") or "")
        segment_punctuation_count = sum(
            1 for text in segment_texts for char in text if char in PUNCTUATION_CHARS
        )
        full_text_punctuation_count = sum(1 for char in full_text if char in PUNCTUATION_CHARS)
        if full_text_punctuation_count and not segment_punctuation_count:
            issues.append(f"{name}: Qwen subtitles lost punctuation from ASR text")
    return issues


def comparison_cases(
    old_app: str,
    new_app: str,
    qwen_context: str = "",
    old_cost_gpu: str = "T4",
    new_cost_gpu: str = "L4",
) -> List[Dict[str, Any]]:
    return [
        {
            "name": "old_whisper",
            "app": old_app,
            "asr_backend": "whisper",
            "language": "auto",
            "cost_gpu": old_cost_gpu,
        },
        {
            "name": "new_whisper",
            "app": new_app,
            "asr_backend": "whisper",
            "language": "auto",
            "cost_gpu": new_cost_gpu,
        },
        {
            "name": "new_qwen3",
            "app": new_app,
            "asr_backend": "qwen3_asr",
            "language": "auto",
            "qwen_context": qwen_context,
            "cost_gpu": new_cost_gpu,
        },
        {
            "name": "fallback",
            "app": new_app,
            "asr_backend": "qwen3_asr",
            "language": "th",
            "cost_gpu": new_cost_gpu,
        },
    ]


def selected_clips(output_dir: Path, names: Iterable[str]) -> Dict[str, tuple[Path, int]]:
    all_clips = {
        "smoke": (output_dir / "clips" / "smoke_90s.mp3", 90),
        "long": (output_dir / "clips" / "long_300s.mp3", 300),
        "stress": (output_dir / "clips" / "stress_1800s.mp3", 1800),
    }
    return {name: all_clips[name] for name in names}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apple-url", default=DEFAULT_APPLE_URL)
    parser.add_argument("--audio-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--old-app", default=os.getenv("OLD_GPU_APP_NAME", DEFAULT_OLD_APP))
    parser.add_argument("--new-app", default=os.getenv("NEW_GPU_APP_NAME", DEFAULT_NEW_APP))
    parser.add_argument(
        "--clips",
        default="smoke,long,stress",
        help="Comma-separated clip names: smoke,long,stress",
    )
    parser.add_argument(
        "--cases",
        default="old_whisper,new_whisper,new_qwen3,fallback",
        help="Comma-separated case names to run",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only download and cut clips; do not call Modal.",
    )
    parser.add_argument(
        "--qwen-context",
        default=os.getenv("QWEN_ASR_CONTEXT", ""),
        help="Optional Qwen3-ASR context hints for proper nouns/acronyms.",
    )
    parser.add_argument(
        "--old-cost-gpu",
        default=os.getenv("OLD_COST_GPU", "T4"),
        help="GPU type used for old-app Modal cost estimates.",
    )
    parser.add_argument(
        "--new-cost-gpu",
        default=os.getenv("NEW_COST_GPU", "L4"),
        help="GPU type used for new-app Modal cost estimates.",
    )
    parser.add_argument(
        "--cost-cpu-count",
        type=float,
        default=float(os.getenv("MODAL_CPU", "4")),
        help="CPU count used for Modal cost estimates.",
    )
    parser.add_argument(
        "--cost-memory-gib",
        type=float,
        default=float(os.getenv("MODAL_MEMORY", "8192")) / 1024.0,
        help="Memory GiB used for Modal cost estimates.",
    )
    parser.add_argument(
        "--region-multiplier",
        type=float,
        default=float(os.getenv("MODAL_REGION_PRICE_MULTIPLIER", "1.0")),
        help="Modal region price multiplier. Use 1.0 when the GPU function has no region pin.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.audio_file:
        original = args.audio_file
        if not original.exists():
            raise RuntimeError(f"--audio-file does not exist: {original}")
    else:
        original = output_dir / "original.m4a"
        audio_url = find_apple_audio_url(args.apple_url)
        download_audio(audio_url, original)

    clip_names = [item.strip() for item in args.clips.split(",") if item.strip()]
    clips = selected_clips(output_dir, clip_names)
    for clip_path, seconds in clips.values():
        cut_clip(original, clip_path, seconds)

    if args.prepare_only:
        _log("prepare-only complete")
        return 0

    allowed_cases = {
        case["name"]: case
        for case in comparison_cases(
            args.old_app,
            args.new_app,
            args.qwen_context,
            old_cost_gpu=args.old_cost_gpu,
            new_cost_gpu=args.new_cost_gpu,
        )
    }
    case_names = [item.strip() for item in args.cases.split(",") if item.strip()]
    cases = [allowed_cases[name] for name in case_names]

    all_results: Dict[str, Any] = {
        "apple_url": args.apple_url,
        "original_audio": str(original),
        "old_app": args.old_app,
        "new_app": args.new_app,
        "qwen_context_enabled": bool(args.qwen_context.strip()),
        "qwen_context_length": len(args.qwen_context.strip()),
        "cost_model": {
            "gpu_prices_per_second": MODAL_GPU_PRICE_PER_SECOND,
            "cpu_price_per_second": MODAL_CPU_PRICE_PER_SECOND,
            "memory_gib_price_per_second": MODAL_MEMORY_GIB_PRICE_PER_SECOND,
            "old_cost_gpu": args.old_cost_gpu,
            "new_cost_gpu": args.new_cost_gpu,
            "cpu_count": args.cost_cpu_count,
            "memory_gib": args.cost_memory_gib,
            "region_multiplier": args.region_multiplier,
        },
        "clips": {},
        "issues": [],
    }

    for clip_name, (clip_path, clip_seconds) in clips.items():
        clip_result: Dict[str, Any] = {
            "clip_path": str(clip_path),
            "clip_seconds": clip_seconds,
            "cases": {},
        }
        all_results["clips"][clip_name] = clip_result
        for case in cases:
            case_name = case["name"]
            full_name = f"{clip_name}.{case_name}"
            _log(f"Calling {full_name}: app={case['app']} backend={case.get('asr_backend')}")
            payload = build_payload(clip_path, clip_seconds, case)
            result = call_modal(case["app"], payload)
            attach_benchmark_metrics(
                result,
                clip_seconds=clip_seconds,
                gpu_type=case.get("cost_gpu", args.new_cost_gpu),
                cpu_count=args.cost_cpu_count,
                memory_gib=args.cost_memory_gib,
                region_multiplier=args.region_multiplier,
            )
            clip_result["cases"][case_name] = result

            case_dir = output_dir / "artifacts" / clip_name / case_name
            case_dir.mkdir(parents=True, exist_ok=True)
            (case_dir / "result.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            (case_dir / "text.txt").write_text(
                str(result.get("text") or ""),
                encoding="utf-8",
            )
            write_srt(result, case_dir / "segments.srt")

            issues = validate_result(full_name, clip_seconds, result)
            all_results["issues"].extend(issues)
            if issues:
                _log("  issues: " + "; ".join(issues))
            else:
                validation = result.get("_validation") or {}
                benchmark = result.get("_benchmark") or {}
                _log(
                    "  ok: "
                    f"segments={len(result.get('segments') or [])} "
                    f"text_len={validation.get('text_len')} "
                    f"elapsed={result.get('_elapsed_seconds', 0):.1f}s "
                    f"throughput={benchmark.get('audio_seconds_per_processing_second', 0):.2f}x "
                    f"cost=${benchmark.get('modal_cost_usd', 0):.5f}"
                )

    (output_dir / "results.json").write_text(
        json.dumps(all_results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _log(f"Wrote results: {output_dir / 'results.json'}")

    if all_results["issues"]:
        _log("Validation finished with issues:")
        for issue in all_results["issues"]:
            _log(f" - {issue}")
        return 2
    _log("Validation passed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
