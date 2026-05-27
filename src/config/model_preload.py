"""Build-time model preloading shared by Modal and local Docker images."""

from __future__ import annotations

import os
import hashlib
import shutil
import subprocess
import time
from pathlib import Path
from urllib.parse import urlparse


# Standard HF cache layout. Same path is used at build time (so weights are
# baked into the image) and at runtime (so offline lookups resolve correctly).
HF_HOME_PATH = "/model/hf-cache"
HF_HUB_CACHE_PATH = "/model/hf-cache/hub"
DIARIZATION_REPO_ID = "pyannote/speaker-diarization-community-1"
EMBEDDING_REPO_ID = "pyannote/embedding"
DEFAULT_QWEN_ASR_MODEL_ID = "Qwen/Qwen3-ASR-1.7B"
DEFAULT_QWEN_FORCED_ALIGNER_MODEL_ID = "Qwen/Qwen3-ForcedAligner-0.6B"


def hf_snapshot_repo_ids() -> tuple[str, str, str, str]:
    """Repos whose HF snapshots must be present for offline runtime use."""

    return (
        DIARIZATION_REPO_ID,
        EMBEDDING_REPO_ID,
        os.getenv("QWEN_ASR_MODEL_ID", DEFAULT_QWEN_ASR_MODEL_ID),
        os.getenv("QWEN_FORCED_ALIGNER_MODEL_ID", DEFAULT_QWEN_FORCED_ALIGNER_MODEL_ID),
    )


def _retry(label: str, operation, *, attempts_override: int | None = None):
    attempts_raw = os.getenv("MODEL_DOWNLOAD_RETRIES", "4")
    delay_raw = os.getenv("MODEL_DOWNLOAD_RETRY_DELAY_SECONDS", "20")
    if attempts_override is not None:
        attempts = max(1, attempts_override)
    else:
        try:
            attempts = max(1, int(float(attempts_raw)))
        except (TypeError, ValueError):
            attempts = 4
    try:
        delay_seconds = max(1.0, float(delay_raw))
    except (TypeError, ValueError):
        delay_seconds = 20.0

    last_exc = None
    for attempt in range(1, attempts + 1):
        try:
            if attempt > 1:
                print(f"Retrying {label} (attempt {attempt}/{attempts})...", flush=True)
            return operation()
        except Exception as exc:
            last_exc = exc
            if attempt >= attempts:
                print(f"[ERROR] {label} failed after {attempts} attempts: {exc!r}", flush=True)
                raise
            sleep_for = min(delay_seconds * (2 ** (attempt - 1)), 180.0)
            print(
                f"[WARN] {label} failed on attempt {attempt}/{attempts}: {exc!r}; "
                f"retrying in {sleep_for:.0f}s",
                flush=True,
            )
            time.sleep(sleep_for)
    raise RuntimeError(f"{label} failed") from last_exc


def _normalise_hf_endpoint(endpoint: str | None) -> str:
    return (endpoint or "").rstrip("/")


def _set_hf_endpoint(endpoint: str | None) -> None:
    """Update HF Hub's process-global endpoint, including imported constants."""

    endpoint = _normalise_hf_endpoint(endpoint) or "https://huggingface.co"
    os.environ["HF_ENDPOINT"] = endpoint
    try:
        import huggingface_hub.constants as hf_constants

        hf_constants.ENDPOINT = endpoint
        hf_constants.HUGGINGFACE_CO_URL_TEMPLATE = (
            endpoint + "/{repo_id}/resolve/{revision}/{filename}"
        )
    except Exception as exc:
        print(f"[WARN] Could not update Hugging Face endpoint constants: {exc!r}", flush=True)


def _hf_snapshot_endpoints() -> list[str]:
    primary = _normalise_hf_endpoint(os.getenv("HF_ENDPOINT"))
    fallback = _normalise_hf_endpoint(
        os.getenv("HF_SNAPSHOT_FALLBACK_ENDPOINT", "https://huggingface.co")
    )
    endpoints = [primary] if primary else ["https://huggingface.co"]
    if fallback and fallback not in endpoints:
        endpoints.append(fallback)
    return endpoints


def _snapshot_with_heartbeat(repo_id: str, label: str) -> str:
    """HF parallel downloads occasionally stall; fewer workers + heartbeats help."""

    import threading

    from huggingface_hub import snapshot_download

    stop = threading.Event()

    def _beat() -> None:
        # Long quiet periods are normal while a single multi-GB shard downloads.
        while not stop.wait(45.0):
            print(
                f"  ... heartbeat: {label} still downloading "
                "(large weights can keep one tqdm bar unchanged for many minutes) ...",
                flush=True,
            )

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_TOKEN")
    max_workers_raw = os.getenv("HF_SNAPSHOT_MAX_WORKERS", "1")
    try:
        max_workers = max(1, int(float(max_workers_raw)))
    except (TypeError, ValueError):
        max_workers = 1

    th = threading.Thread(target=_beat, daemon=True)
    original_endpoint = os.getenv("HF_ENDPOINT")
    th.start()
    try:
        endpoints = _hf_snapshot_endpoints()
        for index, endpoint in enumerate(endpoints):
            is_last_endpoint = index == len(endpoints) - 1
            endpoint_attempts = None if is_last_endpoint else 1
            _set_hf_endpoint(endpoint)
            try:
                return _retry(
                    f"HF snapshot {label} via {endpoint}",
                    lambda: snapshot_download(
                        repo_id,
                        token=hf_token,
                        max_workers=max_workers,
                        endpoint=endpoint,
                    ),
                    attempts_override=endpoint_attempts,
                )
            except Exception as exc:
                if is_last_endpoint:
                    raise
                next_endpoint = endpoints[index + 1]
                print(
                    f"[WARN] HF snapshot {label} failed via {endpoint}: {exc!r}; "
                    f"trying {next_endpoint}",
                    flush=True,
                )
        raise RuntimeError(f"HF snapshot {label} failed")
    finally:
        _set_hf_endpoint(original_endpoint)
        stop.set()


def download_transcription_models() -> None:
    """Download Whisper, pyannote, Qwen3-ASR, and ForcedAligner weights.

    This runs during image build only. It needs ``HF_TOKEN`` because pyannote
    weights are gated. The token must not be persisted in the final runtime
    image; Docker builds should pass it as a BuildKit secret.
    """

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_TOKEN")
    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN missing during image build. For local Docker builds, run "
            "`HF_TOKEN=hf_... scripts/build_aliyun_gpu_image.sh <acr-image>` "
            "or pass `docker build --secret id=hf_token,env=HF_TOKEN` directly. "
            "The token is only used to bake gated pyannote weights into the image."
        )

    model_cache_dir = Path("/model")
    model_cache_dir.mkdir(parents=True, exist_ok=True)
    Path(HF_HUB_CACHE_PATH).mkdir(parents=True, exist_ok=True)

    import whisper

    model_size = os.getenv("DEFAULT_MODEL_SIZE", "large-v3")
    print(f"Downloading Whisper {model_size} model...")

    def _whisper_target() -> tuple[str | None, Path | None]:
        url = getattr(whisper, "_MODELS", {}).get(model_size)
        if not url:
            return None, None
        return url, model_cache_dir / Path(urlparse(url).path).name

    def _valid_whisper_checkpoint(url: str, target: Path) -> bool:
        if not target.exists():
            return False
        expected_sha = url.rstrip("/").split("/")[-2]
        if len(expected_sha) != 64:
            return False
        actual = hashlib.sha256()
        with target.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                actual.update(chunk)
        return actual.hexdigest() == expected_sha

    def _curl_whisper_checkpoint() -> None:
        if os.getenv("WHISPER_PRELOAD_WITH_CURL", "1").lower() in {"0", "false", "no"}:
            return
        curl_bin = shutil.which("curl")
        if not curl_bin:
            print("[WARN] curl not found; falling back to whisper downloader", flush=True)
            return
        url, target = _whisper_target()
        if not url or not target:
            return
        if _valid_whisper_checkpoint(url, target):
            print(f"Whisper checkpoint already present: {target}", flush=True)
            return
        if target.exists():
            print(f"Removing invalid or partial Whisper checkpoint before curl: {target}", flush=True)
            target.unlink()
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                curl_bin,
                "--location",
                "--fail",
                "--retry",
                "8",
                "--retry-all-errors",
                "--retry-delay",
                "5",
                "--connect-timeout",
                "30",
                "--speed-limit",
                "1024",
                "--speed-time",
                "120",
                "--noproxy",
                "openaipublic.azureedge.net",
                "--output",
                str(target),
                url,
            ],
            check=True,
        )

    def _remove_partial_whisper_download() -> None:
        try:
            _, target = _whisper_target()
            if not target:
                return
            if target.exists():
                print(f"Removing partial Whisper checkpoint before retry: {target}", flush=True)
                target.unlink()
        except Exception as exc:
            print(f"[WARN] Could not remove partial Whisper checkpoint: {exc!r}", flush=True)

    def _load_whisper_with_cleanup():
        try:
            _curl_whisper_checkpoint()
            return whisper.load_model(model_size, download_root="/model")
        except Exception:
            _remove_partial_whisper_download()
            raise

    _retry(f"Whisper {model_size} download", _load_whisper_with_cleanup)
    print(f"Whisper {model_size} model downloaded and cached")

    # Modal / non-TTY logs often swallow tqdm carriage-return updates, so we
    # pre-fetch with snapshot_download and set hub verbosity to INFO.
    os.environ.setdefault("HF_HUB_VERBOSITY", "info")

    import huggingface_hub.utils.logging as hf_log

    hf_log.set_verbosity_info()

    diarization_repo, embedding_repo, qwen_asr_id, qwen_align_id = hf_snapshot_repo_ids()

    print(
        "Prefetching speaker diarization HF repos (snapshot_download; "
        "may take several minutes; watch tqdm % or INFO lines below)...",
        flush=True,
    )
    print(f"  snapshot: {diarization_repo}", flush=True)
    _snapshot_with_heartbeat(diarization_repo, diarization_repo)
    print(f"  snapshot done: {diarization_repo}", flush=True)
    print(f"  snapshot: {embedding_repo}", flush=True)
    _snapshot_with_heartbeat(embedding_repo, embedding_repo)
    print(f"  snapshot done: {embedding_repo}", flush=True)

    os.environ["PYANNOTE_PROGRESS_HOOK"] = os.getenv("PYANNOTE_PROGRESS_HOOK", "true")
    print(
        "Instantiating pyannote Pipeline / Model from HF cache "
        "(may download linked sub-checkpoints)...",
        flush=True,
    )
    from pyannote.audio import Model, Pipeline

    original_endpoint = os.getenv("HF_ENDPOINT")
    _set_hf_endpoint(os.getenv("HF_SNAPSHOT_FALLBACK_ENDPOINT", "https://huggingface.co"))
    # token= is the explicit pyannote 4.x argument; the env var is a fallback
    # for any internal HF Hub call.
    try:
        Pipeline.from_pretrained(
            diarization_repo,
            token=hf_token,
        )
        print("Speaker diarization pipeline cached", flush=True)

        Model.from_pretrained(embedding_repo, token=hf_token)
        print("Speaker embedding model cached", flush=True)
    finally:
        _set_hf_endpoint(original_endpoint)

    print("Downloading Qwen3-ASR + ForcedAligner snapshots into HF cache...")

    _snapshot_with_heartbeat(qwen_asr_id, qwen_asr_id)
    print(f"Cached ASR weights: {qwen_asr_id}")
    _snapshot_with_heartbeat(qwen_align_id, qwen_align_id)
    print(f"Cached ForcedAligner weights: {qwen_align_id}")


if __name__ == "__main__":
    download_transcription_models()
