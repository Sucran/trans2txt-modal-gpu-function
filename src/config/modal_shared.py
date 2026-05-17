"""
Shared Modal configuration for the GPU project.
Defines the Modal App, transcription image, volume, and model preloading.

Hugging Face token policy
-------------------------

The HF token is **bridge-only** and **build-only**:

* During ``modal deploy``, the bridge (transcribe-modal-bridge) injects
  ``HF_TOKEN`` into the deploy subprocess env. We pass it into the image build
  via an *inline* :class:`modal.Secret.from_dict` bound to ``run_function``
  only — this is **not** a named workspace Secret. Users running ``modal
  secret list`` in their Modal workspace will see no Hugging Face entry.
* The image build then runs :func:`download_transcription_models` once,
  which downloads gated pyannote weights into the standard Hugging Face cache
  (``HF_HOME = /model/hf-cache``) baked into the image layer.
* After the build step we set ``HF_HUB_OFFLINE=1`` on the image so runtime
  ``Pipeline.from_pretrained`` resolves entirely from the baked cache — no
  Hub call, no token required at runtime, and ``secrets = []`` on the
  function so nothing is attached to user containers.

Operators running ``modal deploy`` locally (without the bridge) must
``export HF_TOKEN=...`` themselves; end users never need to.
"""

import modal
import os
from dotenv import load_dotenv
from pathlib import Path


project_root = Path(__file__).resolve().parents[2]
config_env_path = project_root / "config.env"
env_path = project_root / ".env"

if config_env_path.exists():
    load_dotenv(str(config_env_path), override=False)
    print(f"Loaded config from {config_env_path}")

if env_path.exists():
    load_dotenv(str(env_path), override=False)
    print(f"Loaded config from {env_path}")

# Modal App name for the GPU side. Must match what the CPU project uses
# when calling `modal.Function.from_name(MODAL_APP_NAME, ...)`.
MODAL_APP_NAME = os.getenv("MODAL_APP_NAME", "transcribe-modal-gpu")
app = modal.App(name=MODAL_APP_NAME)

volume = modal.Volume.from_name("cache-volume", create_if_missing=True)
cache_dir = "/root/cache"

PYTHON_VERSION = "3.11"

# Standard HF cache layout. Same path is used at build time (so weights are
# baked into the image) and at runtime (so offline lookups resolve correctly).
HF_HOME_PATH = "/model/hf-cache"
HF_HUB_CACHE_PATH = "/model/hf-cache/hub"

# Read the HF token at module import time. During ``modal deploy`` the bridge
# (or a local operator) sets this; in the runtime container it is absent — and
# that is fine, because ``run_function`` is only executed at build time and the
# function itself runs offline. We store an empty string when missing rather
# than raising here, so that re-importing this module inside a runtime container
# does not crash.
_hf_token_for_build = (
    os.environ.get("HF_TOKEN", "").strip()
    or os.environ.get("HUGGING_FACE_TOKEN", "").strip()
)

# Inline build secret. This is **not** a named workspace secret: it is bundled
# into the deployment graph for this single deploy and is not visible in
# ``modal secret list`` for the target workspace. ``run_function`` is the only
# consumer; the runtime function declares ``secrets=[]``.
hf_build_secret = modal.Secret.from_dict({"HF_TOKEN": _hf_token_for_build})

runtime_env = {
    "DEFAULT_MODEL_SIZE": os.getenv("DEFAULT_MODEL_SIZE", "large-v3"),
    "MODEL_DOWNLOAD_RETRIES": os.getenv("MODEL_DOWNLOAD_RETRIES", "4"),
    "MODEL_DOWNLOAD_RETRY_DELAY_SECONDS": os.getenv(
        "MODEL_DOWNLOAD_RETRY_DELAY_SECONDS", "20"
    ),
    "ASR_BACKEND": os.getenv("ASR_BACKEND", "whisper"),
    "QWEN_ASR_MODEL_ID": os.getenv(
        "QWEN_ASR_MODEL_ID", "Qwen/Qwen3-ASR-1.7B"
    ),
    "QWEN_ALLOWED_ASR_MODEL_IDS": os.getenv("QWEN_ALLOWED_ASR_MODEL_IDS", ""),
    "QWEN_FORCED_ALIGNER_MODEL_ID": os.getenv(
        "QWEN_FORCED_ALIGNER_MODEL_ID", "Qwen/Qwen3-ForcedAligner-0.6B"
    ),
    "QWEN_ASR_RUNTIME": os.getenv("QWEN_ASR_RUNTIME", "vllm"),
    "QWEN_TRANSFORMERS_BATCH_SIZE": os.getenv("QWEN_TRANSFORMERS_BATCH_SIZE", "1"),
    "QWEN_VLLM_DTYPE": os.getenv("QWEN_VLLM_DTYPE", "bfloat16"),
    "QWEN_ALIGNER_DTYPE": os.getenv("QWEN_ALIGNER_DTYPE", ""),
    "QWEN_VLLM_BATCH_SIZE": os.getenv("QWEN_VLLM_BATCH_SIZE", "4"),
    "QWEN_VLLM_GPU_MEMORY_UTILIZATION": os.getenv(
        "QWEN_VLLM_GPU_MEMORY_UTILIZATION", "0.70"
    ),
    "QWEN_VLLM_MAX_MODEL_LEN": os.getenv("QWEN_VLLM_MAX_MODEL_LEN", "8192"),
    "QWEN_ALIGNER_MAX_SEGMENT_SECONDS": os.getenv(
        "QWEN_ALIGNER_MAX_SEGMENT_SECONDS", "60"
    ),
    "QWEN_MAX_NEW_TOKENS": os.getenv("QWEN_MAX_NEW_TOKENS", "1024"),
    "QWEN_ASR_CONTEXT": os.getenv("QWEN_ASR_CONTEXT", ""),
    "QWEN_SUBTITLE_MAX_SECONDS": os.getenv("QWEN_SUBTITLE_MAX_SECONDS", "1.6"),
    "QWEN_SUBTITLE_MAX_CHARS": os.getenv("QWEN_SUBTITLE_MAX_CHARS", "12"),
    "QWEN_SUBTITLE_GAP_SECONDS": os.getenv("QWEN_SUBTITLE_GAP_SECONDS", "0.35"),
    "PYANNOTE_SEGMENTATION_BATCH_SIZE": os.getenv("PYANNOTE_SEGMENTATION_BATCH_SIZE", "256"),
    "PYANNOTE_EMBEDDING_BATCH_SIZE": os.getenv("PYANNOTE_EMBEDDING_BATCH_SIZE", "256"),
    "PYANNOTE_USE_MEMORY_INPUT": os.getenv("PYANNOTE_USE_MEMORY_INPUT", "true"),
    "PYANNOTE_PROGRESS_HOOK": os.getenv("PYANNOTE_PROGRESS_HOOK", "true"),
}


def download_transcription_models() -> None:
    """Download Whisper + pyannote weights into the image at build time.

    Runs **only** during ``modal deploy`` as part of the image build (via
    :meth:`modal.Image.run_function`). Reads ``HF_TOKEN`` from the inline build
    secret and writes pyannote weights into the standard Hugging Face cache at
    ``HF_HOME=/model/hf-cache``. Whisper weights go into ``/model`` (Whisper
    uses its own download root, not HF Hub).

    Fails the build with a clear message if ``HF_TOKEN`` is missing — this is
    the only place we need to fail fast, because module-level import also runs
    in runtime containers where the token is intentionally absent.
    """
    import os as _os
    import time as _time
    from urllib.parse import urlparse as _urlparse
    from pathlib import Path as _Path

    def _retry(label: str, operation):
        attempts_raw = _os.getenv("MODEL_DOWNLOAD_RETRIES", "4")
        delay_raw = _os.getenv("MODEL_DOWNLOAD_RETRY_DELAY_SECONDS", "20")
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
                    print(f"❌ {label} failed after {attempts} attempts: {exc!r}", flush=True)
                    raise
                sleep_for = min(delay_seconds * (2 ** (attempt - 1)), 180.0)
                print(
                    f"⚠️ {label} failed on attempt {attempt}/{attempts}: {exc!r}; "
                    f"retrying in {sleep_for:.0f}s",
                    flush=True,
                )
                _time.sleep(sleep_for)
        raise RuntimeError(f"{label} failed") from last_exc

    hf_token = _os.environ.get("HF_TOKEN") or _os.environ.get("HUGGING_FACE_TOKEN")
    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN missing during image build. The transcribe-modal-bridge "
            "service must set HF_TOKEN on its environment before invoking "
            "`modal deploy`. For local deploys, run `export HF_TOKEN=hf_...` in "
            "your shell. The token is only used to bake gated pyannote weights "
            "into the image and is never persisted as a named Modal Secret."
        )

    model_cache_dir = _Path("/model")
    model_cache_dir.mkdir(parents=True, exist_ok=True)
    _Path(HF_HUB_CACHE_PATH).mkdir(parents=True, exist_ok=True)

    import whisper

    model_size = _os.getenv("DEFAULT_MODEL_SIZE", "large-v3")
    print(f"Downloading Whisper {model_size} model...")

    def _remove_partial_whisper_download() -> None:
        try:
            url = getattr(whisper, "_MODELS", {}).get(model_size)
            if not url:
                return
            target = model_cache_dir / _Path(_urlparse(url).path).name
            if target.exists():
                print(f"Removing partial Whisper checkpoint before retry: {target}", flush=True)
                target.unlink()
        except Exception as exc:
            print(f"⚠️ Could not remove partial Whisper checkpoint: {exc!r}", flush=True)

    def _load_whisper_with_cleanup():
        try:
            return whisper.load_model(model_size, download_root="/model")
        except Exception:
            _remove_partial_whisper_download()
            raise

    _retry(f"Whisper {model_size} download", _load_whisper_with_cleanup)
    print(f"Whisper {model_size} model downloaded and cached")

    # Modal / non-TTY logs often swallow tqdm carriage-return updates, so we
    # pre-fetch with ``snapshot_download`` (shows hub progress) and set hub
    # verbosity to INFO. ``Pipeline.from_pretrained`` may still pull linked
    # checkpoints; prints bracket that phase.
    _os.environ.setdefault("HF_HUB_VERBOSITY", "info")

    import huggingface_hub.utils.logging as _hf_log

    _hf_log.set_verbosity_info()

    from huggingface_hub import snapshot_download

    def _snapshot_with_heartbeat(repo_id: str, label: str) -> str:
        """HF parallel downloads occasionally stall; fewer workers + heartbeats help."""

        import threading

        stop = threading.Event()

        def _beat() -> None:
            # Long quiet periods are normal while a single multi-GB shard downloads.
            while not stop.wait(45.0):
                print(
                    f"  ... heartbeat: {label} — still downloading "
                    "(large weights can keep one tqdm bar unchanged for many minutes) ...",
                    flush=True,
                )

        th = threading.Thread(target=_beat, daemon=True)
        th.start()
        try:
            return _retry(
                f"HF snapshot {label}",
                lambda: snapshot_download(repo_id, token=hf_token, max_workers=2),
            )
        finally:
            stop.set()

    diarization_repo = "pyannote/speaker-diarization-community-1"
    embedding_repo = "pyannote/embedding"

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

    _os.environ["PYANNOTE_PROGRESS_HOOK"] = _os.getenv("PYANNOTE_PROGRESS_HOOK", "true")
    print(
        "Instantiating pyannote Pipeline / Model from HF cache "
        "(may download linked sub-checkpoints)...",
        flush=True,
    )
    from pyannote.audio import Pipeline, Model

    # ``token=`` is the explicit pyannote 4.x argument; the env var is a fallback
    # for any internal HF Hub call. PYANNOTE_CACHE is intentionally not set —
    # it is a no-op in pyannote >= 4 and the standard HF cache is the only
    # location consulted at runtime.
    Pipeline.from_pretrained(
        diarization_repo,
        token=hf_token,
    )
    print("Speaker diarization pipeline cached", flush=True)

    Model.from_pretrained(embedding_repo, token=hf_token)
    print("Speaker embedding model cached", flush=True)

    print("Downloading Qwen3-ASR + ForcedAligner snapshots into HF cache...")

    qwen_asr_id = _os.getenv("QWEN_ASR_MODEL_ID", "Qwen/Qwen3-ASR-1.7B")
    qwen_align_id = _os.getenv(
        "QWEN_FORCED_ALIGNER_MODEL_ID", "Qwen/Qwen3-ForcedAligner-0.6B"
    )
    _snapshot_with_heartbeat(qwen_asr_id, qwen_asr_id)
    print(f"Cached ASR weights: {qwen_asr_id}")
    _snapshot_with_heartbeat(qwen_align_id, qwen_align_id)
    print(f"Cached ForcedAligner weights: {qwen_align_id}")


# Image construction. Order matters:
#   1. Set HF_HOME / HF_HUB_CACHE so the build *and* runtime read the same
#      cache path.
#   2. Install system + Python deps.
#   3. ``run_function`` with the inline build secret to populate the cache.
#   4. After the build, flip ``HF_HUB_OFFLINE=1`` so runtime never reaches out
#      to huggingface.co — even if a stray network call were attempted.
transcription_image = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    .env({"HF_HOME": HF_HOME_PATH, "HF_HUB_CACHE": HF_HUB_CACHE_PATH, **runtime_env})
    .apt_install(
        "ffmpeg",
        "wget",
        "curl",
        "unzip",
        "gnupg2",
        "git",
    )
    .pip_install(
        "git+https://github.com/openai/whisper.git",
        "ffmpeg-python",
        "torchaudio",
        "numpy",
        "librosa",
        "soundfile",
        "dacite",
        "jiwer",
        "pandas",
        "python-dotenv",
        "fastapi[standard]",
        "beautifulsoup4",
        "requests",
        "psutil",
        "huggingface_hub",
        "pyannote.audio>=4.0.0",
        "omegaconf",
        "qwen-asr[vllm]",
    )
    .run_function(
        download_transcription_models,
        secrets=[hf_build_secret],
    )
    .env({"HF_HUB_OFFLINE": "1", **runtime_env})
    .add_local_dir(str(project_root / "src"), remote_path="/root/src")
)

# No runtime secrets attached. The token-bearing inline secret above is
# deploy-time only; runtime resolves models from the baked HF cache in offline
# mode. Keeping this list explicit (and exported) so that callers don't need
# to know whether HF is configured or not.
secrets: list[modal.Secret] = []
