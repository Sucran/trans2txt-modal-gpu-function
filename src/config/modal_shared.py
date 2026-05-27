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

from src.config.modal_image_spec import APT_PACKAGES, PIP_PACKAGES, PYTHON_VERSION
from src.config.model_preload import (
    HF_HOME_PATH,
    HF_HUB_CACHE_PATH,
    download_transcription_models,
)


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
    "QWEN_VLLM_ENABLE_SLEEP_MODE": os.getenv(
        "QWEN_VLLM_ENABLE_SLEEP_MODE", "true"
    ),
    "QWEN_ALIGNER_MAX_SEGMENT_SECONDS": os.getenv(
        "QWEN_ALIGNER_MAX_SEGMENT_SECONDS", "60"
    ),
    "QWEN_MAX_NEW_TOKENS": os.getenv("QWEN_MAX_NEW_TOKENS", "1024"),
    "QWEN_ASR_CONTEXT": os.getenv("QWEN_ASR_CONTEXT", ""),
    "QWEN_SUBTITLE_MAX_SECONDS": os.getenv("QWEN_SUBTITLE_MAX_SECONDS", "2.4"),
    "QWEN_SUBTITLE_MAX_CHARS": os.getenv("QWEN_SUBTITLE_MAX_CHARS", "28"),
    "QWEN_SUBTITLE_GAP_SECONDS": os.getenv("QWEN_SUBTITLE_GAP_SECONDS", "0.50"),
    "PYANNOTE_SEGMENTATION_BATCH_SIZE": os.getenv("PYANNOTE_SEGMENTATION_BATCH_SIZE", "256"),
    "PYANNOTE_EMBEDDING_BATCH_SIZE": os.getenv("PYANNOTE_EMBEDDING_BATCH_SIZE", "256"),
    "PYANNOTE_USE_MEMORY_INPUT": os.getenv("PYANNOTE_USE_MEMORY_INPUT", "true"),
    "PYANNOTE_PROGRESS_HOOK": os.getenv("PYANNOTE_PROGRESS_HOOK", "true"),
}

_torch_alloc_conf = os.getenv("PYTORCH_ALLOC_CONF") or os.getenv(
    "PYTORCH_CUDA_ALLOC_CONF", ""
)
if _torch_alloc_conf:
    runtime_env["PYTORCH_ALLOC_CONF"] = _torch_alloc_conf


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
    .apt_install(*APT_PACKAGES)
    .pip_install(*PIP_PACKAGES)
    .run_function(
        download_transcription_models,
        secrets=[hf_build_secret],
    )
    .env({"HF_HUB_OFFLINE": "1", **runtime_env})
    .add_local_dir(str(project_root / "src"), remote_path="/root/src")
)

# Lightweight image for the backwards-compatible forwarding function. It does
# not load models or pull GPU dependencies; it only needs Modal + dotenv so old
# ``modal.Function.from_name`` callers can be forwarded to the snapshot-backed
# class runtime without paying a second GPU cold start.
dispatcher_image = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    .pip_install("modal>=1.0.3", "python-dotenv>=1.0.0")
    .add_local_dir(str(project_root / "src"), remote_path="/root/src")
)

# No runtime secrets attached. The token-bearing inline secret above is
# deploy-time only; runtime resolves models from the baked HF cache in offline
# mode. Keeping this list explicit (and exported) so that callers don't need
# to know whether HF is configured or not.
secrets: list[modal.Secret] = []
