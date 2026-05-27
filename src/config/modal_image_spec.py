"""Shared dependency spec for the Modal and local Docker GPU images."""

from __future__ import annotations


PYTHON_VERSION = "3.11"
MODAL_DOCKER_BASE_IMAGE = "python:3.11.12-slim-bookworm"

TORCHCODEC_VERSION = "0.9.1"

APT_PACKAGES = (
    "ffmpeg",
    "wget",
    "curl",
    "unzip",
    "gnupg2",
    "git",
)

PIP_PACKAGES = (
    "git+https://github.com/openai/whisper.git",
    "ffmpeg-python",
    "torchaudio",
    # qwen-asr[vllm] currently resolves torch 2.9.x in the Modal image.
    # TorchCodec's binary extension must match torch's minor version or
    # pyannote emits a noisy libtorchcodec warning during startup.
    f"torchcodec=={TORCHCODEC_VERSION}",
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
