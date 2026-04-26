"""
Shared Modal configuration for the GPU project.
Defines the Modal App, transcription image, volume, and model preloading.
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

# Opt-in: `Secret.from_name` does not validate at import time; deploy would fail
# if this is set without creating the secret in the target workspace.
hf_secret = None
if os.getenv("ATTACH_HUGGINGFACE_SECRET", "false").lower() == "true":
    try:
        hf_secret = modal.Secret.from_name("huggingface-secret")
        print("Attached Hugging Face secret (ATTACH_HUGGINGFACE_SECRET=true)")
    except Exception as e:
        print(f"ATTACH_HUGGINGFACE_SECRET=true but lookup failed: {e}")
        hf_secret = None
else:
    print(
        "Hugging Face secret not attached (set ATTACH_HUGGINGFACE_SECRET=true "
        "after creating modal secret 'huggingface-secret')"
    )

volume = modal.Volume.from_name("cache-volume", create_if_missing=True)
cache_dir = "/root/cache"

PYTHON_VERSION = "3.11"


def download_transcription_models() -> None:
    """Download and cache Whisper and speaker diarization models during image build."""
    import whisper
    import os as _os
    from pathlib import Path as _Path

    model_cache_dir = _Path("/model")
    model_cache_dir.mkdir(exist_ok=True)

    model_size = _os.getenv("DEFAULT_MODEL_SIZE", "large-v3")
    print(f"Downloading Whisper {model_size} model...")
    whisper.load_model(model_size, download_root="/model")
    print(f"Whisper {model_size} model downloaded and cached")

    hf_token = _os.environ.get("HF_TOKEN") or _os.environ.get("HUGGING_FACE_TOKEN")
    if hf_token:
        try:
            print("Downloading speaker diarization models...")
            from pyannote.audio import Pipeline, Model

            speaker_cache_dir = "/model/speaker-diarization"
            embedding_cache_dir = "/model/speaker-embedding"

            _os.environ["PYANNOTE_CACHE"] = speaker_cache_dir

            speaker_dir = _Path(speaker_cache_dir)
            embedding_dir = _Path(embedding_cache_dir)
            speaker_dir.mkdir(parents=True, exist_ok=True)
            embedding_dir.mkdir(parents=True, exist_ok=True)

            print("Downloading speaker diarization pipeline...")
            Pipeline.from_pretrained("pyannote/speaker-diarization-community-1")
            print("Speaker diarization pipeline downloaded successfully")

            print("Downloading speaker embedding model...")
            Model.from_pretrained("pyannote/embedding")
            print("Speaker embedding model downloaded successfully")

            import json
            marker = {
                "model_name": "pyannote/speaker-diarization-community-1",
                "embedding_model_name": "pyannote/embedding",
                "cached_at": str(speaker_dir),
                "embedding_cached_at": str(embedding_dir),
                "cache_complete": True,
                "download_successful": True,
            }
            (speaker_dir / "download_complete.json").write_text(json.dumps(marker, indent=2))
            print("Speaker diarization and embedding models downloaded and cached")
        except Exception as e:
            print(f"Failed to download speaker diarization models: {e}")
            import traceback

            traceback.print_exc()
    else:
        print("No HF_TOKEN or HUGGING_FACE_TOKEN found, skipping speaker diarization model download")


transcription_image = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
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
    )
    .run_function(
        download_transcription_models,
        secrets=[hf_secret] if hf_secret else [],
    )
    .add_local_dir(str(project_root / "src"), remote_path="/root/src")
)

secrets = [hf_secret] if hf_secret else []
