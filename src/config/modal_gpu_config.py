"""
Modal GPU endpoint configuration
Handles GPU endpoint deployment and configuration
"""

import os
import sys
from pathlib import Path
from typing import Any

import modal


def _resolve_repo_root(current_file: Path | None = None) -> Path:
    """Find the directory that owns the copied ``src`` tree."""

    resolved_file = (current_file or Path(__file__)).resolve()
    for candidate in resolved_file.parents:
        if (candidate / "src" / "config" / "modal_shared.py").is_file():
            return candidate
    return resolved_file.parent


# `modal deploy` on this file loads it as a top-level module (not as `src.config.*`),
# so `from .modal_shared` fails. Prefer imports from the repo root.
_repo_root = _resolve_repo_root()
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

# Import shared configuration
from src.config.modal_shared import (
    app,
    volume,
    cache_dir,
    dispatcher_image,
    transcription_image,
    secrets,
)

# GPU endpoint configuration from environment variables
MODAL_GPU_TYPE = os.getenv("MODAL_GPU_TYPE", "L4")
MODAL_GPU_CPU = int(os.getenv("MODAL_CPU", "4"))
MODAL_GPU_MEMORY = int(os.getenv("MODAL_MEMORY", "8192"))  # Default 8GB
MODAL_GPU_TIMEOUT = int(os.getenv("MODAL_GPU_TIMEOUT", "1800"))  # 30 minutes
MODAL_GPU_SCALEDOWN_WINDOW = int(os.getenv("MODAL_SCALEDOWN_WINDOW", "30"))
MODAL_GPU_ENABLE_MEMORY_SNAPSHOT = os.getenv(
    "MODAL_GPU_ENABLE_MEMORY_SNAPSHOT",
    "true",
).strip().lower() in {"1", "true", "yes", "y", "on"}
MODAL_GPU_ENABLE_GPU_SNAPSHOT = os.getenv(
    "MODAL_GPU_ENABLE_GPU_SNAPSHOT",
    "true",
).strip().lower() in {"1", "true", "yes", "y", "on"}
MODAL_GPU_SNAPSHOT_PRELOAD = os.getenv(
    "MODAL_GPU_SNAPSHOT_PRELOAD",
    "qwen3_asr,diarization",
)


def _snapshot_experimental_options(
    memory_enabled: bool | None = None,
    gpu_enabled: bool | None = None,
) -> dict[str, Any] | None:
    memory_enabled = (
        MODAL_GPU_ENABLE_MEMORY_SNAPSHOT
        if memory_enabled is None
        else memory_enabled
    )
    gpu_enabled = (
        MODAL_GPU_ENABLE_GPU_SNAPSHOT
        if gpu_enabled is None
        else gpu_enabled
    )
    if memory_enabled and gpu_enabled:
        return {"enable_gpu_snapshot": True}
    return None


MODAL_GPU_SNAPSHOT_EXPERIMENTAL_OPTIONS = _snapshot_experimental_options()


def _resolve_snapshot_preload_targets(
    raw_preload: str | None = None,
    asr_backend: str | None = None,
) -> list[str]:
    """
    Resolve the GPU-heavy services to load during ``@modal.enter(snap=True)``.

    Only one ASR backend is preloaded because ``TranscriptionEndpointService``
    intentionally keeps one GPU-heavy backend resident per container to avoid
    Whisper + Qwen + ForcedAligner VRAM co-residency.
    """

    raw_value = (raw_preload if raw_preload is not None else MODAL_GPU_SNAPSHOT_PRELOAD)
    value = str(raw_value or "auto").strip().lower()
    if value in {"", "0", "false", "off", "none", "disabled", "disable"}:
        return []

    if value == "auto":
        backend = str(asr_backend or os.getenv("ASR_BACKEND", "whisper")).strip().lower()
        if backend in {"qwen", "qwen_asr", "qwen3", "qwen3_asr"}:
            return ["qwen3_asr"]
        if backend in {"diarization", "speaker_diarization", "speaker"}:
            return ["diarization"]
        return ["whisper"]

    aliases = {
        "qwen": "qwen3_asr",
        "qwen3": "qwen3_asr",
        "qwen_asr": "qwen3_asr",
        "qwen3_asr": "qwen3_asr",
        "whisper": "whisper",
        "diarization": "diarization",
        "speaker": "diarization",
        "speaker_diarization": "diarization",
    }

    if value == "all":
        tokens = ["qwen3_asr", "diarization"]
    else:
        tokens = [
            item.strip().lower()
            for item in value.replace(";", ",").split(",")
            if item.strip()
        ]

    targets: list[str] = []
    selected_asr: str | None = None
    for token in tokens:
        target = aliases.get(token)
        if target is None:
            print(f"⚠️ Ignoring unknown MODAL_GPU_SNAPSHOT_PRELOAD target: {token}")
            continue
        if target in {"whisper", "qwen3_asr"}:
            if selected_asr is not None and selected_asr != target:
                print(
                    "⚠️ Ignoring additional ASR snapshot target "
                    f"{target}; already selected {selected_asr}"
                )
                continue
            selected_asr = target
        if target not in targets:
            targets.append(target)
    return targets

print(f"🔧 Modal GPU Configuration:")
print(f"   GPU Type: {MODAL_GPU_TYPE}")
print(f"   CPU: {MODAL_GPU_CPU}")
print(f"   Memory: {MODAL_GPU_MEMORY}MB")
print(f"   Timeout: {MODAL_GPU_TIMEOUT}s")
print(f"   Scaledown Window: {MODAL_GPU_SCALEDOWN_WINDOW}s")
print(f"   Memory Snapshot: {MODAL_GPU_ENABLE_MEMORY_SNAPSHOT}")
print(f"   GPU Snapshot: {MODAL_GPU_ENABLE_GPU_SNAPSHOT}")
print(f"   Snapshot Preload: {MODAL_GPU_SNAPSHOT_PRELOAD}")

# ==================== Unified GPU Endpoint Configuration ====================

def _transcribe_and_diarization_audio_endpoint(
    request_data: dict,
    *,
    transcription_service: Any | None = None,
    diarization_service: Any | None = None,
):
    """
    Unified GPU endpoint handler: Process transcription or speaker diarization request
    统一的 GPU 端点处理函数，根据请求数据自动判断是转录还是说话人识别
    
    Args:
        request_data: Request data dictionary
            - For transcription: contains 'chunk_start_time' or 'model_size'
            - For diarization: only contains 'audio_file_data' and 'audio_file_name'
            - Or explicitly set 'request_type' to 'transcribe' or 'diarization'
    
    Returns:
        Result dictionary from corresponding service
    """
    import sys
    sys.path.append('/root')
    
    request_type = _infer_request_type(request_data)
    
    if request_type == "transcribe":
        # Handle transcription request
        if transcription_service is None:
            from src.services.transcription_endpoint_service import TranscriptionEndpointService

            transcription_service = TranscriptionEndpointService(cache_dir=str(cache_dir))
        service = transcription_service
        return service.process_chunk_request(request_data)
    
    elif request_type == "diarization":
        # Handle speaker diarization request
        if diarization_service is None:
            from src.services.speaker_diarization_service import SpeakerDiarizationService

            diarization_service = SpeakerDiarizationService(cache_dir=str(cache_dir))
        service = diarization_service
        return service.process_diarization_request(request_data)
    
    else:
        return {
            "processing_status": "failed",
            "error_message": f"Unknown request_type: {request_type}. Must be 'transcribe' or 'diarization'"
        }


def _infer_request_type(request_data: dict) -> str:
    request_type = request_data.get("request_type")
    if request_type:
        return str(request_type).strip().lower()
    if (
        "chunk_start_time" in request_data
        or "chunk_end_time" in request_data
        or "model_size" in request_data
        or "asr_backend" in request_data
        or "asr_model_id" in request_data
    ):
        return "transcribe"
    return "diarization"


def _runtime_class_name_for_request(request_data: dict) -> str:
    request_type = _infer_request_type(request_data)
    if request_type == "transcribe":
        return "TranscribeAudioRuntime"
    if request_type == "diarization":
        return "SpeakerDiarizationAudioRuntime"
    return "unknown"


def _asr_snapshot_preload_targets() -> list[str]:
    return [
        target
        for target in _resolve_snapshot_preload_targets()
        if target in {"qwen3_asr", "whisper"}
    ]


def _diarization_snapshot_preload_enabled() -> bool:
    return "diarization" in _resolve_snapshot_preload_targets()

@app.cls(
    region=["ap"],
    image=transcription_image,
    volumes={cache_dir: volume},
    cpu=MODAL_GPU_CPU,
    memory=MODAL_GPU_MEMORY,
    gpu=MODAL_GPU_TYPE,
    timeout=MODAL_GPU_TIMEOUT,
    scaledown_window=MODAL_GPU_SCALEDOWN_WINDOW,
    secrets=secrets,
    enable_memory_snapshot=MODAL_GPU_ENABLE_MEMORY_SNAPSHOT,
    experimental_options=MODAL_GPU_SNAPSHOT_EXPERIMENTAL_OPTIONS,
)
class TranscribeAudioRuntime:
    """
    Snapshot-backed GPU runtime for ASR.

    This class intentionally does not load pyannote. Diarization runs in its own
    Modal class/container so Qwen/Whisper and pyannote never compete for VRAM in
    the same L4 process.
    """

    def _ensure_transcription_service(self):
        if getattr(self, "transcription_service", None) is None:
            from src.services.transcription_endpoint_service import TranscriptionEndpointService

            self.transcription_service = TranscriptionEndpointService(
                cache_dir=str(cache_dir)
            )
        return self.transcription_service

    def _preload_whisper(self) -> None:
        model_size = os.getenv("DEFAULT_MODEL_SIZE", "large-v3")
        print(f"📸 Snapshot preload target: Whisper {model_size}")
        self._ensure_transcription_service()._get_or_create_whisper_service(model_size)

    def _preload_qwen(self) -> None:
        asr_model_id = os.getenv("QWEN_ASR_MODEL_ID", "Qwen/Qwen3-ASR-1.7B")
        aligner_model_id = os.getenv(
            "QWEN_FORCED_ALIGNER_MODEL_ID",
            "Qwen/Qwen3-ForcedAligner-0.6B",
        )
        print(
            "📸 Snapshot preload target: "
            f"Qwen3-ASR {asr_model_id} + ForcedAligner {aligner_model_id}"
        )
        self._ensure_transcription_service()._get_or_create_qwen_service(asr_model_id)
        self._ensure_transcription_service().prepare_cached_backend_for_snapshot()

    @modal.enter(snap=True)
    def preload_for_snapshot(self) -> None:
        targets = _asr_snapshot_preload_targets()
        if not targets:
            print("📸 ASR memory snapshot enabled with no ASR preload target")
            self._ensure_transcription_service()
            return

        print(f"📸 Preparing ASR GPU memory snapshot with targets: {targets}")
        for target in targets:
            if target == "qwen3_asr":
                self._preload_qwen()
            elif target == "whisper":
                self._preload_whisper()
        print("✅ ASR snapshot preload complete")

    @modal.enter()
    def restore_after_snapshot(self) -> None:
        self._ensure_transcription_service().restore_cached_backend_after_snapshot()
        print("♻️ ASR GPU runtime restored and ready")

    @modal.method()
    def process(self, request_data: dict):
        if _infer_request_type(request_data) != "transcribe":
            return {
                "processing_status": "failed",
                "error_message": "TranscribeAudioRuntime only accepts transcribe requests",
            }
        return _transcribe_and_diarization_audio_endpoint(
            request_data,
            transcription_service=self._ensure_transcription_service(),
            diarization_service=None,
        )


@app.cls(
    region=["ap"],
    image=transcription_image,
    volumes={cache_dir: volume},
    cpu=MODAL_GPU_CPU,
    memory=MODAL_GPU_MEMORY,
    gpu=MODAL_GPU_TYPE,
    timeout=MODAL_GPU_TIMEOUT,
    scaledown_window=MODAL_GPU_SCALEDOWN_WINDOW,
    secrets=secrets,
    enable_memory_snapshot=MODAL_GPU_ENABLE_MEMORY_SNAPSHOT,
    experimental_options=MODAL_GPU_SNAPSHOT_EXPERIMENTAL_OPTIONS,
)
class SpeakerDiarizationAudioRuntime:
    """
    Snapshot-backed GPU runtime for pyannote speaker diarization.

    This class intentionally does not load Qwen or Whisper. It gets its own
    Modal container and its own snapshot, matching the CPU orchestration model.
    """

    def _ensure_diarization_service(self):
        if getattr(self, "diarization_service", None) is None:
            from src.services.speaker_diarization_service import SpeakerDiarizationService

            self.diarization_service = SpeakerDiarizationService(
                cache_dir=str(cache_dir)
            )
        return self.diarization_service

    @modal.enter(snap=True)
    def preload_for_snapshot(self) -> None:
        if not _diarization_snapshot_preload_enabled():
            print("📸 Diarization memory snapshot enabled with no diarization preload target")
            self._ensure_diarization_service()
            return

        print("📸 Preparing diarization GPU memory snapshot")
        pipeline = self._ensure_diarization_service()._load_pipeline()
        if pipeline is None:
            raise RuntimeError("Speaker diarization pipeline failed to load for snapshot")
        print("✅ Diarization snapshot preload complete")

    @modal.enter()
    def restore_after_snapshot(self) -> None:
        self._ensure_diarization_service()
        print("♻️ Diarization GPU runtime restored and ready")

    @modal.method()
    def process(self, request_data: dict):
        if _infer_request_type(request_data) != "diarization":
            return {
                "processing_status": "failed",
                "error_message": "SpeakerDiarizationAudioRuntime only accepts diarization requests",
            }
        return _transcribe_and_diarization_audio_endpoint(
            request_data,
            transcription_service=None,
            diarization_service=self._ensure_diarization_service(),
        )


def _dispatch_audio_request(request_data: dict):
    request_type = _infer_request_type(request_data)
    if request_type == "transcribe":
        return TranscribeAudioRuntime().process.remote(request_data)
    if request_type == "diarization":
        return SpeakerDiarizationAudioRuntime().process.remote(request_data)
    return {
        "processing_status": "failed",
        "error_message": (
            f"Unknown request_type: {request_type}. Must be 'transcribe' or 'diarization'"
        ),
    }


# Compatibility entrypoint for existing CPU callers that still use
# ``modal.Function.from_name(..., "transcribe_and_diarization_audio_function")``.
# New callers can avoid this lightweight forwarding hop by using
# ``TranscribeAudioRuntime`` or ``SpeakerDiarizationAudioRuntime`` directly.
@app.function(
    image=dispatcher_image,
    cpu=0.25,
    memory=512,
    timeout=MODAL_GPU_TIMEOUT,
    scaledown_window=min(MODAL_GPU_SCALEDOWN_WINDOW, 60),
    secrets=secrets,
)
def transcribe_and_diarization_audio_function(request_data: dict):
    """
    Backward-compatible Modal function for SDK .call() / .spawn() users.

    The actual GPU work is routed to the dedicated ASR or diarization runtime.
    """
    return _dispatch_audio_request(request_data)

# Export GPU endpoint configuration singleton for service layer
GPU_ENDPOINT_CONFIG = {
    "gpu_type": MODAL_GPU_TYPE,
    "cpu_count": MODAL_GPU_CPU,
    "memory_mb": MODAL_GPU_MEMORY,
    "timeout_seconds": MODAL_GPU_TIMEOUT,
    "scaledown_window_seconds": MODAL_GPU_SCALEDOWN_WINDOW,
    "memory_snapshot_enabled": MODAL_GPU_ENABLE_MEMORY_SNAPSHOT,
    "gpu_snapshot_enabled": MODAL_GPU_ENABLE_GPU_SNAPSHOT,
    "snapshot_preload": MODAL_GPU_SNAPSHOT_PRELOAD,
    "snapshot_preload_targets": _resolve_snapshot_preload_targets(),
    "asr_snapshot_preload_targets": _asr_snapshot_preload_targets(),
    "diarization_snapshot_preload_enabled": _diarization_snapshot_preload_enabled(),
    "runtime_classes": {
        "asr": "TranscribeAudioRuntime",
        "diarization": "SpeakerDiarizationAudioRuntime",
    },
}
