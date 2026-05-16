"""
Modal GPU endpoint configuration
Handles GPU endpoint deployment and configuration
"""

import os
import sys
from pathlib import Path

import modal

# `modal deploy` on this file loads it as a top-level module (not as `src.config.*`),
# so `from .modal_shared` fails. Prefer imports from the repo root.
_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

# Import shared configuration
from src.config.modal_shared import (
    app,
    volume,
    cache_dir,
    transcription_image,
    secrets,
)

# GPU endpoint configuration from environment variables
MODAL_GPU_TYPE = os.getenv("MODAL_GPU_TYPE", "T4")
MODAL_GPU_CPU = int(os.getenv("MODAL_CPU", "4"))
MODAL_GPU_MEMORY = int(os.getenv("MODAL_MEMORY", "8192"))  # Default 8GB
MODAL_GPU_TIMEOUT = int(os.getenv("MODAL_GPU_TIMEOUT", "1800"))  # 30 minutes

print(f"🔧 Modal GPU Configuration:")
print(f"   GPU Type: {MODAL_GPU_TYPE}")
print(f"   CPU: {MODAL_GPU_CPU}")
print(f"   Memory: {MODAL_GPU_MEMORY}MB")
print(f"   Timeout: {MODAL_GPU_TIMEOUT}s")

# ==================== Unified GPU Endpoint Configuration ====================

def _transcribe_and_diarization_audio_endpoint(request_data: dict):
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
    
    # Determine request type from request_data
    request_type = request_data.get("request_type")
    
    # If request_type is not explicitly set, infer from request data structure
    if not request_type:
        # Check for transcription-specific fields
        if (
            "chunk_start_time" in request_data
            or "chunk_end_time" in request_data
            or "model_size" in request_data
            or "asr_backend" in request_data
            or "asr_model_id" in request_data
        ):
            request_type = "transcribe"
        else:
            # Default to diarization if no transcription-specific fields found
            request_type = "diarization"
    
    if request_type == "transcribe":
        # Handle transcription request
        from src.services.transcription_endpoint_service import TranscriptionEndpointService
        
        service = TranscriptionEndpointService(cache_dir=str(cache_dir))
        return service.process_chunk_request(request_data)
    
    elif request_type == "diarization":
        # Handle speaker diarization request
        from src.services.speaker_diarization_service import SpeakerDiarizationService
        
        service = SpeakerDiarizationService(cache_dir=str(cache_dir))
        return service.process_diarization_request(request_data)
    
    else:
        return {
            "processing_status": "failed",
            "error_message": f"Unknown request_type: {request_type}. Must be 'transcribe' or 'diarization'"
        }

# Pure Modal function for SDK .call() / .spawn() only (no HTTP; long work fits Modal timeouts, not request gateways)
@app.function(
    region=["ap"],
    image=transcription_image,
    volumes={cache_dir: volume},
    cpu=MODAL_GPU_CPU,
    memory=MODAL_GPU_MEMORY,
    gpu=MODAL_GPU_TYPE,
    timeout=MODAL_GPU_TIMEOUT,
    scaledown_window=40,  # 15 minutes before scaling down
    secrets=secrets,
)
def transcribe_and_diarization_audio_function(request_data: dict):
    """
    Transcribe and diarization audio function: Pure Modal function for both transcription and speaker diarization
    Can be used with SDK .spawn() calls for both types of requests
    
    The function automatically determines the request type from the request data:
    - If 'chunk_start_time', 'chunk_end_time', or 'model_size' is present -> transcription
    - Otherwise -> speaker diarization
    - Or explicitly set 'request_type' to 'transcribe' or 'diarization'
    """
    return _transcribe_and_diarization_audio_endpoint(request_data)

# Export GPU endpoint configuration singleton for service layer
GPU_ENDPOINT_CONFIG = {
    "gpu_type": MODAL_GPU_TYPE,
    "cpu_count": MODAL_GPU_CPU,
    "memory_mb": MODAL_GPU_MEMORY,
    "timeout_seconds": MODAL_GPU_TIMEOUT,
}
