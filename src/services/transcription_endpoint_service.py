"""
Transcription Endpoint Service
Handles processing of transcription endpoint requests (base64 decoding, file handling, service orchestration)
This is the Service layer that contains core business logic for endpoint request processing
"""

import base64
import os
import threading
import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

from ..utils.file_utils import write_file_bytes, cleanup_temp_file, ensure_directory_exists_path

# Module-level cache: keep one GPU-heavy backend loaded per container.
# Switching between Whisper and Qwen releases the previous backend to avoid VRAM
# pressure from large-v3 + Qwen3-ASR + ForcedAligner co-residency.
@dataclass
class _CachedBackend:
    key: Tuple[str, str]
    service: Any


_cached_backend: Optional[_CachedBackend] = None
_cache_lock = threading.Lock()


def _read_env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return float(default)
    try:
        return float(str(raw).strip())
    except (TypeError, ValueError):
        return float(default)


def _read_env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return int(default)
    try:
        return int(float(str(raw).strip()))
    except (TypeError, ValueError):
        return int(default)


class TranscriptionEndpointService:
    """Service for processing transcription endpoint requests (Service layer)"""
    
    def __init__(self, cache_dir: str = "/tmp"):
        """
        Initialize transcription endpoint service
        
        Args:
            cache_dir: Cache directory for temporary files
        """
        self.cache_dir = cache_dir
    
    def _release_service(self, service: Any) -> None:
        """
        Release a backend service and clear GPU memory.
        
        Args:
            service: backend service instance
        """
        try:
            print(f"🧹 Releasing backend service instance...")

            if hasattr(service, "release"):
                service.release()
                return
            
            if hasattr(service, 'model') and service.model is not None:
                try:
                    import torch
                    if hasattr(service.model, 'to'):
                        service.model.to('cpu')
                    del service.model
                    print(f"   ✅ Backend model released")
                except Exception as e:
                    print(f"   ⚠️ Error releasing backend model: {e}")
            
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    print(f"   ✅ CUDA cache cleared")
            except Exception as e:
                print(f"   ⚠️ Error clearing CUDA cache: {e}")
            
            del service
            
            gc.collect()
            print(f"✅ Backend service instance fully released")
            
        except Exception as e:
            print(f"⚠️ Error during backend service release: {e}")

    def _replace_backend(self, key: Tuple[str, str], service_factory: Any) -> Any:
        global _cached_backend

        if _cached_backend is not None and _cached_backend.key == key:
            print(f"♻️ Reusing cached backend service: {key}")
            return _cached_backend.service

        with _cache_lock:
            if _cached_backend is not None and _cached_backend.key == key:
                print(f"♻️ Reusing cached backend service after lock: {key}")
                return _cached_backend.service

            if _cached_backend is not None:
                print(f"🔄 Backend changed from {_cached_backend.key} to {key}, releasing old service...")
                old_service = _cached_backend.service
                _cached_backend = None
                self._release_service(old_service)

            print(f"🆕 Creating backend service: {key}")
            service = service_factory()
            _cached_backend = _CachedBackend(key=key, service=service)
            print(f"✅ Backend service cached: {key}")
            return service
    
    def _get_or_create_whisper_service(self, model_size: str) -> Any:
        """
        获取或创建 WhisperService 实例（使用缓存避免重复加载模型）
        限制：只能存在一个实例，如果 model_size 改变则释放旧实例
        线程安全，支持并发请求
        
        Args:
            model_size: Whisper model size
            
        Returns:
            WhisperService 实例
        """
        return self._replace_backend(
            ("whisper", model_size),
            lambda: self._create_whisper_service(model_size),
        )

    def _create_whisper_service(self, model_size: str) -> Any:
        from .whisper_service import WhisperService

        return WhisperService(
            cache_dir=self.cache_dir,
            model_size=model_size,
        )

    def _get_or_create_qwen_service(self, asr_model_id: str) -> Any:
        aligner_model_id = os.getenv(
            "QWEN_FORCED_ALIGNER_MODEL_ID",
            "Qwen/Qwen3-ForcedAligner-0.6B",
        )
        return self._replace_backend(
            ("qwen3_asr", f"{asr_model_id}|{aligner_model_id}"),
            lambda: self._create_qwen_service(asr_model_id, aligner_model_id),
        )

    def _create_qwen_service(self, asr_model_id: str, aligner_model_id: str) -> Any:
        from .qwen_asr_service import QwenAsrService

        return QwenAsrService(
            cache_dir=self.cache_dir,
            asr_model_id=asr_model_id,
            aligner_model_id=aligner_model_id,
        )
    
    def process_chunk_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process chunk transcription request - decode base64, save file, transcribe chunk
        
        This is the Service layer method that handles chunk processing logic.
        Called by Modal endpoints.
        
        Args:
            request_data: Chunk transcription request data dictionary
            
        Returns:
            Transcription result dictionary with chunk information
        """
        try:
            # Extract request parameters
            audio_file_data = request_data.get("audio_file_data")
            audio_file_name = request_data.get("audio_file_name", "chunk.mp3")
            # Get model_size from request, or use default from configuration.
            # model_size remains Whisper-only; Qwen uses asr_model_id.
            model_size = request_data.get("model_size") or os.getenv("DEFAULT_MODEL_SIZE", "large-v3")
            asr_backend = str(
                request_data.get("asr_backend")
                or os.getenv("ASR_BACKEND", "whisper")
            ).strip().lower()
            asr_model_id = request_data.get("asr_model_id") or os.getenv(
                "QWEN_ASR_MODEL_ID",
                "Qwen/Qwen3-ASR-0.6B",
            )
            language = request_data.get("language", "auto")  # Default to "auto" for automatic detection
            enable_speaker_diarization = request_data.get("enable_speaker_diarization", False)
            chunk_start_time = request_data.get("chunk_start_time", 0)
            chunk_end_time = request_data.get("chunk_end_time", 0)
            
            if not audio_file_data:
                return {
                    "processing_status": "failed",
                    "error_message": "No audio data provided",
                    "chunk_start_time": chunk_start_time,
                    "chunk_end_time": chunk_end_time
                }
            
            # Decode audio data and save to temporary file
            audio_bytes = base64.b64decode(audio_file_data)
            temp_dir = Path(self.cache_dir)
            ensure_directory_exists_path(temp_dir)
            
            temp_audio_path = temp_dir / audio_file_name
            write_file_bytes(str(temp_audio_path), audio_bytes)
            
            print(f"🎤 Processing chunk on server: {audio_file_name}")
            print(f"   ASR backend: {asr_backend}")
            print(f"   Time range: {chunk_start_time:.2f}s - {chunk_end_time:.2f}s")
            print(f"   Size: {len(audio_bytes) / (1024*1024):.2f} MB")
            
            try:
                if asr_backend == "qwen3_asr":
                    result = self._transcribe_with_qwen_or_fallback(
                        audio_file_path=str(temp_audio_path),
                        asr_model_id=asr_model_id,
                        model_size=model_size,
                        language=language,
                        enable_speaker_diarization=enable_speaker_diarization,
                    )
                elif asr_backend == "whisper":
                    result = self._transcribe_with_whisper(
                        audio_file_path=str(temp_audio_path),
                        model_size=model_size,
                        language=language,
                        enable_speaker_diarization=enable_speaker_diarization,
                    )
                else:
                    return {
                        "processing_status": "failed",
                        "error_message": f"Unknown asr_backend: {asr_backend}",
                        "chunk_start_time": chunk_start_time,
                        "chunk_end_time": chunk_end_time,
                    }
                
                # Add chunk timing information
                if result.get("processing_status") == "success":
                    result["chunk_start_time"] = chunk_start_time
                    result["chunk_end_time"] = chunk_end_time

                    # Ensure pause fields are present. ``transcribe_audio``
                    # already attaches them on the success path, but a legacy
                    # ``WhisperService`` (or a future variant) might not. As a
                    # defensive fallback, run silencedetect best-effort here
                    # and degrade to ``[]`` on any failure.
                    if "pause_intervals" not in result:
                        min_dur_s = _read_env_float(
                            "PAUSE_DETECT_MIN_DUR", 0.4
                        )
                        noise_db = _read_env_int(
                            "PAUSE_DETECT_NOISE_DB", -35
                        )
                        try:
                            from .whisper_service import WhisperService

                            fallback_intervals = WhisperService._detect_pauses(
                                str(temp_audio_path),
                                min_dur_s=min_dur_s,
                                noise_db=noise_db,
                            )
                        except Exception as fallback_err:
                            print(
                                "⚠️ pause-detect fallback raised, degrading to []: "
                                f"{fallback_err!r}"
                            )
                            fallback_intervals = []
                        result["pause_intervals"] = fallback_intervals
                        result.setdefault(
                            "pause_detect_meta",
                            {
                                "min_dur_s": float(min_dur_s),
                                "noise_db": int(noise_db),
                                "schema_version": 1,
                            },
                        )

                    pause_intervals = result.get("pause_intervals") or []
                    total_pause_ms = sum(
                        int(item.get("dur_ms", 0) or 0)
                        for item in pause_intervals
                    )
                    print(
                        f"🛑 chunk pause_intervals count={len(pause_intervals)} "
                        f"total≈{total_pause_ms / 1000.0:.2f}s "
                        f"(chunk_start={chunk_start_time:.2f}s)"
                    )

                print(f"✅ Chunk transcription completed on server")
                return result
                
            finally:
                # Clean up temporary file
                cleanup_temp_file(temp_audio_path)
            
        except Exception as e:
            print(f"❌ Error processing chunk request: {e}")
            return {
                "processing_status": "failed",
                "error_message": f"Server chunk processing error: {str(e)}",
                "chunk_start_time": request_data.get("chunk_start_time", 0),
                "chunk_end_time": request_data.get("chunk_end_time", 0)
            }

    def _transcribe_with_whisper(
        self,
        audio_file_path: str,
        model_size: str,
        language: Any,
        enable_speaker_diarization: bool,
    ) -> Dict[str, Any]:
        service = self._get_or_create_whisper_service(model_size)
        return service.transcribe_audio(
            audio_file_path=audio_file_path,
            model_size=model_size,
            language=language,
            enable_speaker_diarization=enable_speaker_diarization,
        )

    def _transcribe_with_qwen_or_fallback(
        self,
        audio_file_path: str,
        asr_model_id: str,
        model_size: str,
        language: Any,
        enable_speaker_diarization: bool,
    ) -> Dict[str, Any]:
        try:
            from .qwen_asr_service import AlignerLanguageUnsupported

            service = self._get_or_create_qwen_service(asr_model_id)
            return service.transcribe_audio(
                audio_file_path=audio_file_path,
                language=language,
                enable_speaker_diarization=enable_speaker_diarization,
            )
        except AlignerLanguageUnsupported as exc:
            print(
                "⚠️ Qwen3 ForcedAligner language unsupported; "
                f"falling back to Whisper {model_size}: {exc}"
            )
            result = self._transcribe_with_whisper(
                audio_file_path=audio_file_path,
                model_size=model_size,
                # The original language may be unsupported only by Qwen's
                # aligner. Let Whisper auto-detect so fallback remains useful
                # for real audio and for the unsupported-language smoke test.
                language="auto",
                enable_speaker_diarization=enable_speaker_diarization,
            )
            if result.get("processing_status") == "success":
                result["model_used"] = f"qwen3_asr_fallback_whisper:{model_size}"
            return result
