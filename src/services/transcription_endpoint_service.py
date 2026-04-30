"""
Transcription Endpoint Service
Handles processing of transcription endpoint requests (base64 decoding, file handling, service orchestration)
This is the Service layer that contains core business logic for endpoint request processing
"""

import base64
import os
import threading
import gc
from pathlib import Path
from typing import Dict, Any, Optional, Callable

from ..utils.file_utils import write_file_bytes, cleanup_temp_file, ensure_directory_exists_path

# 模块级缓存：缓存 WhisperService 实例以避免重复加载模型
# 限制：只能存在一个实例，如果 model_size 改变则释放旧实例
# 在 Modal 容器生命周期内，实例会被复用
# 使用线程锁保证并发安全
_cached_whisper_service: Optional[Any] = None
_cached_model_size: Optional[str] = None
_cache_lock = threading.Lock()


class TranscriptionEndpointService:
    """Service for processing transcription endpoint requests (Service layer)"""
    
    def __init__(self, cache_dir: str = "/tmp"):
        """
        Initialize transcription endpoint service
        
        Args:
            cache_dir: Cache directory for temporary files
        """
        self.cache_dir = cache_dir
    
    def _release_whisper_service(self, service: Any) -> None:
        """
        释放 WhisperService 实例，清理 GPU 显存
        
        Args:
            service: WhisperService 实例
        """
        try:
            print(f"🧹 Releasing WhisperService instance...")
            
            # 1. 释放 Whisper 模型
            if hasattr(service, 'model') and service.model is not None:
                try:
                    import torch
                    # 将模型移到 CPU（如果模型在 GPU 上）
                    if hasattr(service.model, 'to'):
                        service.model.to('cpu')
                    # 删除模型引用
                    del service.model
                    print(f"   ✅ Whisper model released")
                except Exception as e:
                    print(f"   ⚠️ Error releasing Whisper model: {e}")
            
            # 3. 清理 CUDA 缓存
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    print(f"   ✅ CUDA cache cleared")
            except Exception as e:
                print(f"   ⚠️ Error clearing CUDA cache: {e}")
            
            # 4. 删除服务实例引用
            del service
            
            # 5. 强制垃圾回收
            gc.collect()
            print(f"✅ WhisperService instance fully released")
            
        except Exception as e:
            print(f"⚠️ Error during WhisperService release: {e}")
    
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
        global _cached_whisper_service, _cached_model_size
        
        # 先检查缓存（不加锁，快速路径）
        if _cached_whisper_service is not None and _cached_model_size == model_size:
            print(f"♻️ Reusing cached WhisperService instance for model: {model_size}")
            return _cached_whisper_service
        
        # 需要创建新实例时，使用锁保证线程安全
        with _cache_lock:
            # 双重检查：可能在等待锁时其他线程已经创建了实例
            if _cached_whisper_service is not None and _cached_model_size == model_size:
                print(f"♻️ Reusing cached WhisperService instance for model: {model_size} (after lock)")
                return _cached_whisper_service
            
            # 如果 model_size 改变，需要释放旧实例
            if _cached_whisper_service is not None and _cached_model_size != model_size:
                print(f"🔄 Model size changed from {_cached_model_size} to {model_size}, releasing old instance...")
                # 使用专门的释放方法清理旧实例
                old_service = _cached_whisper_service
                _cached_whisper_service = None  # 先清空缓存引用
                _cached_model_size = None
                self._release_whisper_service(old_service)
            
            # 创建新实例并缓存
            print(f"🆕 Creating new WhisperService instance for model: {model_size}")
            from .whisper_service import WhisperService
            
            service = WhisperService(
                cache_dir=self.cache_dir,
                model_size=model_size
            )
            
            _cached_whisper_service = service
            _cached_model_size = model_size
            print(f"✅ WhisperService instance cached for model: {model_size}")
            return service
    
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
            # Get model_size from request, or use default from configuration
            model_size = request_data.get("model_size") or os.getenv("DEFAULT_MODEL_SIZE", "large-v3")
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
            print(f"   Time range: {chunk_start_time:.2f}s - {chunk_end_time:.2f}s")
            print(f"   Size: {len(audio_bytes) / (1024*1024):.2f} MB")
            
            try:
                # 使用缓存机制获取或创建 WhisperService 实例
                # 这样可以避免每次请求都重新加载模型到 GPU 显存
                service = self._get_or_create_whisper_service(model_size)
                
                result = service.transcribe_audio(
                    audio_file_path=str(temp_audio_path),
                    model_size=model_size,  # Use request model_size if provided, otherwise uses preloaded model
                    language=language,
                    enable_speaker_diarization=enable_speaker_diarization
                )
                
                # Add chunk timing information
                if result.get("processing_status") == "success":
                    result["chunk_start_time"] = chunk_start_time
                    result["chunk_end_time"] = chunk_end_time
                
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

