"""
Whisper Service (GPU-side)
Handles audio transcription using Whisper model on GPU
"""

import whisper
import os
from typing import Dict, Any, List


class WhisperService:
    """
    GPU-side Whisper transcription service
    Runs Whisper model and optionally extracts speaker embeddings
    """
    
    def __init__(self, cache_dir: str = "/tmp", model_size: str = "turbo"):
        self.cache_dir = cache_dir
        self.model_size = model_size
        
        # Preload Whisper model at initialization
        print(f"🔄 Preloading Whisper model ({model_size}) at initialization...")
        self.model = self._load_cached_model(model_size)
        print(f"✅ Whisper model ({model_size}) preloaded successfully")
    
    def _load_cached_model(self, model_size: str = "turbo"):
        """Load Whisper model from cache directory if available"""
        try:
            # Try to load from preloaded cache first
            model_cache_dir = "/model"
            if os.path.exists(model_cache_dir):
                print(f"📦 Loading {model_size} model from cache: {model_cache_dir}")
                # Set download root to cache directory
                model = whisper.load_model(model_size, download_root=model_cache_dir)
                print(f"✅ Successfully loaded {model_size} model from cache")
                return model
            else:
                print(f"⚠️ Cache directory not found, downloading {model_size} model...")
                return whisper.load_model(model_size)
        except Exception as e:
            print(f"⚠️ Failed to load cached model, downloading: {e}")
            return whisper.load_model(model_size)
    
    def transcribe_audio(
        self,
        audio_file_path: str,
        model_size: str = None,
        language: str = None,
        enable_speaker_diarization: bool = False
    ) -> Dict[str, Any]:
        """
        Transcribe audio file using Whisper with optional speaker diarization and embedding extraction
        
        Args:
            audio_file_path: Path to audio file
            model_size: Whisper model size (if None, uses preloaded model)
            language: Language code (optional)
            enable_speaker_diarization: Enable speaker identification and embedding extraction
            
        Returns:
            Transcription result dictionary with segments (including embeddings only if diarization enabled)
            Note: In distributed processing scenarios, file paths are not returned as files are generated on client side
        """
        try:
            # Use model_size parameter or fall back to preloaded model
            if model_size is None:
                model_size = self.model_size
                model = self.model
            elif model_size == self.model_size:
                # Requested model matches preloaded model
                model = self.model
            else:
                # Different model requested, need to load it
                print(f"⚠️ Requested model {model_size} differs from preloaded {self.model_size}, loading new model...")
                model = self._load_cached_model(model_size)
            
            print(f"🎤 [WhisperService] Starting transcription for: {audio_file_path}")
            print(f"🚀 Using model: {model_size}")
            print(f"🎤 Speaker diarization: {enable_speaker_diarization}")
        
            # Check if file exists
            if not os.path.exists(audio_file_path):
                raise FileNotFoundError(f"Audio file not found: {audio_file_path}")
            
            # Transcribe audio
            # Convert "auto" to None for automatic language detection
            transcribe_language = None if (language is None or str(language).strip().lower() == "auto") else language
            
            transcribe_options = {
                "language": transcribe_language,
                "task": "transcribe",
                "verbose": True
            }
            
            print(f"🔄 Transcribing with options: {transcribe_options}")
            result = model.transcribe(audio_file_path, **transcribe_options)
            
            # Extract information
            text = result.get("text", "").strip()
            segments = result.get("segments", [])
            language_detected = result.get("language", "unknown")
            
            # Speaker diarization is disabled - services have been removed
            global_speaker_count = 0
            speaker_summary = {}
            if enable_speaker_diarization:
                print("⚠️ Speaker diarization parameter is set but speaker services have been removed")
                enable_speaker_diarization = False
            
            # Get audio duration
            audio_duration = 0.0
            if segments:
                audio_duration = max(seg.get("end", 0) for seg in segments)
            
            print(f"✅ Transcription completed successfully")
            print(f"   Text length: {len(text)} characters")
            print(f"   Segments: {len(segments)}")
            print(f"   Duration: {audio_duration:.2f}s")
            print(f"   Language: {language_detected}")
            
            # Build segments list
            segments_list = []
            for seg in segments:
                segment_dict = {
                    "start": seg.get("start", 0),
                    "end": seg.get("end", 0),
                    "text": seg.get("text", "").strip(),
                    "speaker": seg.get("speaker", None)
                }
                segments_list.append(segment_dict)
            
            return {
                "model_used": model_size,
                "segment_count": len(segments),
                "audio_duration": audio_duration,
                "processing_status": "success",
                "speaker_diarization_enabled": enable_speaker_diarization,
                "global_speaker_count": global_speaker_count,
                "speaker_summary": speaker_summary,
                "language_detected": language_detected,
                "text": text,
                "segments": segments_list
            }
            
        except Exception as e:
            print(f"❌ Transcription failed: {e}")
            return self._create_error_result(audio_file_path, model_size, str(e))
    
    def _create_error_result(self, audio_file_path: str, model_size: str, error_message: str) -> Dict[str, Any]:
        """Create error result dictionary"""
        return {
            "model_used": model_size,
            "segment_count": 0,
            "audio_duration": 0,
            "processing_status": "failed",
            "speaker_diarization_enabled": False,
            "global_speaker_count": 0,
            "speaker_summary": {},
            "language_detected": "unknown",
            "error_message": error_message,
            "text": "",
            "segments": []
        }

