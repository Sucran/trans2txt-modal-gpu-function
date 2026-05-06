"""
Whisper Service (GPU-side)
Handles audio transcription using Whisper model on GPU
"""

import whisper
import os
import re
import subprocess
from typing import Dict, Any, List


# Parse ffmpeg ``silencedetect`` stderr lines, e.g.
#   [silencedetect @ 0x...] silence_end: 12.345 | silence_duration: 0.789
# Mirrors the regex used in transcribe-modal-cpu's ``split_audio_by_silence``
# for parity. Tolerant of optional whitespace around the colon and pipe so
# minor ffmpeg-version differences in formatting do not silently break us.
_SILENCE_END_RE = re.compile(
    r"silence_end:\s*(?P<end>[0-9]+(?:\.[0-9]+)?)"
    r"\s*\|\s*"
    r"silence_duration:\s*(?P<dur>[0-9]+(?:\.[0-9]+)?)"
)


class WhisperService:
    """
    GPU-side Whisper transcription service
    Runs Whisper model and optionally extracts speaker embeddings
    """
    
    def __init__(self, cache_dir: str = "/tmp", model_size: str = "large-v3"):
        self.cache_dir = cache_dir
        self.model_size = model_size
        
        # Preload Whisper model at initialization
        print(f"🔄 Preloading Whisper model ({model_size}) at initialization...")
        self.model = self._load_cached_model(model_size)
        print(f"✅ Whisper model ({model_size}) preloaded successfully")
    
    def _load_cached_model(self, model_size: str = "large-v3"):
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

    @staticmethod
    def _detect_pauses(
        audio_file_path: str,
        *,
        min_dur_s: float,
        noise_db: int,
    ) -> List[Dict[str, int]]:
        """Run ffmpeg ``silencedetect`` against ``audio_file_path``.

        Returns a list of ``{start_ms, end_ms, dur_ms, mid_ms}`` intervals with
        all times **CHUNK-LOCAL** — relative to the wav passed in. The CPU
        orchestrator is responsible for shifting them to the global timeline
        with ``chunk_start_time``; do NOT apply that offset here.

        Failure isolation: any exception (ffmpeg missing, parse error, etc.)
        is caught and the function returns ``[]`` after logging. It must
        never propagate, since pause detection is best-effort and must not
        fail transcription.
        """
        try:
            cmd = [
                "ffmpeg",
                "-nostdin",
                "-hide_banner",
                "-i",
                audio_file_path,
                "-af",
                f"silencedetect=noise={int(noise_db)}dB:duration={float(min_dur_s)}",
                "-f",
                "null",
                "-",
            ]

            process = subprocess.Popen(
                cmd,
                stderr=subprocess.PIPE,
                stdout=subprocess.PIPE,
                text=True,
            )

            intervals: List[Dict[str, int]] = []
            stderr_stream = process.stderr
            if stderr_stream is not None:
                for line in stderr_stream:
                    try:
                        match = _SILENCE_END_RE.search(line)
                        if not match:
                            continue
                        silence_end = float(match.group("end"))
                        silence_dur = float(match.group("dur"))
                    except Exception as line_err:
                        print(
                            f"⚠️ pause line parse failed (skipped): {line_err!r}"
                        )
                        continue

                    end_ms = int(round(silence_end * 1000))
                    dur_ms = int(round(silence_dur * 1000))
                    if dur_ms <= 0 or end_ms <= 0:
                        continue
                    start_ms = end_ms - dur_ms
                    if start_ms < 0:
                        # Clamp at chunk start while preserving end_ms.
                        start_ms = 0
                        dur_ms = end_ms
                    mid_ms = start_ms + dur_ms // 2
                    intervals.append(
                        {
                            "start_ms": start_ms,
                            "end_ms": end_ms,
                            "dur_ms": dur_ms,
                            "mid_ms": mid_ms,
                        }
                    )

            try:
                process.wait()
            except Exception as wait_err:
                print(f"⚠️ ffmpeg wait failed (ignored): {wait_err!r}")

            intervals.sort(key=lambda item: item["start_ms"])
            return intervals

        except FileNotFoundError as err:
            print(f"⚠️ pause detection unavailable (ffmpeg missing?): {err!r}")
            return []
        except Exception as err:
            print(f"⚠️ pause detection failed for {audio_file_path}: {err!r}")
            return []

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

            # Pause detection (chunk-local). Failure must not break transcription.
            min_dur_s = _read_env_float("PAUSE_DETECT_MIN_DUR", 0.4)
            noise_db = _read_env_int("PAUSE_DETECT_NOISE_DB", -35)
            try:
                pause_intervals = self._detect_pauses(
                    audio_file_path,
                    min_dur_s=min_dur_s,
                    noise_db=noise_db,
                )
            except Exception as pause_err:
                # Defensive: _detect_pauses already swallows internally, but
                # keep an outer guard so a programming error here cannot
                # bubble up and fail the transcription path.
                print(f"⚠️ pause detection raised unexpectedly: {pause_err!r}")
                pause_intervals = []

            pause_detect_meta = {
                "min_dur_s": float(min_dur_s),
                "noise_db": int(noise_db),
                "schema_version": 1,
            }
            total_pause_ms = sum(
                int(item.get("dur_ms", 0) or 0) for item in pause_intervals
            )
            print(
                f"   Pause intervals (chunk-local): {len(pause_intervals)} "
                f"≈{total_pause_ms / 1000.0:.2f}s "
                f"(min_dur={min_dur_s}s noise={noise_db}dB)"
            )

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
                "segments": segments_list,
                "pause_intervals": pause_intervals,
                "pause_detect_meta": pause_detect_meta,
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
            "segments": [],
            "pause_intervals": [],
            "pause_detect_meta": {},
        }


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
        # Tolerate values like ``"-35.0"`` from older configs.
        return int(float(str(raw).strip()))
    except (TypeError, ValueError):
        return int(default)
