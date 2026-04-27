"""
Speaker Diarization Service
Handles speaker diarization using pyannote.audio for entire audio files
"""

import base64
import os
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional
import asyncio

try:
    import ffmpeg
    FFMPEG_AVAILABLE = True
except ImportError:
    FFMPEG_AVAILABLE = False


class SpeakerDiarizationService:
    """Service for speaker diarization using pyannote.audio"""
    
    def __init__(self, cache_dir: str = "/tmp"):
        """
        Initialize speaker diarization service
        
        Args:
            cache_dir: Cache directory for temporary files
        """
        self.cache_dir = cache_dir
        self._pipeline = None  # Lazy loading
    
    def _load_pipeline(self):
        """
        Lazy load pyannote.audio pipeline
        
        Returns:
            Pipeline instance or None if HF_TOKEN is not available
        """
        if self._pipeline is not None:
            return self._pipeline
        
        # 确保 HF_TOKEN 已设置
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_TOKEN")
        if not hf_token:
            print("⚠️ HF_TOKEN not found, speaker diarization will be disabled")
            return None
        
        try:
            from pyannote.audio import Pipeline
            import torch
            
            # 设置缓存目录
            os.environ["PYANNOTE_CACHE"] = "/model/speaker-diarization"
            
            # 启用 TF32 以提高性能和精度（根据 pyannote.audio 的建议）
            # 警告提示禁用 TF32 可能导致精度问题和较低准确性
            # 注意：必须只使用一种 API（旧 API），因为 pyannote.audio 内部使用旧 API 检查
            # 混合使用新旧 API 会导致 RuntimeError
            if torch.cuda.is_available():
                try:
                    # 只使用旧 API，避免与 pyannote.audio 内部检查冲突
                    # pyannote.audio 的 fix_reproducibility 函数使用旧 API 检查 TF32 状态
                    if hasattr(torch.backends.cuda.matmul, 'allow_tf32'):
                        torch.backends.cuda.matmul.allow_tf32 = True
                    if hasattr(torch.backends.cudnn, 'allow_tf32'):
                        torch.backends.cudnn.allow_tf32 = True
                    print("✅ TF32 enabled for better performance and accuracy (using legacy API)")
                except Exception as e:
                    print(f"⚠️ Failed to enable TF32: {e}")
            
            print("📥 Loading speaker diarization pipeline...")
            # 加载 pipeline
            self._pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-community-1"
            )
            
            # 移动到 GPU（如果可用）
            if torch.cuda.is_available():
                self._pipeline.to(torch.device("cuda"))
                print("✅ Pipeline loaded and moved to GPU")
            else:
                print("⚠️ CUDA not available, using CPU")
            
            return self._pipeline
            
        except Exception as e:
            print(f"❌ Failed to load pipeline: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _preprocess_audio_for_diarization(self, audio_file_path: str) -> str:
        """
        预处理音频文件，转换为 pyannote.audio 需要的标准格式
        
        pyannote.audio 要求：
        - 格式：WAV
        - 采样率：16000 Hz
        - 声道：单声道
        
        Args:
            audio_file_path: 原始音频文件路径
            
        Returns:
            预处理后的音频文件路径（如果不需要预处理，返回原路径）
        """
        if not FFMPEG_AVAILABLE:
            print("⚠️ FFmpeg not available, skipping audio preprocessing")
            return audio_file_path
        
        try:
            # 检查文件格式和采样率
            probe = ffmpeg.probe(audio_file_path)
            audio_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'audio'), None)
            
            if audio_stream is None:
                print("⚠️ No audio stream found, skipping preprocessing")
                return audio_file_path
            
            sample_rate = int(audio_stream.get('sample_rate', 0))
            channels = int(audio_stream.get('channels', 0))
            codec = audio_stream.get('codec_name', '')
            
            # 如果已经是标准格式，不需要预处理
            if (codec == 'pcm_s16le' and 
                sample_rate == 16000 and 
                channels == 1 and
                Path(audio_file_path).suffix.lower() == '.wav'):
                print("✅ Audio already in standard format (16kHz mono WAV), skipping preprocessing")
                return audio_file_path
            
            print(f"🔄 Preprocessing audio: {codec} {sample_rate}Hz {channels}ch -> WAV 16kHz mono")
            
            # 创建预处理后的临时文件
            preprocessed_path = tempfile.NamedTemporaryFile(
                delete=False,
                suffix='.wav',
                dir=self.cache_dir
            )
            preprocessed_path.close()
            preprocessed_file_path = preprocessed_path.name
            
            # 使用 FFmpeg 转换音频
            (
                ffmpeg
                .input(audio_file_path)
                .output(
                    preprocessed_file_path,
                    acodec='pcm_s16le',  # 16-bit PCM
                    ar=16000,           # 采样率 16000 Hz
                    ac=1                 # 单声道
                )
                .overwrite_output()
                .run(quiet=True, capture_stdout=True, capture_stderr=True)
            )
            
            print(f"✅ Audio preprocessed: {preprocessed_file_path}")
            return preprocessed_file_path
            
        except Exception as e:
            print(f"⚠️ Audio preprocessing failed: {e}, using original file")
            import traceback
            traceback.print_exc()
            return audio_file_path
    
    def process_diarization_request(self, request_data: dict) -> dict:
        """
        处理说话人识别请求
        
        Args:
            request_data: {
                "audio_file_data": base64_encoded_audio,
                "audio_file_name": "audio.wav"
            }
        
        Returns:
            {
                "processing_status": "success" | "failed",
                "segments": [
                    {"start": float, "end": float, "speaker": str},
                    ...
                ],
                "error_message": str (if failed)
            }
        """
        temp_file = None
        try:
            # 1. 解码 base64 音频数据
            audio_file_data = request_data.get("audio_file_data")
            audio_file_name = request_data.get("audio_file_name", "audio.wav")
            
            if not audio_file_data:
                return {
                    "processing_status": "failed",
                    "error_message": "audio_file_data is required",
                    "segments": []
                }
            
            print(f"📥 Decoding base64 audio data ({len(audio_file_data)} chars)...")
            try:
                audio_bytes = base64.b64decode(audio_file_data)
                print(f"✅ Decoded audio data: {len(audio_bytes)} bytes")
            except Exception as e:
                return {
                    "processing_status": "failed",
                    "error_message": f"Failed to decode base64 audio: {e}",
                    "segments": []
                }
            
            # 2. 保存到临时文件
            temp_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=Path(audio_file_name).suffix or ".wav",
                dir=self.cache_dir
            )
            temp_file.write(audio_bytes)
            temp_file.close()
            temp_file_path = temp_file.name
            
            print(f"💾 Saved audio to temporary file: {temp_file_path}")
            
            # 3. 预处理音频（转换为标准格式）
            preprocessed_path = self._preprocess_audio_for_diarization(temp_file_path)
            preprocessed_file_created = (preprocessed_path != temp_file_path)

            # 3.5 Pipeline 健康检查：HF_TOKEN 缺失或加载失败时直接 failed，
            # 避免上游把 "[] segments + status=success" 误读为 "音频里没说话人"。
            if self._load_pipeline() is None:
                hf_token_present = bool(
                    os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_TOKEN")
                )
                msg = (
                    "Speaker diarization pipeline unavailable: "
                    + ("pyannote pipeline failed to load" if hf_token_present
                       else "HF_TOKEN not found (huggingface-secret not attached?)")
                )
                print(f"❌ {msg}")
                return {
                    "processing_status": "failed",
                    "error_message": msg,
                    "segments": [],
                }

            # 4. 调用 diarize_audio()
            segments = self.diarize_audio(preprocessed_path)
            
            # 5. 清理临时文件
            try:
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
                    print(f"🧹 Cleaned up temporary file: {temp_file_path}")
                if preprocessed_file_created and os.path.exists(preprocessed_path):
                    os.unlink(preprocessed_path)
                    print(f"🧹 Cleaned up preprocessed file: {preprocessed_path}")
            except Exception as e:
                print(f"⚠️ Failed to clean up temp file: {e}")
            
            # 5. 返回结果
            return {
                "processing_status": "success",
                "segments": segments
            }
            
        except Exception as e:
            error_message = str(e)
            print(f"❌ Error in process_diarization_request: {error_message}")
            import traceback
            traceback.print_exc()
            
            # 清理临时文件
            if temp_file and os.path.exists(temp_file.name):
                try:
                    os.unlink(temp_file.name)
                except Exception:
                    pass
            
            return {
                "processing_status": "failed",
                "error_message": error_message,
                "segments": []
            }
    
    def diarize_audio(self, audio_file_path: str) -> List[Dict]:
        """
        执行说话人识别
        
        Args:
            audio_file_path: 音频文件路径
        
        Returns:
            List[Dict]: [{"start": float, "end": float, "speaker": str}, ...]
        """
        try:
            # 1. 加载 pipeline（lazy loading）
            pipeline = self._load_pipeline()
            if pipeline is None:
                print("⚠️ Pipeline not available, returning empty segments")
                return []
            
            # 检查文件是否存在
            if not os.path.exists(audio_file_path):
                raise FileNotFoundError(f"Audio file not found: {audio_file_path}")
            
            # 2. 预处理音频（如果还没有预处理）
            # 注意：如果是从 process_diarization_request 调用，音频已经预处理过
            # 但如果是直接调用 diarize_audio，需要在这里预处理
            preprocessed_path = self._preprocess_audio_for_diarization(audio_file_path)
            print(f"🎤 Running speaker diarization on: {preprocessed_path}")
            
            # 3. 运行 diarization（按照之前的代码，直接对结果调用 itertracks）
            diarization_result = pipeline(preprocessed_path)
            
            
            # 4. 格式化结果（按照之前的代码逻辑）
            segments = []
            for turn, speaker in diarization_result.exclusive_speaker_diarization:
                segments.append({
                    "start": turn.start,
                    "end": turn.end,
                    "speaker": speaker
                })
            
            print(f"✅ Speaker diarization completed: {len(segments)} segments, {len(set(s['speaker'] for s in segments))} speakers")
            
            # 4. 返回 segments 列表
            return segments
            
        except Exception as e:
            print(f"❌ Error in diarize_audio: {e}")
            import traceback
            traceback.print_exc()
            return []
