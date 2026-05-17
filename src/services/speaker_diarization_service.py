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

    @staticmethod
    def _env_bool(name: str, default: bool = False) -> bool:
        value = os.getenv(name)
        if value is None:
            return default
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}

    @staticmethod
    def _positive_int(value: Any) -> Optional[int]:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return None
        return parsed if parsed > 0 else None

    def _diarization_batch_config(self, request_data: Optional[dict] = None) -> Dict[str, int]:
        request_data = request_data or {}
        hints = request_data.get("speaker_hints")
        if not isinstance(hints, dict):
            hints = {}

        segmentation_batch_size = (
            self._positive_int(request_data.get("segmentation_batch_size"))
            or self._positive_int(hints.get("segmentation_batch_size"))
            or self._positive_int(os.getenv("PYANNOTE_SEGMENTATION_BATCH_SIZE"))
            or 256
        )
        embedding_batch_size = (
            self._positive_int(request_data.get("embedding_batch_size"))
            or self._positive_int(hints.get("embedding_batch_size"))
            or self._positive_int(os.getenv("PYANNOTE_EMBEDDING_BATCH_SIZE"))
            or 256
        )
        return {
            "segmentation_batch_size": segmentation_batch_size,
            "embedding_batch_size": embedding_batch_size,
        }

    def _apply_pipeline_batch_config(self, pipeline: Any, batch_config: Dict[str, int]) -> None:
        segmentation_batch_size = batch_config["segmentation_batch_size"]
        embedding_batch_size = batch_config["embedding_batch_size"]

        if hasattr(pipeline, "segmentation_batch_size"):
            pipeline.segmentation_batch_size = segmentation_batch_size
        if hasattr(pipeline, "embedding_batch_size"):
            pipeline.embedding_batch_size = embedding_batch_size

        print(
            "⚙️ Pyannote batch config: "
            f"segmentation_batch_size={getattr(pipeline, 'segmentation_batch_size', segmentation_batch_size)}, "
            f"embedding_batch_size={getattr(pipeline, 'embedding_batch_size', embedding_batch_size)}"
        )

    def _speaker_count_kwargs(self, request_data: Optional[dict] = None) -> Dict[str, int]:
        request_data = request_data or {}
        hints = request_data.get("speaker_hints")
        if not isinstance(hints, dict):
            hints = {}

        num_speakers = (
            self._positive_int(request_data.get("num_speakers"))
            or self._positive_int(hints.get("num_speakers"))
        )
        if num_speakers:
            return {"num_speakers": num_speakers}

        min_speakers = (
            self._positive_int(request_data.get("min_speakers"))
            or self._positive_int(hints.get("min_speakers"))
        )
        max_speakers = (
            self._positive_int(request_data.get("max_speakers"))
            or self._positive_int(hints.get("max_speakers"))
        )
        if min_speakers and max_speakers and min_speakers > max_speakers:
            print(
                "⚠️ Ignoring invalid speaker bounds: "
                f"min_speakers={min_speakers}, max_speakers={max_speakers}"
            )
            return {}

        kwargs: Dict[str, int] = {}
        if min_speakers:
            kwargs["min_speakers"] = min_speakers
        if max_speakers:
            kwargs["max_speakers"] = max_speakers
        return kwargs

    def _pipeline_audio_input(self, audio_file_path: str) -> Any:
        if not self._env_bool("PYANNOTE_USE_MEMORY_INPUT", True):
            return audio_file_path

        try:
            import torchaudio

            waveform, sample_rate = torchaudio.load(audio_file_path)
            print(
                "📦 Loaded diarization audio into memory: "
                f"shape={tuple(waveform.shape)}, sample_rate={sample_rate}"
            )
            return {"waveform": waveform, "sample_rate": sample_rate}
        except Exception as e:
            print(f"⚠️ Failed to load audio into memory, falling back to file path: {e}")
            return audio_file_path
    
    def _load_pipeline(self):
        """
        Lazy load the pyannote.audio pipeline from the baked HF cache.

        The image build (``download_transcription_models``) preloads weights
        into ``HF_HOME=/model/hf-cache`` and the image sets
        ``HF_HUB_OFFLINE=1`` afterwards, so this loader does **not** need an
        HF token at runtime. If loading fails, the cache is broken and the
        operator should redeploy the GPU image.

        Returns:
            Pipeline instance, or ``None`` if the pipeline could not be loaded.
        """
        if self._pipeline is not None:
            return self._pipeline

        try:
            from pyannote.audio import Pipeline
            import torch

            # 启用 TF32 以提高性能和精度（pyannote.audio 的 fix_reproducibility
            # 内部使用旧 API 检查，所以这里也只能用旧 API，避免冲突）。
            if torch.cuda.is_available():
                try:
                    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
                        torch.backends.cuda.matmul.allow_tf32 = True
                    if hasattr(torch.backends.cudnn, "allow_tf32"):
                        torch.backends.cudnn.allow_tf32 = True
                    print("✅ TF32 enabled for better performance and accuracy (using legacy API)")
                except Exception as e:
                    print(f"⚠️ Failed to enable TF32: {e}")

            print("📥 Loading speaker diarization pipeline from baked HF cache...")
            self._pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-community-1"
            )

            if torch.cuda.is_available():
                self._pipeline.to(torch.device("cuda"))
                print("✅ Pipeline loaded and moved to GPU")
            else:
                print("⚠️ CUDA not available, using CPU")

            self._apply_pipeline_batch_config(
                self._pipeline,
                self._diarization_batch_config(),
            )

            return self._pipeline

        except Exception as e:
            print(f"❌ Failed to load pipeline from cache: {e}")
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
            audio_file_data = None
            del audio_bytes
            
            print(f"💾 Saved audio to temporary file: {temp_file_path}")
            
            # 3. 预处理音频（转换为标准格式）
            preprocessed_path = self._preprocess_audio_for_diarization(temp_file_path)
            preprocessed_file_created = (preprocessed_path != temp_file_path)

            # 3.5 Pipeline 健康检查：加载失败时直接 failed，避免上游把
            # "[] segments + status=success" 误读为 "音频里没说话人"。模型权重
            # 在镜像构建时就烘焙进了 HF 缓存，运行时使用 HF_HUB_OFFLINE=1，
            # 所以这里失败几乎只可能是镜像缓存损坏，需要重新部署 GPU 镜像。
            if self._load_pipeline() is None:
                msg = (
                    "Speaker diarization pipeline unavailable: failed to load "
                    "pyannote weights from the baked HF cache "
                    "(/model/hf-cache). The GPU image is likely missing the "
                    "build-time download step — redeploy the GPU app via the "
                    "transcribe-modal-bridge so the image is rebuilt with the "
                    "Hugging Face cache populated."
                )
                print(f"❌ {msg}")
                return {
                    "processing_status": "failed",
                    "error_message": msg,
                    "segments": [],
                }

            batch_config = self._diarization_batch_config(request_data)
            speaker_kwargs = self._speaker_count_kwargs(request_data)
            print(
                "🎚️ Diarization request config: "
                f"batch={batch_config}, speaker_kwargs={speaker_kwargs or '{}'}"
            )

            # 4. 调用 diarize_audio()
            segments = self.diarize_audio(
                preprocessed_path,
                batch_config=batch_config,
                speaker_kwargs=speaker_kwargs,
                already_preprocessed=True,
            )
            
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
                "segments": segments,
                "diarization_config": {
                    **batch_config,
                    **speaker_kwargs,
                    "memory_input": self._env_bool("PYANNOTE_USE_MEMORY_INPUT", True),
                },
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
    
    def diarize_audio(
        self,
        audio_file_path: str,
        batch_config: Optional[Dict[str, int]] = None,
        speaker_kwargs: Optional[Dict[str, int]] = None,
        already_preprocessed: bool = False,
    ) -> List[Dict]:
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
            
            batch_config = batch_config or self._diarization_batch_config()
            speaker_kwargs = speaker_kwargs or {}
            self._apply_pipeline_batch_config(pipeline, batch_config)

            # 2. 预处理音频（如果还没有预处理）
            preprocessed_path = (
                audio_file_path
                if already_preprocessed
                else self._preprocess_audio_for_diarization(audio_file_path)
            )
            print(f"🎤 Running speaker diarization on: {preprocessed_path}")

            pipeline_input = self._pipeline_audio_input(preprocessed_path)
            use_progress_hook = self._env_bool("PYANNOTE_PROGRESS_HOOK", True)

            # 3. 运行 diarization。Batch size 在 pipeline 属性上配置；speaker count
            # constraints 作为 apply 参数传入，pyannote 4.x 支持这些参数。
            ProgressHook = None
            if use_progress_hook:
                try:
                    from pyannote.audio.pipelines.utils.hook import ProgressHook
                except Exception as e:
                    print(f"⚠️ ProgressHook unavailable, running without hook: {e}")

            if ProgressHook is not None:
                with ProgressHook() as hook:
                    diarization_result = pipeline(
                        pipeline_input,
                        hook=hook,
                        **speaker_kwargs,
                    )
            else:
                diarization_result = pipeline(pipeline_input, **speaker_kwargs)
            
            
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
