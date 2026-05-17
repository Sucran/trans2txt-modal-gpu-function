"""
Qwen3-ASR service for GPU-side transcription.

The CPU merge layer requires every successful chunk to return timestamped
segments. For Qwen3-ASR that means the ForcedAligner is part of the runtime
contract, not an optional enhancement.
"""

from __future__ import annotations

import gc
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import ffmpeg


MAX_ALIGNER_SEGMENT_SECONDS = int(os.getenv("QWEN_ALIGNER_MAX_SEGMENT_SECONDS", "60"))
DEFAULT_SUBTITLE_MAX_SECONDS = 1.6
DEFAULT_SUBTITLE_MAX_CHARS = 12
DEFAULT_SUBTITLE_GAP_SECONDS = 0.35
DEFAULT_VLLM_BATCH_SIZE = 4
DEFAULT_VLLM_GPU_MEMORY_UTILIZATION = 0.70

SENTENCE_ENDING_PUNCTUATION = set(".!?。！？…")
CLAUSE_ENDING_PUNCTUATION = set(",;:，；：、")
NO_SPACE_BEFORE = set(".,!?;:%)]}，。！？；：、》）】”’")
NO_SPACE_AFTER = set("([{\"'“‘《（【")
ALIGNMENT_PUNCTUATION = SENTENCE_ENDING_PUNCTUATION | CLAUSE_ENDING_PUNCTUATION | NO_SPACE_BEFORE | NO_SPACE_AFTER

ALIGNER_LANGUAGE_NAMES = {
    "chinese",
    "english",
    "cantonese",
    "french",
    "german",
    "italian",
    "japanese",
    "korean",
    "portuguese",
    "russian",
    "spanish",
}

LANGUAGE_ALIASES = {
    "zh": "Chinese",
    "zh-cn": "Chinese",
    "zh-hans": "Chinese",
    "zh-hant": "Chinese",
    "cn": "Chinese",
    "chinese": "Chinese",
    "mandarin": "Chinese",
    "en": "English",
    "en-us": "English",
    "en-gb": "English",
    "english": "English",
    "yue": "Cantonese",
    "cantonese": "Cantonese",
    "fr": "French",
    "french": "French",
    "de": "German",
    "german": "German",
    "it": "Italian",
    "italian": "Italian",
    "ja": "Japanese",
    "jp": "Japanese",
    "japanese": "Japanese",
    "ko": "Korean",
    "korean": "Korean",
    "pt": "Portuguese",
    "portuguese": "Portuguese",
    "ru": "Russian",
    "russian": "Russian",
    "es": "Spanish",
    "spanish": "Spanish",
}


class AlignerLanguageUnsupported(Exception):
    """Raised when Qwen3-ASR would produce a language the aligner cannot handle."""


class QwenTimestampParseError(Exception):
    """Raised when Qwen timestamp objects cannot be mapped into CPU segments."""


def _normalise_language(language: Optional[str]) -> Optional[str]:
    if language is None:
        return None
    value = str(language).strip()
    if not value or value.lower() == "auto":
        return None
    return LANGUAGE_ALIASES.get(value.lower(), value)


def _normalise_context(context: Optional[str]) -> str:
    if context is None:
        return ""
    return str(context).strip()


def _is_aligner_language(language: Optional[str]) -> bool:
    if language is None:
        return True
    return str(language).strip().lower() in ALIGNER_LANGUAGE_NAMES


def _get_attr_or_key(item: Any, *names: str) -> Any:
    for name in names:
        if isinstance(item, dict) and name in item:
            return item[name]
        if hasattr(item, name):
            return getattr(item, name)
    return None


def _read_env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return float(default)
    try:
        value = float(str(raw).strip())
    except (TypeError, ValueError):
        return float(default)
    return value if value > 0 else float(default)


def _read_env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return int(default)
    try:
        value = int(float(str(raw).strip()))
    except (TypeError, ValueError):
        return int(default)
    return value if value > 0 else int(default)


def _normalise_qwen_runtime(runtime: Optional[str]) -> str:
    value = str(runtime or "transformers").strip().lower().replace("-", "_")
    return "vllm" if value == "vllm" else "transformers"


def _cuda_supports_bf16(torch_module: Any) -> bool:
    try:
        cuda = getattr(torch_module, "cuda", None)
        return bool(
            cuda
            and cuda.is_available()
            and hasattr(cuda, "is_bf16_supported")
            and cuda.is_bf16_supported()
        )
    except Exception:
        return False


def _select_torch_dtype_name(torch_module: Any) -> str:
    return "bfloat16" if _cuda_supports_bf16(torch_module) else "float16"


def _resolve_torch_dtype_name(
    torch_module: Any,
    raw_value: Optional[str],
    default: str = "auto",
) -> str:
    raw = str(raw_value or default or "auto").strip().lower()
    aliases = {
        "auto": "auto",
        "bf16": "bfloat16",
        "bfloat16": "bfloat16",
        "fp16": "float16",
        "float16": "float16",
        "half": "float16",
    }
    value = aliases.get(raw, aliases.get(str(default).strip().lower(), "auto"))
    if value == "auto":
        return _select_torch_dtype_name(torch_module)
    if value == "bfloat16" and not _cuda_supports_bf16(torch_module):
        print(
            "⚠️ Requested bfloat16 but CUDA device does not support BF16; "
            "falling back to float16"
        )
        return "float16"
    return value


def _torch_dtype_from_name(torch_module: Any, dtype_name: str) -> Any:
    if dtype_name == "bfloat16":
        return torch_module.bfloat16
    return torch_module.float16


def _contains_cjk(text: str) -> bool:
    return any(
        "\u3400" <= char <= "\u9fff"
        or "\uf900" <= char <= "\ufaff"
        for char in text
    )


def _join_piece(left: str, right: str) -> str:
    right = right.strip()
    if not right:
        return left
    if not left:
        return right

    left_tail = left[-1]
    right_head = right[0]
    if (
        right_head in NO_SPACE_BEFORE
        or left_tail in NO_SPACE_AFTER
        or _contains_cjk(left_tail)
        or _contains_cjk(right_head)
    ):
        return f"{left}{right}"
    return f"{left} {right}"


def _ends_sentence(text: str) -> bool:
    stripped = text.rstrip()
    return bool(stripped) and stripped[-1] in SENTENCE_ENDING_PUNCTUATION


def _ends_clause(text: str) -> bool:
    stripped = text.rstrip()
    return bool(stripped) and stripped[-1] in CLAUSE_ENDING_PUNCTUATION


def _is_alignment_ignorable(char: str) -> bool:
    return char.isspace() or char in ALIGNMENT_PUNCTUATION


def _select_torch_dtype(torch_module: Any) -> Any:
    """Pick the best inference dtype for the active CUDA device.

    T4 does not support BF16, while newer cards such as L4/A10G/A100 do. The
    Modal test deployment intentionally stays on T4, so Qwen must use FP16
    there instead of hard-coding BF16.
    """

    return _torch_dtype_from_name(torch_module, _select_torch_dtype_name(torch_module))


class QwenAsrService:
    """Qwen3-ASR backend with mandatory ForcedAligner timestamps."""

    def __init__(
        self,
        cache_dir: str = "/tmp",
        asr_model_id: str = "Qwen/Qwen3-ASR-1.7B",
        aligner_model_id: str = "Qwen/Qwen3-ForcedAligner-0.6B",
    ):
        self.cache_dir = cache_dir
        self.asr_model_id = asr_model_id
        self.aligner_model_id = aligner_model_id
        self.runtime = _normalise_qwen_runtime(
            os.getenv("QWEN_ASR_RUNTIME", "transformers")
        )
        self.inference_batch_size = self._configured_inference_batch_size()
        print(
            "🔄 Preloading Qwen3-ASR model "
            f"({asr_model_id}) runtime={self.runtime} "
            f"batch={self.inference_batch_size} "
            f"with ForcedAligner ({aligner_model_id})..."
        )
        self.model = self._load_model()
        print("✅ Qwen3-ASR + ForcedAligner preloaded successfully")

    def _configured_inference_batch_size(self) -> int:
        if self.runtime == "vllm":
            return _read_env_int("QWEN_VLLM_BATCH_SIZE", DEFAULT_VLLM_BATCH_SIZE)
        return _read_env_int("QWEN_TRANSFORMERS_BATCH_SIZE", 1)

    def _load_model(self) -> Any:
        if self.runtime == "vllm":
            return self._load_vllm_model()
        return self._load_transformers_model()

    def _load_transformers_model(self) -> Any:
        import torch
        from qwen_asr import Qwen3ASRModel

        dtype = _select_torch_dtype(torch)
        print(f"   Qwen3-ASR transformers dtype selected: {dtype}")
        return Qwen3ASRModel.from_pretrained(
            self.asr_model_id,
            dtype=dtype,
            device_map="cuda:0",
            forced_aligner=self.aligner_model_id,
            forced_aligner_kwargs={
                "dtype": dtype,
                "device_map": "cuda:0",
            },
            max_inference_batch_size=self.inference_batch_size,
            max_new_tokens=int(os.getenv("QWEN_MAX_NEW_TOKENS", "1024")),
        )

    def _load_vllm_model(self) -> Any:
        import torch
        from qwen_asr import Qwen3ASRModel

        dtype_name = _resolve_torch_dtype_name(
            torch,
            os.getenv("QWEN_VLLM_DTYPE"),
            default="bfloat16",
        )
        aligner_dtype_name = _resolve_torch_dtype_name(
            torch,
            os.getenv("QWEN_ALIGNER_DTYPE"),
            default=dtype_name,
        )
        aligner_dtype = _torch_dtype_from_name(torch, aligner_dtype_name)
        gpu_memory_utilization = _read_env_float(
            "QWEN_VLLM_GPU_MEMORY_UTILIZATION",
            DEFAULT_VLLM_GPU_MEMORY_UTILIZATION,
        )
        kwargs: Dict[str, Any] = {
            "dtype": dtype_name,
            "gpu_memory_utilization": gpu_memory_utilization,
        }
        max_model_len = _read_env_int("QWEN_VLLM_MAX_MODEL_LEN", 0)
        if max_model_len > 0:
            kwargs["max_model_len"] = max_model_len

        print(
            "   Qwen3-ASR vLLM config: "
            f"dtype={dtype_name}, aligner_dtype={aligner_dtype_name}, "
            f"gpu_memory_utilization={gpu_memory_utilization}, "
            f"batch={self.inference_batch_size}"
        )
        return Qwen3ASRModel.LLM(
            model=self.asr_model_id,
            forced_aligner=self.aligner_model_id,
            forced_aligner_kwargs={
                "dtype": aligner_dtype,
                "device_map": "cuda:0",
            },
            max_inference_batch_size=self.inference_batch_size,
            max_new_tokens=int(os.getenv("QWEN_MAX_NEW_TOKENS", "1024")),
            **kwargs,
        )

    def transcribe_audio(
        self,
        audio_file_path: str,
        language: Optional[str] = None,
        context: Optional[str] = None,
        enable_speaker_diarization: bool = False,
    ) -> Dict[str, Any]:
        try:
            if enable_speaker_diarization:
                print("⚠️ Qwen3-ASR chunk diarization is ignored; CPU merges full-audio pyannote diarization")

            requested_language = _normalise_language(language)
            context_text = _normalise_context(context)
            if requested_language and not _is_aligner_language(requested_language):
                raise AlignerLanguageUnsupported(
                    f"Qwen3-ForcedAligner does not support requested language: {requested_language}"
                )

            duration = self._probe_duration(audio_file_path)
            print(
                f"🎤 [QwenAsrService] Starting transcription: {audio_file_path} "
                f"duration={duration:.2f}s language={requested_language or 'auto'} "
                f"context={'on' if context_text else 'off'} len={len(context_text)}"
            )

            raw_segments: List[Dict[str, Any]] = []
            texts: List[str] = []
            languages: List[str] = []

            with tempfile.TemporaryDirectory(prefix="qwen_asr_", dir=self.cache_dir) as tmp_dir:
                aligner_inputs = list(
                    self._iter_aligner_inputs(audio_file_path, duration, tmp_dir)
                )
                for batch in self._iter_transcription_batches(aligner_inputs):
                    paths = [item[0] for item in batch]
                    results = self._transcribe_many(paths, requested_language, context_text)
                    if len(results) != len(batch):
                        raise QwenTimestampParseError(
                            "Qwen3-ASR result count did not match input batch size"
                        )

                    for result, (_sub_path, sub_start, _sub_end) in zip(results, batch):
                        detected_language = str(getattr(result, "language", "") or "unknown")
                        if not _is_aligner_language(_normalise_language(detected_language)):
                            raise AlignerLanguageUnsupported(
                                f"Qwen3-ForcedAligner does not support detected language: {detected_language}"
                            )

                        text = str(getattr(result, "text", "") or "").strip()
                        if text:
                            texts.append(text)
                        if detected_language and detected_language != "unknown":
                            languages.append(detected_language)

                        raw_segments.extend(
                            self._parse_time_stamps(
                                getattr(result, "time_stamps", None),
                                sub_start,
                            )
                        )

            if not raw_segments:
                raise QwenTimestampParseError("Qwen3-ASR returned no timestamped segments")

            raw_segments.sort(key=lambda item: (float(item["start"]), float(item["end"])))
            full_text = " ".join(texts).strip()
            segments = self._aggregate_subtitle_segments(raw_segments, full_text)
            if not segments:
                raise QwenTimestampParseError("Qwen3-ASR aggregation returned no segments")

            language_detected = self._most_common(languages) or "unknown"
            audio_duration = max(float(seg["end"]) for seg in segments) if segments else duration

            print(
                "✅ Qwen3-ASR transcription completed: "
                f"{len(raw_segments)} raw alignment segments -> {len(segments)} subtitle segments, "
                f"language={language_detected}, duration={audio_duration:.2f}s"
            )

            return {
                "model_used": f"qwen3_asr:{self.asr_model_id}",
                "qwen_asr_runtime": self.runtime,
                "qwen_inference_batch_size": self.inference_batch_size,
                "segment_count": len(segments),
                "raw_segment_count": len(raw_segments),
                "audio_duration": audio_duration,
                "processing_status": "success",
                "speaker_diarization_enabled": False,
                "global_speaker_count": 0,
                "speaker_summary": {},
                "language_detected": language_detected,
                "text": full_text,
                "segments": segments,
                "segment_aggregation": self._aggregation_metadata(),
                "qwen_context_enabled": bool(context_text),
                "qwen_context_length": len(context_text),
            }

        except AlignerLanguageUnsupported:
            raise
        except Exception as exc:
            print(f"❌ Qwen3-ASR transcription failed: {exc}")
            return {
                "model_used": f"qwen3_asr:{self.asr_model_id}",
                "qwen_asr_runtime": getattr(self, "runtime", "unknown"),
                "qwen_inference_batch_size": getattr(self, "inference_batch_size", 0),
                "segment_count": 0,
                "audio_duration": 0,
                "processing_status": "failed",
                "speaker_diarization_enabled": False,
                "global_speaker_count": 0,
                "speaker_summary": {},
                "language_detected": "unknown",
                "error_message": str(exc),
                "text": "",
                "segments": [],
            }

    def _transcribe_one(self, audio_path: str, language: Optional[str], context: Optional[str]) -> Any:
        try:
            results = self.model.transcribe(
                audio=audio_path,
                context=_normalise_context(context),
                language=language,
                return_time_stamps=True,
            )
        except Exception as exc:
            message = str(exc).lower()
            if "language" in message and ("support" in message or "unsupported" in message):
                raise AlignerLanguageUnsupported(str(exc)) from exc
            raise
        if not results:
            raise QwenTimestampParseError("Qwen3-ASR returned an empty result list")
        return results[0]

    def _transcribe_many(
        self,
        audio_paths: List[str],
        language: Optional[str],
        context: Optional[str],
    ) -> List[Any]:
        if len(audio_paths) == 1:
            return [self._transcribe_one(audio_paths[0], language, context)]

        try:
            context_text = _normalise_context(context)
            results = self.model.transcribe(
                audio=audio_paths,
                context=[context_text] * len(audio_paths),
                language=[language] * len(audio_paths),
                return_time_stamps=True,
            )
        except Exception as exc:
            message = str(exc).lower()
            if "language" in message and ("support" in message or "unsupported" in message):
                raise AlignerLanguageUnsupported(str(exc)) from exc
            raise

        if not results:
            raise QwenTimestampParseError("Qwen3-ASR returned an empty batch result")
        return list(results)

    def _parse_time_stamps(
        self,
        time_stamps: Optional[Iterable[Any]],
        offset_seconds: float,
    ) -> List[Dict[str, Any]]:
        if not time_stamps:
            raise QwenTimestampParseError("Qwen3-ASR returned no time_stamps")

        segments: List[Dict[str, Any]] = []
        for item in time_stamps:
            if isinstance(item, (tuple, list)) and len(item) >= 3:
                start, end, text = item[0], item[1], item[2]
            else:
                start = _get_attr_or_key(item, "start_time", "start")
                end = _get_attr_or_key(item, "end_time", "end")
                text = _get_attr_or_key(item, "text", "word", "token")

            if start is None or end is None or text is None:
                raise QwenTimestampParseError(f"Unsupported Qwen time_stamps item: {item!r}")

            start_f = max(0.0, float(start)) + offset_seconds
            end_f = max(start_f + 0.01, float(end) + offset_seconds)
            text_s = str(text).strip()
            if not text_s:
                continue

            segments.append(
                {
                    "start": start_f,
                    "end": end_f,
                    "text": text_s,
                    "speaker": None,
                }
            )

        if not segments:
            raise QwenTimestampParseError("Qwen3-ASR time_stamps had no non-empty text")
        return segments

    def _aggregation_metadata(self) -> Dict[str, Any]:
        return {
            "enabled": True,
            "max_seconds": _read_env_float(
                "QWEN_SUBTITLE_MAX_SECONDS", DEFAULT_SUBTITLE_MAX_SECONDS
            ),
            "max_chars": _read_env_int(
                "QWEN_SUBTITLE_MAX_CHARS", DEFAULT_SUBTITLE_MAX_CHARS
            ),
            "gap_seconds": _read_env_float(
                "QWEN_SUBTITLE_GAP_SECONDS", DEFAULT_SUBTITLE_GAP_SECONDS
            ),
            "strategy": "punctuated_text_projection",
            "schema_version": 2,
        }

    def _aggregate_subtitle_segments(
        self,
        raw_segments: List[Dict[str, Any]],
        punctuated_text: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Merge Qwen forced-align timestamps into subtitle blocks.

        ForcedAligner timestamps are word/character-level and usually omit
        punctuation. The ASR text itself keeps punctuation, so project those
        marks back onto the aligned tokens before choosing subtitle breaks.
        """

        if not raw_segments:
            return []

        config = self._aggregation_metadata()
        max_seconds = float(config["max_seconds"])
        max_chars = int(config["max_chars"])
        gap_seconds = float(config["gap_seconds"])

        ordered = self._project_punctuation_onto_timestamps(raw_segments, punctuated_text)
        aggregated: List[Dict[str, Any]] = []
        current_start: Optional[float] = None
        current_end: Optional[float] = None
        current_text = ""

        def flush() -> None:
            nonlocal current_start, current_end, current_text
            text = current_text.strip()
            if current_start is not None and current_end is not None and text:
                aggregated.append(
                    {
                        "start": current_start,
                        "end": max(current_start + 0.01, current_end),
                        "text": text,
                        "speaker": None,
                    }
                )
            current_start = None
            current_end = None
            current_text = ""

        for segment in ordered:
            piece = str(segment.get("text", "") or "").strip()
            if not piece:
                continue
            start = float(segment.get("start", 0.0) or 0.0)
            end = float(segment.get("end", start) or start)

            if current_start is None or current_end is None:
                current_start = start
                current_end = max(start + 0.01, end)
                current_text = piece
                continue

            candidate_text = _join_piece(current_text, piece)
            gap = max(0.0, start - current_end)
            candidate_duration = max(end, current_end) - current_start
            current_duration = current_end - current_start
            hard_max_seconds = max(max_seconds * 2.0, max_seconds + 1.2)
            hard_max_chars = max(max_chars * 2, max_chars + 8)
            gap_break = gap >= gap_seconds and (
                len(current_text) >= 4
                or _ends_sentence(current_text)
                or _ends_clause(current_text)
                or gap >= max(gap_seconds * 2.0, 0.75)
            )
            should_break = (
                gap_break
                or (_ends_sentence(current_text) and len(current_text) >= 4)
                or (_ends_clause(current_text) and (len(current_text) >= 6 or current_duration >= 0.8))
                or (candidate_duration > hard_max_seconds and len(current_text) >= 4)
                or (len(candidate_text) > hard_max_chars and len(current_text) >= 4)
            )

            if should_break:
                flush()
                current_start = start
                current_end = max(start + 0.01, end)
                current_text = piece
            else:
                current_text = candidate_text
                current_end = max(current_end, end)

        flush()
        return aggregated

    def _project_punctuation_onto_timestamps(
        self,
        raw_segments: List[Dict[str, Any]],
        punctuated_text: Optional[str],
    ) -> List[Dict[str, Any]]:
        if not punctuated_text:
            return sorted(raw_segments, key=lambda item: (float(item["start"]), float(item["end"])))

        ordered = sorted(raw_segments, key=lambda item: (float(item["start"]), float(item["end"])))
        text = str(punctuated_text)
        text_index = 0
        matched = 0
        total = 0
        projected: List[Dict[str, Any]] = []

        def append_punctuation(mark: str, current_text: str, emitted_current: bool) -> str:
            if mark.isspace():
                return current_text
            if emitted_current or not projected:
                return f"{current_text}{mark}"
            projected[-1]["text"] = f"{projected[-1]['text']}{mark}"
            return current_text

        for segment in ordered:
            piece = str(segment.get("text", "") or "").strip()
            if not piece:
                continue

            content_chars = [char for char in piece if not _is_alignment_ignorable(char)]
            if not content_chars:
                projected.append(dict(segment))
                continue

            total += len(content_chars)
            output_piece = ""
            emitted_current = False

            for raw_char in content_chars:
                while text_index < len(text) and _is_alignment_ignorable(text[text_index]):
                    output_piece = append_punctuation(
                        text[text_index], output_piece, emitted_current
                    )
                    text_index += 1

                if text_index >= len(text) or text[text_index] != raw_char:
                    print(
                        "⚠️ Qwen punctuation projection mismatch; "
                        "falling back to raw forced-align text"
                    )
                    return ordered

                output_piece = f"{output_piece}{raw_char}"
                emitted_current = True
                matched += 1
                text_index += 1

            projected_segment = dict(segment)
            projected_segment["text"] = output_piece.strip()
            if projected_segment["text"]:
                projected.append(projected_segment)

        while text_index < len(text) and _is_alignment_ignorable(text[text_index]):
            if projected and not text[text_index].isspace():
                projected[-1]["text"] = f"{projected[-1]['text']}{text[text_index]}"
            text_index += 1

        if total and matched / total < 0.95:
            print(
                "⚠️ Qwen punctuation projection matched too little text; "
                "falling back to raw forced-align text"
            )
            return ordered

        return projected

    def _iter_aligner_inputs(
        self,
        audio_file_path: str,
        duration: float,
        tmp_dir: str,
    ) -> Iterable[Tuple[str, float, float]]:
        if duration <= MAX_ALIGNER_SEGMENT_SECONDS:
            yield audio_file_path, 0.0, duration
            return

        print(
            f"✂️ Chunk duration {duration:.2f}s exceeds aligner window; "
            f"splitting into <= {MAX_ALIGNER_SEGMENT_SECONDS}s subsegments"
        )
        segment_count = int(math.ceil(duration / MAX_ALIGNER_SEGMENT_SECONDS))
        for index in range(segment_count):
            start = float(index * MAX_ALIGNER_SEGMENT_SECONDS)
            end = min(duration, start + MAX_ALIGNER_SEGMENT_SECONDS)
            if end <= start:
                continue
            sub_path = str(Path(tmp_dir) / f"qwen_subchunk_{index:04d}.wav")
            (
                ffmpeg.input(audio_file_path, ss=start, t=end - start)
                .output(
                    sub_path,
                    format="wav",
                    acodec="pcm_s16le",
                    ac=1,
                    ar=16000,
                    loglevel="error",
                )
                .overwrite_output()
                .run(quiet=True)
            )
            yield sub_path, start, end

    def _iter_transcription_batches(
        self,
        inputs: List[Tuple[str, float, float]],
    ) -> Iterable[List[Tuple[str, float, float]]]:
        batch_size = max(1, int(getattr(self, "inference_batch_size", 1) or 1))
        if getattr(self, "runtime", "transformers") != "vllm":
            batch_size = 1
        for start in range(0, len(inputs), batch_size):
            yield inputs[start : start + batch_size]

    def _probe_duration(self, audio_file_path: str) -> float:
        try:
            probe = ffmpeg.probe(audio_file_path)
            return float(probe["format"]["duration"])
        except Exception as exc:
            raise RuntimeError(f"Failed to probe audio duration for Qwen3-ASR: {exc}") from exc

    def _most_common(self, values: List[str]) -> Optional[str]:
        if not values:
            return None
        return max(set(values), key=values.count)

    def release(self) -> None:
        try:
            if getattr(self, "model", None) is not None:
                try:
                    if hasattr(self.model, "to"):
                        self.model.to("cpu")
                except Exception as exc:
                    print(f"⚠️ Error moving Qwen3-ASR model to CPU: {exc}")
                del self.model

            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as exc:
                print(f"⚠️ Error clearing CUDA cache after Qwen3-ASR release: {exc}")

            gc.collect()
            print("✅ Qwen3-ASR service released")
        except Exception as exc:
            print(f"⚠️ Error releasing Qwen3-ASR service: {exc}")
