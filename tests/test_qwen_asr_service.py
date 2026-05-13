"""Offline tests for Qwen3-ASR service helpers.

These tests avoid importing the real qwen-asr package or running ffmpeg. They
cover the lightweight logic that can break T4 compatibility and timestamp
contracts before any GPU deployment is attempted.
"""

from __future__ import annotations

import sys
import types
import unittest
import os
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if "ffmpeg" not in sys.modules:
    sys.modules["ffmpeg"] = types.ModuleType("ffmpeg")

from src.services import qwen_asr_service as qwen  # noqa: E402


class _FakeCuda:
    def __init__(self, available: bool, bf16: bool):
        self._available = available
        self._bf16 = bf16

    def is_available(self) -> bool:
        return self._available

    def is_bf16_supported(self) -> bool:
        return self._bf16


class _FakeTorch:
    bfloat16 = "bf16"
    float16 = "fp16"

    def __init__(self, available: bool, bf16: bool):
        self.cuda = _FakeCuda(available, bf16)


class _StampObj:
    def __init__(self, start: float, end: float, text: str):
        self.start_time = start
        self.end_time = end
        self.text = text


class _FakeFfmpegChain:
    def __init__(self, calls: list[dict[str, object]]):
        self.calls = calls
        self.kwargs: dict[str, object] = {}

    def output(self, path: str, **kwargs: object) -> "_FakeFfmpegChain":
        self.kwargs = {"path": path, **kwargs}
        return self

    def overwrite_output(self) -> "_FakeFfmpegChain":
        return self

    def run(self, quiet: bool = True) -> None:
        self.calls.append({"quiet": quiet, **self.kwargs})


class _FakeFfmpeg:
    def __init__(self):
        self.inputs: list[dict[str, float | str]] = []
        self.outputs: list[dict[str, object]] = []

    def input(self, path: str, ss: float, t: float) -> _FakeFfmpegChain:
        self.inputs.append({"path": path, "ss": ss, "t": t})
        return _FakeFfmpegChain(self.outputs)


class QwenAsrServiceHelperTests(unittest.TestCase):
    def test_dtype_auto_uses_fp16_when_bf16_is_not_supported(self) -> None:
        self.assertEqual(qwen._select_torch_dtype(_FakeTorch(True, False)), "fp16")

    def test_dtype_auto_uses_bf16_when_supported(self) -> None:
        self.assertEqual(qwen._select_torch_dtype(_FakeTorch(True, True)), "bf16")

    def test_language_aliases_gate_aligner_support(self) -> None:
        self.assertEqual(qwen._normalise_language("zh-CN"), "Chinese")
        self.assertTrue(qwen._is_aligner_language(qwen._normalise_language("zh-CN")))
        self.assertFalse(qwen._is_aligner_language(qwen._normalise_language("th")))

    def test_parse_time_stamps_accepts_tuples_dicts_and_objects(self) -> None:
        service = object.__new__(qwen.QwenAsrService)
        segments = service._parse_time_stamps(
            [
                (0.0, 0.5, "hello"),
                {"start": 0.5, "end": 1.0, "text": "world"},
                _StampObj(1.0, 1.5, "again"),
            ],
            offset_seconds=10.0,
        )
        self.assertEqual(
            segments,
            [
                {"start": 10.0, "end": 10.5, "text": "hello", "speaker": None},
                {"start": 10.5, "end": 11.0, "text": "world", "speaker": None},
                {"start": 11.0, "end": 11.5, "text": "again", "speaker": None},
            ],
        )

    def test_iter_aligner_inputs_splits_long_audio_at_limit(self) -> None:
        service = object.__new__(qwen.QwenAsrService)
        fake = _FakeFfmpeg()
        with mock.patch.object(qwen, "MAX_ALIGNER_SEGMENT_SECONDS", 180):
            with mock.patch.object(qwen, "ffmpeg", fake):
                items = list(service._iter_aligner_inputs("/tmp/in.m4a", 370.0, "/tmp/qwen"))

        self.assertEqual(
            [(round(start), round(end)) for _path, start, end in items],
            [(0, 180), (180, 360), (360, 370)],
        )
        self.assertEqual(
            [(round(call["ss"]), round(call["t"])) for call in fake.inputs],
            [(0, 180), (180, 180), (360, 10)],
        )
        self.assertEqual(len(fake.outputs), 3)

    def test_aggregate_chinese_alignment_segments_by_punctuation(self) -> None:
        service = object.__new__(qwen.QwenAsrService)
        chars = list("药价背后的谈判。今天聊聊医保。")
        raw = [
            {"start": i * 0.12, "end": i * 0.12 + 0.08, "text": char, "speaker": None}
            for i, char in enumerate(chars)
        ]

        with mock.patch.dict(os.environ, {"QWEN_SUBTITLE_MAX_CHARS": "42"}, clear=False):
            segments = service._aggregate_subtitle_segments(raw)

        self.assertEqual([seg["text"] for seg in segments], ["药价背后的谈判。", "今天聊聊医保。"])
        self.assertEqual(segments[0]["start"], raw[0]["start"])
        self.assertEqual(segments[0]["end"], raw[7]["end"])
        self.assertEqual(segments[1]["start"], raw[8]["start"])

    def test_project_punctuated_text_onto_alignment_segments(self) -> None:
        service = object.__new__(qwen.QwenAsrService)
        raw_text = "五块这不是特例哈阿莫西林从六毛多一票降到两毛阿司匹林"
        raw = [
            {"start": i * 0.1, "end": i * 0.1 + 0.08, "text": char, "speaker": None}
            for i, char in enumerate(raw_text)
        ]

        projected = service._project_punctuation_onto_timestamps(
            raw,
            "五块，这不是特例哈。阿莫西林从六毛多一票降到两毛，阿司匹林",
        )

        projected_text = "".join(segment["text"] for segment in projected)
        self.assertEqual(
            projected_text,
            "五块，这不是特例哈。阿莫西林从六毛多一票降到两毛，阿司匹林",
        )

    def test_punctuated_aggregation_does_not_split_drug_name(self) -> None:
        service = object.__new__(qwen.QwenAsrService)
        raw_text = "五块这不是特例哈阿莫西林从六毛多一票降到两毛阿司匹林最便宜只要三分钱一票"
        raw = [
            {"start": i * 0.1, "end": i * 0.1 + 0.08, "text": char, "speaker": None}
            for i, char in enumerate(raw_text)
        ]

        segments = service._aggregate_subtitle_segments(
            raw,
            "五块，这不是特例哈。阿莫西林从六毛多一票降到两毛，阿司匹林最便宜只要三分钱一票，",
        )
        texts = [segment["text"] for segment in segments]

        self.assertIn("五块，这不是特例哈。", texts)
        self.assertIn("阿莫西林从六毛多一票降到两毛，", texts)
        self.assertFalse(any(text.endswith("阿莫") for text in texts))
        self.assertFalse(any(text.startswith("西林") for text in texts))

    def test_aggregate_english_alignment_segments_with_spaces(self) -> None:
        service = object.__new__(qwen.QwenAsrService)
        raw = [
            {"start": 0.0, "end": 0.2, "text": "hello", "speaker": None},
            {"start": 0.2, "end": 0.4, "text": "world", "speaker": None},
            {"start": 0.4, "end": 0.5, "text": ".", "speaker": None},
            {"start": 0.6, "end": 0.8, "text": "next", "speaker": None},
            {"start": 0.8, "end": 1.0, "text": "line", "speaker": None},
        ]

        segments = service._aggregate_subtitle_segments(raw)

        self.assertEqual([seg["text"] for seg in segments], ["hello world.", "next line"])

    def test_aggregate_breaks_on_gap_and_length(self) -> None:
        service = object.__new__(qwen.QwenAsrService)
        raw = [
            {"start": 0.0, "end": 0.2, "text": "one", "speaker": None},
            {"start": 0.2, "end": 0.4, "text": "two", "speaker": None},
            {"start": 1.5, "end": 1.7, "text": "after", "speaker": None},
            {"start": 1.7, "end": 1.9, "text": "gap", "speaker": None},
            {"start": 2.0, "end": 2.2, "text": "long", "speaker": None},
            {"start": 2.2, "end": 2.4, "text": "phrase", "speaker": None},
        ]

        with mock.patch.dict(
            os.environ,
            {"QWEN_SUBTITLE_GAP_SECONDS": "0.8", "QWEN_SUBTITLE_MAX_CHARS": "4"},
            clear=False,
        ):
            segments = service._aggregate_subtitle_segments(raw)

        self.assertEqual([seg["text"] for seg in segments], ["one two", "after gap", "long phrase"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
