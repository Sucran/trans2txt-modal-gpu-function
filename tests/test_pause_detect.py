"""Offline tests for ``WhisperService._detect_pauses``.

These tests never invoke ffmpeg. We stub :mod:`subprocess.Popen` so the
function reads canned ffmpeg ``silencedetect`` stderr lines, then assert the
parsed intervals match the expected ms math. ``whisper`` is also stubbed so
``whisper_service`` can be imported without the heavy native dependency.
"""

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from typing import Iterable, List
from unittest import mock

# ---------------------------------------------------------------------------
# Make ``src`` importable without installing the package, and stub ``whisper``
# so importing ``whisper_service`` does not require the heavy native package.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if "whisper" not in sys.modules:
    _whisper_stub = types.ModuleType("whisper")

    def _load_model(*_args, **_kwargs):  # pragma: no cover - never called
        raise RuntimeError("whisper.load_model should not be invoked in tests")

    _whisper_stub.load_model = _load_model  # type: ignore[attr-defined]
    sys.modules["whisper"] = _whisper_stub

from src.services.whisper_service import WhisperService  # noqa: E402


class _FakePopen:
    """Minimal stand-in for ``subprocess.Popen`` used in tests."""

    def __init__(self, stderr_lines: Iterable[str]):
        # ``Popen.stderr`` is iterable; an iterator over strings is enough
        # because ``_detect_pauses`` only does ``for line in stderr_stream``.
        self.stderr = iter(list(stderr_lines))
        self.stdout = iter([])
        self.returncode = 0
        self.wait_calls = 0

    def wait(self) -> int:
        self.wait_calls += 1
        return self.returncode


def _silence_lines(events: List[tuple[float, float]]) -> List[str]:
    """Build canned ffmpeg stderr lines for a list of (end_s, dur_s)."""
    lines: List[str] = [
        "ffmpeg version N-12345-gabc Copyright (c) 2000-2024 the FFmpeg developers\n",
        "  configuration: --enable-x\n",
        "[silencedetect @ 0x55a1] silence_start: 0.10\n",
    ]
    for end_s, dur_s in events:
        lines.append(
            f"[silencedetect @ 0x55a1] silence_end: {end_s} | "
            f"silence_duration: {dur_s}\n"
        )
    lines.append("size=N/A time=00:01:00.00 bitrate=N/A speed=1x\n")
    return lines


class DetectPausesTests(unittest.TestCase):
    """Happy-path + defensive coverage for ``_detect_pauses``."""

    def test_parses_multiple_silence_events(self) -> None:
        events = [
            (1.500, 0.500),   # short pause near the start
            (12.345, 0.789),  # mid-chunk pause
            (60.000, 1.250),  # longer pause near the end
        ]
        fake = _FakePopen(_silence_lines(events))

        with mock.patch(
            "src.services.whisper_service.subprocess.Popen",
            return_value=fake,
        ) as popen_mock:
            intervals = WhisperService._detect_pauses(
                "/tmp/chunk.wav",
                min_dur_s=0.4,
                noise_db=-35,
            )

        # ffmpeg was invoked exactly once with our flags.
        self.assertEqual(popen_mock.call_count, 1)
        invoked_cmd = popen_mock.call_args.args[0]
        self.assertIn("ffmpeg", invoked_cmd[0])
        self.assertIn("silencedetect=noise=-35dB:duration=0.4", invoked_cmd)
        self.assertEqual(fake.wait_calls, 1)

        # Three events → three intervals, sorted by start_ms ascending.
        self.assertEqual(len(intervals), 3)

        # Event 1: end=1.500s, dur=0.500s → end_ms=1500, dur_ms=500,
        #          start_ms=1000, mid_ms=1250.
        self.assertEqual(
            intervals[0],
            {
                "start_ms": 1000,
                "end_ms": 1500,
                "dur_ms": 500,
                "mid_ms": 1250,
            },
        )

        # Event 2: end=12.345s, dur=0.789s → end_ms=12345, dur_ms=789,
        #          start_ms=11556, mid_ms=11556 + 394 = 11950.
        self.assertEqual(
            intervals[1],
            {
                "start_ms": 11556,
                "end_ms": 12345,
                "dur_ms": 789,
                "mid_ms": 11556 + 789 // 2,
            },
        )

        # Event 3: end=60.000s, dur=1.250s → end_ms=60000, dur_ms=1250,
        #          start_ms=58750, mid_ms=58750 + 625 = 59375.
        self.assertEqual(
            intervals[2],
            {
                "start_ms": 58750,
                "end_ms": 60000,
                "dur_ms": 1250,
                "mid_ms": 59375,
            },
        )

        # Times are CHUNK-LOCAL: nothing should exceed our largest event_ms.
        for entry in intervals:
            self.assertGreaterEqual(entry["start_ms"], 0)
            self.assertGreater(entry["end_ms"], entry["start_ms"])
            self.assertEqual(
                entry["dur_ms"], entry["end_ms"] - entry["start_ms"]
            )

    def test_malformed_stderr_yields_empty_list(self) -> None:
        """Garbage stderr must NOT raise; just return ``[]``."""
        garbage = [
            "this is not a valid ffmpeg line\n",
            "silence_end without_duration_field\n",
            "silence_duration: 0.5 but no end\n",
            "\n",
            "  silence_end: not-a-number | silence_duration: 0.4\n",
        ]
        fake = _FakePopen(garbage)
        with mock.patch(
            "src.services.whisper_service.subprocess.Popen",
            return_value=fake,
        ):
            intervals = WhisperService._detect_pauses(
                "/tmp/chunk.wav",
                min_dur_s=0.4,
                noise_db=-30,
            )
        self.assertEqual(intervals, [])

    def test_subprocess_failure_returns_empty_list(self) -> None:
        """If ffmpeg can't even be spawned, return ``[]`` (no exception)."""
        with mock.patch(
            "src.services.whisper_service.subprocess.Popen",
            side_effect=FileNotFoundError("ffmpeg not found"),
        ):
            intervals = WhisperService._detect_pauses(
                "/tmp/chunk.wav",
                min_dur_s=0.5,
                noise_db=-40,
            )
        self.assertEqual(intervals, [])

    def test_zero_duration_event_is_dropped(self) -> None:
        """A 0ms silence (dur_ms <= 0) should be skipped, not emitted."""
        # 0.0004s rounds to 0ms with the ``round`` rule → drop.
        fake = _FakePopen(_silence_lines([(2.000, 0.0004)]))
        with mock.patch(
            "src.services.whisper_service.subprocess.Popen",
            return_value=fake,
        ):
            intervals = WhisperService._detect_pauses(
                "/tmp/chunk.wav",
                min_dur_s=0.4,
                noise_db=-35,
            )
        self.assertEqual(intervals, [])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
