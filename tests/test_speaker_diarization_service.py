"""Offline tests for speaker diarization helper defaults."""

from __future__ import annotations

import os
import sys
import types
import unittest
import base64
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.services.speaker_diarization_service import (  # noqa: E402
    DiarizationRuntimeError,
    SpeakerDiarizationService,
)


class SpeakerDiarizationServiceHelperTests(unittest.TestCase):
    def test_diarization_batch_defaults_are_l4_sized(self) -> None:
        service = SpeakerDiarizationService()

        with mock.patch.dict(
            os.environ,
            {
                "PYANNOTE_SEGMENTATION_BATCH_SIZE": "",
                "PYANNOTE_EMBEDDING_BATCH_SIZE": "",
            },
            clear=False,
        ):
            config = service._diarization_batch_config({})

        self.assertEqual(
            config,
            {
                "segmentation_batch_size": 256,
                "embedding_batch_size": 256,
            },
        )

    def test_diarization_batch_request_overrides_defaults(self) -> None:
        service = SpeakerDiarizationService()

        config = service._diarization_batch_config(
            {
                "segmentation_batch_size": 64,
                "embedding_batch_size": 96,
            }
        )

        self.assertEqual(
            config,
            {
                "segmentation_batch_size": 64,
                "embedding_batch_size": 96,
            },
        )

    def test_pipeline_audio_input_prefers_soundfile_memory_payload(self) -> None:
        class FakeArray:
            def __init__(self, shape: tuple[int, int]):
                self.shape = shape

            @property
            def T(self) -> "FakeArray":
                return FakeArray((self.shape[1], self.shape[0]))

            def copy(self) -> "FakeArray":
                return self

        service = SpeakerDiarizationService()
        fake_soundfile = types.SimpleNamespace(
            read=lambda *_args, **_kwargs: (FakeArray((3, 1)), 16000)
        )
        fake_torch = types.SimpleNamespace(from_numpy=lambda array: array)

        with mock.patch.dict(
            sys.modules,
            {
                "soundfile": fake_soundfile,
                "torch": fake_torch,
            },
            clear=False,
        ):
            payload = service._pipeline_audio_input("/tmp/audio.wav")

        self.assertEqual(payload["sample_rate"], 16000)
        self.assertEqual(payload["waveform"].shape, (1, 3))

    def test_pipeline_audio_input_raises_when_memory_preload_fails(self) -> None:
        service = SpeakerDiarizationService()
        fake_soundfile = types.SimpleNamespace(
            read=lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("sf boom"))
        )
        fake_torch = types.SimpleNamespace(from_numpy=lambda array: array)
        fake_torchaudio = types.SimpleNamespace(
            load=lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("tc boom"))
        )

        with mock.patch.dict(
            sys.modules,
            {
                "soundfile": fake_soundfile,
                "torch": fake_torch,
                "torchaudio": fake_torchaudio,
            },
            clear=False,
        ):
            with self.assertRaises(DiarizationRuntimeError):
                service._pipeline_audio_input("/tmp/audio.wav")

    def test_process_diarization_request_reports_runtime_failure(self) -> None:
        service = SpeakerDiarizationService()
        request = {
            "audio_file_data": base64.b64encode(b"fake audio").decode("ascii"),
            "audio_file_name": "audio.mp3",
        }

        with mock.patch.object(
            service,
            "_preprocess_audio_for_diarization",
            side_effect=lambda path: path,
        ):
            with mock.patch.object(service, "_load_pipeline", return_value=object()):
                with mock.patch.object(
                    service,
                    "diarize_audio",
                    side_effect=DiarizationRuntimeError("pyannote failed"),
                ):
                    result = service.process_diarization_request(request)

        self.assertEqual(result["processing_status"], "failed")
        self.assertIn("pyannote failed", result["error_message"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
