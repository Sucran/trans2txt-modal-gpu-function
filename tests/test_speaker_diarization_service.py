"""Offline tests for speaker diarization helper defaults."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.services.speaker_diarization_service import SpeakerDiarizationService  # noqa: E402


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


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
