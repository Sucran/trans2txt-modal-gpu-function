"""Offline tests for transcription endpoint request guards."""

from __future__ import annotations

import base64
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.services import transcription_endpoint_service as endpoint  # noqa: E402


class TranscriptionEndpointServiceTests(unittest.TestCase):
    def tearDown(self) -> None:
        endpoint._cached_backend = None

    def _request(self, **overrides: object) -> dict[str, object]:
        request: dict[str, object] = {
            "request_type": "transcribe",
            "audio_file_data": base64.b64encode(b"not-real-audio").decode("ascii"),
            "audio_file_name": "dummy.mp3",
            "chunk_start_time": 0,
            "chunk_end_time": 1,
            "language": "auto",
        }
        request.update(overrides)
        return request

    def test_qwen_model_allowlist_defaults_to_configured_model(self) -> None:
        with mock.patch.dict(os.environ, {"QWEN_ALLOWED_ASR_MODEL_IDS": ""}, clear=False):
            self.assertEqual(
                endpoint._allowed_qwen_asr_model_ids("Qwen/Qwen3-ASR-1.7B"),
                {"Qwen/Qwen3-ASR-1.7B"},
            )

    def test_qwen_model_allowlist_accepts_configured_plus_extra_models(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"QWEN_ALLOWED_ASR_MODEL_IDS": "Qwen/Qwen3-ASR-0.6B, custom/model"},
            clear=False,
        ):
            self.assertEqual(
                endpoint._allowed_qwen_asr_model_ids("Qwen/Qwen3-ASR-1.7B"),
                {
                    "Qwen/Qwen3-ASR-1.7B",
                    "Qwen/Qwen3-ASR-0.6B",
                    "custom/model",
                },
            )

    def test_qwen_request_rejects_unallowed_model_id(self) -> None:
        service = endpoint.TranscriptionEndpointService(cache_dir="/tmp")

        with mock.patch.dict(
            os.environ,
            {
                "QWEN_ASR_MODEL_ID": "Qwen/Qwen3-ASR-1.7B",
                "QWEN_ALLOWED_ASR_MODEL_IDS": "",
            },
            clear=False,
        ):
            result = service.process_chunk_request(
                self._request(asr_backend="qwen3_asr", asr_model_id="not/allowed")
            )

        self.assertEqual(result["processing_status"], "failed")
        self.assertIn("not allowed", result["error_message"])

    def test_request_backend_selects_qwen_even_when_env_default_is_whisper(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = endpoint.TranscriptionEndpointService(cache_dir=tmp_dir)
            with mock.patch.dict(
                os.environ,
                {
                    "ASR_BACKEND": "whisper",
                    "QWEN_ASR_MODEL_ID": "Qwen/Qwen3-ASR-1.7B",
                    "QWEN_ALLOWED_ASR_MODEL_IDS": "",
                },
                clear=False,
            ), mock.patch.object(
                service,
                "_transcribe_with_qwen_or_fallback",
                return_value={
                    "processing_status": "success",
                    "segments": [],
                    "text": "",
                    "pause_intervals": [],
                },
            ) as qwen_call, mock.patch.object(
                service, "_transcribe_with_whisper"
            ) as whisper_call:
                result = service.process_chunk_request(
                    self._request(asr_backend="qwen3_asr", qwen_context="CFO 陈震")
                )

        self.assertEqual(result["processing_status"], "success")
        qwen_call.assert_called_once()
        whisper_call.assert_not_called()
        self.assertEqual(qwen_call.call_args.kwargs["qwen_context"], "CFO 陈震")

    def test_request_backend_selects_whisper_even_when_env_default_is_qwen(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = endpoint.TranscriptionEndpointService(cache_dir=tmp_dir)
            with mock.patch.dict(
                os.environ,
                {"ASR_BACKEND": "qwen3_asr"},
                clear=False,
            ), mock.patch.object(
                service,
                "_transcribe_with_whisper",
                return_value={
                    "processing_status": "success",
                    "segments": [],
                    "text": "",
                    "pause_intervals": [],
                },
            ) as whisper_call, mock.patch.object(
                service, "_transcribe_with_qwen_or_fallback"
            ) as qwen_call:
                result = service.process_chunk_request(
                    self._request(asr_backend="whisper")
                )

        self.assertEqual(result["processing_status"], "success")
        whisper_call.assert_called_once()
        qwen_call.assert_not_called()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
