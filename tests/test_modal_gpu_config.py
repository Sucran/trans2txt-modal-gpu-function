"""Tests for Modal entrypoint import path handling."""

from __future__ import annotations

import importlib
import os
import sys
import unittest
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _resolve_repo_root(current_file: Path) -> Path:
    with mock.patch.dict(os.environ, {}, clear=False):
        module = importlib.import_module("src.config.modal_gpu_config")
    return module._resolve_repo_root(current_file)


class ModalGpuConfigPathTests(unittest.TestCase):
    def test_resolves_repo_root_from_source_tree_entry(self) -> None:
        repo_root = PROJECT_ROOT
        entry = repo_root / "src" / "config" / "modal_gpu_config.py"

        self.assertEqual(_resolve_repo_root(entry), repo_root)

    def test_resolves_repo_root_from_modal_flattened_entry(self) -> None:
        entry = PROJECT_ROOT / "modal_gpu_config.py"

        self.assertEqual(_resolve_repo_root(entry), PROJECT_ROOT)


class ModalGpuSnapshotConfigTests(unittest.TestCase):
    def _module(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            return importlib.import_module("src.config.modal_gpu_config")

    def test_gpu_snapshot_options_require_memory_snapshot(self) -> None:
        module = self._module()

        self.assertEqual(
            module._snapshot_experimental_options(
                memory_enabled=True,
                gpu_enabled=True,
            ),
            {"enable_gpu_snapshot": True},
        )
        self.assertIsNone(
            module._snapshot_experimental_options(
                memory_enabled=False,
                gpu_enabled=True,
            )
        )
        self.assertIsNone(
            module._snapshot_experimental_options(
                memory_enabled=True,
                gpu_enabled=False,
            )
        )

    def test_snapshot_preload_auto_follows_asr_backend(self) -> None:
        module = self._module()

        self.assertEqual(
            module._resolve_snapshot_preload_targets("auto", "qwen3_asr"),
            ["qwen3_asr"],
        )
        self.assertEqual(
            module._resolve_snapshot_preload_targets("auto", "whisper"),
            ["whisper"],
        )
        self.assertEqual(
            module._resolve_snapshot_preload_targets("auto", "diarization"),
            ["diarization"],
        )

    def test_snapshot_preload_keeps_one_asr_backend(self) -> None:
        module = self._module()

        self.assertEqual(
            module._resolve_snapshot_preload_targets("qwen3_asr,whisper,diarization"),
            ["qwen3_asr", "diarization"],
        )
        self.assertEqual(
            module._resolve_snapshot_preload_targets("none"),
            [],
        )

    def test_snapshot_preload_splits_asr_and_diarization_targets(self) -> None:
        module = self._module()

        with mock.patch.object(
            module,
            "MODAL_GPU_SNAPSHOT_PRELOAD",
            "qwen3_asr,diarization",
        ):
            self.assertEqual(module._asr_snapshot_preload_targets(), ["qwen3_asr"])
            self.assertTrue(module._diarization_snapshot_preload_enabled())

    def test_request_payload_routes_to_dedicated_runtime_class(self) -> None:
        module = self._module()

        self.assertEqual(
            module._runtime_class_name_for_request(
                {"request_type": "transcribe", "audio_file_data": "..."}
            ),
            "TranscribeAudioRuntime",
        )
        self.assertEqual(
            module._runtime_class_name_for_request(
                {"request_type": "diarization", "audio_file_data": "..."}
            ),
            "SpeakerDiarizationAudioRuntime",
        )
        self.assertEqual(
            module._runtime_class_name_for_request(
                {"chunk_start_time": 0, "audio_file_data": "..."}
            ),
            "TranscribeAudioRuntime",
        )
        self.assertEqual(
            module._runtime_class_name_for_request({"audio_file_data": "..."}),
            "SpeakerDiarizationAudioRuntime",
        )


if __name__ == "__main__":
    unittest.main()
