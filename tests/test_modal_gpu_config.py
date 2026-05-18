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


if __name__ == "__main__":
    unittest.main()
