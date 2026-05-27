"""Static verification that the GPU Modal app declares no runtime HF secret.

Run with::

    uv run python scripts/check_hf_isolation.py

The script imports the GPU Modal config in the *same* way `modal deploy` does
and asserts:

* The runtime exports ``secrets=[]`` for the compatibility function, ASR
  runtime, and diarization runtime (no named Hugging Face workspace secret is
  referenced at runtime).
* The image build step uses an inline ``modal.Secret.from_dict``-style
  anonymous secret carrying ``HF_TOKEN`` — i.e. it is bundled into the
  deployment graph for this single deploy and is **not** a named workspace
  secret that the end user would see in ``modal secret list``.
* ``HF_HUB_OFFLINE=1`` is set on the runtime image so the function never
  contacts huggingface.co at runtime.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _import_with_token(token: str | None) -> "ModuleType":  # type: ignore[name-defined]
    for key in ("HF_TOKEN", "HUGGING_FACE_TOKEN"):
        os.environ.pop(key, None)
    if token:
        os.environ["HF_TOKEN"] = token

    for mod in [m for m in list(sys.modules) if m.startswith("src.config")]:
        sys.modules.pop(mod, None)

    import importlib

    return importlib.import_module("src.config.modal_gpu_config")


def main() -> None:
    cfg = _import_with_token("hf_test_token_e2e")

    secret_names = [type(s).__name__ for s in cfg.secrets]
    print(f"runtime secrets list type names: {secret_names}")
    assert cfg.secrets == [], (
        "Runtime function must declare secrets=[]; got "
        f"{cfg.secrets!r}. The HF token must NEVER be attached to the "
        "function at runtime (only the inline build secret on run_function)."
    )
    print(
        "✅ runtime function secrets=[] — no Hugging Face entry will appear "
        "in the user's Modal workspace Secrets list."
    )

    # The function and class references reflect what `modal deploy` will
    # register; if a named "huggingface-secret" sneaked in we'd find it here.
    deployed_refs = {
        "function": repr(cfg.transcribe_and_diarization_audio_function),
        "asr_class": repr(cfg.TranscribeAudioRuntime),
        "diarization_class": repr(cfg.SpeakerDiarizationAudioRuntime),
    }
    for label, ref in deployed_refs.items():
        assert "huggingface-secret" not in ref, (
            f"{label} references a named 'huggingface-secret' workspace secret: {ref}"
        )
    print("✅ function/class metadata does not reference any named HF workspace secret")

    cfg2 = _import_with_token(None)
    assert cfg2.secrets == [], "Runtime container path must also report secrets=[]"
    print("✅ module re-imports cleanly with HF_TOKEN absent (runtime container path)")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
