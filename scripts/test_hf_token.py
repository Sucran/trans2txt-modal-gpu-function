#!/usr/bin/env python3
"""Verify HF_TOKEN can reach Hugging Face and gated pyannote repos.

Usage (never commit a token; do not paste tokens into chat):

  export HF_TOKEN=hf_...
  uv run --with huggingface_hub python scripts/test_hf_token.py

Exit code 0 if checks pass; non-zero on failure.
"""

from __future__ import annotations

import os
import sys


def main() -> int:
    token = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_TOKEN") or "").strip()
    if not token:
        print("error: set HF_TOKEN (or HUGGING_FACE_TOKEN) in the environment", file=sys.stderr)
        return 2

    try:
        from huggingface_hub import model_info, whoami
    except ImportError:
        print(
            "error: install huggingface_hub, e.g.\n"
            "  uv run --with huggingface_hub python scripts/test_hf_token.py",
            file=sys.stderr,
        )
        return 2

    try:
        from huggingface_hub.errors import HfHubHTTPError
    except ImportError:
        from huggingface_hub.utils import HfHubHTTPError

    print("1) whoami ...")
    try:
        info = whoami(token=token)
        name = info.get("name") if isinstance(info, dict) else getattr(info, "name", str(info))
        print(f"   ok: logged in as {name}")
    except Exception as e:
        print(f"   fail: {e}", file=sys.stderr)
        return 1

    repos = [
        "pyannote/speaker-diarization-community-1",
        "pyannote/embedding",
    ]
    for repo_id in repos:
        print(f"2) model_info({repo_id}) ...")
        try:
            meta = model_info(repo_id, token=token)
            rid = getattr(meta, "id", None) or repo_id
            print(f"   ok: {rid}")
        except HfHubHTTPError as e:
            print(f"   fail (HTTP): {e}", file=sys.stderr)
            if e.response is not None and e.response.status_code == 403:
                print(
                    "   hint: open the model page in a browser while logged in as this user, "
                    "accept any 'Gated model' / terms, then retry.",
                    file=sys.stderr,
                )
            return 1
        except Exception as e:
            print(f"   fail: {e}", file=sys.stderr)
            return 1

    print("all checks passed (token + pyannote repo access).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
