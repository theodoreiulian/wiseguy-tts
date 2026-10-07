"""Fetch the shipped voice from Hugging Face into models/wiseguy.

    uv run wiseguy-download                # optional: `uv run wiseguy` does this automatically
    WISEGUY_HF_REPO=user/repo uv run wiseguy-download
"""

from __future__ import annotations

import os

from pathlib import Path

HF_REPO = os.environ.get("WISEGUY_HF_REPO", "theodoreiulian/wiseguy-tts")


def fetch(model_dir: str | Path) -> None:
    from huggingface_hub import snapshot_download

    print(f"downloading {HF_REPO} -> {model_dir} (~1.8 GB, one time)", flush=True)
    snapshot_download(repo_id=HF_REPO, local_dir=model_dir)


def main() -> None:
    from .engine import MODEL_DIR

    fetch(MODEL_DIR)
    print("done. run `uv run wiseguy`")


if __name__ == "__main__":
    main()
