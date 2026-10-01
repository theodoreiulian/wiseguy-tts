"""Where the fine-tuning pipeline keeps things.

Big, re-downloadable things live in a work directory outside the repo
($WISEGUY_WORK, default ~/wiseguy-work; `setup.sh` fills it). The trainable
checkpoint of the shipped voice and its replay data live in the repo's
gitignored models/finetune/.
"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
WORK = Path(os.environ.get("WISEGUY_WORK", Path.home() / "wiseguy-work"))

QWEN_SRC = WORK / "qwen3-tts-src"                                  # QwenLM/Qwen3-TTS checkout (dataset.py)
BASE = WORK / "base" / "Qwen3-TTS-12Hz-0.6B-Base"                   # speaker encoder for new embeddings
TOKENIZER = WORK / "base" / "Qwen3-TTS-Tokenizer-12Hz"              # 12 Hz speech codec (audio -> codes)
ENHANCE_WEIGHTS = WORK / "enhance-weights" / "enhancer_stage2"      # Resemble-Enhance
VENV_PY = WORK / ".venv" / "bin" / "python"                         # training/data venv (PyTorch)
DATA = WORK / "data"                                                # your recordings, clips, manifests
RUNS = WORK / "runs"                                                # training checkpoints

FINETUNE = REPO / "models" / "finetune"
CHECKPOINT = FINETUNE / "checkpoint"                                # the shipped voice, full precision (start here)
REPLAY = FINETUNE / "replay_codes.jsonl"                            # the 1345 clean clips it was trained on, as codes
SPEAKERS = FINETUNE / "speakers.pt"                                 # speaker embeddings + the shipped voice blend
