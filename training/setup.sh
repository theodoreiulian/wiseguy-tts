#!/bin/bash
# Build the fine-tuning work directory: $WISEGUY_WORK (default ~/wiseguy-work).
#
#   training/setup.sh              # everything, incl. ~3.7 GB of model downloads
#   training/setup.sh --no-models  # environment only
#
# Re-runnable: skips what's already there.
set -euo pipefail

WORK="${WISEGUY_WORK:-$HOME/wiseguy-work}"
QWEN_COMMIT=022e286b98fbec7e1e916cb940cdf532cd9f488e   # the QwenLM/Qwen3-TTS commit the voice was trained with
mkdir -p "$WORK"/{base,data,runs}
cd "$WORK"
echo "work dir: $WORK"

# 1. Qwen3-TTS source: the qwen_tts package + finetuning/dataset.py that train_wiseguy.py imports
if [ ! -d qwen3-tts-src ]; then
  git clone -q https://github.com/QwenLM/Qwen3-TTS.git qwen3-tts-src
fi
git -C qwen3-tts-src checkout -q "$QWEN_COMMIT"

# 2. One PyTorch venv for data prep, training and evaluation (inference uses the repo's own MLX venv)
if [ ! -x .venv/bin/python ]; then
  uv venv -q -p 3.12 .venv
fi
PY=.venv/bin/python
uv pip install -q -p "$PY" torch torchaudio "transformers==4.57.3" "accelerate==1.12.0" \
  librosa soundfile einops onnxruntime safetensors huggingface_hub tensorboard \
  mlx-whisper praat-parselmouth phonemizer espeakng-loader speechbrain yt-dlp \
  celluloid omegaconf ptflops rich resampy tabulate scipy
uv pip install -q -p "$PY" -e qwen3-tts-src
# resemble-enhance pins torch 2.1 + deepspeed; install it without deps and patch it (below)
uv pip install -q -p "$PY" --no-deps resemble-enhance

SP=$("$PY" -c "import site; print(site.getsitepackages()[0])")

# 2a. resemble-enhance imports deepspeed at module level, but only uses it for training: stub it
if [ ! -f "$SP/deepspeed/__init__.py" ]; then
  mkdir -p "$SP/deepspeed/runtime"
  cat > "$SP/deepspeed/__init__.py" <<'EOF'
"""Import-only stub: resemble-enhance imports deepspeed for training; inference never calls it."""
class DeepSpeedConfig:  # noqa: D101
    def __init__(self, *a, **k): raise RuntimeError("deepspeed stub: training not supported")
def init_distributed(*a, **k): raise RuntimeError("deepspeed stub")
EOF
  printf 'def get_accelerator():\n    raise RuntimeError("deepspeed stub")\n' > "$SP/deepspeed/accelerator.py"
  touch "$SP/deepspeed/runtime/__init__.py"
  echo 'class DeepSpeedEngine: pass' > "$SP/deepspeed/runtime/engine.py"
  echo 'def clip_grad_norm_(*a, **k): raise RuntimeError("deepspeed stub")' > "$SP/deepspeed/runtime/utils.py"
fi

# 2b. resemble-enhance + NumPy 2: float() of a 1-element array
CFM="$SP/resemble_enhance/enhancer/lcfm/cfm.py"
if ! grep -q 'x0=0)\[0\])' "$CFM"; then
  sed -i '' 's/x0=0))/x0=0)[0])/' "$CFM"
fi

# 2c. espeak paths for the phoneme recognizer used by nyjudge.py
"$PY" - > env.sh <<'EOF'
import espeakng_loader as e
print(f"export PHONEMIZER_ESPEAK_LIBRARY={e.get_library_path()} ESPEAK_DATA_PATH={e.get_data_path()}")
EOF

# 3. Models
if [ "${1:-}" != "--no-models" ]; then
  HF=.venv/bin/hf
  [ -f base/Qwen3-TTS-12Hz-0.6B-Base/model.safetensors ] || "$HF" download Qwen/Qwen3-TTS-12Hz-0.6B-Base --local-dir base/Qwen3-TTS-12Hz-0.6B-Base > /dev/null
  [ -f base/Qwen3-TTS-Tokenizer-12Hz/model.safetensors ] || "$HF" download Qwen/Qwen3-TTS-Tokenizer-12Hz --local-dir base/Qwen3-TTS-Tokenizer-12Hz > /dev/null
  if [ ! -d enhance-weights/enhancer_stage2 ]; then
    "$HF" download ResembleAI/resemble-enhance --include "enhancer_stage2/*" --local-dir enhance-weights > /dev/null
  fi
fi

"$PY" - <<'EOF'
import torch, qwen_tts, speechbrain, mlx_whisper, parselmouth  # noqa: F401
from resemble_enhance.enhancer.inference import enhance  # noqa: F401
print(f"ok: torch {torch.__version__}, MPS {torch.backends.mps.is_available()}")
EOF
echo "done. export WISEGUY_WORK=$WORK  (and 'source $WORK/env.sh' before running nyjudge.py)"
