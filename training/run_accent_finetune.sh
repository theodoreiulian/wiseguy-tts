#!/bin/zsh
# Run the long, restartable part of the accent fine-tuning pipeline unattended.
#
# The enhancement worker is launched separately because it is resumable per
# clip. Pass its PID here and this script will wait for it, validate that every
# curated clip was produced, then score, encode, train, and install a candidate.
# The shipped models/wiseguy directory is never modified.
#
# Usage:
#   WISEGUY_WORK=/path/to/work training/run_accent_finetune.sh ENHANCE_PID
#   WISEGUY_WORK=/path/to/work training/run_accent_finetune.sh launchd:LABEL

set -euo pipefail

ROOT=${0:A:h:h}
: "${WISEGUY_WORK:?set WISEGUY_WORK to the prepared training work directory}"
source "$WISEGUY_WORK/env.sh"

PY="$WISEGUY_WORK/.venv/bin/python"
GUARD=("$ROOT/.venv/bin/python" "$ROOT/training/guard.py")
DATA="$WISEGUY_WORK/data"
SEG="$DATA/seg_youtube_all"
RAW="$SEG/manifest_curated_raw.jsonl"
ENH_ROOT="$DATA/seg_youtube_enh"
ENH_MANIFEST="$SEG/manifest_curated_enh.jsonl"
SCORES="$SEG/curated_enh_scores.jsonl"
FINAL="$SEG/manifest_final.jsonl"
CODES="$DATA/youtube_codes.jsonl"
SPK="$DATA/youtube_speakers.pt"
RUN="$WISEGUY_WORK/runs/youtube-r1"
CANDIDATE="$ROOT/models/wiseguy-youtube-candidate"
WAIT_FOR=${1:-}

stage() {
  print -r -- "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"
}

if [[ "$WAIT_FOR" == launchd:* ]]; then
  wait_label=${WAIT_FOR#launchd:}
  stage "waiting for enhancement launchd service $wait_label"
  while launchctl print "gui/$UID/$wait_label" >/dev/null 2>&1; do
    sleep 30
  done
elif [[ -n "$WAIT_FOR" ]]; then
  stage "waiting for enhancement PID $WAIT_FOR"
  while kill -0 "$WAIT_FOR" 2>/dev/null; do
    sleep 30
  done
fi

expected=$(wc -l < "$RAW" | tr -d ' ')
actual=$(find "$ENH_ROOT" -type f -name '*.wav' | wc -l | tr -d ' ')
if [[ "$actual" != "$expected" ]]; then
  stage "ERROR enhancement incomplete: $actual/$expected clips"
  exit 2
fi
stage "enhancement complete: $actual clips"

score_count=0
[[ -f "$SCORES" ]] && score_count=$(wc -l < "$SCORES" | tr -d ' ')
if [[ -s "$FINAL" && "$score_count" == "$expected" ]]; then
  stage "reusing completed enhanced scores and quality-filtered manifest"
else
  stage "building enhanced manifest"
  "$PY" "$ROOT/training/enh_manifest.py" "$RAW" "$ENH_ROOT" "$ENH_MANIFEST"

  stage "scoring enhanced clips"
  "${GUARD[@]}" --max-gb 9 --min-free 25 --threads 4 -- \
    "$PY" "$ROOT/training/score_clips.py" \
    "$ENH_MANIFEST" "$SCORES" --keep "$FINAL"
fi

read -r final_clips final_seconds <<< "$("$PY" -c \
  'import json,sys; rows=[json.loads(x) for x in open(sys.argv[1])]; print(len(rows), int(sum(r["dur"] for r in rows)))' \
  "$FINAL")"
stage "quality gate retained $final_clips clips / $(( final_seconds / 60 )) minutes"
if (( final_clips < 100 || final_seconds < 900 )); then
  stage "ERROR too little quality-filtered material to train safely"
  exit 3
fi

code_count=0
[[ -f "$CODES" ]] && code_count=$(wc -l < "$CODES" | tr -d ' ')
if [[ -s "$SPK" && "$code_count" == "$final_clips" ]]; then
  stage "reusing completed codec tokens and speaker embeddings"
else
  stage "encoding codec tokens and speaker embeddings"
  cd "$ROOT/training"
  "${GUARD[@]}" --max-gb 10 --min-free 25 --mps-ratio 0.5 -- \
    "$PY" prep.py "$FINAL" "$CODES" "$SPK" \
    --base-spk "$ROOT/models/finetune/speakers.pt" \
    --voice grimm:0.5,king:0.5
fi

if [[ -s "$RUN/epoch1/model.safetensors" ]]; then
  stage "reusing completed two-epoch training checkpoint"
else
  stage "training two guarded epochs with replay data"
  mkdir -p "$RUN"
  cd "$ROOT/training"
  "${GUARD[@]}" --max-gb 16 --min-free 20 --mps-ratio 0.7 --threads 4 -- \
    "$PY" train_wiseguy.py \
    --init "$ROOT/models/finetune/checkpoint" \
    --data "$CODES" "$ROOT/models/finetune/replay_codes.jsonl" \
    --spk "$SPK" --out "$RUN" \
    --epochs 2 --batch 2 --accum 8 --lr 5e-6 --freeze_text --save_every_epoch
fi

if [[ -s "$CANDIDATE/model.safetensors" ]]; then
  stage "reusing converted 8-bit MLX candidate"
else
  stage "converting epoch1 to a separate 8-bit MLX candidate"
  cd "$ROOT"
  "${GUARD[@]}" --max-gb 10 --min-free 25 -- \
    uv run python training/install_model.py \
    "$RUN/epoch1" "$SPK" grimm:0.5,king:0.5 \
    --bits 8 --out "$CANDIDATE"
fi

stage "pipeline complete; candidate ready at $CANDIDATE"
