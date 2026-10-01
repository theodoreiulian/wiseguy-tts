# Fine-tuning the wiseguy voice on more data

Instructions for an AI agent (or person) continuing to train this voice.
Read all of it before running anything. Several of the rules exist because
breaking them froze or crashed the owner's laptop.

---

## 1. What exists

| thing | where | notes |
|---|---|---|
| shipped voice (what `uv run wiseguy` uses) | `models/wiseguy/` | 8-bit MLX, ~1.8 GB. Don't train this one. |
| **trainable checkpoint of the shipped voice** | `models/finetune/checkpoint/` | full-precision PyTorch Qwen3-TTS 0.6B, custom-voice format, speaker `wiseguy` in codec slot 3000. **Start every fine-tune from here.** |
| speaker embeddings + voice blend | `models/finetune/speakers.pt` | `{"names", "emb", "voice", "voice_blend"}`. Speakers: desposito, grimm, king, pascrell. Shipped blend: grimm 0.5 + king 0.5. |
| replay data | `models/finetune/replay_codes.jsonl` | the 1,345 clips (3.1 h) the shipped voice was trained on, as codec tokens + text + speaker. No audio needed. Mix it into every new run so the voice doesn't drift. |
| pipeline scripts | `training/` | all take explicit paths; shared locations are in `training/paths.py` |
| inference engine | `wiseguy/engine.py` | MLX; contains a required prompt-layout patch (see §7) |

Everything big and re-downloadable lives in a work directory outside the repo:
`$WISEGUY_WORK` (default `~/wiseguy-work`). `training/setup.sh` creates it.
`models/` and the work dir are gitignored. Never commit audio or weights.

How the voice was built (for context):

1. House floor speeches from four NY/NJ congressmen were downloaded from their
   official channels.
2. Transcribed with Whisper, cut into 2–14 s clips, and filtered to each
   folder's main speaker with ECAPA.
3. Cleaned with Resemble-Enhance to remove chamber reverb and PA-mic sound,
   then quality-filtered.
4. Encoded to Qwen 12 Hz codec tokens and fully fine-tuned on Apple MPS, each
   clip conditioned on its own speaker's x-vector.
5. The shipped voice is a blend of the speaker embeddings, so it isn't a clone
   of any single real person.

---

## 2. Hard rules

### Machine safety (owner: M1 Max, 32 GB, a laptop they're using at the same time)

- **Run every heavy job through `training/guard.py`.** It runs the job at low
  priority, caps threads and the PyTorch-MPS allocator, and kills the job if
  its RSS exceeds `--max-gb`, if system free memory drops under `--min-free`
  percent, or if swap grows. GPU memory does **not** show in RSS; the free-memory
  check is what protects against GPU jobs.
  ```bash
  .venv/bin/python training/guard.py --max-gb 12 --min-free 25 -- <command...>
  ```
- **One GPU job at a time.** Two concurrent Metal jobs crashed transcription
  ("Impacting Interactivity") and starved the machine.
- **Resemble-Enhance runs on CPU only.** On MPS it outputs garbage (gibberish,
  peak amplitude 0.03). With `PYTORCH_ENABLE_MPS_FALLBACK=1` on an old torch, it
  froze and then crashed the laptop. `guard.py` sets the fallback off; leave it off.
- Long jobs: tie `caffeinate -i -w <pid>` to the job, or the Mac sleeps and
  training stalls for hours.
- Ask the owner before starting anything that will run more than about 30
  minutes or use more than about 12 GB. Tell them the expected memory and
  duration.
- Free disk was often under 30 GB. Each full checkpoint is ~2.4 GB; delete the
  ones you don't keep.

### Voice identity

Don't let the output become a clone of one identifiable real person. Keep the
shipped voice a blend of several speakers' embeddings (§5.8).

---

## 3. Setup

```bash
cd <repo>
export WISEGUY_WORK=~/wiseguy-work
training/setup.sh            # ~10 min + ~3.7 GB downloads; --no-models for env only
source $WISEGUY_WORK/env.sh  # espeak paths for the phoneme recognizer (nyjudge.py)
W=$WISEGUY_WORK; PY=$W/.venv/bin/python; GUARD="$PWD/.venv/bin/python $PWD/training/guard.py"
```

`setup.sh` does the following:
- clones QwenLM/Qwen3-TTS at the pinned commit
- builds one PyTorch venv for data prep, training and evaluation
- installs Resemble-Enhance without its pinned deps and applies two patches (a
  deepspeed import stub, and a NumPy 2 `float()` fix in `lcfm/cfm.py`)
- downloads the Qwen3-TTS 0.6B Base model (its speaker encoder makes the
  embeddings), the 12 Hz codec, and the Resemble-Enhance weights

Inference (the CLI) uses the repo's own MLX venv (`uv sync`), not this one.

---

## 4. Choosing data

The goal is a heavy North Jersey / New York Italian-American accent, loud and
animated. What matters, in order:

1. **Accent strength.** Check a candidate speaker's real audio with the judge
   (§6) before downloading hours of it. The single most audible feature is
   r-dropping: `F3min_r_ratio` around 0.85 or higher means non-rhotic (good),
   and about 0.70 means r-pronouncing. Of the original speakers, King (0.84–0.89)
   and Grimm (0.89) drop r's; Pascrell and D'Esposito (about 0.71) don't, which
   is why the blend is Grimm + King.
2. **Recording quality.** Clean, close mic. Reverberant hall audio taught v1 an
   "old courtroom microphone" sound until the data was cleaned (§5.5).
3. **Delivery.** Loud, fast, emphatic speakers. Floor speeches are a formal
   register; casual speech would help.
4. **Amount.** About 1–2 h of clean audio per new speaker is plenty. Total
   training was 3–6 h.

Where the original data came from: the official YouTube channels of Reps.
Bill Pascrell (@RepPascrell), Michael Grimm (@repmichaelgrimm), Anthony
D'Esposito (@RepDEsposito) and Peter King (@RepPeterKing). Floor speeches and
official statements, downloaded with yt-dlp.

---

## 5. The pipeline

Put each speaker's recordings in their own folder: `$W/data/<speaker>/*.wav`.
Speaker names are lowercase, one per person. Don't reuse an existing name for a
different person.

### 5.1 Get audio (24 kHz mono WAV)

```bash
mkdir -p $W/data/newguy && cd $W/data/newguy
$W/.venv/bin/yt-dlp -x --audio-format wav --postprocessor-args "ffmpeg:-ar 24000 -ac 1" -o "%(id)s.%(ext)s" -a ids.txt
# or for local files: ffmpeg -i in.mp3 -ar 24000 -ac 1 out.wav
```

### 5.2 Transcribe (Whisper large-v3-turbo, word timestamps; GPU)

```bash
cd $W/data && $GUARD --max-gb 8 --min-free 25 -- $PY <repo>/training/transcribe.py newguy
```
This writes `<file>.json` next to each wav. It's resumable: done files are
skipped. Speed is about 1 minute per 5–10 minutes of audio.

### 5.3 Cut utterances

```bash
$PY <repo>/training/segment.py $W/data/seg_new newguy:$W/data/newguy
```
Clips are 2–14 s, cut at pauses and sentence ends. Low-confidence words and
clipped audio are dropped, and loudness is normalized to about −20 dBFS. Output
is `seg_new/<speaker>/*.wav` plus `seg_new/manifest.jsonl`
(`audio, text, speaker, dur`).

### 5.4 Keep only the target speaker (ECAPA)

```bash
$GUARD --max-gb 8 --min-free 25 -- $PY <repo>/training/ecapa_filter.py $W/data/seg_new/manifest.jsonl $W/data/seg_new/manifest_clean.jsonl 0.45
```
This drops presiding officers, reporters, news anchors and so on. Expect it to
keep 50–95% depending on the source. Don't use Qwen's own x-vectors for this:
their raw cosine is about 0.98 for everyone.

### 5.5 Clean the audio (Resemble-Enhance, **CPU only**)

The most important quality step. It takes about 16 s per clip per worker.

```bash
# split into 2 shards, run 2 guarded CPU workers (4 threads each)
$PY - "$W/data/seg_new" <<'EOF'
import json, random, sys
d = sys.argv[1]
rows = open(f"{d}/manifest_clean.jsonl").readlines(); random.seed(0); random.shuffle(rows)
for k in range(2): open(f"{d}/shard{k}.jsonl", "w").writelines(rows[k::2])
EOF
for k in 0 1; do
  nohup $GUARD --max-gb 12 --min-free 25 --threads 4 -- $PY <repo>/training/enhance_batch.py \
    $W/data/seg_new/shard$k.jsonl $W/data/seg_new_enh > $W/data/enh$k.log 2>&1 &
  nohup caffeinate -i -w $! >/dev/null 2>&1 &
done
```
Notes:
- Use `--max-gb 12`, not 9. Long clips peak at about 9.2 GB and get killed.
- It's resumable, so if a worker is killed, just rerun the same command.
- Then point a manifest at the cleaned copies:

```bash
$PY <repo>/training/enh_manifest.py $W/data/seg_new/manifest_clean.jsonl $W/data/seg_new_enh $W/data/seg_new/manifest_enh.jsonl
```

### 5.6 Score and filter

```bash
$GUARD --max-gb 9 --min-free 25 -- $PY <repo>/training/score_clips.py \
  $W/data/seg_new/manifest_enh.jsonl $W/data/seg_new/scores.jsonl --keep $W/data/seg_new/manifest_final.jsonl
```
The defaults keep clips with UTMOS ≥ 3.0 and WER ≤ 0.15. Last time that kept
1,345 of 1,999 clips, at a mean UTMOS of 3.45. Raw floor audio was about 3.1;
the cleanup alone took it to about 3.25.

### 5.7 Encode + speaker embeddings (GPU)

```bash
cd <repo>/training && $GUARD --max-gb 10 --min-free 25 --mps-ratio 0.5 -- $PY prep.py \
  $W/data/seg_new/manifest_final.jsonl $W/data/new_codes.jsonl $W/data/spk.pt \
  --base-spk ../models/finetune/speakers.pt --voice grimm:0.5,king:0.5
```
`--base-spk` keeps the four existing speakers' embeddings and adds the new
speaker(s), so the replay data and the new data can be trained together.
`--voice` sets the blend written into slot 3000. You can change it later
without retraining (§5.9).

### 5.8 Train (GPU, about 10 s per optimizer step)

Warm-start from the shipped voice and **always include the replay data**:

```bash
cd <repo>/training && nohup $GUARD --max-gb 16 --min-free 20 --mps-ratio 0.7 --threads 4 -- $PY train_wiseguy.py \
  --init ../models/finetune/checkpoint \
  --data $W/data/new_codes.jsonl ../models/finetune/replay_codes.jsonl \
  --spk $W/data/spk.pt --out $W/runs/r6 \
  --epochs 2 --batch 2 --accum 8 --lr 5e-6 --freeze_text --save_every_epoch > $W/runs/r6.log 2>&1 &
nohup caffeinate -i -w $! >/dev/null 2>&1 &
```

Settings that worked:
- **Learning rate:** 1e-5 for 2 epochs from the Base model; 5e-6 for warm
  starts; 3e-6 for gentle nudges.
- **Batch:** batch 2 × accum 8 (effective 16).
- **Text embedding:** keep it frozen (`--freeze_text`). It's half the
  parameters and doesn't need to change.
- **Memory:** about 12.5–13.7 GB of MPS. `--mps-ratio 0.7` is needed: 0.5 OOMs.
  The guard may kill a run if the owner's apps push free memory under 20%. Each
  epoch is saved, so resume from the last `epoch*` with `--init`.
- **Time:** steps ≈ clips × epochs / 16. For 1,345 clips × 2 epochs that's
  about 170 steps, about 30 minutes.
- **Loss** starts around 2.7–2.9 for a warm start and falls to about 2.2–2.3.
  From Base it starts at about 2.9–3.2. Main and sub-talker loss are summed as
  main + 0.3 × sub.

### 5.9 Pick the voice blend, install, compare

```bash
# re-blend without retraining (writes a copy with a new slot-3000 embedding)
$PY <repo>/training/revoice.py $W/runs/r6/epoch1 $W/runs/r6_blend $W/data/spk.pt grimm:0.5,king:0.5
# convert to 8-bit MLX as a *candidate* (never overwrite models/wiseguy until it wins)
cd <repo> && $GUARD --max-gb 10 --min-free 25 -- uv run python training/install_model.py \
  $W/runs/r6/epoch1 $W/data/spk.pt grimm:0.5,king:0.5 --bits 8 --out models/wiseguy-candidate
```
How the blend behaves:
- Weighting toward the non-rhotic speakers (Grimm, King) is what makes him drop
  his r's. Equal blends with Pascrell come out rhotic (F3 about 0.72–0.79).
- Always generate at least 2 renders per blend before judging. Single samples
  are noisy.

When the candidate beats the current voice on §6, swap it in: move
`models/wiseguy` aside and rename the candidate. Then refresh
`models/finetune/`:
- copy the winning **PyTorch** epoch dir, revoiced, into `checkpoint/`
  (`cp -RL`, no symlinks)
- copy the new `spk.pt` (with `voice_blend` set) into `speakers.pt`
- append the new clean codes to `replay_codes.jsonl`

That way the next agent starts from the new voice.

---

## 6. Evaluation (do all of it before shipping)

Generate through the real MLX engine, so you test what ships:

```bash
cd <repo>
$GUARD --max-gb 9 --min-free 25 -- uv run python training/eval_engine.py models/wiseguy-candidate /tmp/eval 0   # diag.txt + a boast line, with speed
$GUARD --max-gb 8 --min-free 25 -- uv run python training/stability.py models/wiseguy-candidate /tmp/stab 0.9    # 12 lines x 2
$GUARD --max-gb 6 --min-free 25 -- uv run python training/bench.py models/wiseguy-candidate                     # latency/CPU/RAM
source $W/env.sh && $GUARD --max-gb 9 --min-free 25 -- $PY training/nyjudge.py /tmp/eval/s0__diag.wav --json /tmp/j.json
```
- `nyjudge.py`: accent features. Keep each file at 35 s or less, or the
  phoneme recognizer exceeds about 8 GB.
- `score_clips.py` (on a manifest of the generated wavs): UTMOS + WER.

Ship only if the candidate holds or beats these (the shipped v2 values):

| metric | target | shipped v2 |
|---|---|---|
| `F3min_r_ratio` (dropped r's) | ≥ 0.84 | 0.86 |
| `thought_raised` ("caw-fee") | ≥ 0.7 | 0.96 |
| `g_dropped` (talkin') | ≥ 0.8 | 0.8–1.0 |
| UTMOS on stability lines (mean / worst) | ≥ 4.1 / ≥ 3.3 | 4.19 / 3.40 |
| Whisper WER, plain text | ≤ 2 % | 1.4 % |
| runaways (duration ≫ text) | 0 in 24+ gens | 0 |
| first audio / RTF / peak RAM | ≤ 200 ms / ≤ 0.4 / ≤ 2.5 GB | ~130 ms / 0.34 / 2.1 GB |
| pitch range at temp 0.9 | ≥ 14 semitones | ~15.3 |

The phone-level `r_dropped` and `th_stopped` numbers are noisy because Whisper
alignment is approximate. Trust `F3min_r_ratio`. In the end the owner listens:
render the six lines in §8 and let them decide.

---

## 7. Gotchas already solved (don't re-break them)

1. **Upstream fine-tuning script bugs** (QwenLM `finetuning/sft_12hz.py`); both
   are fixed in `train_wiseguy.py`:
   - *Double label shift.* Pre-shifted labels go into a loss that shifts again,
     so the model trains on targets 2 frames ahead (starting loss about 7.7
     instead of 1.1). Fix: cross-entropy computed by hand.
   - *Code-predictor off-by-one.* The sub-talker must get the talker hidden
     state from the position *before* each code (`codec_mask[:, 1:]`), not the
     code's own position. Without this fix the model babbles and never emits
     EOS.
   - Also: the 0.6B model needs `text_projection` on text embeddings (the
     upstream script only works for 1.7B), and batching different-length
     `ref_mels` fails, so precomputed per-speaker embeddings are used instead.
2. **Prompt layout.** The model is trained on the *text-first* (non-streaming)
   layout. mlx-audio builds CustomVoice prompts in the streaming layout;
   `wiseguy/engine.py::_text_first_inputs` rebuilds them. Remove it and output
   degrades. Also use `lang_code="auto"`: a language tag the model never saw
   makes it babble.
3. **Custom-voice checkpoints have no speaker encoder.** Make new embeddings
   with the Base model (`prep.py --init` defaults to it).
4. **Runaways.** `engine.py` caps tokens per sentence (`len/8 + 3` s). Keep it.
5. **Respelling** (`wiseguy/respell.py`, `--accent 1-3`): off by default. Eye
   dialect like "duh cops" was heard as the wrong word. The trained accent
   carries plain text.
6. **Temperature 0.9** is the default. It measured more expressive *and* more
   natural than 0.8. 1.0 is too random. `--loud` (sentence ends read as "!")
   gives the widest pitch range.
7. **Shell traps** (zsh):
   - An unmatched glob aborts the whole command line, `rm` included.
   - `until ! pgrep -f X` matches its own command line and never ends. Wait on
     log markers instead.
   - A command stored in a variable isn't word-split; use a function.
8. **mlx-audio warnings** like "model of type qwen3_tts to instantiate a model
   of type" are harmless.

---

## 8. Already tried (save yourself the time)

| idea | result |
|---|---|
| Kokoro-82M + rule-based phoneme accent (v0) | intelligible but obviously synthetic, generic accent; replaced |
| zero-shot cloning (Qwen3 0.6B/1.7B, Chatterbox, Turbo, VoxCPM2, OmniVoice) | Qwen3 carried the accent best; none invented r-dropping absent from the prompt |
| equal 3-speaker blend (Pascrell-heavy data) | rhotic (F3 0.72) |
| VoiceFixer for cleanup | worse than raw (UTMOS 2.6 vs 2.9) |
| Resemble-Enhance on MPS | corrupt output; CPU only |
| bf16 instead of 8-bit MLX | not better (4.04 vs 4.19), bigger and slower |
| short fine-tune on the most animated half of the clips | no wider pitch range, voice got deeper; not shipped |
| top_k 30 | slightly more "natural" score but weaker r-dropping; kept 50 |

Ideas not yet tried, roughly in order of expected value:
- Add casual, conversational (not floor-speech) audio from strongly non-rhotic
  NY/NJ Italian-American men.
- A hired voice actor reading a script written for this character. That's the
  cleanest route to "loud, boastful".
- LoRA fine-tune of the 1.7B model. Better prosody, but about 2× RAM and slower
  inference. Ask the owner first.

The six lines the owner judges by:
```
Hey, are you talking to me? I'm the best there is in this whole neighborhood, and everybody knows it.
Listen to me. The boss wants his money by Friday, no excuses, you understand?
Forget about it. I'm not going to that wedding, not after what his brother did.
My mother makes the best gravy in the whole state of New Jersey. Nobody touches her recipe.
What are you, a comedian? Get out of here before I lose my temper.
Nothing. I got nothing to say to the cops, and neither do you.
```
Render them with `uv run wiseguy "<line>" -o NN.wav` and `--loud`.
