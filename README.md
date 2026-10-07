# wiseguy-tts

A local text-to-speech voice with a heavy North Jersey / New York
Italian-American accent: loud, fast-talking, straight out of a mob picture.
It's a [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) 0.6B model
fine-tuned on public footage and movie clips, running on the Mac's GPU through
[MLX](https://github.com/Blaizzy/mlx-audio).

```bash
uv run wiseguy                                  # interactive: type a line, hear it
uv run wiseguy "Ay, how you doin'?"             # say one line
uv run wiseguy "Forget about it." -o line.wav   # write a WAV instead
uv run wiseguy --loud                           # most animated: every sentence an exclamation
uv run wiseguy --accent 2                       # optional: exaggerated street-talk respelling
```

The first run downloads the ~1.8 GB voice weights automatically from
[Hugging Face: theodoreiulian/wiseguy-tts](https://huggingface.co/theodoreiulian/wiseguy-tts)
into `models/wiseguy/` (Apache 2.0, a fine-tune of Qwen3-TTS 0.6B). To fetch them
ahead of time, run `uv run wiseguy-download`. Set `WISEGUY_MODEL` to use weights
elsewhere.

Inside the demo:

| command | does |
|---|---|
| `:accent 0-3` | optional street-talk respelling on top of the trained accent: **0 none (default)**, 1 slang, 2 heavy, 3 old-school |
| `:loud on/off` | read sentences as exclamations: the most animated delivery |
| `:temp 0.9` | sampling temperature: higher is livelier, lower is steadier |
| `:save x.wav` | save the last line |
| `:stats` | peak RAM |
| Ctrl+C | shut him up mid-sentence |

## How it works

**1. A voice that learned the accent from real speech.** The model was
fine-tuned on public footage and movie clips of North Jersey / New York
speakers. The audio was filtered for speaker identity, accent, and audio quality,
and enhanced where needed. Each clip is conditioned on its own speaker's
embedding during training, so the shipped voice is a new blend, not a clone of
any one person.

The current v3 model adds a second, replay-protected fine-tune sourced from
official or authorized YouTube film/TV clips and actor interviews. The source
pool covered *The Godfather*, *Donnie Brasco*, *Goodfellas*, *Casino*, and
*The Sopranos*. From 7.89 raw hours, speaker verification, accent auditing,
audio enhancement, UTMOS, and transcript matching retained 295 clips (27.7
minutes). Those clips were trained alongside all 3.1 hours of the v2 replay
set, so the stronger conversational accent did not erase the clean base voice.

**2. Optional street talk on top (off by default).** A respelling layer
(`wiseguy/respell.py`) can push an even broader read:
gonna, whaddaya, lemme, 'cause, "fuhget about it", dese/dem/dose, tink,
wit', brudduh/mudduh, nuttin, and dropped g's. Every spelling was A/B-tested
on the model: a phoneme recognizer checked that the accent sound comes out,
and Whisper checked the word is still understood. Spellings that garbled
were dropped, e.g. "dere", "brotha", and "Fuhgeddaboudit" written as one word.
It's off by default because listening showed that even the surviving
respellings can sound like the wrong word: "duh cops" instead of a Jersey
"the cops". The trained accent carries the voice on plain text.

The training recordings and transcripts are not in the repo. See
[FINETUNING.md](FINETUNING.md) for how to fine-tune further.

## Results

The accent is measured, not just claimed. `training/nyjudge.py` aligns
Whisper words with a wav2vec2 phoneme recognizer and Praat formants, then
scores the classic New York / Jersey features. It was calibrated on real
accented speech, plus aggregate measurements of reference dialogue used as a
yardstick.

| | dropped r's (F3 ratio¹) | raised "caw-fee" vowel | dropped g's | Whisper WER |
|---|---|---|---|---|
| target: heavy NY/NJ reference speech | 0.85–0.87 | 43–80 % | 100 % | n/a |
| stock Kokoro v1 (the first attempt) | n/a | 0 % | 0 % | n/a |
| Qwen3-TTS zero-shot clone | 0.76 | 73 % | 80 % | n/a |
| wiseguy v1 | 0.86 | 91 % | 100 % | 0 % |
| wiseguy v2 (cleaned training audio) | 0.84–0.86 | 91–96 % | 80–100 % | 1.0–1.4 % |
| **wiseguy v3 (default)** | **0.88** | **73 %** | **100 %** | **2.0 %** |

¹ The lowest third formant in the r-part of words like *car, here, brother*,
relative to the speaker's typical F3. Pronouncing the r pulls F3 down; about
0.7 means r-pronouncing, 0.85 and up means r-dropping.
Naturalness (UTMOS, a predicted 1–5 listener score) on 24 conversational lines:

| | UTMOS (mean / worst) | energy above 4 kHz |
|---|---|---|
| raw training audio | 3.1 | |
| stock Qwen3-TTS | 4.0 | |
| wiseguy v1 ("old courtroom mic") | 3.6 / 2.5 | −8.3 dB |
| wiseguy v2 | 4.1–4.2 / 3.0–3.4 | −6.4 dB |
| **wiseguy v3 (default, 48 conversational renders)** | **4.12 / 3.37** | |

v2 fixes the "old courtroom microphone" sound. Its training audio was run
through [Resemble-Enhance](https://github.com/resemble-ai/resemble-enhance)
(on CPU) to strip room reverb, mic coloration and compression. Clips the
cleanup couldn't fix were filtered out (UTMOS < 3.0, or Whisper no longer
matching the transcript). The v1 model was then fine-tuned on that audio.

### Expressiveness

Measured on 8 conversational lines, sampled 2–3 times each:

| delivery | pitch range | UTMOS | WER |
|---|---|---|---|
| temperature 0.8 (old default) | 11.7 st | 4.10 | 0.7 % |
| v2, temperature 0.9 | 15.2–15.5 st | 4.13–4.27 | 0.7 % |
| **v3, temperature 0.9 (default)** | **14.9 st** | **4.12** | **2.0 %** |
| temperature 0.9 + `--loud` | 17.5 st | 4.14 | 0 % |
| temperature 1.0 | 13.1 st | 4.00 | 1.9 % (too random) |

A short fine-tune on the most animated half of the training clips was also
tried. It didn't widen the pitch range, and it made the voice deeper, so it
wasn't shipped. Full-precision (bf16) weights weren't better than 8-bit either.

## Speed and resources (M1 Max, 32 GB)

The default model is 8-bit quantized MLX. It runs on the GPU, so the CPU
stays mostly free.

| | |
|---|---|
| time to first audio | about 120–190 ms |
| generation speed | about 3× real time (RTF about 0.34) |
| CPU while talking | about 25 % of one core, about 2.5 % of the machine |
| CPU idle at the prompt | 0 % |
| memory | about 2.1 GB peak (model plus codec, in unified memory) |
| load time | about 3 s |

## Fine-tuning further / rebuilding

See **[FINETUNING.md](FINETUNING.md)**: setup (`training/setup.sh`), the data
pipeline, training, evaluation targets, and the guardrails that keep the Mac
responsive. The trainable checkpoint of the shipped voice, its speaker
embeddings and its training data (as codec tokens) are in `models/finetune/`.

Two fixes to the upstream Qwen3-TTS fine-tuning recipe were needed, both
documented in `training/train_wiseguy.py`. Without them, the fine-tuned model
babbles and never stops:

- The loss was shifted twice, so the model trained against targets two frames
  ahead.
- The code predictor was conditioned on the talker state one step late.
