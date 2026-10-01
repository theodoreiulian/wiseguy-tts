# wiseguy-tts

A local text-to-speech voice with a heavy North Jersey / New York
Italian-American accent: loud, fast-talking, straight out of a mob picture.
It's a [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) 0.6B model
fine-tuned on real accented speech, running on the Mac's GPU through
[MLX](https://github.com/Blaizzy/mlx-audio).

```bash
uv run wiseguy                                  # interactive: type a line, hear it
uv run wiseguy "Ay, how you doin'?"             # say one line
uv run wiseguy "Forget about it." -o line.wav   # write a WAV instead
uv run wiseguy --loud                           # most animated: every sentence an exclamation
uv run wiseguy --accent 2                       # optional: exaggerated street-talk respelling
```

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

**1. A voice that learned the accent from real speakers.** The model is
fine-tuned on about 6 hours of public-domain speech: House floor speeches from
official congressional channels, which are US government works. The speakers
are four New York / New Jersey men chosen for their accents:

| speaker | from | brings |
|---|---|---|
| Rep. Bill Pascrell | Paterson, NJ (Italian-American) | the Jersey vowels, a loud, fiery delivery |
| Rep. Michael Grimm | Brooklyn / Staten Island (Italian-American) | a dropped-r New York accent |
| Rep. Anthony D'Esposito | Long Island (Italian-American) | New York vowels |
| Rep. Peter King | Queens / Long Island | a heavily dropped-r accent |

**2. A new voice, not a clone.** Each clip is conditioned on its own
speaker's embedding during training. The shipped voice is a *blend* of the
speaker embeddings, weighted toward the speakers who drop their r's. It sounds
like one guy from the neighborhood, not like any of the four.

**3. Optional street talk on top (off by default).** A respelling layer
(`wiseguy/respell.py`) can push an even broader read:
gonna, whaddaya, lemme, 'cause, "fuhget about it", dese/dem/dose, tink,
wit', brudduh/mudduh, nuttin, and dropped g's. Every spelling was A/B-tested
on the model: a phoneme recognizer checked that the accent sound comes out,
and Whisper checked the word is still understood. Spellings that garbled
were dropped, e.g. "dere", "brotha", and "Fuhgeddaboudit" written as one word.
It's off by default because listening showed that even the surviving
respellings can sound like the wrong word: "duh cops" instead of a Jersey
"the cops". The trained accent carries the voice on plain text.

The source recordings, transcripts and weights are not in the repo. See
[FINETUNING.md](FINETUNING.md) for how to make them.

## Results

The accent is measured, not just claimed. `training/nyjudge.py` aligns
Whisper words with a wav2vec2 phoneme recognizer and Praat formants, then
scores the classic New York / Jersey features. It was calibrated on real
accented speech: the four source speakers, plus aggregate measurements of
reference TV dialogue that were used as a yardstick only, never for training.

| | dropped r's (F3 ratio¹) | raised "caw-fee" vowel | dropped g's | Whisper WER |
|---|---|---|---|---|
| target: heavy NY/NJ reference speech | 0.85–0.87 | 43–80 % | 100 % | n/a |
| stock Kokoro v1 (the first attempt) | n/a | 0 % | 0 % | n/a |
| Qwen3-TTS zero-shot clone | 0.76 | 73 % | 80 % | n/a |
| wiseguy v1 (trained on the raw floor recordings) | 0.86 | 91 % | 100 % | 0 % |
| **wiseguy v2 (default, trained on cleaned audio)** | **0.86** | **96 %** | **80–100 %** | **1 %** |

¹ The lowest third formant in the r-part of words like *car, here, brother*,
relative to the speaker's typical F3. Pronouncing the r pulls F3 down; about
0.7 means r-pronouncing, 0.85 and up means r-dropping.
Naturalness (UTMOS, a predicted 1–5 listener score) on 24 conversational lines:

| | UTMOS (mean / worst) | energy above 4 kHz |
|---|---|---|
| raw congressional training audio | 3.1 | |
| stock Qwen3-TTS | 4.0 | |
| wiseguy v1 ("old courtroom mic") | 3.6 / 2.5 | −8.3 dB |
| **wiseguy v2 (default)** | **4.2 / 3.4** | **−6.4 dB** |

v2 fixes the "old courtroom microphone" sound. Its training audio was run
through [Resemble-Enhance](https://github.com/resemble-ai/resemble-enhance)
(on CPU) to strip the chamber reverb, PA-mic coloration and compression.
Clips the cleanup couldn't fix were filtered out (UTMOS < 3.0, or Whisper no
longer matching the transcript), which left 3.1 h. The v1 model was then
fine-tuned on that audio. The voice blend shifted to Grimm 0.5 / King 0.5 to
keep the dropped r's at full strength.

### Expressiveness

Measured on 8 conversational lines, sampled 2–3 times each:

| delivery | pitch range | UTMOS | WER |
|---|---|---|---|
| temperature 0.8 (old default) | 11.7 st | 4.10 | 0.7 % |
| **temperature 0.9 (default)** | **15.2–15.5 st** | **4.13–4.27** | **0.7 %** |
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
