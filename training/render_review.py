"""Render the six listening-review lines from FINETUNING.md in one model load.

usage: render_review.py model_dir output_dir
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

from wiseguy.engine import SAMPLE_RATE, Wiseguy


LINES = [
    "Hey, are you talking to me? I'm the best there is in this whole neighborhood, and everybody knows it.",
    "Listen to me. The boss wants his money by Friday, no excuses, you understand?",
    "Forget about it. I'm not going to that wedding, not after what his brother did.",
    "My mother makes the best gravy in the whole state of New Jersey. Nobody touches her recipe.",
    "What are you, a comedian? Get out of here before I lose my temper.",
    "Nothing. I got nothing to say to the cops, and neither do you.",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    voice = Wiseguy(args.model)
    list(voice.stream("Ay."))
    for loud in (False, True):
        voice.loud = loud
        mode = "loud" if loud else "plain"
        for index, line in enumerate(LINES, 1):
            audio = np.concatenate(list(voice.stream(line)))
            path = args.output / f"{index:02d}_{mode}.wav"
            sf.write(path, audio, SAMPLE_RATE)
            print(f"{path}: {len(audio) / SAMPLE_RATE:.1f}s", flush=True)


if __name__ == "__main__":
    main()
