"""The wiseguy voice: a Qwen3-TTS model fine-tuned on North Jersey /
New York Italian-American speech, run on the Apple GPU through MLX."""

from __future__ import annotations

import os
import re
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from .respell import respell

SAMPLE_RATE = 24000
ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = Path(os.environ.get("WISEGUY_MODEL", ROOT / "models" / "wiseguy"))

_SENTENCE = re.compile(r"(?<=[.!?…])\s+")


def sentences(text: str, first_max: int = 80) -> list[str]:
    """Split into sentences so audio starts after the first one is generated.
    A long first sentence is split at its first comma for a faster start."""
    parts = [p.strip() for p in _SENTENCE.split(text.strip()) if p.strip()]
    if parts and len(parts[0]) > first_max and "," in parts[0][20:]:
        head, _, tail = parts[0].partition(",")
        if len(head) >= 12:
            parts[0:1] = [head + ",", tail.strip()]
    return parts


def _text_first_inputs(model, orig):
    """mlx-audio builds CustomVoice prompts in Qwen's streaming layout (text fed
    one token per audio step). The wiseguy model was fine-tuned on the
    text-first layout (all text, then audio), so build that instead. Audio is
    still streamed out as it's generated."""
    import mlx.core as mx

    def prepare(text, language="auto", speaker=None, ref_audio=None, ref_text=None, instruct=None):
        embeds, _, pad = orig(text, language=language, speaker=speaker, ref_audio=ref_audio, ref_text=ref_text, instruct=instruct)
        cfg = model.config
        ids = mx.array(model.tokenizer.encode(f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"))[None]
        proj = lambda x: model.talker.text_projection(model.talker.get_text_embeddings()(x))  # noqa: E731
        text_embed = proj(ids)
        eos = proj(mx.array([[cfg.tts_eos_token_id]]))
        codec = model.talker.get_input_embeddings()
        body = mx.concatenate([text_embed[:, 3:-5], eos], axis=1)
        body = body + codec(mx.array([[cfg.talker_config.codec_pad_id]]))
        start = pad + codec(mx.array([[cfg.talker_config.codec_bos_id]]))
        # drop the streaming layout's "first text token + codec bos" step
        return mx.concatenate([embeds[:, :-1], body.astype(embeds.dtype), start.astype(embeds.dtype)], axis=1), pad, pad

    return prepare


class Wiseguy:
    def __init__(
        self,
        model_dir: str | Path = MODEL_DIR,
        strength: int = 0,
        temperature: float = 0.9,  # measured best: livelier pitch AND higher naturalness than 0.8
        loud: bool = False,
        top_k: int = 50,
    ) -> None:
        from mlx_audio.tts.utils import load_model

        if not (Path(model_dir) / "config.json").exists():
            from .download import fetch

            fetch(model_dir)
        self.model = load_model(Path(model_dir))
        self.model._prepare_generation_inputs = _text_first_inputs(self.model, self.model._prepare_generation_inputs)
        self.speaker = self.model.get_supported_speakers()[0]
        self.strength = strength
        self.temperature = temperature
        self.top_k = top_k
        self.loud = loud

    def text(self, text: str) -> str:
        """What he'll actually read, after the respelling layer."""
        text = respell(text, self.strength)
        if self.loud:
            # He doesn't do statements: read sentence ends as exclamations.
            # Measured: widest pitch range of all settings tried (17.5 st vs 11.7).
            text = re.sub(r"(?<!\.)\.(\s|$)", r"!\1", text)
        return text

    def stream(self, text: str, interval: float = 0.32) -> Iterator[np.ndarray]:
        """Yield float32 audio chunks as they're generated."""
        for sentence in sentences(self.text(text)):
            for r in self.model.generate(
                text=sentence,
                voice=self.speaker,
                lang_code="auto",  # trained without a language tag
                temperature=self.temperature,
                top_k=self.top_k,
                stream=True,
                streaming_interval=interval,
                # 12.5 codec frames per second; even a slow reading of a sentence
                # fits in len/8 + 3 seconds. Caps the rare sampling runaway.
                max_tokens=int(12.5 * (len(sentence) / 8 + 3)),
            ):
                audio = np.asarray(r.audio, dtype=np.float32).reshape(-1)
                if audio.size:
                    yield audio
            yield np.zeros(int(0.08 * SAMPLE_RATE), np.float32)  # breath between sentences

    def say(self, text: str) -> np.ndarray:
        parts = list(self.stream(text, interval=2.0))
        return np.concatenate(parts) if parts else np.zeros(0, np.float32)
