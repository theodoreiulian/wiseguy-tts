"""Encode utterances into Qwen3-TTS 12 Hz codec codes and compute speaker embeddings.

usage: prep.py manifest.jsonl out_with_codes.jsonl spk_emb.pt [--voice spk1:w,spk2:w] [--base-spk models/finetune/speakers.pt]

With --base-spk, the existing speakers' embeddings are kept and new speakers are added,
so old (replay) and new data can be trained together.
"""
import argparse, json, random

import librosa, numpy as np, torch

import paths

from qwen_tts import Qwen3TTSTokenizer
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

ap = argparse.ArgumentParser()
ap.add_argument("manifest")
ap.add_argument("out")
ap.add_argument("spk_out")
ap.add_argument("--voice", default=None, help="blend for the new voice, e.g. pascrell:0.5,grimm:0.5 (default: equal)")
ap.add_argument("--init", default=str(paths.BASE), help="Base model (its speaker encoder makes the embeddings)")
ap.add_argument("--tokenizer", default=str(paths.TOKENIZER))
ap.add_argument("--base-spk", default=None, help="existing speakers.pt to extend (e.g. models/finetune/speakers.pt)")
args = ap.parse_args()

rows = [json.loads(l) for l in open(args.manifest)]
tok = Qwen3TTSTokenizer.from_pretrained(args.tokenizer, device_map="mps")
with open(args.out, "w") as f:
    for i in range(0, len(rows), 16):
        batch = rows[i : i + 16]
        enc = tok.encode([r["audio"] for r in batch])
        for r, code in zip(batch, enc.audio_codes):
            r["audio_codes"] = code.cpu().tolist()
            f.write(json.dumps(r) + "\n")
        print(f"codes {i + len(batch)}/{len(rows)}", end="\r", flush=True)
print()
del tok

model = Qwen3TTSModel.from_pretrained(args.init, dtype=torch.float32, attn_implementation="sdpa").model.to("mps")
sr = model.speaker_encoder_sample_rate
names = sorted({r["speaker"] for r in rows})
embs = []
random.seed(0)
for n in names:
    clips = [r for r in rows if r["speaker"] == n and r["dur"] > 4]
    clips = random.sample(clips, min(40, len(clips)))
    e = []
    for r in clips:
        wav, _ = librosa.load(r["audio"], sr=sr)
        with torch.no_grad():
            e.append(model.extract_speaker_embedding(audio=wav, sr=sr).float().cpu())
    e = torch.stack(e)
    c = e.mean(0)
    sims = torch.nn.functional.cosine_similarity(e, c[None], dim=-1)
    print(f"{n}: {len(clips)} clips, cos to centroid mean {sims.mean():.3f} min {sims.min():.3f}, norm {c.norm():.2f}")
    embs.append(c)
if args.base_spk:  # keep existing speakers (replay data) and add the new ones
    old = torch.load(args.base_spk)
    for n, e in zip(old["names"], old["emb"]):
        if n not in names:
            names.append(n); embs.append(e)
    order = sorted(range(len(names)), key=lambda i: names[i])
    names = [names[i] for i in order]; embs = [embs[i] for i in order]
emb = torch.stack(embs)
weights = {n: 1.0 for n in names}
if args.voice:
    weights = {k: float(v) for k, v in (p.split(":") for p in args.voice.split(","))}
w = torch.tensor([weights.get(n, 0.0) for n in names])
voice = (w[:, None] * emb).sum(0) / w.sum()
for n, e in zip(names, emb):
    print(f"new voice vs {n}: cos {torch.nn.functional.cosine_similarity(voice, e, dim=0):.3f}")
torch.save({"names": names, "emb": emb, "voice": voice}, args.spk_out)
