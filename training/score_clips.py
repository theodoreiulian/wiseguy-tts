"""Score clips for realism (UTMOS) and transcript match (Whisper WER), then keep
the good ones. Used after enhancement to drop clips the cleanup couldn't fix.

usage: score_clips.py manifest.jsonl scores.jsonl [--keep out_manifest.jsonl]
                      [--min-utmos 3.0] [--max-wer 0.15]

Run it through guard.py. It uses the GPU (Whisper via MLX) plus ~3 GB of RAM.
"""
import argparse, json, re

import librosa, mlx_whisper, numpy as np, torch

ap = argparse.ArgumentParser()
ap.add_argument("manifest")
ap.add_argument("scores")
ap.add_argument("--keep", default=None)
ap.add_argument("--min-utmos", type=float, default=3.0)
ap.add_argument("--max-wer", type=float, default=0.15)
a = ap.parse_args()
torch.set_num_threads(4)
pred = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)


def norm(s):
    return re.sub(r"[^a-z' ]", " ", s.lower()).replace("'", "").split()


def wer(r, h):
    d = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        nd = [i]
        for j, hw in enumerate(h, 1):
            nd.append(min(d[j] + 1, nd[j - 1] + 1, d[j - 1] + (rw != hw)))
        d = nd
    return d[-1] / max(1, len(r))


rows = [json.loads(l) for l in open(a.manifest)]
kept = []
with open(a.scores, "w") as out:
    for i, r in enumerate(rows):
        y, _ = librosa.load(r["audio"], sr=16000)
        with torch.no_grad():
            m = pred(torch.tensor(y)[None], 16000).item()
        asr = mlx_whisper.transcribe(y, path_or_hf_repo="mlx-community/whisper-large-v3-turbo", language="en")["text"]
        r = {**r, "utmos": round(m, 3), "wer": round(wer(norm(r["text"]), norm(asr)), 3)}
        out.write(json.dumps(r) + "\n")
        if r["utmos"] >= a.min_utmos and r["wer"] <= a.max_wer:
            kept.append(r)
        if i % 250 == 0:
            print(f"{i}/{len(rows)}", flush=True)
print(f"scored {len(rows)}; mean UTMOS {np.mean([json.loads(l)['utmos'] for l in open(a.scores)]):.2f}; "
      f"kept {len(kept)} ({sum(r['dur'] for r in kept) / 3600:.2f} h)")
if a.keep:
    with open(a.keep, "w") as f:
        for r in kept:
            f.write(json.dumps({k: r[k] for k in ("audio", "text", "speaker", "dur")}) + "\n")
