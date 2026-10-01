"""Resemble-Enhance a manifest's clips on MPS, resumably, gently.
usage: enhance_batch.py manifest.jsonl out_root [--limit N] [--nfe 32]"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np, soundfile as sf, torch, librosa
from resemble_enhance.enhancer.inference import enhance
torch.set_num_threads(4)
ap = argparse.ArgumentParser(); ap.add_argument("manifest"); ap.add_argument("out"); ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--nfe", type=int, default=32); ap.add_argument("--every", type=int, default=1)
a = ap.parse_args()
rows = [json.loads(l) for l in open(a.manifest)][:: a.every]
if a.limit: rows = rows[: a.limit]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths  # noqa: E402

RUN = paths.ENHANCE_WEIGHTS
t0, done = time.time(), 0
for i, r in enumerate(rows):
    out = os.path.join(a.out, r["speaker"], os.path.basename(r["audio"]))
    if os.path.exists(out): continue
    os.makedirs(os.path.dirname(out), exist_ok=True)
    wav, sr = sf.read(r["audio"], dtype="float32")
    h, hs = enhance(torch.from_numpy(wav), sr, "cpu", nfe=a.nfe, solver="midpoint", lambd=0.9, tau=0.5, run_dir=RUN)
    y = librosa.resample(h.numpy(), orig_sr=hs, target_sr=24000)
    peak = np.abs(y).max(); rms = np.sqrt(np.mean(y ** 2))
    y = y * min(0.1 / max(rms, 1e-4), 0.95 / max(peak, 1e-4))  # same loudness norm as segment.py
    tmp = out + ".tmp.wav"; sf.write(tmp, y, 24000); os.replace(tmp, out)
    done += 1
    if done % 25 == 0:
        el = time.time() - t0
        print(f"{i+1}/{len(rows)} enhanced, {el/done:.2f}s/clip, eta {(len(rows)-i-1)*el/done/60:.0f} min", flush=True)
print(f"done: {done} clips in {(time.time()-t0)/60:.1f} min", flush=True)
