"""Generate the diagnostic lines through the production MLX engine and time it.
usage: eval_engine.py model_dir out_dir [strength ...]"""
import sys, os, time
import numpy as np, soundfile as sf
from wiseguy.engine import SAMPLE_RATE, Wiseguy

model, out = sys.argv[1], sys.argv[2]
strengths = [int(x) for x in sys.argv[3:]] or [0, 2]
os.makedirs(out, exist_ok=True)
DIAG = open(os.path.join(os.path.dirname(__file__), "diag.txt")).read().strip()
BOAST = "Hey, you talkin' to me? I'm the best there is in this whole neighborhood, and everybody knows it. Forget about it."
t = time.time(); g = Wiseguy(model); list(g.stream("Ay.")); print(f"load+warmup {time.time()-t:.1f}s")
for s in strengths:
    g.strength = s
    for tag, text in (("diag", DIAG), ("boast", BOAST)):
        t = time.time(); first = None; parts = []
        for a in g.stream(text):
            if first is None: first = time.time() - t
            parts.append(a)
        audio = np.concatenate(parts); el = time.time() - t
        sf.write(f"{out}/s{s}__{tag}.wav", audio, SAMPLE_RATE)
        print(f"s{s} {tag}: {len(audio)/SAMPLE_RATE:.1f}s audio, gen {el:.1f}s (RTF {el/(len(audio)/SAMPLE_RATE):.2f}), first audio {first*1000:.0f}ms", flush=True)
