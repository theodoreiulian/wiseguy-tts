"""Stability/intelligibility sweep: varied lines x reps x temperatures through the
production engine. Writes wavs + a jsonl of (line, respelled, temp, duration).
usage: stability.py model_dir out_dir [temps...]"""
import sys, os, json, time
import numpy as np, soundfile as sf
from wiseguy.engine import SAMPLE_RATE, Wiseguy

LINES = [
    "Hey, how you doing? Long time no see.",
    "You know what your problem is? You talk too much.",
    "I've been running this neighborhood for thirty years, and nobody tells me what to do.",
    "Listen to me, the boss wants his money by Friday. No excuses.",
    "Forget about it, I'm not going to the wedding.",
    "Get me a coffee, and none of that fancy stuff.",
    "I'm the best there is, ask anybody on the block.",
    "My mother makes the best gravy in the whole state of New Jersey.",
    "What are you, a comedian? Get out of here.",
    "That guy owes me big, and I'm going to collect.",
    "Nothing, I got nothing to say to the cops.",
    "We're going down the shore this weekend, the whole family.",
]
model, out = sys.argv[1], sys.argv[2]
temps = [float(t) for t in sys.argv[3:]] or [0.8]
os.makedirs(out, exist_ok=True)
g = Wiseguy(model)
list(g.stream("Ay."))
log = open(os.path.join(out, "runs.jsonl"), "w")
for temp in temps:
    g.temperature = temp
    for i, line in enumerate(LINES):
        for rep in range(2):
            t = time.time()
            a = np.concatenate(list(g.stream(line)))
            name = f"t{temp}_{i:02d}_{rep}.wav"
            sf.write(os.path.join(out, name), a, SAMPLE_RATE)
            log.write(json.dumps({"file": name, "line": line, "said": g.text(line), "temp": temp,
                                  "dur": round(len(a) / SAMPLE_RATE, 2), "gen": round(time.time() - t, 2)}) + "\n")
            log.flush()
print("done")
