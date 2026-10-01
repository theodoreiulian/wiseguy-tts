"""Speed / memory / CPU of the production engine.
usage: bench.py model_dir"""
import sys, time, resource
import numpy as np, mlx.core as mx
from wiseguy.engine import SAMPLE_RATE, Wiseguy
LINES = ["Ay, how you doin'?",
         "Listen to me, I been running this neighborhood for thirty years, and nobody, I mean nobody, tells me what to do.",
         "You know what your problem is? You talk too much. Now get outta here before I lose my temper."]
t = time.perf_counter(); g = Wiseguy(sys.argv[1]); list(g.stream("Ay.")); load = time.perf_counter() - t
for line in LINES:
    c0 = time.process_time(); t = time.perf_counter(); first = None; n = 0
    for a in g.stream(line):
        if first is None: first = time.perf_counter() - t
        n += len(a)
    el = time.perf_counter() - t; cpu = time.process_time() - c0; audio = n / SAMPLE_RATE
    wall = max(el, audio)  # when played live, the process runs as long as the audio does
    print(f"{len(line):4d} chars: first audio {first*1000:4.0f} ms, {audio:4.1f}s audio in {el:4.1f}s (RTF {el/audio:.2f}), "
          f"CPU {cpu:4.1f}s = {cpu/wall*100:3.0f}% of one core while talking ({cpu/wall*10:.1f}% of the machine)")
print(f"load {load:.1f}s, peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**30:.2f} GB, "
      f"peak GPU memory {mx.get_peak_memory()/2**30:.2f} GB")
