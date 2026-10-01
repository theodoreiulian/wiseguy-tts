"""Run a heavy job without freezing the Mac.

    guard.py [--max-gb 10] [--min-free 25] [--max-swap-gb 1.5] [--threads 4] [--mps-ratio 0.5] -- cmd args...

- starts the job at low priority (nice 10) with capped CPU threads and a capped
  PyTorch-MPS / MLX memory budget
- a watchdog samples every 2 s and kills the whole job if:
    * the job's memory (RSS of its process tree) exceeds --max-gb
    * system free memory drops under --min-free percent
    * swap use grows by more than --max-swap-gb since the start
- every kill is logged with the reason; exit code 99 means "killed by guard"
"""

import argparse
import os
import re
import signal
import subprocess
import sys
import time


def free_percent() -> float:
    out = subprocess.run(["memory_pressure", "-Q"], capture_output=True, text=True).stdout
    m = re.search(r"free percentage:\s*(\d+)%", out)
    return float(m.group(1)) if m else 100.0


def swap_gb() -> float:
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    m = re.search(r"used = ([\d.]+)M", out)
    return float(m.group(1)) / 1024 if m else 0.0


def tree_rss_gb(pid: int) -> float:
    out = subprocess.run(["ps", "-Ao", "pid=,ppid=,rss="], capture_output=True, text=True).stdout
    kids, rss = {}, {}
    for line in out.split("\n"):
        parts = line.split()
        if len(parts) == 3:
            p, pp, r = map(int, parts)
            kids.setdefault(pp, []).append(p)
            rss[p] = r
    total, stack = 0, [pid]
    while stack:
        p = stack.pop()
        total += rss.get(p, 0)
        stack += kids.get(p, [])
    return total / 1024 / 1024


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-gb", type=float, default=10)
    ap.add_argument("--min-free", type=float, default=25)
    ap.add_argument("--max-swap-gb", type=float, default=1.5)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--mps-ratio", type=float, default=0.5, help="PyTorch MPS allocator cap (fraction of the GPU working set)")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd

    env = dict(os.environ)
    t = str(a.threads)
    env.update(OMP_NUM_THREADS=t, MKL_NUM_THREADS=t, VECLIB_MAXIMUM_THREADS=t, OPENBLAS_NUM_THREADS=t,
               # PyTorch MPS: cap the allocator (fraction of the GPU's recommended working set)
               PYTORCH_MPS_HIGH_WATERMARK_RATIO=str(a.mps_ratio), PYTORCH_MPS_LOW_WATERMARK_RATIO=str(round(a.mps_ratio * 0.8, 3)),
               PYTORCH_ENABLE_MPS_FALLBACK="0",  # never silently bounce ops to the CPU
               WISEGUY_GUARD_MAX_GB=str(a.max_gb))
    swap0 = swap_gb()
    proc = subprocess.Popen(["nice", "-n", "10", *cmd], env=env, start_new_session=True)
    reason, peak = None, 0.0
    while proc.poll() is None:
        time.sleep(2)
        rss, free, swap = tree_rss_gb(proc.pid), free_percent(), swap_gb() - swap0
        peak = max(peak, rss)
        if rss > a.max_gb:
            reason = f"job memory {rss:.1f} GB > {a.max_gb} GB"
        elif free < a.min_free:
            reason = f"system free memory {free:.0f}% < {a.min_free}%"
        elif swap > a.max_swap_gb:
            reason = f"swap grew {swap:.1f} GB > {a.max_swap_gb} GB"
        if reason:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            print(f"[guard] KILLED: {reason} (peak job memory {peak:.1f} GB)", file=sys.stderr, flush=True)
            return 99
    print(f"[guard] finished with code {proc.returncode}, peak job memory {peak:.1f} GB", file=sys.stderr, flush=True)
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
