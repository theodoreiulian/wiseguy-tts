"""Interactive demo: type a line, hear it said like you're from Newark.

    uv run wiseguy                      # interactive
    uv run wiseguy "Ay, how you doin'?" # say one line and exit
    uv run wiseguy "..." -o line.wav    # write a WAV instead of playing
"""

from __future__ import annotations

import argparse
import queue
import resource
import sys
import threading
import time

import numpy as np

from .engine import MODEL_DIR, SAMPLE_RATE, Wiseguy

BANNER = """
  ╔══════════════════════════════════════════════╗
  ║   W I S E G U Y   ·   Jersey accent TTS      ║
  ║   "Ay, you type it, I say it. Capeesh?"      ║
  ╚══════════════════════════════════════════════╝

  Type something and hit Enter (Ctrl+C stops him mid-sentence).
  Commands: :accent 0-3  :loud on|off  :temp 0.9  :save file.wav  :stats  :quit
"""


def rss_mb() -> float:
    # ru_maxrss is bytes on macOS, KiB on Linux
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1 << 20) if sys.platform == "darwin" else peak / 1024


class Speaker:
    """Plays chunks as they are generated so the first words come out fast."""

    def __init__(self) -> None:
        import sounddevice as sd

        self.sd = sd

    def play(self, chunks) -> dict:
        q: queue.Queue = queue.Queue()
        stop = threading.Event()
        stats = {"first": None, "audio": 0.0}
        start = time.perf_counter()

        def produce():
            try:
                for audio in chunks:
                    if stop.is_set():
                        return
                    if stats["first"] is None:
                        stats["first"] = time.perf_counter() - start
                    stats["audio"] += len(audio) / SAMPLE_RATE
                    q.put(audio)
            finally:
                q.put(None)

        worker = threading.Thread(target=produce, daemon=True)
        worker.start()
        try:
            with self.sd.OutputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32") as out:
                while (audio := q.get()) is not None:
                    out.write(audio.reshape(-1, 1))
        finally:
            stop.set()  # Ctrl+C: stop generating the rest of the line too
        stats["total"] = time.perf_counter() - start
        return stats


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="wiseguy", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("text", nargs="*", help="say this and exit (omit for interactive mode)")
    ap.add_argument("-o", "--out", help="write a WAV file instead of playing")
    ap.add_argument("--accent", type=int, default=0, choices=range(4),
                    help="optional street-talk respelling on top of the trained accent: "
                         "0 none (default, most natural), 1 slang, 2 heavy, 3 old-school")
    ap.add_argument("--temp", type=float, default=0.9, help="sampling temperature (more = livelier, less = steadier)")
    ap.add_argument("--loud", action="store_true", help="read every sentence as an exclamation: the most animated delivery")
    ap.add_argument("--model", default=str(MODEL_DIR), help="model directory")
    ap.add_argument("--show-text", action="store_true", help="print the respelled text he actually reads")
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    engine = Wiseguy(args.model, strength=args.accent, temperature=args.temp, loud=args.loud)
    for _ in engine.stream("Ay."):  # warm-up: compile the GPU kernels before the first real line
        pass
    load = time.perf_counter() - t0

    def generate(text: str):
        yield from engine.stream(text)

    if args.text:
        text = " ".join(args.text)
        if args.show_text:
            print(engine.text(text))
        if args.out:
            import soundfile as sf

            audio = np.concatenate(list(generate(text)))
            sf.write(args.out, audio, SAMPLE_RATE)
            print(f"wrote {args.out} ({len(audio) / SAMPLE_RATE:.1f}s)")
        else:
            Speaker().play(generate(text))
        return

    print(BANNER)
    print(f" loaded in {load:.1f}s · peak RAM {rss_mb():.0f} MB\n")
    speaker = Speaker()
    last: list[np.ndarray] = []
    while True:
        try:
            line = input("\033[1myou>\033[0m ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nFuhgeddaboudit. Ciao.")
            return
        if not line:
            continue
        if line.startswith(":"):
            cmd, _, arg = line[1:].partition(" ")
            if cmd in ("q", "quit", "exit"):
                return
            elif cmd == "accent" and arg.isdigit():
                engine.strength = min(3, int(arg))
                print(f"accent {engine.strength}")
            elif cmd == "loud":
                engine.loud = arg != "off"
                print(f"loud {'on' if engine.loud else 'off'}")
            elif cmd == "temp" and arg:
                engine.temperature = min(1.5, max(0.1, float(arg)))
                print(f"temperature {engine.temperature}")
            elif cmd == "save" and arg:
                import soundfile as sf

                if last:
                    sf.write(arg, np.concatenate(last), SAMPLE_RATE)
                    print(f"saved {arg}")
                else:
                    print("nothin' to save yet")
            elif cmd == "stats":
                print(f"peak RAM {rss_mb():.0f} MB")
            else:
                print("commands: :accent 0-3  :loud on|off  :temp 0.9  :save file.wav  :stats  :quit")
            continue
        if args.show_text:
            print(f"  \033[2m{engine.text(line)}\033[0m")
        last = []

        def gen(text=line):
            for audio in generate(text):
                last.append(audio)
                yield audio

        cpu0 = time.process_time()
        try:
            stats = speaker.play(gen())
        except KeyboardInterrupt:
            print()
            continue
        cpu = time.process_time() - cpu0
        print(
            f"  \033[2mfirst audio {stats['first'] * 1000:.0f} ms · {stats['audio']:.1f}s spoken · "
            f"CPU {cpu:.1f}s ({cpu / max(stats['total'], 1e-6) * 100:.0f}% of one core) · peak RAM {rss_mb():.0f} MB\033[0m"
        )


if __name__ == "__main__":
    main()
