"""Cut long recordings into clean 2-14 s training utterances.

Uses the Whisper word timestamps (<file>.json next to each wav): cut at pauses
and sentence ends, drop low-confidence words, pad the edges, loudness-normalise.
Writes <out>/<speaker>/<file>_<n>.wav and a manifest.jsonl.

usage: segment.py out_dir speaker:dir [speaker:dir ...]
"""
import sys, os, json, glob, re
import numpy as np, soundfile as sf

SR = 24000
MIN_S, MAX_S, GAP = 2.0, 14.0, 0.45


def utterances(words):
    cur = []
    for w in words:
        if cur:
            gap = w["start"] - cur[-1]["end"]
            dur = w["end"] - cur[0]["start"]
            sentence_end = cur[-1]["word"].strip()[-1:] in ".?!"
            if gap > GAP or dur > MAX_S or (sentence_end and dur > 6 and gap > 0.15):
                yield cur
                cur = []
        cur.append(w)
    if cur:
        yield cur


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    manifest = open(os.path.join(out, "manifest.jsonl"), "w")
    total = {}
    for arg in sys.argv[2:]:
        spk, d = arg.split(":")
        os.makedirs(os.path.join(out, spk), exist_ok=True)
        n_spk = 0.0
        for js in sorted(glob.glob(f"{d}/*.json")):
            wav = js[:-5] + ".wav"
            if not os.path.exists(wav) or js.endswith(".segs.json"):
                continue
            audio, sr = sf.read(wav, dtype="float32")
            assert sr == SR, wav
            words = [w for s in json.load(open(js)) for w in s.get("words", [])]
            for k, utt in enumerate(utterances(words)):
                s, e = utt[0]["start"] - 0.12, utt[-1]["end"] + 0.18
                dur = e - s
                text = "".join(w["word"] for w in utt).strip()
                probs = [w.get("probability", 1) for w in utt]
                if dur < MIN_S or dur > MAX_S + 0.5 or np.mean(probs) < 0.75 or min(probs) < 0.2:
                    continue
                if len(text.split()) < 4 or not re.search(r"[a-zA-Z]", text):
                    continue
                seg = audio[max(0, int(s * SR)) : int(e * SR)]
                peak = np.abs(seg).max()
                if peak < 0.02 or (np.abs(seg) > 0.99).mean() > 0.001:  # silence or clipped
                    continue
                rms = np.sqrt(np.mean(seg ** 2))
                seg = seg * min(0.1 / max(rms, 1e-4), 0.95 / peak)  # ~ -20 dBFS, no clipping
                name = f"{os.path.basename(wav)[:-4]}_{k:04d}.wav"
                path = os.path.abspath(os.path.join(out, spk, name))
                sf.write(path, seg, SR)
                manifest.write(json.dumps({"audio": path, "text": text, "speaker": spk, "dur": round(dur, 2)}) + "\n")
                n_spk += dur
        total[spk] = n_spk / 3600
    print({k: f"{v:.2f} h" for k, v in total.items()})


if __name__ == "__main__":
    main()
