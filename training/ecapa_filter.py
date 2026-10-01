"""Speaker filter with SpeechBrain ECAPA (VoxCeleb). Per folder: iterative centroid,
keep utterances with cosine > thr to their own speaker's centroid AND closer to it
than to any other speaker's centroid.
usage: ecapa_filter.py manifest.jsonl out.jsonl [thr]"""
import sys, json, torch, torchaudio, librosa
from speechbrain.inference.speaker import EncoderClassifier
enc = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="/tmp/ecapa", run_opts={"device": "cpu"})
rows = [json.loads(l) for l in open(sys.argv[1])]
thr = float(sys.argv[3]) if len(sys.argv) > 3 else 0.45
E = []
for i, r in enumerate(rows):
    y, _ = librosa.load(r["audio"], sr=16000)
    with torch.no_grad():
        E.append(enc.encode_batch(torch.tensor(y)[None]).squeeze())
    if i % 300 == 0: print("emb", i, len(rows), flush=True)
E = torch.nn.functional.normalize(torch.stack(E), dim=-1)
torch.save(E, sys.argv[2] + ".ecapa.pt")
names = sorted({r["speaker"] for r in rows})
idx = {n: [i for i, r in enumerate(rows) if r["speaker"] == n] for n in names}
cent = {}
for n in names:
    c = E[idx[n]].mean(0)
    for _ in range(4):
        s = E[idx[n]] @ torch.nn.functional.normalize(c, dim=0)
        c = E[[i for i, v in zip(idx[n], s) if v > thr] or idx[n]].mean(0)
    cent[n] = torch.nn.functional.normalize(c, dim=0)
C = torch.stack([cent[n] for n in names])
for n in names: print(n, " ".join(f"{m}:{cent[n]@cent[m]:.2f}" for m in names))
keep = []
for n in names:
    S = E[idx[n]] @ C.T
    own = S[:, names.index(n)]
    ok = [i for i, o, row in zip(idx[n], own, S) if o > thr and o >= row.max()]
    print(f"{n}: kept {len(ok)}/{len(idx[n])} ({sum(rows[i]['dur'] for i in ok)/3600:.2f} h), own-sim median {own.median():.2f} p10 {own.quantile(.1):.2f}")
    dropped = [(float(o), rows[i]["text"][:80]) for i, o in zip(idx[n], own) if i not in ok][:4]
    for d in dropped: print("   drop", f"{d[0]:.2f}", d[1])
    keep += ok
with open(sys.argv[2], "w") as f:
    for i in sorted(keep): f.write(json.dumps(rows[i]) + "\n")
