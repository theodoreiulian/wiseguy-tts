"""Make a copy of a checkpoint with a different blended voice in slot 3000.
usage: revoice.py src_ckpt dst spk.pt name:w,name:w"""
import os, sys, torch
from safetensors.torch import load_file, save_file
src, dst, spkf, blend = sys.argv[1:5]
spk = torch.load(spkf)
w = dict((k, float(v)) for k, v in (p.split(":") for p in blend.split(",")))
wt = torch.tensor([w.get(n, 0.0) for n in spk["names"]])
voice = (wt[:, None] * spk["emb"]).sum(0) / wt.sum()
os.makedirs(dst, exist_ok=True)
for f in os.listdir(src):
    if f != "model.safetensors" and not os.path.exists(os.path.join(dst, f)):
        os.symlink(os.path.abspath(os.path.join(src, f)), os.path.join(dst, f))
sd = load_file(os.path.join(src, "model.safetensors"))
sd["talker.model.codec_embedding.weight"][3000] = voice.to(sd["talker.model.codec_embedding.weight"].dtype)
save_file(sd, os.path.join(dst, "model.safetensors"))
print("wrote", dst, dict(zip(spk["names"], wt.tolist())))
