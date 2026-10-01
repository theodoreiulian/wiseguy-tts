"""Point a manifest at the enhanced copies of its clips (only those that exist).
usage: enh_manifest.py manifest.jsonl enh_root out.jsonl"""
import json, os, sys
src, root, out = sys.argv[1:4]
n = 0
with open(out, "w") as f:
    for line in open(src):
        r = json.loads(line)
        p = os.path.join(root, r["speaker"], os.path.basename(r["audio"]))
        if os.path.exists(p):
            r["audio"] = os.path.abspath(p)
            f.write(json.dumps(r) + "\n"); n += 1
print(f"{n} enhanced clips -> {out}")
