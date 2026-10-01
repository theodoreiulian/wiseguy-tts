"""Install a trained checkpoint as the CLI's voice: re-blend the voice, convert
to MLX (optionally quantized) and write it to models/wiseguy.

usage: install_model.py CKPT SPK_PT BLEND [--bits 8] [--out models/wiseguy]
   e.g. install_model.py ~/wiseguy-work/runs/r6/epoch1 ~/wiseguy-work/data/spk.pt grimm:0.5,king:0.5
"""
import argparse, os, shutil, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("ckpt")
ap.add_argument("spk")
ap.add_argument("blend")
ap.add_argument("--bits", type=int, default=0, help="quantize to this many bits (0 = keep bf16)")
ap.add_argument("--out", default=os.path.join(HERE, "..", "models", "wiseguy"))
sys.path.insert(0, HERE)
import paths  # noqa: E402

ap.add_argument("--train-python", default=str(paths.VENV_PY))
args = ap.parse_args()

with tempfile.TemporaryDirectory() as tmp:
    voiced = os.path.join(tmp, "voiced")
    subprocess.run([args.train_python, os.path.join(HERE, "revoice.py"), args.ckpt, voiced, args.spk, args.blend], check=True)
    if os.path.exists(args.out):
        shutil.rmtree(args.out)
    cmd = [sys.executable, "-m", "mlx_audio.convert", "--hf-path", voiced, "--mlx-path", args.out, "--dtype", "bfloat16", "--model-domain", "tts"]
    if args.bits:
        cmd += ["-q", "--q-bits", str(args.bits)]
    subprocess.run(cmd, check=True)
print("installed", os.path.abspath(args.out))
