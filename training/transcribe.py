import sys, glob, json, os, mlx_whisper, warnings
warnings.filterwarnings("ignore")
for d in sys.argv[1:]:
    for f in sorted(glob.glob(f"{d}/*.wav")):
        out = f[:-4] + ".json"
        if os.path.exists(out): continue
        r = mlx_whisper.transcribe(f, path_or_hf_repo="mlx-community/whisper-large-v3-turbo", language="en", word_timestamps=True, condition_on_previous_text=False)
        json.dump(r["segments"], open(out, "w"))
        print(f, len(r["segments"]), flush=True)
