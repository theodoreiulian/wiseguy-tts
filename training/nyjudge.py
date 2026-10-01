"""Objective New York / North Jersey accent judge.

For each wav: Whisper word timestamps + wav2vec2 phoneme CTC (frame-timed)
-> per-word recognized phones -> rates of the classic NYC features, plus
Praat formants at the vowel of THOUGHT/LOT/TRAP words.

usage: nyjudge.py file_or_dir [...] [--json out.json]
"""
import sys, os, glob, json, re, collections, warnings
import numpy as np, librosa, torch, parselmouth, mlx_whisper
warnings.filterwarnings("ignore")
from transformers import AutoProcessor, AutoModelForCTC

P = AutoProcessor.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft")
M = AutoModelForCTC.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft").eval()
ID2TOK = {v: k for k, v in P.tokenizer.get_vocab().items()}

W = lambda s: set(s.split())
THOUGHT = W("talk talks talking talked walk walked walking call calls called calling all always also almost small tall ball wall fall law laws saw caught taught bought brought thought fought coffee dog dogs off office officer often long wrong strong song boss loss lost cost costs cross across water daughter because auto august author awful draw along belong")
LOT = W("got not hot stop stopped job jobs lot lots top block problem problems shot doctor god john rock clock body probably possible bottom common honest policy economy concept contract")
TRAP_TENSE = W("bad man mad ask asked class half last laugh plan stand bag glad fast past pass after can't answer dance chance aunt math grass")
TRAP_LAX = W("back cat happen black that at hat map bat cash sat pack attack")
NURSE = W("work worked working first word words world person her were heard earn learn turn serve nurse jersey early")
R_FINAL = W("car cars far are here there where their they're four more door floor year years never better brother mother father other over under after water number power together sure poor hard part heart start party card yard sort short court board order force north forget forward morning corner york sister dollar dollars hour our your you're for or nor matter member leader leaders support important")
TH = W("the this that these those them they there their then than though with think thing things three through thank thanks nothing something anything everything")
ING_EXCL = W("thing king sing ring bring spring string wing sting swing during nothing something anything everything")

VOWELS = set("aeiouæɑɒɔəɛɜɪʊʌɐɚɝʉɨøœɤ")
BACK_ROUNDED = ("o", "ɔ", "ʊ", "u")  # raised THOUGHT -> NYC
LOW_UNROUNDED = ("ɑ", "a", "ɒ", "ʌ", "æ")
RHOTIC = set("ɹɚɝrɻ")


def ctc_tokens(y):
    with torch.no_grad():
        logits = M(P(y, sampling_rate=16000, return_tensors="pt").input_values).logits[0]
    ids = logits.argmax(-1).numpy()
    out, prev = [], None
    for i, t in enumerate(ids):
        if t != prev and t != P.tokenizer.pad_token_id:
            tok = ID2TOK.get(int(t), "")
            if tok and tok not in ("<s>", "</s>", "<unk>", "|", " "):
                out.append((i * 0.02, tok))
        prev = t
    return out


def formants_at(snd, t):
    f = snd.to_formant_burg(time_step=0.005, max_number_of_formants=5, maximum_formant=5000)
    vals = [[f.get_value_at_time(k, tt) for k in (1, 2, 3)] for tt in np.linspace(t - 0.02, t + 0.02, 5)]
    v = np.nanmedian(np.array(vals, dtype=float), axis=0)
    return v


def f3_min(snd, t0, t1):
    f = snd.to_formant_burg(time_step=0.005, max_number_of_formants=5, maximum_formant=5000)
    ts = np.linspace(t0 + 0.4 * (t1 - t0), t1, 8)
    v = [f.get_value_at_time(3, t) for t in ts]
    v = [x for x in v if x == x]
    return min(v) if v else np.nan


def judge(path):
    y, _ = librosa.load(path, sr=16000)
    res = mlx_whisper.transcribe(y, path_or_hf_repo="mlx-community/whisper-large-v3-turbo", language="en", word_timestamps=True)
    words = [(re.sub(r"[^a-z']", "", w["word"].lower().replace("’", "'")), w["start"], w["end"]) for s in res["segments"] for w in s["words"]]
    toks = ctc_tokens(y)
    snd = parselmouth.Sound(y, 16000)
    stats = collections.defaultdict(list)
    ex = collections.defaultdict(list)
    for i, (w, s, e) in enumerate(words):
        ph = [(t, p) for t, p in toks if s - 0.04 <= t <= e + 0.02]
        phs = "".join(p for _, p in ph)
        vow = [(t, p) for t, p in ph if p[0] in VOWELS]
        if not w or not ph:
            continue
        if w in THOUGHT and vow:
            v = vow[0][1]
            stats["thought_raised"].append(v.startswith(BACK_ROUNDED))
            stats["F_thought"].append(formants_at(snd, vow[0][0] + 0.02))
            ex["thought"].append(f"{w}:{phs}")
        if w in LOT and vow:
            stats["lot_rounded"].append(vow[0][1].startswith(BACK_ROUNDED))
            stats["F_lot"].append(formants_at(snd, vow[0][0] + 0.02))
        if w in TRAP_TENSE and vow:
            stats["F_trap_tense"].append(formants_at(snd, vow[0][0] + 0.02))
            ex["trap"].append(f"{w}:{phs}")
        if w in TRAP_LAX and vow:
            stats["F_trap_lax"].append(formants_at(snd, vow[0][0] + 0.02))
        nxt = words[i + 1][0] if i + 1 < len(words) else ""
        if w in R_FINAL and not (nxt[:1] in "aeiou" and w[-1] in "re"):
            stats["r_dropped"].append(not any(c in RHOTIC for c in phs))
            stats["F3min_r"].append(f3_min(snd, s, e))
            ex["r"].append(f"{w}:{phs}")
        if w in NURSE:
            stats["nurse_rless"].append(not any(c in RHOTIC for c in phs))
        if w in TH:
            first = next((p for _, p in ph if p not in "ˈˌ"), "")
            stats["th_stopped"].append(first[:1] in ("d", "t"))
            ex["th"].append(f"{w}:{phs}")
        if w.endswith(("ing", "in'", "in")) and len(w) > 4 and w not in ING_EXCL and (w.endswith("ing") or w.endswith("in'")):
            stats["g_dropped"].append(not phs.endswith(("ŋ", "ŋɡ", "ŋk")))
    # all-vowel F3 median for normalising the r measure
    f_all = snd.to_formant_burg(time_step=0.01, max_number_of_formants=5, maximum_formant=5000)
    f3s = [f_all.get_value_at_time(3, t) for t in np.arange(0, snd.duration, 0.01)]
    f3med = np.nanmedian(np.array(f3s, dtype=float))
    out = {"file": os.path.basename(path), "text": res["text"].strip()}
    for k, v in stats.items():
        if k.startswith("F_"):
            a = np.array(v, dtype=float)
            out[k] = [round(float(x), 0) for x in np.nanmedian(a, axis=0)] if len(a) else None
        elif k == "F3min_r":
            a = np.array(v, dtype=float)
            out["F3min_r_ratio"] = round(float(np.nanmedian(a) / f3med), 3) if len(a) else None
        else:
            out[k] = f"{np.mean(v):.2f} (n={len(v)})"
    ft, fl = out.get("F_thought"), out.get("F_lot")
    if ft and fl:
        out["LOT-THOUGHT_F1"] = fl[0] - ft[0]  # NYC: big positive (THOUGHT raised)
    tt, tl = out.get("F_trap_tense"), out.get("F_trap_lax")
    if tt and tl:
        out["TRAP_tense-lax_F1"] = tt[0] - tl[0]  # NYC: negative (tensed = raised)
    out["examples"] = {k: v[:8] for k, v in ex.items()}
    return out


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    jout = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else None
    files = []
    for a in args:
        if a == jout:
            continue
        files += sorted(glob.glob(a + "/*.wav")) if os.path.isdir(a) else [a]
    allr = []
    for f in files:
        r = judge(f)
        allr.append(r)
        ex = r.pop("examples")
        print(json.dumps({k: v for k, v in r.items() if k != "text"}, ensure_ascii=False))
        print("   ", r["text"][:160])
        for k, v in ex.items():
            print(f"    {k}: {' '.join(v)}")
    if jout:
        json.dump(allr, open(jout, "w"), indent=1, ensure_ascii=False)
