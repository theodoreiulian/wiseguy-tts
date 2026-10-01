"""Eye-dialect respelling: push the heaviest wiseguy features the model's
learned accent doesn't produce on its own.

The fine-tuned model already has the Jersey vowels, rhythm and melody from
real speakers. What formal-register training speech lacks is the full street
register: "dese/dem/dose", "brudduh", "nuttin". A language-model TTS
reads spellings like these the way they're written, in the accent it knows.

Off by default: the fine-tuned model already has the accent, and a listener
heard respellings like "duh" come out as a different word rather than as a
Jersey "the". Kept as an opt-in for a more exaggerated read.

strength 0: text untouched (default)
         1: slang and contractions only (gonna, whaddaya)
         2: + TH-stopping (dis, dat, dose, tink, brudduh) and dropped g's
         3: + dey/den, "ya" for you, and spelled-out "cawfee, tawk, dawg"
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------- level 1
_DET = r"(?:the|a|an|my|your|his|her|our|their|this|that|these|those|work|school|church|jail|bed|town|court|jersey|new|brooklyn|home)"
SLANG = [
    # "forget about it" stays as written: every respelling tried ("Fuhgeddaboudit",
    # "fuhget about it") garbled or came out sounding like an expletive; the
    # trained accent already says it the Jersey way.
    (r"\bwhat are you\b", "whaddaya"),
    (r"\bwhat do you\b", "whaddaya"),
    (r"\bwhat did you\b", "whadja"),
    (r"\bdon't you\b", "doncha"),
    (r"\bdid you\b", "didja"),
    (r"\bgot you\b", "gotcha"),
    (r"\byou know\b", "ya know"),
    (r"\bgoing to\b(?! " + _DET + r"\b)", "gonna"),
    (r"\bwant to\b(?! " + _DET + r"\b)", "wanna"),
    (r"\bgot to\b(?! " + _DET + r"\b)", "gotta"),
    (r"\bkind of\b", "kinda"),
    (r"\bsort of\b", "sorta"),
    (r"\bout of\b", "outta"),
    (r"\ba lot of\b", "a lotta"),
    (r"\bdon't know\b", "dunno"),
    (r"\blet me\b", "lemme"),
    (r"\bgive me\b", "gimme"),
    (r"\bbecause\b", "'cause"),
    (r"\byou guys\b", "youse guys"),
    (r"\bcapisce\b|\bcapiche\b|\bcapish\b", "capeesh"),
]

# ---------------------------------------------------------------- level 2
# Every spelling here was A/B-tested on the fine-tuned model: it has to carry
# the accent (checked with a phoneme recognizer) and still be heard as the
# right word (checked with Whisper). Spellings that garbled ("dere", "brotha",
# "Fuhgeddaboudit" as one word) or changed nothing ("betta" stays r-coloured)
# were dropped.
TH_WORDS = {
    "the": "duh", "this": "dis", "that": "dat", "that's": "dat's",
    "those": "dose", "them": "dem", "with": "wit", "without": "witout",
    "think": "tink", "thinks": "tinks", "thinking": "tinkin'", "thing": "ting",
    "things": "tings", "nothing": "nuttin", "something": "somethin", "anything": "anythin",
    "everything": "everythin", "brother": "brudduh", "brothers": "brudduhs",
    "mother": "mudduh", "father": "fadduh", "other": "udduh", "three": "tree",
}
NO_ING = {"thing", "king", "sing", "ring", "bring", "spring", "string", "wing", "sting", "swing", "during"}

# ---------------------------------------------------------------- level 3
OLD_SCHOOL = {
    "these": "dese", "forget": "fahget", "they": "dey", "they're": "dey're", "then": "den", "their": "deir",
    "you": "ya", "your": "ya", "you're": "ya", "coffee": "cawfee", "talk": "tawk",
    "talking": "tawkin'", "walk": "wawk", "dog": "dawg",
}


def _case(src: str, repl: str) -> str:
    if src.isupper() and len(src) > 1:
        return repl.upper()
    if src[:1].isupper():
        return repl[:1].upper() + repl[1:]
    return repl


def _word_sub(text: str, table: dict[str, str]) -> str:
    return re.sub(
        r"\b[A-Za-z']+\b",
        lambda m: _case(m.group(0), table.get(m.group(0).lower(), m.group(0))),
        text,
    )


def _drop_g(m: re.Match) -> str:
    w = m.group(0)
    if w.lower() in NO_ING or len(w) <= 5:
        return w
    return w[:-1] + "'"


def respell(text: str, strength: int = 2) -> str:
    if strength <= 0:
        return text
    text = text.replace("’", "'")
    for pattern, repl in SLANG:
        text = re.sub(pattern, lambda m, r=repl: _case(m.group(0), r), text, flags=re.IGNORECASE)
    if strength >= 2:
        table = dict(TH_WORDS)
        if strength >= 3:
            table.update(OLD_SCHOOL)
        text = _word_sub(text, table)
        text = re.sub(r"\b[A-Za-z]+ing\b", _drop_g, text)
    return text
