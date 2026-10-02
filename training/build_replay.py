"""Merge new encoded clips into a portable replay JSONL.

Audio paths are reduced to a basename because replay training only needs text,
speaker labels, durations, and codec tokens. Duplicate source utterances are
skipped so rerunning a corpus merge is idempotent.

usage: build_replay.py existing.jsonl new_codes.jsonl output.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def portable(row: dict) -> dict:
    source = row.get("source") or Path(row.get("audio", "unknown.wav")).name
    return {
        "text": row["text"],
        "speaker": row["speaker"],
        "dur": row["dur"],
        "source": source,
        "audio_codes": row["audio_codes"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("existing", type=Path)
    parser.add_argument("new", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    merged = []
    seen = set()
    for path in (args.existing, args.new):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = portable(json.loads(line))
            key = (row["speaker"], row["source"], row["text"])
            if key not in seen:
                seen.add(key)
                merged.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row) + "\n" for row in merged))
    print(f"merged {len(merged)} replay clips -> {args.output}")


if __name__ == "__main__":
    main()
