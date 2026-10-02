"""Download a provenance-checked YouTube source list as 24 kHz mono WAVs.

The input is JSONL. Required fields are ``url``, ``speaker``, and
``expected_channel``; descriptive fields such as ``work``, ``kind``, and
``notes`` are copied into the output provenance manifest.

Each source is inspected before download.  A source is rejected when its
reported channel does not exactly match ``expected_channel`` or when it is a
live stream.  This prevents search results and unofficial reuploads from
silently entering a training corpus.

usage: ingest_youtube.py sources.jsonl output_dir provenance.jsonl
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, check=True, text=True, capture_output=True)


def inspect(url: str) -> dict:
    result = run(
        "yt-dlp",
        "--no-playlist",
        "--no-warnings",
        "--dump-single-json",
        url,
    )
    return json.loads(result.stdout)


def download(url: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    common = [
        "yt-dlp",
        "--no-playlist",
        "--no-overwrites",
        "--continue",
        "--write-info-json",
        "-x",
        "--audio-format",
        "wav",
        "--postprocessor-args",
        "ffmpeg:-ar 24000 -ac 1",
        "-o",
        str(out_dir / "%(id)s.%(ext)s"),
    ]
    try:
        subprocess.run([*common, "-f", "bestaudio/best", url], check=True)
    except subprocess.CalledProcessError:
        # Some official uploads advertise Opus audio that intermittently
        # returns HTTP 403. Prefer the English-original M4A on retry; the
        # final fallback also covers uploads without language metadata.
        subprocess.run(
            [*common, "-f", "ba[ext=m4a][language^=en]/ba[ext=m4a]/ba", url],
            check=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sources", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("provenance", type=Path)
    parser.add_argument("--inspect-only", action="store_true")
    args = parser.parse_args()

    sources = [json.loads(line) for line in args.sources.read_text().splitlines() if line.strip()]
    seen: set[str] = set()
    records: list[dict] = []
    failures = 0

    for index, source in enumerate(sources, 1):
        try:
            info = inspect(source["url"])
            video_id = info["id"]
            channel = info.get("channel") or info.get("uploader")
            if video_id in seen:
                raise ValueError(f"duplicate video id: {video_id}")
            seen.add(video_id)
            if channel != source["expected_channel"]:
                raise ValueError(
                    f"channel mismatch: expected {source['expected_channel']!r}, got {channel!r}"
                )
            if info.get("is_live") or info.get("live_status") == "is_live":
                raise ValueError("live sources are not accepted")

            speaker_dir = args.output_dir / source["speaker"]
            if not args.inspect_only:
                download(source["url"], speaker_dir)
            wav = speaker_dir / f"{video_id}.wav"
            record = {
                **source,
                "id": video_id,
                "title": info.get("title"),
                "channel": channel,
                "channel_id": info.get("channel_id"),
                "duration": info.get("duration"),
                "upload_date": info.get("upload_date"),
                "webpage_url": info.get("webpage_url") or source["url"],
                "audio": str(wav.resolve()),
                "status": "inspected" if args.inspect_only else "downloaded",
            }
            records.append(record)
            print(
                f"[{index}/{len(sources)}] {video_id} {channel}: "
                f"{info.get('title')} ({info.get('duration')}s)",
                flush=True,
            )
        except (subprocess.CalledProcessError, KeyError, ValueError, json.JSONDecodeError) as exc:
            failures += 1
            records.append({**source, "status": "rejected", "error": str(exc)})
            print(f"[{index}/{len(sources)}] REJECTED {source.get('url')}: {exc}", file=sys.stderr)

        args.provenance.parent.mkdir(parents=True, exist_ok=True)
        args.provenance.write_text("".join(json.dumps(row) + "\n" for row in records))

    total = sum((row.get("duration") or 0) for row in records if row["status"] != "rejected")
    print(f"accepted {len(records) - failures}/{len(records)} sources, {total / 3600:.2f} raw hours")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
