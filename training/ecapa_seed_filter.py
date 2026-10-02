"""Filter mixed dialogue sources using hand-audited target-speaker seeds.

``ecapa_filter.py`` works well when one speaker dominates every input folder.
Film scenes violate that assumption, and a long interview can also pull an
actor centroid away from a much younger character voice.  This companion
filter builds separate centroids for roles/eras from a few unmistakable lines.

The JSON config contains a ``groups`` list.  Each group specifies the manifest
speaker label, source video IDs, case-insensitive text fragments identifying
seed utterances, and an optional cosine threshold (default 0.35).  An optional
``max_hours`` cap keeps the highest-similarity clips when a clean but weaker
accent source would otherwise dominate the corpus.  Embeddings are loaded from
the tensor written by ``ecapa_filter.py`` so no audio is encoded twice.

usage: ecapa_seed_filter.py manifest.jsonl manifest.ecapa.pt seeds.json out.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def matches_source(row: dict, source_ids: list[str]) -> bool:
    name = Path(row["audio"]).name
    return any(name.startswith(source_id + "_") for source_id in source_ids)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("embeddings", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    embeddings = torch.nn.functional.normalize(torch.load(args.embeddings), dim=-1)
    if len(rows) != len(embeddings):
        raise ValueError(f"row/embedding mismatch: {len(rows)} != {len(embeddings)}")
    groups = json.loads(args.config.read_text())["groups"]

    kept: dict[str, dict] = {}
    for group in groups:
        candidate_indices = [
            i
            for i, row in enumerate(rows)
            if row["speaker"] == group["speaker"] and matches_source(row, group["sources"])
        ]
        patterns = [pattern.casefold() for pattern in group["seeds"]]
        seed_indices = [
            i
            for i in candidate_indices
            if any(pattern in rows[i]["text"].casefold() for pattern in patterns)
        ]
        if not seed_indices:
            raise ValueError(f"{group['name']}: none of the configured seed lines matched")

        seed_centroid = torch.nn.functional.normalize(embeddings[seed_indices].mean(0), dim=0)
        similarities = embeddings[candidate_indices] @ seed_centroid
        threshold = float(group.get("threshold", 0.35))
        selected_with_scores = [
            (float(score), i)
            for i, score in zip(candidate_indices, similarities)
            if score >= threshold
        ]
        selected_with_scores.sort(reverse=True)

        max_hours = group.get("max_hours")
        if max_hours is not None:
            capped = []
            seconds = 0.0
            limit = float(max_hours) * 3600
            for score, i in selected_with_scores:
                if seconds + rows[i]["dur"] > limit and capped:
                    continue
                capped.append((score, i))
                seconds += rows[i]["dur"]
            selected_with_scores = capped

        selected = [i for _, i in selected_with_scores]
        for i in selected:
            kept[rows[i]["audio"]] = rows[i]

        ranked = sorted(zip(similarities.tolist(), candidate_indices), reverse=True)
        print(
            f"{group['name']}: seeds {len(seed_indices)}, kept {len(selected)}/{len(candidate_indices)} "
            f"({sum(rows[i]['dur'] for i in selected) / 3600:.2f} h), threshold {threshold:.2f}"
            + (f", cap {float(max_hours):.2f} h" if max_hours is not None else "")
        )
        for score, i in ranked[:3]:
            print(f"  top {score:.2f} {rows[i]['text'][:110]}")
        for i in seed_indices:
            score = float(embeddings[i] @ seed_centroid)
            print(f"  seed {score:.2f} {rows[i]['text'][:110]}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row) + "\n" for row in kept.values()))
    print(f"kept {len(kept)} unique clips ({sum(row['dur'] for row in kept.values()) / 3600:.2f} h)")


if __name__ == "__main__":
    main()
