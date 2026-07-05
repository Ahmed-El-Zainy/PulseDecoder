#!/usr/bin/env python3
"""Merges benchmark/data/*.json into all_papers.json and rebuilds benchmark.html.

Run this after adding/editing any file in benchmark/data/.
Font files must exist under benchmark/fonts/*.b64 (base64-encoded woff2).
"""
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OVERVIEW_IDS = {"2020ChallengePaper", "CinC2020-236"}
OUT_OF_SCOPE_IDS = {"CinC2020-340"}


def merge():
    records = []
    for f in sorted(glob.glob(os.path.join(HERE, "data", "*.json"))):
        if f.endswith("_leaderboard_reference.json"):
            continue
        r = json.load(open(f))
        pid = r["paper_id"]
        if pid in OVERVIEW_IDS:
            r["entry_type"] = "challenge_overview"
        elif pid in OUT_OF_SCOPE_IDS:
            r["entry_type"] = "out_of_scope"
        else:
            r["entry_type"] = "team_entry"
        records.append(r)

    def sort_key(r):
        order = {"team_entry": 0, "challenge_overview": 1, "out_of_scope": 2}[r["entry_type"]]
        rank = r["result"].get("official_rank")
        rank_val = rank if isinstance(rank, (int, float)) else 999
        return (order, rank_val, r["paper_id"])

    records.sort(key=sort_key)
    out_path = os.path.join(HERE, "all_papers.json")
    with open(out_path, "w") as out:
        json.dump(records, out, indent=2)
    return records


def build_html():
    fonts_dir = os.path.join(HERE, "fonts")

    def read_b64(name):
        with open(os.path.join(fonts_dir, f"{name}.b64")) as f:
            return f.read().strip()

    template = open(os.path.join(HERE, "benchmark_template.html")).read()
    replacements = {
        "__FONT_NEWSREADER_ITALIC__": read_b64("newsreader-italic-400"),
        "__FONT_NEWSREADER_REGULAR__": read_b64("newsreader-regular-500"),
        "__FONT_PLEX_SANS_400__": read_b64("plex-sans-400"),
        "__FONT_PLEX_SANS_600__": read_b64("plex-sans-600"),
        "__FONT_PLEX_MONO_400__": read_b64("plex-mono-400"),
        "__FONT_PLEX_MONO_500__": read_b64("plex-mono-500"),
        "__PAPER_DATA_JSON__": open(os.path.join(HERE, "all_papers.json")).read(),
    }
    for k, v in replacements.items():
        template = template.replace(k, v)

    out_path = os.path.join(HERE, "benchmark.html")
    with open(out_path, "w") as f:
        f.write(template)
    print(f"Wrote {out_path} ({len(template)} bytes)")


if __name__ == "__main__":
    records = merge()
    print(f"Merged {len(records)} records into all_papers.json")
    build_html()
