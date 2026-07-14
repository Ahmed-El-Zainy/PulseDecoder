#!/usr/bin/env python3
"""Cross-paper comparative analysis over benchmark/data/*.json.

Complements build.py's merge step: where build.py just assembles the raw
records for the interactive dashboard, this script derives comparative
statistics (architecture vs. performance/generalization, technique efficacy,
internal rank/score consistency) and writes them to analysis_report.md.

Two fields used here are NOT structured in the source JSON and are
recovered heuristically:
  - the official validation-leaderboard score (needed for the
    generalization gap), which only exists as free prose inside
    result.evaluation_split / result.self_reported_metrics.other
  - technique usage (augmentation/ensembling/pretraining/threshold tuning/
    domain generalization), via keyword matching over the relevant text
    fields.
Every heuristic value is stored alongside the matched snippet and a
confidence label so a reader can audit it; see the "Data quality notes"
section of the generated report.

Re-run after adding/editing files in benchmark/data/.
"""
import glob
import json
import os
import re

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
REPORT_PATH = os.path.join(HERE, "analysis_report.md")

# Mirrors benchmark/build.py's entry-type split so cross-paper comparisons
# exclude the challenge overview paper(s) and the out-of-scope (non-ECG)
# entry rather than treating them as ordinary team submissions.
OVERVIEW_IDS = {"2020ChallengePaper", "CinC2020-236"}
OUT_OF_SCOPE_IDS = {"CinC2020-340"}

HEDGE_WORDS = [
    "not clearly distinguished",
    "apparently",
    "ambiguous",
    "not distinguished",
    "may be",
    "seemingly",
    "default/error score",
    "not a true measure",
]

# Ordered from most to least specific; first match in the text wins.
VALIDATION_PATTERNS = [
    r"(?:official\s+)?(?:challenge\s+)?validation(?:[\s-]*(?:leaderboard|phase|set|split)?)?\s*(?:score)?\s*(?:was|of|is|:)\s*(-?\d+\.\d+)",
    r"(-?\d+\.\d+)\s*(?:challenge\s+)?on\s+(?:the|their(?:\s+own)?)\s+(?:official\s+)?validation\s*(?:set|split|leaderboard|data(?:set)?)?",
    r"validation[^.;]{0,25}?\((-?\d+\.\d+)\)",
]
CV_PATTERNS = [
    r"(?:\d+[\s-]fold\s+)?cross[\s-]?validation[^.;]{0,40}?(-?\d+\.\d+)",
    r"(-?\d+\.\d+)[^.;]{0,15}?(?:\d+[\s-]fold\s+)?cross[\s-]?validation",
]

TECHNIQUE_KEYWORDS = {
    "augmentation": ["augment"],
    "ensemble": ["ensemble", "bagging", "boosting", "majority vote", "majority-vote", "voting scheme"],
    "external_pretraining": ["pretrain", "pre-train", "pre train", "external data", "transfer learning"],
    "threshold_tuning": ["threshold"],
    "domain_generalization": ["domain generalization", "domain adaptation", "adversarial", "gradient reversal"],
}


def load_records():
    records = []
    for f in sorted(glob.glob(os.path.join(DATA_DIR, "*.json"))):
        if f.endswith("_leaderboard_reference.json"):
            continue
        records.append(json.load(open(f)))
    return records


def classify_bucket(paper_id):
    if paper_id in OVERVIEW_IDS:
        return "challenge_overview"
    if paper_id in OUT_OF_SCOPE_IDS:
        return "out_of_scope"
    return "team_entry"


def first_match(patterns, text):
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            start = max(0, m.start() - 40)
            end = min(len(text), m.end() + 20)
            return float(m.group(1)), text[start:end].strip()
    return None, None


def extract_validation_score(record):
    """Best-effort recovery of the official validation-leaderboard score.

    Returns (value, confidence, snippet). confidence is one of
    "high" / "low" / "none" -- "low" means either a hedge word was found
    near the match, or the value had to fall back to a k-fold CV score
    (which is not the same thing as the official validation split).
    """
    text = " ".join(
        [
            record["result"].get("evaluation_split", "") or "",
            record["result"]["self_reported_metrics"].get("other", "") or "",
        ]
    )
    if not text.strip():
        return None, "none", None

    value, snippet = first_match(VALIDATION_PATTERNS, text)
    if value is not None:
        hedged = any(h in text.lower() for h in HEDGE_WORDS)
        return value, ("low" if hedged else "high"), snippet

    value, snippet = first_match(CV_PATTERNS, text)
    if value is not None:
        return value, "low", snippet

    return None, "none", None


def tag_techniques(record):
    haystacks = " ".join(
        [
            record["data"].get("augmentation_techniques", "") or "",
            record["data"].get("class_imbalance_handling", "") or "",
            record["training"].get("cross_validation_strategy", "") or "",
            record["model"].get("architecture_details", "") or "",
            record["model"].get("key_novelty", "") or "",
            record["contribution"].get("summary", "") or "",
        ]
    ).lower()

    tags = {}
    for tag, keywords in TECHNIQUE_KEYWORDS.items():
        matched = next((kw for kw in keywords if kw in haystacks), None)
        tags[f"uses_{tag}"] = matched is not None
        tags[f"uses_{tag}_snippet"] = matched
    return tags


def build_dataframe(records):
    rows = []
    for r in records:
        bucket = classify_bucket(r["paper_id"])
        score = r["result"].get("official_challenge_metric_score")
        rank = r["result"].get("official_rank")
        val_score, val_confidence, val_snippet = extract_validation_score(r)

        row = {
            "paper_id": r["paper_id"],
            "bucket": bucket,
            "architecture_family": r["model"].get("architecture_family"),
            "official_challenge_metric_score": score if isinstance(score, (int, float)) else None,
            "has_hidden_test": isinstance(score, (int, float)),
            "official_rank": rank if isinstance(rank, (int, float)) else None,
            "validation_score": val_score,
            "validation_confidence": val_confidence,
            "validation_snippet": val_snippet,
        }
        row.update(tag_techniques(r))

        if row["official_challenge_metric_score"] is not None and val_score is not None:
            row["generalization_gap"] = val_score - row["official_challenge_metric_score"]
        else:
            row["generalization_gap"] = None

        rows.append(row)
    return pd.DataFrame(rows)


def analysis_architecture(df):
    team = df[df["bucket"] == "team_entry"]
    grouped = (
        team.groupby("architecture_family")
        .agg(
            n=("paper_id", "count"),
            n_with_hidden_test=("has_hidden_test", "sum"),
            mean_hidden_test_score=("official_challenge_metric_score", "mean"),
            median_hidden_test_score=("official_challenge_metric_score", "median"),
            mean_generalization_gap=("generalization_gap", "mean"),
            n_with_gap=("generalization_gap", lambda s: s.notna().sum()),
        )
        .sort_values("mean_hidden_test_score", ascending=False)
    )
    return grouped


def analysis_techniques(df):
    team = df[df["bucket"] == "team_entry"]
    rows = []
    for tag in TECHNIQUE_KEYWORDS:
        col = f"uses_{tag}"
        for used in (True, False):
            subset = team[team[col] == used]
            rows.append(
                {
                    "technique": tag,
                    "used": used,
                    "n": len(subset),
                    "mean_hidden_test_score": subset["official_challenge_metric_score"].mean(),
                    "mean_generalization_gap": subset["generalization_gap"].mean(),
                }
            )
    return pd.DataFrame(rows)


def analysis_consistency(df):
    team = df[df["bucket"] == "team_entry"]
    notes = []

    ranked = team[team["official_rank"].notna()]
    dupes = ranked[ranked.duplicated(subset="official_rank", keep=False)]
    if not dupes.empty:
        for rank, grp in dupes.groupby("official_rank"):
            notes.append(
                f"Duplicate official_rank {int(rank)}: "
                + ", ".join(grp["paper_id"])
            )
    else:
        notes.append("No duplicate official_rank values among team entries.")

    both = ranked[ranked["official_challenge_metric_score"].notna()].sort_values("official_rank")
    prev_score = None
    disagreements = []
    for _, row in both.iterrows():
        if prev_score is not None and row["official_challenge_metric_score"] > prev_score:
            disagreements.append(
                f"{row['paper_id']} (rank {int(row['official_rank'])}, "
                f"score {row['official_challenge_metric_score']}) scores higher than the "
                f"previous (better) rank's score {prev_score}"
            )
        prev_score = row["official_challenge_metric_score"]
    if disagreements:
        notes.append("Rank/score order disagreements: " + "; ".join(disagreements))
    else:
        notes.append("Rank order agrees with score order for all ranked+scored team entries.")

    hedged = df[df["validation_confidence"] == "low"]
    notes.append(
        f"{len(hedged)} paper(s) have a low-confidence extracted validation score "
        "(hedge language in source text, or fell back to a k-fold CV score instead "
        "of the official validation-leaderboard score): "
        + ", ".join(hedged["paper_id"])
    )

    notes.append(
        "benchmark/data/_leaderboard_reference.json is empty, so self-reported "
        "rank/score cannot be cross-checked against an independent official "
        "leaderboard -- only the internal consistency checks above are possible."
    )

    return notes


def to_markdown_table(df, float_cols=()):
    df = df.copy()
    for c in float_cols:
        if c in df.columns:
            df[c] = df[c].map(lambda v: f"{v:.3f}" if pd.notna(v) else "n/a")
    df = df.astype(str)
    headers = list(df.columns)
    rows = df.values.tolist()
    widths = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h) for i, h in enumerate(headers)]
    header_line = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)) + " |"
    sep_line = "| " + " | ".join("-" * w for w in widths) + " |"
    body_lines = [
        "| " + " | ".join(cell.ljust(w) for cell, w in zip(row, widths)) + " |"
        for row in rows
    ]
    return "\n".join([header_line, sep_line, *body_lines])


def write_report(df, arch_table, tech_table, consistency_notes):
    team = df[df["bucket"] == "team_entry"]
    lines = []
    lines.append("# PhysioNet/CinC 2020 benchmark -- comparative analysis\n")
    lines.append(
        f"Generated from {len(team)} team-entry papers "
        f"({team['has_hidden_test'].sum()} with a numeric official hidden-test "
        f"score, {team['official_rank'].notna().sum()} with a numeric official rank). "
        f"{len(df) - len(team)} entries (challenge overview / out-of-scope) excluded.\n"
    )

    lines.append("## 1. Architecture family vs. performance & generalization gap\n")
    lines.append(to_markdown_table(arch_table.reset_index(), float_cols=[
        "mean_hidden_test_score", "median_hidden_test_score", "mean_generalization_gap",
    ]))
    lines.append(
        "\n`mean_generalization_gap` is validation-leaderboard score minus official "
        "hidden-test score, computed only for papers where both a hidden-test score "
        "and a heuristically-extracted validation score were available "
        "(`n_with_gap`); take low-`n_with_gap` rows with a grain of salt.\n"
    )

    lines.append("## 2. Technique usage vs. performance & generalization gap\n")
    lines.append(to_markdown_table(tech_table, float_cols=[
        "mean_hidden_test_score", "mean_generalization_gap",
    ]))
    lines.append(
        "\nSample sizes per group are small (n≈5-25) and many papers combine several "
        "techniques at once, so these are directional comparisons, not controlled "
        "experiments or significance tests.\n"
    )

    lines.append("## 3. Internal rank/score consistency\n")
    for note in consistency_notes:
        lines.append(f"- {note}")
    lines.append("")

    lines.append("## Data quality notes\n")
    lines.append(
        "- Only papers with a numeric `result.official_challenge_metric_score` are "
        "counted as having a hidden-test score; papers that failed to submit/score "
        "are excluded from score-based comparisons but still counted in `n`.\n"
        "- The validation-leaderboard score used for `generalization_gap` is not a "
        "structured field in the source JSON -- it is recovered via regex over "
        "`result.evaluation_split` / `result.self_reported_metrics.other` prose, "
        "which varies in phrasing per paper. Each row's `validation_confidence` "
        "(`high`/`low`/`none`) and matched snippet are available in the underlying "
        "dataframe (`analyze.py`'s `build_dataframe`) for audit; only `high`- and "
        "`low`-confidence values are used in the aggregate tables above.\n"
        "- Technique tags (augmentation/ensemble/pretraining/threshold-tuning/"
        "domain-generalization) are keyword matches over free text, not a "
        "structured field -- false negatives are likely where a paper uses a "
        "technique but describes it without the matched keywords.\n"
        "- `_leaderboard_reference.json` is empty, so rank/score cross-checking is "
        "internal-only (see section 3), not verified against an independent source.\n"
    )

    lines.append("## Implications for PulseDecoder\n")
    top_families = arch_table.head(3)
    lines.append(
        "Based on the data available (small-n, partial coverage -- treat as "
        "directional, not conclusive):\n"
    )
    for fam, row in top_families.iterrows():
        lines.append(
            f"- **{fam}** entries (n={int(row['n'])}) averaged a hidden-test score "
            f"of {row['mean_hidden_test_score']:.3f}" +
            (f" with a mean generalization gap of {row['mean_generalization_gap']:.3f} "
             f"(n={int(row['n_with_gap'])})" if pd.notna(row['mean_generalization_gap']) else
             " (generalization gap not computable for this group)") +
            "."
        )
    lines.append(
        "\nSee sections 2 and 3 above before treating any single technique or "
        "architecture family as a settled recommendation -- coverage gaps in the "
        "source data limit how far these numbers can be trusted."
    )

    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    records = load_records()
    df = build_dataframe(records)

    arch_table = analysis_architecture(df)
    tech_table = analysis_techniques(df)
    consistency_notes = analysis_consistency(df)

    print(f"Loaded {len(df)} papers ({(df['bucket'] == 'team_entry').sum()} team entries).\n")
    print("=== Architecture vs. performance/gap ===")
    print(arch_table)
    print("\n=== Technique efficacy ===")
    print(tech_table.to_string(index=False))
    print("\n=== Internal consistency ===")
    for note in consistency_notes:
        print(f"- {note}")

    write_report(df, arch_table, tech_table, consistency_notes)
    print(f"\nWrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
