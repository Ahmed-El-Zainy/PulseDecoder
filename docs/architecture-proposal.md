# PulseDecoder — Architecture & Pretraining Proposal (v1)

**Status:** Draft, for senior review
**Basis:** Structured extraction + comparative analysis of all 42 papers from
the PhysioNet/Computing in Cardiology Challenge 2020 ("Classification of
12-lead ECGs"). See `benchmark/data/*.json` (raw extractions, one per paper,
per `benchmark/rubric.md`), `benchmark/analyze.py` (analysis script), and
`benchmark/analysis_report.md` (generated tables this proposal draws on).

## 1. Goal

Decide PulseDecoder's initial model architecture and pretraining strategy
using evidence from how 42 independent teams approached the same 12-lead ECG
multi-label classification task, rather than starting from a blank-page
architecture choice.

## 2. Recommended backbone: 1D (SE-)ResNet

The single best-scoring paper across all 42 is **CinC2020-281**
(SE-ResNet + rule-based correction, official hidden-test score 0.514,
rank 3/41). ResNet was also the strongest-performing architecture family on
average (mean hidden-test 0.252, vs. 0.094 for plain CNN and 0.118 for
Ensemble — see `analysis_report.md` §1). Recommendation: build PulseDecoder's
core model on a **1D ResNet backbone with squeeze-and-excitation blocks**,
not a plain CNN or a from-scratch ensemble-of-heterogeneous-models approach.

## 3. Techniques to include (each grounded in a specific top-performing paper)

| # | Technique | Evidence | Why it matters here |
|---|-----------|----------|----------------------|
| 1 | **Domain-generalization training** (adversarial/gradient-reversal-style, or heavy multi-source augmentation) | CinC2020-445 — smallest high-confidence generalization gap in the corpus (validation → hidden-test drop of only 0.172) | The corpus's single most common failure mode is validation-to-hidden-test collapse (see §4). This is the technique that most directly targets it. |
| 2 | **Hybrid rule-based correction layer on top of the DL output** | CinC2020-281 (the #1 paper) combines SE-ResNet with a rule-based bradycardia correction step and a dual 10s/30s segment ensemble | The top individual result in the whole corpus isn't pure deep learning — it's DL + a thin domain-knowledge correction layer. |
| 3 | **Class-specific threshold tuning** | Technique-efficacy comparison: threshold-tuned papers averaged 0.199 vs. 0.098 for untuned (`analysis_report.md` §2). CinC2020-328's own ablation attributes +0.055 to threshold tuning alone | Cheap, well-evidenced, should be a default, not an afterthought. |
| 4 | **Augmentation matched to the actual failure mode** | Temporal stretch / voltage scaling (CinC2020-189); lead dropout / shuffle / inversion (CinC2020-445) | Augmentation-users averaged a higher score (0.199 vs. 0.144) in the corpus — but the *kind* of augmentation should target cross-source shift, not generic image-style transforms. |
| 5 | **Metric-aligned loss — with a generalization safeguard** | CinC2020-189 built a custom loss directly from the challenge's weighted metric and *topped validation* (0.696) — then suffered the **worst** validation→test collapse in the corpus (down to 0.202) | Cautionary tale, not a rejected idea: metric-aligned loss is good, but only paired with #1 above. Loss alignment alone overfits to validation-set idiosyncrasies. |

## 4. Pretraining strategy

Two papers in the corpus used external pretraining data — treat this as a
natural experiment rather than a "pretraining helps" assumption:

- **CinC2020-374**: pretrained on 1.27M records (MUSE, University of
  Michigan) — the largest external dataset in the corpus — then fully
  retrained on the challenge data. Result: rank **27/41** (0.141), likely
  because diagnosis labels were auto-mapped from free text via UMLS/SNOMED
  heuristics (label noise).
- **CinC2020-253**: pretrained on a much smaller but **physician-annotated,
  same-modality** dataset (254k ECGs, UMCU). Result: rank **7/41** (0.417);
  the paper's own ablation credits pretraining with only ~0.02 metric points
  of the total.

**Conclusion:** in this corpus, label quality and modality match dominate raw
pretraining scale. Neither of these two institutional datasets is public, so
neither checkpoint is reusable directly.

**Action for PulseDecoder:** none of the 42 corpus papers (all 2020-era,
task-specific) ship an open pretrained checkpoint. If we want a genuine
reusable pretrained starting point, that means looking outside this corpus at
newer open ECG models trained on public data (e.g. PTB-XL, CODE-15,
MIMIC-IV-ECG). This needs a short, separate spike to identify and verify a
specific candidate (license, actual public availability, input format match)
before committing — not a recommendation we can respond with a name only.

## 5. Known limitations of this evidence base (be upfront with reviewers)

- Only 28/39 team-entry papers report a numeric official hidden-test score;
  the rest failed to submit/score and are excluded from score comparisons.
- The "generalization gap" figures above are heuristically extracted from
  free-text prose (not a structured field in the source papers) — see
  `analysis_report.md`'s Data quality notes for per-paper confidence levels.
- Self-reported ranks/scores are internally consistency-checked only; there
  is no independent leaderboard file to verify them against
  (`benchmark/data/_leaderboard_reference.json` is empty).
- n per architecture family / technique group is small (roughly 5–25) —
  treat comparisons as directional, not statistically significant.

## 6. Proposed build phases

**Phase 1 — Baseline.**
Reproduce a working 1D SE-ResNet trained on the official PhysioNet/CinC 2020
training data (publicly available via PhysioNet), targeting the challenge's
official weighted-accuracy metric. Goal: get in the neighborhood of the
corpus's top result (~0.5 on hidden-test-style holdout) before adding any
novel technique, so later phases have a real baseline to measure against.

**Phase 2 — Generalization hardening.**
Add domain-generalization training (adversarial or multi-source augmentation
per §3.1) and the source-matched augmentation set (§3.4). Evaluate
specifically on a held-out data *source*, not just a random split — the
corpus's core lesson is that random-split validation hides the real failure
mode.

**Phase 3 — Domain-knowledge correction + tuning.**
Add the rule-based correction layer (§3.2) and class-specific threshold
tuning (§3.3). These are cheap and well-evidenced; low risk, should land
before more speculative work.

**Phase 4 — Pretraining spike.**
Time-boxed investigation into a public pretrained ECG checkpoint (§4). Decide
go/no-go based on: license compatibility, input format match (leads, sampling
rate), and whether fine-tuning it beats the Phase 3 model on our own
held-out-source evaluation. Do not adopt a pretrained model on scale alone.

## 7. Open questions for senior review

1. Do we have (or need to acquire) a held-out-source validation split from
   day one, given that's the corpus's clearest predictor of hidden-test
   performance?
2. Compute budget — several top corpus papers used single V100-class GPUs
   with training measured in hours; is that the right budget assumption?
3. Is a pretraining spike (Phase 4) worth scheduling now, or deferred until
   Phases 1–3 establish our own baseline?

## References (papers cited above)

| Paper ID | Score / Rank | Role in this proposal |
|---|---|---|
| CinC2020-281 | 0.514 / 3rd | Best overall result; backbone + hybrid-correction template |
| CinC2020-445 | 0.437 / 5th | Smallest generalization gap; domain-generalization template |
| CinC2020-189 | 0.202 / 21st (validation 0.696) | Cautionary tale: metric-aligned loss without generalization safeguard |
| CinC2020-328 | 0.420 / 6th | Threshold-tuning ablation evidence |
| CinC2020-253 | 0.417 / 7th | Pretraining on smaller, high-quality, same-modality data |
| CinC2020-374 | 0.141 / 27th | Pretraining on larger but noisily-labeled data — did not help |

Full extraction detail for any of these is in `benchmark/data/<paper_id>.json`;
the source PDFs are in `papers/<paper_id>.pdf`.
