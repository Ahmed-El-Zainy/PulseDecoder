# Extraction Rubric — PhysioNet/CinC2020 Paper Benchmark

You are extracting structured data from one or more papers submitted to the
**PhysioNet/Computing in Cardiology Challenge 2020** ("Classification of
12-lead ECGs"). For each assigned PDF, produce exactly one JSON file at
`benchmark/data/<paper_id>.json` (where `<paper_id>` is the PDF filename
without the `.pdf` extension, e.g. `CinC2020-134`).

## Ground rules

- **Never fabricate.** If a field is not explicitly stated in the paper and
  cannot be directly read off a table/figure, set its value to the literal
  string `"not reported"`. Do not guess or infer beyond what the text
  supports.
- Numbers should be extracted as reported (keep original units/precision).
  If a metric is reported for multiple splits (e.g. cross-validation vs.
  official hidden test set), capture the official hidden test set number in
  `result.official_challenge_metric_score` and put any others under
  `result.self_reported_metrics` or `result.evaluation_split` notes.
- The official Challenge metric is a custom weighted accuracy metric defined
  in the challenge (NOT plain classification accuracy) — see
  `2020ChallengePaper.pdf` for its exact definition. Most team papers report
  their score for this metric directly, often alongside their leaderboard
  rank out of 41 teams.
- `architecture_family` must be one of: `"CNN"`, `"ResNet"`, `"Transformer"`,
  `"RNN/LSTM"`, `"Hybrid"`, `"Ensemble"`, `"Other"` — pick the closest fit
  and use `architecture_details` to add nuance.
- Output must be **valid JSON** (no trailing commas, no comments). Write it
  with the Write tool to the exact path given.
- After writing all your assigned files, report back briefly: which files
  you wrote, and flag any PDF that was unreadable, scanned-image-only, or
  otherwise degraded.

## JSON Schema

```json
{
  "paper_id": "string — PDF filename without extension",
  "team_name": "string — team/entry name as given in the paper, or \"not reported\"",
  "authors_affiliation": "string — short author/affiliation summary",

  "result": {
    "official_challenge_metric_score": "number or \"not reported\"",
    "official_rank": "number (out of 41) or \"not reported\"",
    "evaluation_split": "string — e.g. \"official hidden test set\", \"5-fold CV on training data\"",
    "self_reported_metrics": {
      "f1": "number or \"not reported\"",
      "auroc": "number or \"not reported\"",
      "sensitivity": "number or \"not reported\"",
      "specificity": "number or \"not reported\"",
      "accuracy": "number or \"not reported\"",
      "other": "string — any other notable self-reported metric, or \"not reported\""
    }
  },

  "model": {
    "architecture_family": "CNN | ResNet | Transformer | RNN/LSTM | Hybrid | Ensemble | Other",
    "architecture_details": "string — brief technical description",
    "key_novelty": "string — 1-2 sentences on the standout architectural idea, or \"not reported\""
  },

  "input": {
    "leads_used": "string — e.g. \"12\" or \"reduced subset: I, II, V2\"",
    "signal_length_window": "string — e.g. \"10s fixed window\"",
    "sampling_rate": "string — e.g. \"500 Hz\"",
    "preprocessing": "string — filtering, resampling, normalization steps"
  },

  "data": {
    "training_datasets": "string — which official challenge datasets used, plus any external data (e.g. PTB-XL)",
    "class_imbalance_handling": "string — or \"not reported\"",
    "augmentation_techniques": "string — or \"not reported\""
  },

  "training": {
    "loss_function": "string — or \"not reported\"",
    "optimizer": "string — or \"not reported\"",
    "cross_validation_strategy": "string — or \"not reported\"",
    "hardware": "string — or \"not reported\""
  },

  "contribution": {
    "summary": "string — 1-2 sentence plain-language summary of the paper's one novel idea",
    "code_available": "true | false | \"not reported\"",
    "code_url": "string — URL if given, else \"not reported\""
  }
}
```

## Example (illustrative only — do not treat as real data)

```json
{
  "paper_id": "EXAMPLE-DO-NOT-USE",
  "team_name": "Example Team",
  "authors_affiliation": "J. Doe et al., Example University",
  "result": {
    "official_challenge_metric_score": 0.641,
    "official_rank": 5,
    "evaluation_split": "official hidden test set",
    "self_reported_metrics": {
      "f1": 0.72,
      "auroc": "not reported",
      "sensitivity": "not reported",
      "specificity": "not reported",
      "accuracy": "not reported",
      "other": "not reported"
    }
  },
  "model": {
    "architecture_family": "ResNet",
    "architecture_details": "34-layer 1D ResNet with squeeze-excitation blocks",
    "key_novelty": "Adds a squeeze-excitation attention block after each residual stage to reweight lead-wise features."
  },
  "input": {
    "leads_used": "12",
    "signal_length_window": "10s fixed window, zero-padded",
    "sampling_rate": "500 Hz",
    "preprocessing": "Bandpass filter 0.5-40Hz, per-lead z-score normalization"
  },
  "data": {
    "training_datasets": "All 6 official challenge training sets, no external data",
    "class_imbalance_handling": "Weighted BCE loss by inverse class frequency",
    "augmentation_techniques": "Random crop, amplitude scaling, lead dropout"
  },
  "training": {
    "loss_function": "Weighted binary cross-entropy",
    "optimizer": "Adam, lr=1e-3 with cosine decay",
    "cross_validation_strategy": "5-fold stratified CV",
    "hardware": "not reported"
  },
  "contribution": {
    "summary": "Shows squeeze-excitation attention on top of a standard 1D ResNet improves rare-class detection without extra data.",
    "code_available": true,
    "code_url": "https://github.com/example/repo"
  }
}
```

## Special case: `2020ChallengePaper.pdf`

This is the **challenge overview paper**, not a competing team entry. For
this file only:
- Set `team_name` to `"N/A — challenge overview paper"`.
- Set `model`, `input`, `data`, `training` sub-fields to
  `"not applicable — overview paper"` instead of `"not reported"`.
- In `contribution.summary`, summarize the challenge's task definition
  (12-lead ECG multi-label classification, ~27 scored classes) and scoring
  metric instead of a novel technique.
- Additionally, if the paper contains a full leaderboard table (team name →
  official score → rank), extract it **verbatim** into a second file:
  `benchmark/data/_leaderboard_reference.json`, as a JSON array of
  `{"team_name": "...", "official_challenge_metric_score": ..., "official_rank": ...}`
  objects. This file is used later for cross-checking other papers'
  self-reported scores/ranks — do not omit it if the table exists.
