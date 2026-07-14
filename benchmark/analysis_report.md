# PhysioNet/CinC 2020 benchmark -- comparative analysis

Generated from 39 team-entry papers (28 with a numeric official hidden-test score, 25 with a numeric official rank). 3 entries (challenge overview / out-of-scope) excluded.

## 1. Architecture family vs. performance & generalization gap

| architecture_family | n  | n_with_hidden_test | mean_hidden_test_score | median_hidden_test_score | mean_generalization_gap | n_with_gap |
| ------------------- | -- | ------------------ | ---------------------- | ------------------------ | ----------------------- | ---------- |
| Other               | 1  | 1                  | 0.354                  | 0.354                    | 0.081                   | 1          |
| ResNet              | 8  | 6                  | 0.252                  | 0.236                    | 0.374                   | 4          |
| Hybrid              | 9  | 7                  | 0.154                  | 0.167                    | 0.271                   | 4          |
| Ensemble            | 10 | 7                  | 0.118                  | 0.205                    | 0.374                   | 5          |
| CNN                 | 10 | 6                  | 0.094                  | 0.157                    | 0.623                   | 2          |
| RNN/LSTM            | 1  | 1                  | 0.024                  | 0.024                    | n/a                     | 0          |

`mean_generalization_gap` is validation-leaderboard score minus official hidden-test score, computed only for papers where both a hidden-test score and a heuristically-extracted validation score were available (`n_with_gap`); take low-`n_with_gap` rows with a grain of salt.

## 2. Technique usage vs. performance & generalization gap

| technique             | used  | n  | mean_hidden_test_score | mean_generalization_gap |
| --------------------- | ----- | -- | ---------------------- | ----------------------- |
| augmentation          | True  | 10 | 0.199                  | 0.354                   |
| augmentation          | False | 29 | 0.144                  | 0.364                   |
| ensemble              | True  | 17 | 0.158                  | 0.467                   |
| ensemble              | False | 22 | 0.153                  | 0.224                   |
| external_pretraining  | True  | 3  | 0.254                  | n/a                     |
| external_pretraining  | False | 36 | 0.144                  | 0.361                   |
| threshold_tuning      | True  | 16 | 0.199                  | 0.376                   |
| threshold_tuning      | False | 23 | 0.098                  | 0.327                   |
| domain_generalization | True  | 1  | 0.437                  | 0.172                   |
| domain_generalization | False | 38 | 0.145                  | 0.374                   |

Sample sizes per group are small (n≈5-25) and many papers combine several techniques at once, so these are directional comparisons, not controlled experiments or significance tests.

## 3. Internal rank/score consistency

- Duplicate official_rank 16: CinC2020-134, CinC2020-225
- Duplicate official_rank 24: CinC2020-185, CinC2020-198
- Rank/score order disagreements: CinC2020-198 (rank 24, score 0.167) scores higher than the previous (better) rank's score 0.155; CinC2020-406 (rank 40, score -0.179) scores higher than the previous (better) rank's score -0.29
- 5 paper(s) have a low-confidence extracted validation score (hedge language in source text, or fell back to a k-fold CV score instead of the official validation-leaderboard score): CinC2020-144, CinC2020-328, CinC2020-353, CinC2020-356, CinC2020-417
- benchmark/data/_leaderboard_reference.json is empty, so self-reported rank/score cannot be cross-checked against an independent official leaderboard -- only the internal consistency checks above are possible.

## Data quality notes

- Only papers with a numeric `result.official_challenge_metric_score` are counted as having a hidden-test score; papers that failed to submit/score are excluded from score-based comparisons but still counted in `n`.
- The validation-leaderboard score used for `generalization_gap` is not a structured field in the source JSON -- it is recovered via regex over `result.evaluation_split` / `result.self_reported_metrics.other` prose, which varies in phrasing per paper. Each row's `validation_confidence` (`high`/`low`/`none`) and matched snippet are available in the underlying dataframe (`analyze.py`'s `build_dataframe`) for audit; only `high`- and `low`-confidence values are used in the aggregate tables above.
- Technique tags (augmentation/ensemble/pretraining/threshold-tuning/domain-generalization) are keyword matches over free text, not a structured field -- false negatives are likely where a paper uses a technique but describes it without the matched keywords.
- `_leaderboard_reference.json` is empty, so rank/score cross-checking is internal-only (see section 3), not verified against an independent source.

## Implications for PulseDecoder

Based on the data available (small-n, partial coverage -- treat as directional, not conclusive):

- **Other** entries (n=1) averaged a hidden-test score of 0.354 with a mean generalization gap of 0.081 (n=1).
- **ResNet** entries (n=8) averaged a hidden-test score of 0.252 with a mean generalization gap of 0.374 (n=4).
- **Hybrid** entries (n=9) averaged a hidden-test score of 0.154 with a mean generalization gap of 0.271 (n=4).

See sections 2 and 3 above before treating any single technique or architecture family as a settled recommendation -- coverage gaps in the source data limit how far these numbers can be trusted.
