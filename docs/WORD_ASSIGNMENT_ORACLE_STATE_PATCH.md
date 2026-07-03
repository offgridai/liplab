# Word assignment oracle/state patch

Diagnostic-only patch. Runtime playback is unchanged.

Adds:

- `word_oracle_visemes.csv` / `word_oracle_grade.json`: uses MFA gold word spans and CMU/planned phone local positions to establish the upper bound for word assignment when word timing is perfect but phone timing inside words is still locally projected.
- `region_oracle_visemes.csv` / `region_oracle_grade.json`: uses MFA gold speech regions and text-normalized event positions to establish a speech-region-only upper bound.
- `word_state_shadow.csv`: per-update shadow state for the proposed word-state decoder. It logs current word, next word, word span, duration ratio, boundary confidence, duration advance prior, still-current probability, next-word probability, and transition reason.
- `ORACLES ...` summary line in `scripts/summarize.py`.

The word-state diagnostic intentionally does not feed live playback. It is meant to make over-detect vs under-detect behavior visible before promotion.
