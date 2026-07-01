# Regression Policy

The harness should remain simple and strict enough to keep automated iteration honest.

A scheduling change should be rejected if it causes any of the following on the checked-in corpus:

- monotonicity/order violations,
- worse speech-vs-silence ownership,
- worse word-window ownership,
- worse phoneme alignment beyond the configured tolerance.

Regression gating is now hierarchical and follows the review priorities for this project:

1. `speech_*`: overlap and sentence-boundary ownership between predicted speech spans and approved gold speech regions.
2. `word_*`: overlap and timing of predicted word windows against approved gold word windows.
3. `phoneme_*` and `intra_word_*`: phoneme coverage and timing once speech and words are already owned correctly.

The older blended summary metrics are still printed for continuity and debugging, but they are no longer the primary pass/fail contract.

Current threshold values live in `docs/grade_thresholds.json` and are enforced by `scripts/check_grades.py`.

The accepted corpus snapshot lives in `docs/grade_baseline.json`. Future runs are judged as regressions relative to that baseline, with the small allowed deltas from `docs/grade_thresholds.json`.

When the handmade corpus grows, tighten these thresholds rather than adding permissive fallback logic.

When the approved gold corpus is deliberately re-labeled or expanded, refresh the baseline snapshot and then re-tighten the allowed deltas.
