# Regression Policy

The harness should remain simple and strict enough to keep automated iteration honest.

A scheduling change should be rejected if it causes any of the following on the checked-in corpus:

- monotonicity/order violations,
- worse speech-vs-silence ownership,
- worse word-window ownership,
- worse phoneme alignment beyond the configured tolerance.

Regression gating is now hierarchical and follows the review priorities for this project:

1. `streaming_region_boundary_pair_*`: a streaming gap is correct only when its
   close and following resume both match the same MFA gap within 100 ms. The
   complete-pair F1 is the primary speech-region segmentation metric. Initial
   open and final close are reported separately because they do not form a gap
   pair.
2. `speech_*`: overlap and boundary timing for cases whose streaming and gold
   region counts already agree.
3. `word_*`: overlap and timing of predicted word windows against approved gold word windows.
4. `phoneme_*` and `intra_word_*`: phoneme coverage and timing once speech and words are already owned correctly.

Non-gating detector and cursor metrics are retained only when they explain the active runtime path. They must be labeled diagnostic and must not be mistaken for the regression contract.

Current threshold values live in `docs/grade_thresholds.json` and are enforced by `scripts/check_grades.py`.

The accepted corpus snapshot lives in `docs/grade_baseline.json`. Future runs are judged as regressions relative to that baseline, with the small allowed deltas from `docs/grade_thresholds.json`.

When the handmade corpus grows, tighten these thresholds rather than adding permissive fallback logic.

When the approved gold corpus is deliberately re-labeled or expanded, refresh the baseline snapshot and then re-tighten the allowed deltas.
