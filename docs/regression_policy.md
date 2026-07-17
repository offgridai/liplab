# Regression Policy

The harness should remain simple and strict enough to keep automated iteration honest.

A scheduling change should be rejected if it causes any of the following on the checked-in corpus:

- monotonicity/order violations,
- worse speech-vs-silence ownership,
- worse word-window ownership,
- worse phoneme alignment beyond the configured tolerance.

Regression gating is now hierarchical and follows the review priorities for this project:

1. `runtime_word_region_integrity_rate` and
   `runtime_mfa_word_region_strict_accuracy`: words must remain atomic and map
   to the same speech region as MFA. Split-word, incorrect-word, and
   affected-case counts make failures directly reviewable.
2. `streaming_region_boundary_pair_*`: a streaming gap is correct only when its
   close and following resume both match the same MFA gap within 100 ms. The
   complete-pair F1 is the primary speech-region segmentation metric. Initial
   open and final close are reported separately because they do not form a gap
   pair.
3. `region_head_*`, `region_resume_in_pause_*`, and `pause_clean_rate`: first
   viseme alignment at region resumes and visible animation leakage into MFA
   non-speech intervals.
4. `speech_*`: overlap and boundary timing for cases whose streaming and gold
   region counts already agree.
5. `word_head_*` and `word_*`: first-viseme timing and predicted word windows
   against approved MFA-backed gold words.
6. `phoneme_*` and `intra_word_*`: phoneme coverage and timing once speech and
   words are already owned correctly.

`runtime_mfa_boundary_agreement_rate` is preferred over raw runtime/MFA region
index equality. One missing split changes every later ordinal while boundary
agreement continues to identify the actual local error.

Non-gating detector and cursor metrics are retained only when they explain the active runtime path. They must be labeled diagnostic and must not be mistaken for the regression contract.

Current threshold values live in `docs/grade_thresholds.json` and are enforced by `scripts/check_grades.py`.

The accepted corpus snapshot lives in `docs/grade_baseline.json`. Future runs are judged as regressions relative to that baseline, with the small allowed deltas from `docs/grade_thresholds.json`.

When the handmade corpus grows, tighten these thresholds rather than adding permissive fallback logic.

When the approved gold corpus is deliberately re-labeled or expanded, refresh the baseline snapshot and then re-tighten the allowed deltas.
