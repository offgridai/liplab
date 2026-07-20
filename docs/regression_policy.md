# Regression Policy

Corpus acceptance follows the product priorities directly:

1. Region starts: the first visible event must belong to the correct MFA speech
   region and land within 100 ms of its class-aware visual target.
2. Pauses: visible animation should be absent between MFA speech regions.
3. Word starts: performed animation onset should land within 100 ms of MFA word
   onset, and its first visible gesture should land within 100 ms of the target
   selected by phone class: vowel nucleus center, or the relevant visible
   consonant onset/closure landmark.
4. Word-region assignment: every planned event for a word must be committed,
   all events must belong to one runtime speech region, and that complete
   region must map to the word's MFA region. Split words always fail.

The first three are reported as `region_start`, `pause`,
`word_animation_onset`, and `class_aware_visual_anchor`. Mean absolute errors
show the size of timing misses without replacing the 100 ms success rates.
`strict_three_level_word_assignment.success_rate` catches a wrong region even
when an event happens to be close to the correct timestamp.

Event completion and monotonic order are non-negotiable guardrails. Exact and
nearby nucleus metrics remain useful diagnostics for the syllable estimator,
but are not universal visual scores because consonant-led gestures intentionally
target an earlier landmark. Older speech, boundary-pair, detector, and cursor
measurements remain in per-case reports for diagnosis, not as parallel
definitions of success.

`scripts/summarize.py` writes the scorecard to
`outputs/runs/latest/alignment_summary.json` and the review tables to
`alignment_cases.csv` and `alignment_words.csv`. `scripts/check_grades.py`
compares that scorecard with `docs/grade_baseline.json` using the small allowed
deltas in `docs/grade_thresholds.json`.

When approved gold is deliberately changed or expanded, refresh the baseline.
When runtime quality improves, accept the new baseline rather than loosening a
threshold.
