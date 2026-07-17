# Regression Policy

Corpus acceptance follows the product priorities directly:

1. Region starts: the first visible event must belong to the correct MFA speech
   region and be centered within 100 ms of its corresponding MFA phoneme.
2. Pauses: visible animation should be absent between MFA speech regions.
3. Word starts: the first visible event must belong to the word's MFA region
   and be centered within 100 ms of its corresponding MFA phoneme.
4. Word-region assignment: every planned event for a word must be committed,
   and all of its events must map to the word's MFA region.

The first three are reported as `region_start.success_rate`,
`pause.clean_rate`, and `word_start.success_rate`. Region- and word-start mean
absolute errors show the size of timing misses without replacing the success
rates. `word_region_assignment.success_rate` catches a wrong region even when
the event happens to be close to the correct timestamp.

Event completion and monotonic order are non-negotiable guardrails. The older
speech, boundary-pair, phoneme, detector, and cursor measurements remain in
per-case reports for diagnosis, but they are not parallel definitions of
success.

`scripts/summarize.py` writes the scorecard to
`outputs/runs/latest/alignment_summary.json` and the review tables to
`alignment_cases.csv` and `alignment_words.csv`. `scripts/check_grades.py`
compares that scorecard with `docs/grade_baseline.json` using the small allowed
deltas in `docs/grade_thresholds.json`.

When approved gold is deliberately changed or expanded, refresh the baseline.
When runtime quality improves, accept the new baseline rather than loosening a
threshold.
