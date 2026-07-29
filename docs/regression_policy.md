# Regression policy

Corpus acceptance follows current product behavior and includes every approved
reference case. Missing events receive explicit penalties in comprehensive
timing statistics; they are never removed from the mean or median.

The priority scores are:

1. `region_start`: the first visible event belongs to the correct MFA speech
   region and begins within 100 ms of its reference.
2. `pause`: performed animation remains neutral between MFA speech regions.
3. `word_animation_onset`: a word's first performed animation begins within
   100 ms of its MFA word onset.
4. `word_duration`: the actual above-threshold animation span of every word
   agrees with its complete MFA spoken duration within the larger of 80 ms or
   25%. Missing words remain zero-duration failures.
5. `strict_three_level_word_assignment`: all planned events for a word are
   committed, remain in one neural region, and map to the word's MFA region.

Word-to-region ownership is also an explicit non-regression gate independent of
region-start timing. The gate protects the direct assignment success rate and
the counts of early-region thefts, late-region assignments, materially early
intact words, and the worst intact-word lead. This prevents a severe isolated
ownership error from disappearing inside otherwise good aggregate onset or
region-start scores.

Comprehensive mean and median errors expose both aggregate outliers and typical
behavior. Region matching is monotonic and overlap-aware so a missed pause does
not shift all later comparisons.

Event completion and monotonic order are non-negotiable guardrails. Delivery
metrics separately catch missing/late events, empty speech regions, compressed
sentences, and word or sentence loss. Word-duration reports include comprehensive
mean, median, p90, p95, signed bias, duration ratios, compression, and stretching.
Within-word duration scoring separately compares
each visible run's share of its word against MFA and reports run error,
internal-boundary error, and word-level total variation.

`scripts/summarize.py` writes `outputs/runs/latest/alignment_summary.json` plus
case, word, and boundary review tables. `scripts/check_grades.py` compares that
scorecard with `docs/grade_baseline.json` and `docs/grade_thresholds.json`.

Offline checkpoint selection additionally decodes held-out phone and key-viseme
occupancy. It reports onset and exit MAE, spoken coverage, zero-overlap counts,
and under-occupied counts, alongside compact-word interval metrics. These MFA
labels are training and evaluation evidence only; they are not runtime inputs
and cannot override transcript-owned phone or pose identity. A candidate must
still pass the complete corpus gates after winning held-out selection.

When approved gold is deliberately changed or expanded, refresh the baseline.
When runtime quality improves, accept the new baseline rather than loosening a
threshold. Metrics from deleted detectors, candidate estimators, diagnostic
cursors, or offline controllers must not be restored as parallel score paths.
