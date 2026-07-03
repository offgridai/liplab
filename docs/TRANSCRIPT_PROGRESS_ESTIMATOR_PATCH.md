# Transcript Progress Estimator Shadow Patch

Diagnostic-only patch. Runtime playback is unchanged.

This replaces the prior shadow word-state segmentation with a continuous transcript-progress estimator:

- Hidden state is continuous progress through the transcript region, not a discrete current-word advance decision.
- Progress is monotonic and freezes outside detected speech regions.
- Audio features modulate progress density rather than directly commanding word transitions.
- Word boundaries are derived when cumulative progress crosses transcript word-weight thresholds.
- Acoustic boundary salience can softly bend/snap those crossings but cannot skip words or rebase time.

The existing `word_assignment_shadow.csv` and shadow grading now evaluate this progress-derived segmentation. `word_state_shadow.csv` includes additional fields:

- `transcript_progress`
- `prior_transcript_progress`
- `progress_word_float`

These are diagnostic views of the continuous latent progress variable.

The false-positive boundary audit and pruning simulation outputs are preserved from the previous patch.
