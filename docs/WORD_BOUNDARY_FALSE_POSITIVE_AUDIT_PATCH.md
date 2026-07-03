# Word Boundary False-Positive Audit Patch

Diagnostic-only patch. Runtime behavior is unchanged.

Adds corpus-level analysis for audio word-boundary candidates to identify classes of false-positive candidates that are wrong more often than not.

New generated files in `outputs/runs/latest`:

- `audio_word_boundary_false_positive_audit.csv`
  - one row per audio boundary candidate
  - includes audit class, reason combo, MFA proximity, score, and whether it falls into a conservative proposed-prune class
- `audio_word_boundary_false_positive_class_summary.csv`
  - aggregate precision/false-rate by coarse acoustic false-positive class
- `audio_word_boundary_feature_reliability.csv`
  - aggregate reliability by single reason and reason-combination
- `audio_word_boundary_pruning_simulation.json`
  - before/after simulation for conservatively dropping high-count, low-precision, low-recall-contribution classes

The summary step now prints:

- `AUDIO_WORD_BOUNDARY_FALSE_POSITIVE_AUDIT ...`
- `AUDIO_WORD_BOUNDARY_WORST_FP_CLASSES ...`

The pruning simulation is not a runtime rule. It is intended to identify candidates worth reviewing or downweighting before the word-state/Viterbi decoder consumes the boundary stream.
