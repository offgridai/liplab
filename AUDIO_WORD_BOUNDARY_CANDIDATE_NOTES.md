# Audio-only word-boundary candidate diagnostics

Adds a diagnostic-only audio word-boundary candidate stream. It does not alter runtime alignment, commit, viseme timing, speech detection, or duration priors.

Outputs per case:

- `audio_word_boundary_candidates.csv`
- `audio_word_boundary_grade.json`

Summary additions:

- `AUDIO_WORD_BOUNDARY ...` line from `scripts/summarize.py`
- Aggregate metrics in `grade_summary.compute_summary()`:
  - candidate/selected/reference counts
  - precision/recall/F1 at 50, 100, 150 ms
  - nearest-candidate median/p90 errors

Candidate scoring combines audio-only cues commonly used in speech/prosody boundary work:

- RMS/energy valley depth
- low speech evidence
- low-energy trough width
- spectral distribution change
- voicing/periodicity change
- flux recovery after the trough
- stable-voicing penalty to reduce intra-vowel false positives

Candidates are generated and scored from waveform-derived feature frames only. MFA word boundaries are used only for grading/logging nearest-boundary error.
