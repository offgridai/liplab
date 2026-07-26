# Neural-owned streamer training plan

## Runtime ownership

The neural sequence owns these decisions:

1. enter and leave non-rendering silence states;
2. open, extend, close, and resume speech regions from those states;
3. advance into the first visible token of each word;
4. advance through every transcript phone and vowel nucleus;
5. emit the transcript token's authoritative viseme at the decoded center.

Audio never invents viseme identity. The transcript lattice contains every
renderable event in fixed order, plus legal non-rendering silence states.
Deterministic scheduling is absent from the primary experiment.

## Available supervision

- MFA phone spans supervise phone occupancy and vowel-nucleus centers.
- MFA word spans supervise word-start and word-end transitions.
- MFA speech regions label long silence states and pause/resume boundaries.
- The CMU plan supplies phone, vowel, word-position, duration, and viseme
  features available before audio begins.
- The existing causal 10 ms acoustic feature stream is the only audio input.
- Frozen train, validation, unseen-text, and held-out-speaker splits prevent
  generated variants of the same transcript from crossing text splits.

MFA region labels are supervision only. They are deliberately excluded from
token features and inference output.

## Implemented curriculum

The C++/CUDA trainer implements the following fixed-seed curriculum:

- epochs 1-4 emphasize region boundaries, silence occupancy, word starts, and
  vowel nuclei;
- epochs 1-6 repeat training cases containing MFA region gaps, using only the
  training split;
- later epochs return to balanced sequence occupancy plus rotating
  forward-sum loss;
- normalized acoustic features receive small deterministic Gaussian noise to
  reduce dependence on a particular generated voice or recording level;
- checkpoint selection remains validation-only.

The rejected explicit boundary-margin loss is not part of the curriculum. It
reduced unseen-text generalization in the preceding ablation.

## Scorecard and next data work

Every experiment reports, by frozen split:

- exact ordered viseme match and center MAE;
- neural speech-region precision/recall and boundary MAE;
- word-first-viseme start MAE;
- vowel-nucleus/syllable precision, recall, and center MAE;
- region containment, fallback rate, and ordering violations.

Timing metrics use a comprehensive reporting contract. Each metric exposes
coverage, matched-only mean, median, p90, p95, and maximum, plus the same
statistics over all references.
Unmatched references receive an explicit penalty equal to the matching window:
180 ms for visemes, word onsets, and intra-word events; 100 ms for syllable
nuclei. Speech-region matching has no finite window, so missing regions receive
a documented 1000 ms penalty. Extras remain visible through precision, extra
count, and F1 rather than being hidden inside timing error. This preserves the
useful conditional timing diagnostic while ensuring the comprehensive score
cannot improve by excluding difficult or misaligned references.

Speech regions are paired with an order-preserving, overlap-aware sequence
alignment. A region must overlap to qualify as a boundary match; missing and
extra regions remain explicit. The report separately counts merges (one
prediction overlapping multiple references), splits (one reference overlapping
multiple predictions), and clean one-to-one matches. This prevents one missed
pause from shifting every later ordinal pairing while retaining the full cost
of the segmentation failure.

The next data-factory expansion should target examples currently scarce in
the corpus: 120-600 ms mid-sentence pauses, false-pause fricatives, stop/restart
repairs, one-word regions, rapid lists, vowel-initial resumes, and the same
text rendered at contrasting speaking rates. New generated text groups must be
assigned to a new frozen split before training; they must not be appended to
validation or test after results are inspected.

Promotion targets are: aggregate match above 0.93, unseen-text and held-out
speaker above 0.90, speech-region match above 0.94, region start/end MAE below
120/180 ms, syllable recall above 0.90 with center MAE below 40 ms, word onset
below 75 ms, zero ordering violations, and deterministic fallback below 5%.
