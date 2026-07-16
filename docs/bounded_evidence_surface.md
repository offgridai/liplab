# Bounded streaming evidence surface

This experimental branch adds one UE-portable acoustic observation layer. It
does not schedule animation, choose phone identity, or assign observations to
transcript syllables.

## Evidence horizon

- Preroll: the configured future audio buffer, currently 350 ms.
- Postroll: recent elapsed audio, currently 1500 ms.
- History: compact emitted observations only. Raw frames older than postroll are
  not part of a new decision.

The analyzer uses the full bounded surface to establish local energy and class
baselines. A candidate remains revisable until enough following evidence exists
to describe its temporal shape, then becomes an immutable observation.

## Emitted observations

- Syllabic pulses use a multi-scale sonority-weighted energy envelope,
  bilateral valleys, speech evidence, and voicing evidence. Pulses closer than
  120 ms remain distinct only when a measured trough separates their peaks.
- Sibilants use competing articulatory-class evidence and local maxima.
- Labiodentals use the same competing evidence, but remain a moderate signal.
- MBP preserves brief closure/release peaks rather than smoothing them as a
  sustained sound.
- Glides and rounded vowels use sustained low-frequency/voicing evidence.
- Lulls and resumes are emitted once per sustained low-evidence interval.

MBP, glide, and rounded-vowel observations are not emitted unconditionally
because their raw corpus precision is too low. They are exposed only to the
bounded transcript-conditioning pass.

## Scoring

`streaming_evidence_grade.json` grades acoustic ingredients directly against
MFA intervals. It intentionally does not use transcript position:

- pulse observations are matched to MFA vowel intervals,
- phone-family observations are matched to all MFA phones in that family,
- lull/resume observations are matched to MFA speech gaps and annotated
  punctuation gaps.

Transcript-to-syllable assignment is scored separately so sequence priors do
not disguise weak acoustic evidence in the ingredient metrics.

## Transcript-conditioned diagnostic

`streaming_conditioned_evidence_grade.json` is a second, side-by-side score.
The active committed schedule supplies the expected phone family and a bounded
vicinity; permissive phone-family candidates are retained only when they can be
matched monotonically inside a bounded 650 ms history / 350 ms preroll
assignment surface. FV and sibilants compete because both use frication
evidence. MBP, glide, and round are adjudicated independently by their expected
transcript family. The remaining postroll stabilizes acoustic baselines without
admitting distant candidates.

Pulse observations remain purely acoustic because their detector is already
more reliable than the runtime timing prior. Assigning pulses to particular
transcript vowels remains deliberately outside this ingredient experiment.

## Intra-envelope phone-family evidence

`streaming_intra_envelope_phone_grade.json` measures the narrower decision
needed by the proposed pulse-driven runtime. Acoustic pulses are first matched
to MFA syllable nuclei within 100 ms. The authoritative transcript syllable
then selects which phone-family channels are relevant, and acoustic candidates
must confirm and locate those expected families inside that same syllable.
Audio never chooses or invents phone identity.

Candidates are consolidated to the strongest observation per acoustic pulse.
Glide candidates must lie within 120 ms of that pulse and rounded-vowel
candidates within 80 ms; the brief MBP/FV transitions and sustained sibilants
are not given an additional pulse-distance limit. Family thresholds were swept
against the transcript-conditioned objective rather than the harder and
irrelevant blank-slate classification problem.

On the 224 region-qualified cases, the retained advisory detector reports:

| Family | Precision | Recall | F1 | Mean center error | Decision latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| Sibilant | 0.961 | 0.871 | 0.914 | 4.9 ms | 25 ms |
| Glide | 0.827 | 0.842 | 0.834 | 29.2 ms | 25 ms |
| Rounded vowel | 0.776 | 0.845 | 0.809 | 14.7 ms | 25 ms |
| MBP | 0.815 | 0.721 | 0.765 | 36.8 ms | 25 ms |
| FV | 0.862 | 0.674 | 0.756 | 15.5 ms | 25 ms |

This path is enabled explicitly with
`bRefinedIntraEnvelopePhoneCandidates` in the harness diagnostic. The default
remains off for runtime consumers, so these detector gains do not yet alter
animation scheduling.

## Monotonic syllable candidate sets

Candidate generation is graded separately from final syllable ranking. For
each acoustic pulse, a causal beam retains 12 possible monotonic transcript
cursors. Each cursor may skip the observation, assign it to its current
syllable, or assign it to the immediately following syllable. The latter is
needed to recover from one missed acoustic pulse. The six best distinct
syllable identities are emitted as an advisory candidate set; no identity is
committed and playback is not changed.

`streaming_syllable_candidate_set_grade.json` first matches each acoustic pulse
to an MFA syllable nucleus within 100 ms, then reports whether that correct
syllable appears among the first N candidates. On the 224 region-qualified
cases, the selected current-or-next / 12-state beam reports:

| Candidate count | Recall |
| ---: | ---: |
| 1 | 0.532 |
| 2 | 0.718 |
| 3 | 0.828 |
| 4 | 0.897 |
| 5 | 0.933 |
| 6 | 0.956 |

A five-syllable expansion reached only 0.928 recall at six candidates because
false pulses advanced the cursor too aggressively. Current-only expansion
fell to 0.635 because missed pulses left the cursor behind. Widening the beam
beyond 12 had no material effect. The selected candidate stage therefore uses
the smallest forward recovery that preserves high candidate recall.

## Monotonic syllable assignment

Final pulse assignment is graded separately from candidate recall. The
retrospective matcher performs one left-to-right segmental dynamic-programming
alignment between ordered acoustic pulses and ordered transcript syllables. It
may leave an acoustic pulse or transcript syllable unmatched, but it cannot
reorder an assignment. Transcript timing is a weak progression prior;
transcript-conditioned phone-family evidence supplies the local acoustic score.

The retained recurrence follows two established speech-alignment ideas. A
Sakoe-Chiba-style local slope constraint permits at most one skipped pulse and
one skipped transcript syllable between consecutive assignments. An explicit
duration term penalizes disagreement between the observed inter-pulse interval
and the corresponding transcript-prior interval. Strong pulse confidence raises
the local emission score; weak unexpected phone-family evidence lowers it.

`streaming_syllable_assignment_grade.json` first associates emitted pulses
one-to-one with MFA nuclei within 100 ms. Exact precision is the fraction of
all emitted assignments that name the associated transcript syllable. Exact
recall divides exact assignments by all MFA-backed transcript syllables.
Within-one accuracy accepts the immediately adjacent transcript syllable but
keeps every emitted assignment in its denominator.

The matcher penalties and evidence weights were selected on source-log-held-
out cases with `scripts/tune_syllable_matcher_dp.py`. Correcting that tuner to
apply each word-boundary pause once, rather than once per phone, was necessary
for it to reproduce the C++ timing geometry. On the 224 region-qualified cases,
the retained C++ matcher reports:

- exact assignment precision 0.759,
- exact assignment recall 0.675,
- within-one assignment accuracy 0.908,
- 3,046 assignments for 3,425 MFA-backed transcript syllables.

The preceding matcher baseline was 0.572 precision, 0.532 recall, and 0.839
within-one accuracy. A greedy selector using the current pulse plus one future
pulse from preroll was rejected at 0.487 precision, 0.431 recall, and 0.739
within-one accuracy. Candidate-set recall therefore does not imply that a
single bounded greedy decision can safely collapse the alternatives. This
retrospective score is an information upper bound; it does not change playback.
A forward-backward minimum-risk reranker over each selected syllable and its
two immediate neighbors was also rejected: held-out exact assignment improved
slightly, but aggregate F1 fell from 0.7190 to 0.7184 because the posterior was
derived from the same evidence as the Viterbi path and did not independently
resolve adjacent ambiguity.

Research references:

- Sakoe and Chiba, [Dynamic Programming Algorithm Optimization for Spoken Word
  Recognition](https://jeffe.cs.illinois.edu/teaching/compgeom/2022/refs/Sakoe-Chiba-DTW.pdf).
- Hosom, [Speaker-Independent Phoneme Alignment Using Transition-Dependent
  States](https://pmc.ncbi.nlm.nih.gov/articles/PMC2682710/).
- Bonastre et al., [Robust Speaker Verification with State Duration
  Modeling](https://doi.org/10.1016/S0167-6393(01)00044-9).

## Postroll sweep

The corpus sweep compared 650, 1000, 1500, and 2000 ms while holding preroll at
350 ms. A 1500 ms acoustic history was best. At 2000 ms conditioned FV declined,
indicating diminishing returns.

## Final ingredient scores

The retained 1500/350 ms configuration scores as follows on the full corpus:

| Landmark | Precision | Recall | Mean center error |
| --- | ---: | ---: | ---: |
| Pulse (acoustic) | 0.900 | 0.874 | 5.0 ms |
| Sibilant (conditioned) | 0.919 | 0.831 | 0.5 ms |
| Glide (conditioned) | 0.771 | 0.755 | 19.3 ms |
| Rounded vowel (conditioned) | 0.661 | 0.672 | 15.1 ms |
| FV (conditioned) | 0.660 | 0.608 | 3.6 ms |
| MBP (conditioned) | 0.581 | 0.565 | 9.9 ms |

The refined pulse path is enabled explicitly with
`bRefinedPulseCandidates`. It remains advisory while pulse-to-transcript
sequence assignment is evaluated; existing runtime consumers retain the prior
pulse stream so an ingredient-score improvement cannot silently alter
playback.

These scores evaluate ingredients, not transcript progress or playback. The
individual ingredient reports remain diagnostic; runtime consumes only stable
syllable assignments produced by the sequence estimator.

## Refined speech-region detector

`scripts/train_streaming_region_model.py` trains a compact speech/non-speech
classifier from the checked-in MFA speech intervals and exports fixed C++
coefficients to `OffgridAIStreamingRegionModel.inl`. Training uses Python and
scikit-learn, but the shipped detector is dependency-free C++: one 40-term
logistic dot product per finalized 10 ms frame followed by a two-state
run-length decoder.

The classifier sees the current acoustic frame, 100 ms of trailing context,
100 ms of leading context already available in preroll, and their combined
200 ms neighborhood. The decoder requires 20 ms of speech to open and 160 ms
of non-speech to close. Its boundary timestamp is the first quiet frame, not
the later decision time.

Training and tuning are evaluated out of sample with five group folds keyed by
source Offgrid log, so sibling lines from one run cannot straddle train and
test. The retained out-of-fold result is:

- speech-frame precision 0.983, recall 0.981, F1 0.982,
- complete close/resume pair precision 0.911, recall 0.914, F1 0.912 within
  100 ms.

The exported all-corpus model reproduces complete-pair precision 0.907, recall
0.916, and F1 0.912 in the C++ harness. These latter figures are an integration
check, not a held-out claim.

The refined estimate is deliberately separate from causal playback occupancy.
It needs 100 ms of leading context and 160 ms of persistence before confirming
a close, so substituting it directly for the live gate starves the current
scheduler. `GetSpeechRegions()` therefore remains the causal runtime contract;
`GetRefinedSpeechRegions()` and `GetRefinedGapCandidates()` expose the learned
postroll estimate for grading and a future bounded-delay adapter. The harness
writes these as `refined_speech_regions.csv` and
`refined_gap_candidates.csv`.

## Transcript position estimators

The advisory matcher uses syllabic pulses as monotonic sequence events. CMU
phone order supplies expected MBP, FV, glide, sibilant, and rounded-vowel
offsets relative to each vowel nucleus. Nearby acoustic candidates support or
contradict each pulse-to-syllable assignment. Text duration is a weak
progression tie-breaker, not an absolute timestamp.

Two scores are deliberately kept separate:

- `streaming_syllable_position_grade.json` is a retrospective assignment
  diagnostic. It aligns the completed observation sequence and measures the
  information available in the bounded acoustic ingredients, but it is not a
  runtime claim.
- `streaming_dense_position_grade.json` is the deployable causal diagnostic.
  Each update uses only observations already decidable within the 350 ms
  preroll. It emits a current syllable and word estimate throughout detected
  speech. No future line duration or final region count is available.

The causal matcher recomputes a narrow monotonic prefix alignment whenever a
new pulse becomes decidable. Calibrated phone and punctuation-pause durations
provide a weak local progression prior. Between confirmed pulse assignments,
the prior only interpolates; experiments that extrapolated speaking rate or
waited for additional pulse lookahead were rejected because they reduced
position recall.

The newest two syllable assignments remain revisable for at most the 350 ms
preroll horizon. A later prefix may roll back one or two premature advances;
older assignments remain immutable. Deeper rollback, delayed commitment,
permissive pulse candidates, hard region resets, and stronger phone-family
weighting were tested and rejected.

On the qualified corpus, the retrospective upper bound reports:

- exact transcript syllable: 0.600,
- within one syllable: 0.915,
- correct transcript word: 0.715,
- timing precision/recall within 100 ms: 0.645 / 0.553,
- high-confidence exact position: 0.725 at 0.307 coverage.

The causal current-position diagnostic reports:

- active-speech coverage: 0.967,
- exact current-syllable recall: 0.451,
- within one syllable: 0.817,
- correct transcript word: 0.602.

The retrospective score is an upper bound. The causal estimator remains useful
for diagnostics, while runtime correction uses the sparser historical anchors
below. Neither path may hard-skip transcript phones.

## Runtime pulse rebasing

The runtime does not require a correct current-position estimate on every
frame. It retains strict acoustic pulses across the configured 350 ms preroll
and 1500 ms postroll, then monotonically matches stable historical pulses to
transcript syllable nuclei. The first pulse of each speech region anchors its
first transcript syllable. Later assignments must survive two updates and pass
the exact-match score gate.

An accepted pulse centers the matched transcript nucleus on the acoustic pulse
and translates the remaining uncommitted timeline from that point. This also
works when the pulse is already in postroll: committed events remain immutable,
but every still-mutable successor receives the retrospective correction. The
correction never changes transcript identity, phone order, speech-region state,
or duration priors.

On the current corpus, exact runtime pulse rebasing reports:

- 1,307 accepted observations and 860 timing matches within 100 ms,
- timing precision 0.658 and corpus recall 0.222,
- mean absolute applied timeline correction of 107 ms,
- no change to the established playback baseline.

The strong within-one assignment score suggested a speculative fallback after
several unresolved pulses. Trials after two and three unresolved pulses were
not safe: speculative timing precision was 0.432 and 0.381 respectively, and
both reduced visible-phone coverage while increasing word-start error. The
implementation remains instrumented behind a disabled internal switch, but
only exact pulse assignments may steer production playback.

The acoustic reference phone and the first translated phone are separate. If
the next word is wholly mutable, the complete word moves as one unit. If that
word is partly committed, the correction begins at the following word. This
prevents multiple delayed anchors from stretching two halves of one word
differently. An anchor whose ideal correction lies behind the immutable prefix
is projected to the earliest legal suffix center, preserving monotonic order.
The newest accepted anchor remains authoritative until a later confident
anchor or punctuation-section reset replaces it.

## Audio-evidence playback clock experiment

The experimental runtime path now lets acoustic evidence govern progression;
text punctuation does not open a competing playback hold. Transcript identity,
order, and duration priors remain authoritative.

The clock follows four deterministic rules:

- Before the first accepted syllable, a detected syllable envelope starts the
  playhead.
- Each accepted transcript-conditioned pulse grants progress through that
  syllable's local text interval. Later pulse matching is predicted from the
  last accepted acoustic syllable rather than from a potentially stalled
  animation clock.
- Detected non-speech always freezes progress and contributes to the permanent
  audio-lull offset used by the future schedule.
- During confirmed speech, missing pulse evidence can invoke a bounded
  fail-soft after 60 ms. It advances at 0.8x speed and may move at most 250 ms
  beyond the latest evidence frontier. This prevents one missed pulse from
  suppressing the rest of a line without allowing an unrestricted fallback
  scheduler.

An accepted syllable rebase moves the complete mutable syllable span, beginning
after the previous vowel nucleus. Committed events are never reordered or
revised.

On the current 292-case graded corpus this selected configuration reports:

- no order-failure cases,
- speech animation coverage 0.817,
- phoneme coverage 0.521,
- word match rate 0.976 and mean word-head error 246.8 ms,
- mean phoneme-center error 81.2 ms,
- runtime syllable progress exact in 0.731 of cases and within one in 0.889,
- raw envelope support on 0.721 of speech updates,
- gate-open rate 0.897 and fail-soft use on 0.123 of speech updates,
- 61 leaking pause boundaries.

This is an experimental perceptual candidate, not a replacement baseline. It
improves completeness over strict pulse-only gating but still fails the checked
baseline for coverage, timing, and pause leakage. `pause_safety_pair_rate` is
not applicable to this path because that metric grades the disabled
text-punctuation hold resolver; use the audio-clock diagnostics and measured
pause leakage instead.
