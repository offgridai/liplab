# Streaming vs MFA Action Plan

This document summarizes the current streamed runtime behavior against the new MFA-backed gold set and proposes the next work plan.

## Current state

The corpus now evaluates against offline MFA-backed gold timings while keeping transcript-owned viseme identity.

Latest streamed harness aggregate:

- cases: `105`
- matched visemes: `1156 / 1951`
- match rate: `59.25%`
- mean of per-case mean center error: `57.582 ms`
- median per-case mean center error: `52.111 ms`
- worst mean center error: `134.0 ms`
- passing cases: `5`
- failing cases: `100`

Main grade failure counts:

- missing viseme cases: `100`
- order violation cases: `14`
- early-start failure cases: `23`
- late-tail failure cases: `11`
- over-`120 ms` mean-center cases: `1`

This confirms the new gold set is materially stricter than the earlier offline-core fallback labels.

## What MFA is doing vs what the streamed aligner is doing

Offline MFA:

- sees the full utterance
- optimizes a monotonic phone path using future evidence
- uses stronger acoustic features and duration modeling
- exports stable word and phone boundaries after the whole file is known

Current streamed aligner:

- sees only rolling buffered audio
- must emit stable committed timings before full future evidence exists
- uses a lightweight monotonic left-to-right phone segmentation over observed speech
- cannot revise already committed events

That means the remaining gap should be interpreted as an online inference and commitment problem, not a transcript planning problem.

## What the logs suggest

The current failure mix is not dominated by order problems. It is dominated by under-matching:

- many cases miss a substantial fraction of gold visemes
- order violations exist but are secondary
- early and late boundary leakage are meaningful but not the dominant issue

Observed pattern from `committed.csv` and grade outputs:

- visible phones often arrive in the correct neighborhood but commit with too little evidence
- several cases show local overlaps or near-overlaps around adjacent words, which later surface as order violations
- many missing visemes are associated with weak or ambiguous phone evidence rather than total path collapse
- the gold export flags a non-trivial number of MFA mapping mismatches and fallback timings, especially around words where MFA phone labels differ from the transcript planner's expected visible-phone subset

Implication:

- the streamed path is usually following the right transcript path
- it needs better confidence management, better duration handling, and a more explicit left-to-right path score

## Comparison to the attached research notes

The attached notes are directionally correct:

- delay commitment: strongly supported by the current missing and boundary leakage profile
- speaking-rate adaptation: likely valuable because static duration expectations are brittle across the corpus
- duration priors: should be made explicit and scored rather than implicit and heuristic
- transition/path scoring: consistent with the need to reduce local premature choices
- online Viterbi / beam search: matches the design direction already implied by the current streaming forced aligner
- richer diagnostics: immediately useful

One point needs reframing:

- "missing and extra phones are nearly symmetric" is not the best current summary for this harness
- what we directly observe in grading is high missing viseme count against gold, with secondary ordering and boundary issues

## Proposed plan

### Phase A: Logging and diagnostics

Add logging before changing behavior.

Desired new outputs per case:

- `online_phone_alignment.csv`
  - expected phone index
  - expected phone label
  - word index
  - frame time
  - chosen state
  - state score
  - duration in state
  - is_committed
  - commit time
  - commit reason
- `alignment_beam.csv`
  - frame time
  - hypothesis rank
  - phone index
  - phone label
  - accumulated score
  - duration penalty
  - transition penalty
  - confidence
- `commit_decisions.csv`
  - viseme index
  - source phone
  - proposed start/center/end
  - buffer age at commit
  - confidence
  - reason for emitting now instead of waiting

Additional derived metrics:

- P90 center error
- onset error
- offset error
- duration error
- extra committed viseme count
- per-word missing count
- per-phone-class miss rate
- phrase-start and phrase-end boundary leakage

Purpose:

- identify whether misses cluster in specific phone classes
- separate "bad evidence" from "premature commit"
- expose whether ordering failures originate in the aligner or in viseme scheduling after alignment

### Phase B: Correct missing visemes and ordering failures

#### 1. Add a commit lag

Current recommendation:

- keep audio lookahead around the existing `350 ms` buffer
- add explicit commit lag of roughly `80-120 ms`

Expected benefit:

- fewer phones committed before enough future evidence arrives
- fewer missing visemes
- fewer local order inversions

#### 2. Split alignment state from commit state

The aligner should maintain:

- current best online phone path
- committed prefix
- uncommitted suffix

Only the committed prefix becomes immutable.

Expected benefit:

- preserve monotonicity without freezing local decisions too early

#### 3. Make phone duration priors explicit

Each phone or phone class should carry:

- minimum duration
- preferred duration
- maximum duration

Score paths against those priors rather than only clamping heuristically.

Focus first on visible classes:

- bilabial
- labiodental
- front vowels
- open vowels
- round vowels
- glides

Expected benefit:

- fewer dropped visible phones
- fewer compressed clusters inside one acoustic peak

#### 4. Introduce transition costs

Add a small left-to-right transition score between phones.

Reward:

- staying on the current phone when evidence is still plausible
- moving forward when duration is satisfied and the next phone becomes more plausible

Penalize:

- jumping forward too early
- collapsing multiple visible phones into one local feature burst

Expected benefit:

- fewer ordering failures
- fewer missed middle phones in dense words

#### 5. Add confidence-gated commitment

Only commit when one of these is true:

- confidence exceeds threshold, or
- the phone has aged past a configurable horizon and must flush

This should be a runtime tuning layer over the aligner's state confidence, not a second fallback scheduler.

### Phase C: Timing quality improvements

#### 1. Speaking-rate adaptation

Estimate local speech rate from:

- recent observed speech duration
- recent committed phone durations
- phrase-local phone density

Use that to scale preferred phone durations.

Expected benefit:

- better center timing
- less early-start and late-tail leakage

#### 2. Class-specific boundary padding

Tune render spans separately from alignment spans.

Candidates:

- vowels: slightly wider symmetric render windows
- bilabials/plosives: tighter onset anchoring, shorter tails
- glides: shorter visible hold unless supported by long evidence

Expected benefit:

- reduce late tails on bilabials and labiodentals
- reduce early starts on open vowels and glides

#### 3. Phrase-boundary anchoring

Several errors appear near word and phrase boundaries.

Tighten the rule that the first visible phone of a phrase should not be pulled backward into the prior phrase unless confidence is very high.

Expected benefit:

- fewer order failures across phrase edges
- less pause leakage

### Phase D: Tuning experiments

Run controlled experiments rather than bundling changes.

Recommended experiment matrix:

1. commit lag
   - `0 ms`
   - `80 ms`
   - `120 ms`
   - `160 ms`

2. buffer depth
   - `300 ms`
   - `350 ms`
   - `450 ms`
   - `500 ms`

3. beam width
   - `8`
   - `16`
   - `32`

4. duration prior strength
   - weak
   - medium
   - strong

5. transition penalty strength
   - off
   - light
   - medium

Success criteria:

- reduce missing viseme cases first
- preserve or reduce order violations
- then reduce mean and P90 timing error

## Recommended implementation order

1. Add diagnostics and per-phone-class metrics.
2. Add commit lag and split committed prefix from uncommitted path.
3. Add explicit duration priors.
4. Add transition scoring / beam path scoring.
5. Add speaking-rate adaptation.
6. Tune render-window padding after alignment quality improves.

## Non-goals

Do not:

- move gold generation logic into runtime
- let acoustics choose viseme identity
- add overlapping fallback schedulers
- rewrite committed history
- optimize directly against the earlier generated labels now that MFA-backed gold exists
