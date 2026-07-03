# Retrospective Cursor Design

## Goal

Introduce a hidden monotonic cursor that estimates how far the speaker has progressed through the transcript-derived phone and word plan, using delayed evidence from already-observed audio rather than only front-edge streaming guesses.

The cursor is not allowed to invent viseme identity or reorder committed events. Its job is to improve pacing of future events by establishing a better estimate of what has already passed.

## Why This Exists

The current runtime already has:

- transcript-derived expected phones and words
- speech occupancy islands from streamed audio
- a monotonic phone aligner against expected phones
- progress diagnostics over boundaries and word timing

What it lacks is a durable hidden state saying:

- these words and phones are now likely behind us
- these boundaries now look confirmed
- future pacing should be scheduled from this corrected progress estimate

Without that state, early onset mistakes cascade through the rest of the utterance.

## Design Thesis

Use occupancy and energy structure as the primary evidence for coarse progress. Use expected transcript timing as the prior. Use boundary and phone-class evidence as secondary support. Update a delayed monotonic cursor over recently observed audio, then pace future visemes from that cursor.

In short:

- occupancy owns coarse speech mass and valleys
- the text plan owns what boundaries are possible
- the retrospective cursor decides which expected boundaries have probably been crossed
- phoneme evidence refines that decision but does not own it

## Inputs

The cursor consumes:

- `Plan.ExpectedPhones`
- word-level grouping and sentence grouping from the text plan
- speech islands from the detector
- streamed audio feature frames
- expected word weights and phone durations
- local boundary evidence already computed in the runtime

It should not consume:

- external TTS timing
- token indices from another model
- audio-driven viseme identity selection

## Hidden State

Maintain a compact cursor state per active sentence region:

- `ConfirmedWordIndex`
- `ConfirmedPhoneIndex`
- `ConfirmedProgress01`
- `LastConfirmationPlaybackSec`
- `LastConfirmationObservedAudioSec`
- `ConfirmationConfidence`
- `PendingBoundaryCount`

This state is advisory for future pacing only. Already committed events remain immutable.

## Evidence Ledger

Keep a trailing ledger over a bounded look-behind window, for example the last `0.6` to `1.2` seconds or last `2` to `5` words of expected material.

For each update, accumulate:

- occupancy mass in the trailing window
- low-energy valleys inside occupied speech
- voiced/unvoiced balance
- boundary salience near expected word boundaries
- expected phone coverage mass from the aligner
- detected speech region starts and stops

The key difference from the current front-edge logic is that this ledger summarizes what has already passed, not what might happen next.

## Confirmation Rule

At each update:

1. Build the expected boundary candidates inside the trailing window.
2. Score each candidate boundary using:
   - expected timing prior
   - occupancy valley depth
   - local boundary confidence
   - phone-class transition evidence
3. Accept only monotonic boundary crossings.
4. Promote the farthest sufficiently supported boundary as the new confirmed cursor.

This creates bounded hindsight:

- we wait a little for evidence to accumulate
- then we move the hidden cursor decisively
- then future scheduling benefits from that more accurate anchor

## Scheduling Contract

The cursor may:

- adjust pacing of not-yet-committed future visemes
- inform word-start expectations for future words
- help decide that a pause really occurred

The cursor may not:

- rewrite committed timings
- suppress planned visible visemes permanently
- reorder events
- replace the transcript plan with audio-guessed content

## Evidence Priority

Recommended priority order:

1. Speech occupancy and energy valleys
2. Expected boundary locations from transcript structure
3. Boundary confidence around those expected locations
4. Phone-class evidence from the aligner

Reason:

- occupancy is the most reliable coarse cue in streaming
- transcript structure constrains the search space sharply
- boundary confidence is useful but should not float unconstrained
- phoneme evidence is helpful but too noisy to own the cursor alone

## First Implementation Cut

A practical phase-1 version:

1. Detect first speech onset as today.
2. Let `2` to `4` words of evidence accumulate.
3. Compute the most plausible monotonic word boundary prefix that has already passed.
4. Set `ConfirmedWordIndex` and `ConfirmedProgress01`.
5. Pace future words and visemes from that confirmed position.
6. Reconfirm only at strong later boundary opportunities.

This is intentionally not a fully continuous decoder. It is a bounded-delay anchoring system.

## Evaluation

Success should be measured primarily against:

- speech region start alignment
- speech region end alignment
- word start alignment
- reduction in cascading late starts deeper into the line

Secondary metrics:

- intra-word center timing
- progress MAE against gold
- number of strong re-anchor confirmations

## Immediate Next Steps

1. Build a failure audit for bad speech-region starts.
2. Identify whether the worst cases are:
   - late onset confirmation
   - early false attack
   - under-segmented region drift
   - over-segmented false split
3. Add a shadow retrospective cursor diagnostic path before changing live scheduling.
4. Compare shadow cursor progress against gold words and speech regions.
5. Only then let the cursor influence runtime pacing.
