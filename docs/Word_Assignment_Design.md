# Word Assignment Replacement Design

## Objective

Replace the old streaming phone-aligner placement logic with a simpler word-assignment placement model while preserving the original runtime track contract.

The desired runtime logic remains:

```text
speech region starts
identify words that belong to this region
play events from current word
advance when word transition evidence says to advance
```

But implementation must still emit absolute-time events for the existing performer.

## Non-goals

Do not replace:

- `FOffgridAIVisemePerformer`
- `FOffgridAIAlignedVisemeTrack`
- absolute playback-time event semantics
- the public runtime session API
- the concept of committing a bounded future prefix

## Model

The replacement has two layers:

### Layer A: Word assignment

Maps transcript words to observed speech regions.

Inputs:

- speech islands
- transcript words and punctuation
- audio boundary salience
- current playback time
- observed audio end

Outputs:

- absolute word spans: `{word_index, start_sec, end_sec, confidence, reason}`

### Layer B: Word-local viseme placement

Converts a word span into visible viseme events.

Inputs:

- word span
- CMU phones for the word
- visible text-plan events for the word

Outputs:

- absolute event centers and render windows compatible with `FOffgridAIAlignedVisemeEvent`

## Live-region rules

For a closed speech island:

```text
Words assigned to the island may be laid out over the full island duration.
```

For an open/live speech island:

```text
The planner may project future word spans, but may only emit events whose absolute centers are <= CommitTimeHorizonSec.
```

This distinction is crucial. The planner can reason ahead, but the committed track must remain a safe absolute-time prefix.

## Word transition rules

Boundary salience is used to adjust word boundary timing, not to directly emit visemes.

Advance from word N to word N+1 when one of these is true:

1. Strong boundary salience near the projected boundary and minimum word occupancy has elapsed.
2. Current word exceeds late timeout based on word/phone weights.
3. Speech island closes and remaining assigned words must be distributed across the closed region.

Minimum occupancy rule:

```text
Do not close a word before at least its first visible phone has had a plausible absolute-time window.
```

Late timeout rule:

```text
No word can hold indefinitely; after projected max duration, advance even without boundary salience.
```

## Event placement rules

For each word:

1. Collect visible text-plan events with matching `WordIndex`.
2. Get the word's expected phones.
3. Estimate local phone weights using existing `WeightSeconds` from `ExpectedPhones`.
4. Normalize local phone weights to the chosen word span duration.
5. Place event center inside the corresponding phone subspan using existing pose-specific lead/center rules.
6. Clamp event center to absolute monotonic order.
7. Append only if eligible under commit horizon.

## Commit eligibility

Candidate event is eligible iff:

```text
EventIndex >= FirstUncommittedEventIndex
CenterSec > LastCommittedCenterSec + MinGap
CenterSec <= CommitTimeHorizonSec + epsilon
```

Live mode should not require the entire word to be closed before committing early events in that word, but it must never compress all events in the word into already-observed partial time.

## Shadow mode implementation

Add a function:

```cpp
TArray<FOffgridAIWordAssignmentEvent> BuildWordAssignmentShadow(...)
```

It should not mutate `CommittedTrack`.

Write rows comparing old and shadow event placement. This is the first safe implementation step.

## Promotion implementation

Once shadow mode passes contract checks:

1. Replace only the event placement loop in `UpdateCommittedTrack`.
2. Preserve all event fields expected by performer/harness.
3. Preserve `CommitPlaybackSeconds` and `CommitLeadSeconds` diagnostics.
4. Preserve final drain only as `bPlaybackFinalized`, not on ordinary stream close.
5. Remove old Viterbi placement only after promoted output is demonstrably contract-safe.

## Expected advantages

- Word pacing becomes explicit.
- Boundary salience affects word transitions directly.
- Per-word phone placement is deterministic and easier to debug.
- The model avoids global phone-beam behavior that loses or delays phones.

## Main risks

- Bad word-to-island assignment can poison all later timing.
- Overtrusting boundary salience can cause intra-word splits.
- Undertrusting boundary salience degenerates to duration priors.
- Premature promotion can again violate the committed-track contract.

## Required CSVs

### `word_assignment_shadow.csv`

```csv
case_id,update_ordinal,current_playback_sec,observed_audio_end_sec,commit_horizon_sec,event_index,word_index,word,pose_id,phone_index,shadow_center_sec,shadow_start_sec,shadow_end_sec,shadow_reason,eligible,contract_violation
```

### `word_assignment_vs_committed.csv`

```csv
case_id,event_index,word,pose_id,old_center_sec,shadow_center_sec,delta_ms,old_commit_reason,shadow_reason,old_lead_sec,shadow_lead_sec
```

### `word_assignment_contract_summary.csv`

```csv
case_id,total_shadow_events,eligible_events,violations,nonmonotonic_events,relative_time_suspects,lead_p50_ms,lead_p95_ms,center_delta_p50_ms,center_delta_p95_ms
```
