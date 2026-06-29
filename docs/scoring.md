# Scoring and Diagnostics

Scoring exists to prevent a patch from looking good in one metric while getting worse visually or violating runtime invariants.

## Evaluation priority

Evaluate in this order:

1. Hard invariants.
2. Event coverage and uniqueness.
3. Source event order and committed-time monotonicity.
4. Initial speech onset quality.
5. Speech-active vs pause behavior.
6. Unreal human perception.
7. FaceDriver peak timing and magnitude.
8. Tail-drain / final-flush counts.
9. Corpus-wide summaries and worst cases.

The user weights Offgrid perceptual observations and Offgrid logs more highly than broad LipLab summary metrics.

## Hard invariants

These should remain zero:

- missing planned source events in the committed track;
- duplicate source event indices;
- invalid source event indices;
- out-of-order source event indices;
- non-monotonic committed centers;
- already-committed event retiming.

Expected for a healthy line:

```text
CommittedEventCount == PlannedEventCount
SourceEventIndex values are unique
SourceEventIndex values are monotonic
CommittedCenterSec values are monotonic
```

## Primary diagnostics

Use these first:

- `viseme_plan_review.md`
- `runtime_speech_islands.csv`
- `runtime_audio_occupancy_diagnostics.csv`
- `runtime_stream_tail_diagnostics.csv`
- `runtime_commit_events.csv`
- `submitted_poses.csv`
- FaceDriver peak/pose logs from Offgrid

## Audio-occupancy diagnostics

`runtime_audio_occupancy_diagnostics.csv` should answer:

- which planned source event was committed?
- what was its planned center?
- what was its committed center?
- did it commit during `speech_active`, `pause_hold`, `tail_drain`, `final_flush`, or `fallback`?
- which speech island, if any, was active?
- what was the text playhead value?
- how much audio-active time had been observed?

Important warning signs:

- many ordinary words committing as `final_flush`;
- repeated source event indices;
- large jumps in committed center time;
- future sentence words committed during an earlier speech island;
- a visible pose remaining active across unrelated later words.

## Speech island diagnostics

`runtime_speech_islands.csv` should be inspected for:

- first onset time;
- pause/resume structure;
- missed speech starts;
- overly fragmented speech;
- overly merged speech.

The runtime does not require speech islands to match punctuation. It only requires the speech/pause signal to be reasonable enough to advance and hold the text playhead.

## Tail diagnostics

`runtime_stream_tail_diagnostics.csv` helps distinguish:

- real short audio;
- delayed stream close;
- text plan too long for speech-active audio;
- late finalization.

High final-flush counts usually indicate playhead-rate mismatch rather than missing punctuation logic.

## Human review table

For Offgrid lines, a useful review table is:

```text
Line
Speech islands detected
Words that visibly start too early/late
Words/poses that appear wrong or sticky
Final-flush event count
Likely cause: onset, occupancy, playhead rate, tail drain, performer sampling
```

Do not over-index on one aggregate score while the user reports clear visible defects.
