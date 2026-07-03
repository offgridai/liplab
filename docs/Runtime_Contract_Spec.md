# Offgrid Lipsync Runtime Contract Spec

Source reviewed: clean `offgrid_dropin` and harness files from the reverted branch.

## 1. Core conclusion

The original runtime is not a current-state renderer. It is a **monotonic absolute-time event scheduler**.

The performer does not know about words, phones, alignment confidence, audio buffers, or streaming state. It only samples a committed track at a playback-space time:

```cpp
FOffgridAIVisemePerformer::Sample(Track, PlaybackSeconds)
```

Therefore any replacement must preserve this contract:

```text
Producer emits an append-only, sorted sequence of absolute playback-time viseme events.
Consumer samples those events by absolute playback time.
```

The failed word-assignment patches broke this by changing either the timing basis, the append behavior, or the implied event lifecycle.

## 2. Time domains

The original code mixes several time domains. They must not be substituted for each other.

| Name | Meaning | Source | May lead playback? | May be used for visible pose sampling? |
|---|---|---|---:|---:|
| `CurrentPlaybackSec` | Actual line playback clock supplied to `RuntimeSession::Update` | Caller / harness / UE line playback | No | Yes |
| `ObservedAudioBufferEndSec` | Amount of TTS audio received and analyzed | `StreamingSpeechDetector` | Yes | No |
| `PrerollSec` | Buffer/lookahead available before audible playback reaches that point | Begin input | Yes | No, except as commit horizon |
| `CommitTimeHorizon` | Maximum event center allowed to become committed on this update | Runtime adapter | Yes | No |
| `FinalRenderCenterSeconds` | Absolute playback-space center of the visible event | committed event | N/A | Yes |
| `RenderStartSeconds` / `RenderEndSeconds` | Absolute playback-space event envelope | committed event | N/A | Yes |
| `CommitPlaybackSeconds` | Playback time when event was appended to committed track | runtime adapter | N/A | Diagnostics only |
| `CommitLeadSeconds` | `FinalRenderCenterSeconds - CommitPlaybackSeconds` | runtime adapter | N/A | Diagnostics only |
| `AlignedPhoneStartSeconds` / `EndSeconds` | Acoustic phone span estimate in line playback/audio space | online aligner | Yes while provisional | Diagnostics / placement input |
| `TextDiagnosticCenterSeconds` | Text-only nominal event time | text planner | N/A | Diagnostics only |

Critical rule:

```text
Observed audio may justify planning or committing future absolute events.
Observed audio must not become the playback clock.
```

## 3. Data structure contracts

### `FOffgridAITextVisemePlan`

Contract:

- Immutable after `BeginLine`.
- Defines transcript order.
- `Events` are text-derived visible viseme events.
- `ExpectedPhones` are the transcript/CMU phone path.
- `EventIndex` in committed events points into `Plan.Events`.
- `SourcePhoneIndex` or global phone mapping connects a visible event to the expected phone path.
- Word and sentence indices are metadata; they do not themselves authorize timing.

Non-contract:

- `StartNorm` / `EndNorm` are not playback times.
- `EstimatedDurationSeconds` is not authoritative audio duration.

### `FOffgridAIAlignedVisemeEvent`

Contract:

- Represents one visible viseme event in absolute playback time.
- `EventIndex` must be monotonic and refer to a text-plan event.
- `FinalRenderCenterSeconds`, `RenderStartSeconds`, and `RenderEndSeconds` are absolute line playback seconds.
- Once appended, an event is effectively immutable. Consumers and diagnostics assume committed history is stable.
- Events must remain sorted by `FinalRenderCenterSeconds` and monotonic by `EventIndex`.
- `FinalRenderCenterSeconds` should normally be nondecreasing and at least slightly after the previous event center.

Non-contract:

- It is not a current word/phone state.
- It is not relative-to-now.
- It is not relative-to-current-word.
- It is not a buffer-local timestamp.

### `FOffgridAIAlignedVisemeTrack`

Contract:

- Append-only committed prefix of the line.
- `Events` may contain future events relative to current playback, because the performer can safely sample only the active envelope.
- Future committed events are allowed and expected.
- `SpeechStartSeconds` / `SpeechEndSeconds` describe the observed speech envelope for gating and diagnostics, not the end of event generation.
- Runtime arrays (`RuntimeFirstAlignedObservedEndSeconds`, `RuntimeObservedPhoneStartSeconds`, `RuntimeObservedPhoneEndSeconds`) are diagnostics / streaming evidence state. They must not alter the meaning of committed event timestamps.

### `FOffgridAIVisemePerformer::Sample`

Contract:

- Pure function over `Track` and `PlaybackSeconds`.
- Does not know or care when an event was committed.
- Samples event weights by comparing `PlaybackSeconds` to absolute event render windows.
- It is safe for `Track.Events` to contain future events, because future events have zero weight until playback enters their envelope.

Implication:

```text
If visible motion runs faster than realtime, either event centers are too early/compressed or the caller's playback clock is wrong. The performer itself does not advance through events by queue order.
```

### `FOffgridAILipsyncRuntimeSession::Update`

Contract:

- Caller supplies the authoritative `CurrentPlaybackSec`.
- Session passes current playback, observed audio, preroll, and transcript plan into the adapter.
- Adapter may append events up to the allowed commit horizon.
- Adapter must not reinterpret playback time.

### `FOffgridAILipsyncRuntimeAdapter::UpdateCommittedTrack`

Original contract:

- Build/extend the append-only committed event track.
- It must append only events with `EventIndex >= FirstUncommittedEventIndex`.
- It must keep event order monotonic.
- It may commit ahead of playback within bounded lead, because the performer samples by absolute time.
- It must never convert future audio availability into visible current time.
- It must not remove or rewrite previously committed events.

## 4. Commit horizon contract

Original code computes a commit horizon:

```text
PrerollHorizon = CurrentPlaybackSec + PrerollSec
PlaybackLeadHorizon = CurrentPlaybackSec + MaxStreamingCommitLeadSec
CommitTimeHorizon = min(observed/preroll horizon, playback lead horizon)
```

This means:

```text
Commit horizon controls which future events may be appended.
It does not control which events are visible.
Visibility is controlled only by Sample(track, playback_time).
```

A word-assignment replacement may still use a commit horizon, but only as an append boundary for absolute event centers.

## 5. Why the previous replacement failed

The previous word-assignment patches tried to replace the runtime with a word-state machine. That is algorithmically reasonable, but they violated the output contract by producing event times that were effectively tied to one of:

- partial observed region duration,
- buffered audio duration,
- current word-relative time,
- current state rather than append-only event history,
- or a compressed projection of the word sequence.

The performer still sampled absolute event windows. Once event centers were compressed/advanced, the mouth appeared faster than realtime even if the planner was trying to be clocked.

## 6. Replacement boundary

The replacement should not replace `FOffgridAIVisemePerformer`, `FOffgridAIAlignedVisemeTrack`, or the meaning of committed events.

The safe replacement boundary is:

```text
inside RuntimeAdapter event placement only
```

Old:

```text
Plan event -> online phone aligner -> placement heuristics -> append absolute event
```

New:

```text
Plan event -> word assignment state -> word-local phone placement -> append absolute event
```

But the output must still be:

```text
append-only absolute-time FOffgridAIAlignedVisemeEvent sequence
```

## 7. Required invariants before any runtime replacement

Add debug checks or diagnostics for these invariants:

```cpp
// Track-level
EventIndex strictly increases.
FinalRenderCenterSeconds strictly/non-strictly increases with minimum gap.
RenderStartSeconds <= FinalRenderCenterSeconds <= RenderEndSeconds.
No previously committed event changes after the next update.
All committed event centers are absolute line playback seconds.

// Time safety
FinalRenderCenterSeconds <= CommitTimeHorizon + epsilon at append time.
CommitLeadSeconds == FinalRenderCenterSeconds - CommitPlaybackSeconds.
CommitLeadSeconds <= MaxStreamingCommitLeadSec + epsilon unless final playback drain.

// Word-assignment-specific
WordIndex never decreases.
EventIndex never skips backward.
Only words assigned to the current/closed speech region are eligible.
Open live word may append only events whose absolute centers are <= commit horizon.
Boundary salience may advance to next word, but may not rebase previous word times.
```

## 8. Shadow-mode requirement

Before promotion, the new word-assignment planner should run in shadow mode and write:

```text
word_assignment_shadow.csv
word_assignment_contract.csv
word_assignment_vs_committed.csv
```

No runtime behavior change in shadow mode.

Minimum columns:

```csv
update_ordinal,
current_playback_sec,
observed_audio_end_sec,
commit_time_horizon_sec,
event_index,
word_index,
word,
phone_index,
pose_id,
old_committed_center_sec,
shadow_center_sec,
shadow_render_start_sec,
shadow_render_end_sec,
shadow_reason,
shadow_commit_eligible,
shadow_commit_lead_sec,
contract_violation
```

The first promotion criterion is not quality; it is contract safety:

```text
No shadow event appears earlier than old commit safety rules allow.
No shadow track is nonmonotonic.
No shadow event is relative-time/rebased.
No shadow event center is compressed into observed partial region time.
```

## 9. Word-assignment planner contract

The new planner should expose a pure, testable function:

```cpp
struct FOffgridAIWordAssignmentInput
{
    const FOffgridAITextVisemePlan* Plan;
    const TArray<FOffgridAIStreamingSpeechIsland>* SpeechIslands;
    const TArray<FOffgridAIStreamingAudioFeatureFrame>* AudioFeatureFrames;
    float CurrentPlaybackSec;
    float ObservedAudioBufferEndSec;
    float CommitTimeHorizonSec;
    bool bInputStreamClosed;
    bool bPlaybackFinalized;
    int32 FirstUncommittedEventIndex;
    float LastCommittedCenterSec;
};

struct FOffgridAIWordAssignmentEvent
{
    int32 EventIndex;
    int32 WordIndex;
    int32 PhoneIndex;
    FName PoseID;
    float CenterSec;      // absolute line playback seconds
    float StartSec;       // absolute line playback seconds
    float EndSec;         // absolute line playback seconds
    FName Reason;
};
```

Hard contract:

```text
WordAssignmentPlanner never returns current-state poses.
It only returns absolute-time candidate events compatible with FOffgridAIAlignedVisemeEvent.
```

## 10. Word-assignment algorithm that preserves the contract

The algorithm should be phrased as absolute event placement, not realtime pose control.

For each update:

1. Determine closed/active speech islands from the detector.
2. Assign transcript words to speech islands. If ambiguous, do not invent a hard mapping; use current old behavior or shadow only.
3. For each assigned island, derive absolute word spans in line playback space.
4. For closed islands, word spans may fill the full observed island duration.
5. For open islands, word spans may be projected, but emitted event centers are still capped by `CommitTimeHorizonSec`.
6. Inside each word span, distribute that word's visible phones using CMU/text weights.
7. Boundary salience may adjust the boundary between adjacent word spans, but only in absolute playback space.
8. Append only candidate events whose event index is at or after the committed prefix and whose center is <= commit horizon.

The planner is allowed to know about current word state, but the output is still absolute events.

## 11. Deletion plan after contracts pass

Once shadow mode proves safe:

Delete or bypass these old placement responsibilities:

- forced phone Viterbi as the primary live event scheduler,
- final region word drain as a primary placement source,
- same-sentence max-gap compression as a timing authority.

Keep:

- text planner,
- speech detector,
- audio boundary salience detector,
- performer,
- committed track format,
- event diagnostics,
- final drain only as a fallback after playback finalization.

## 12. Promotion tests

Before replacing runtime behavior, run corpus in three modes:

1. Baseline clean branch.
2. Shadow word-assignment only.
3. Promoted word-assignment.

Required summary metrics:

```text
contract_violations = 0
shadow_realtime_lead_p95 <= old_realtime_lead_p95 + 50ms
shadow_word_f1 >= old_word_f1 or clear case-level wins
shadow_phoneme_coverage >= old_phoneme_coverage or explainable regression
no increase in degenerate cases
no visible-speed complaints in case-level CSV inspection
```

## 13. Immediate next patch should be diagnostics-only

The next code patch should not change live behavior. It should add:

- `word_assignment_shadow.csv`
- contract invariant checks
- summary of contract violations
- old-vs-shadow timing deltas

Only after that patch produces intelligible results should we promote the planner.
