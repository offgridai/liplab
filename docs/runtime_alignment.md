# Runtime Alignment

The runtime alignment model is audio occupancy playback.

It does not align punctuation envelopes. It does not match viseme landmarks. It does not warp already-committed events. It advances a monotonic text playhead while streaming audio is speech-active and holds that playhead during pauses.

## Inputs

### Text input

`FOffgridAITextVisemePlanner` creates one complete planned viseme timeline before audio playback begins.

Important fields per planned event include:

- `SourceEventIndex` / event order;
- word;
- MetaHuman pose ID;
- planned center time;
- render window;
- salience and duration metadata.

The planned timeline is a prior, not final runtime timing.

### Audio input

`FOffgridAIStreamingSpeechDetector` receives PCM chunks and produces:

- feature frames;
- speech-active decisions;
- speech islands with start/end times;
- stream-tail metadata.

The detector is the runtime's only active audio timing authority.

## Runtime behavior

### 1. First speech onset

The first speech onset starts audio-occupancy playback. Before first onset, the committed track should not run ahead of speech.

### 2. Speech-active advance

While audio is speech-active, the scheduler advances through the planned text timeline and commits future viseme events monotonically.

### 3. Pause hold

When audio is paused, the playhead holds. This handles comma pauses, hard sentence pauses, and arbitrary TTS hesitations without needing punctuation-specific runtime logic.

### 4. Resume

When speech resumes, the same playhead continues. No text envelope is selected or skipped; the next uncommitted planned event remains next.

### 5. Tail drain

If speech-active audio ends before all planned events have committed, the runtime may continue briefly in tail-drain mode so late text does not collapse entirely into finalization.

Tail drain must remain monotonic and must not duplicate or retime already-committed events.

### 6. Final flush

Final flush is a diagnostic fallback. It should only catch true leftovers at stream close. A patch that increases ordinary final-flush events is suspect.

## Non-goals

The active runtime does not do any of these:

- punctuation horizon gates;
- hard/soft comma release rules;
- text-envelope to audio-envelope assignment;
- coalescing text envelopes;
- global remapping of the entire text timeline after the fact;
- per-audio-island text repartitioning in streaming mode;
- landmark-based nudging;
- retrospective retiming of already-committed events.

## Known limitation

The current text playhead advances according to the planner's duration prior. Actual TTS speech-active duration often differs. When the planner is too slow, tail events drain or final-flush. When the planner is too fast, future words can appear too early.

The safest next direction is bounded adaptive playhead rate:

```text
keep monotonic order
keep committed events fixed
adjust only future playhead speed
use speech-active progress, not punctuation
```

Avoid designs that repartition already-seen text over newly arriving audio islands. Those can create duplicate, sticky, or wrong visemes in streaming playback.
