# OffgridAI Lipsync System Specification

This document is the source-of-truth overview for the shared LipLab / OffgridAI lipsync runtime.

## One-sentence summary

```text
Text creates a complete monotonic MetaHuman viseme score; streaming audio drives a speech-active playhead through that score, holding during pauses and rendering the resulting committed viseme track.
```

## High-level pipeline

Transcript arrives:

1. Normalize text.
2. Convert text to CMU phonemes, using fallback pronunciation when needed.
3. Map CMU phonemes to MetaHuman viseme pose IDs.
4. Use prosody, salience, and duration estimates to draft one monotonic viseme timeline.

Audio stream arrives with about 150 ms preroll/lookahead:

5. Detect first speech onset.
6. Classify incoming audio frames as speech-active or pause.
7. Advance the text/viseme playhead while speech is active.
8. Hold the playhead while audio is paused.
9. If speech-active audio ends before all planned visemes are reached, drain remaining future visemes briefly and monotonically.
10. `VisemePerformer` and the Unreal FaceDriver render committed visemes on the MetaHuman face.

## Runtime model

The runtime is intentionally punctuation-unaware. Punctuation may influence the text/prosody planner's prior timeline, but it is not a runtime gate.

At runtime:

```text
audio speaking -> text playhead advances
audio pause    -> text playhead holds
audio resumes  -> text playhead continues
```

The runtime does not assign text envelopes to audio envelopes. It does not wait specifically for commas, periods, or list pauses. This lets the system follow the TTS performance whether a comma is spoken with a clear pause, a tiny hesitation, or no pause at all.

## Strict invariants

These must remain true in every runtime patch:

- every committed event refers to a valid planned source event;
- source event order remains monotonic;
- committed event centers remain monotonic;
- no event is duplicated in the committed timeline;
- already-played events are never moved;
- audio evidence may only affect future/unplayed timing;
- FaceDriver receives timing and weights; it does not own planning policy;
- host code may feed input and export diagnostics, but shared core owns lipsync timing behavior.

## Current known issue

The planned text timeline often runs at a different speed than the actual TTS speech-active duration.

Typical symptom:

```text
speech-active audio ends before the text playhead reaches all planned visemes
-> remaining tail visemes drain or final-flush late
```

The next safe improvement is an adaptive playhead-rate controller that keeps the monotonic text playhead closer to observed speech-active progress without repartitioning text, using punctuation gates, or retiming already-committed events.

## Optional landmarks

`OffgridAIStreamingLandmarkDetector.*` remains available as an optional helper for future experiments. Landmark detection is not part of the active runtime path.

Do not add landmark timing authority unless a future patch proves it preserves monotonic source order, does not move already-played events, and improves perceptual results in Offgrid logs.
