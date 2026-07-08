# OffgridAI Lipsync

## Overview

OffgridAI Lipsync is a streaming viseme scheduler.

Ownership is strict:

- transcript owns viseme identity and order
- PCM audio occupancy owns when speech is active
- runtime scheduling owns when planned visemes are committed
- the performer and face driver only render committed events

The active runtime does not use TTS hint streams, text-progress estimates, token indices, predicted word schedules, or acoustic identity overrides.

## Runtime Pipeline

```text
Transcript
  -> TextVisemePlanner
  -> Planned phone + viseme chain
  -> RuntimeSession
       -> StreamingSpeechDetector
       -> RuntimeAdapter
  -> Committed viseme track
  -> VisemePerformer
  -> FaceDriver
```

## Active Components

### TextVisemePlanner

The planner converts transcript text into:

- a dense expected phone chain
- ordered visible viseme events
- per-word boundary punctuation metadata
- per-word speech-region ownership metadata

The planner provides a duration prior only. It does not own final timestamps.

### StreamingSpeechDetector

The detector consumes streamed PCM and produces:

- observed speech-region opens
- observed speech-region closes
- audio feature frames used for occupancy and punctuation-hold decisions

The detector is the timing authority for coarse speech activity.

### RuntimeAdapter

The adapter maps planned phones onto observed active speech time and commits visible visemes monotonically.

Current behavior:

1. Wait for the first observed speech-region open.
2. Start a monotonic active-speech playhead from that observed start.
3. Advance through the planned phone chain using duration priors.
4. Before selected punctuation boundaries, open a bounded hold.
5. During the hold, watch the live audio frames for a real lull and resumed speech.
6. If a resumed speech cue is observed, re-anchor the paused clock to that observed resume point.
7. If no useful cue arrives before the deadline, release the hold and continue monotonically.

The adapter never rewrites committed events.

### VisemePerformer

The performer samples committed events and converts them into pose weights.

It does not schedule events.

## Current Pause / Resume Model

Pause and resume are handled only in the runtime adapter.

There is one active mechanism:

- punctuation may open a bounded audio-aware hold
- the hold watches occupancy / low-energy evidence
- if speech resumes after a real lull, the playhead is re-anchored to that resume instant
- otherwise the hold expires and playback continues

Current hold classes:

- `SoftListPause`: used for list-like commas, up to `450ms`
- `HardBreakPause`: used for hard punctuation and clause-break commas, up to `1200ms`

There is no separate older state machine that waits for a text-owned future region. Playback is driven by the observed stream plus the bounded hold.

## Timing Prior

The planner supplies relative durations, not absolute timestamps.

Current prior details:

- phone weights come from the text plan
- runtime scales those weights into active-speech seconds
- adjacent words get a small `20ms` spacer in prior space
- committed centers are lead-adjusted per pose so strongly visible mouth shapes can land slightly ahead of their nominal center

## Core Invariants

- transcript owns viseme identity
- audio owns coarse speech timing
- committed playback is monotonic
- committed events are append-only
- planned visible visemes are not permanently suppressed as a timing shortcut
- no overlapping fallback scheduler owns the same decision

## Harness Relationship

`offgrid_dropin` is the authoritative runtime implementation shared with OffgridAI.

The standalone harness exists to:

- stream corpus audio through the same code
- export inspectable logs
- compare committed output against gold labels

It should not diverge into a second lipsync implementation.
