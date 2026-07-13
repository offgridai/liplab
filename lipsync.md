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
       -> AcousticEvidence
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
- syllable membership and strong-phone landmarks
- punctuation fences that may require pause/resume resolution

The planner provides a duration prior only. It does not own final timestamps.

### StreamingSpeechDetector

The detector consumes streamed PCM and produces causal evidence:

- observed speech-region opens
- observed speech-region closes
- quiet/resume evidence
- syllabic envelopes and feature frames

The detector does not choose phone or viseme identity.

### AcousticEvidence

The acoustic evidence classifier maps streamed feature frames to broad articulatory probabilities used by the runtime cursor. It is deliberately small and deterministic. It is not a forced aligner, beam search, or alternate transcript.

### RuntimeAdapter

The adapter maps planned phones onto observed active speech time and commits visible visemes monotonically.

Current behavior:

1. Wait for observed speech onset and anchor the first uncommitted event to it.
2. Advance one transcript cursor through the planned chain using relative duration priors.
3. At punctuation fences, require matching quiet/resume evidence before crossing the boundary.
4. Within active speech, match syllable envelopes and strong-phone evidence as soft timing anchors.
5. Rebase the uncommitted suffix when an anchor is accepted.
6. Commit events monotonically once their placement is no longer revisable.

The adapter never rewrites committed events.

### VisemePerformer

The performer samples committed events and converts them into pose weights.

It does not schedule events.

## Current Pause / Resume Model

Pause and resume are resolved by the same transcript cursor that handles syllable timing. Punctuation supplies a strict fence; streamed audio determines whether a lull occurred and where speech resumed. A fence does not silently time out and let downstream visemes cross unresolved silence.

Speech occupancy remains the coarse safety boundary. Punctuation evidence explains where a transcript boundary maps into that observed stream; it does not pre-create an independent text-owned speech-region schedule.

## Timing Prior

The planner supplies relative durations, not absolute timestamps.

Current prior details:

- phone weights come from the text plan
- runtime advances through those relative durations and may rebase the uncommitted suffix from accepted audio anchors
- adjacent words get a small `20ms` spacer in prior space
- committed centers are lead-adjusted per pose so strongly visible mouth shapes can land slightly ahead of their nominal center

## Core Invariants

- transcript owns viseme identity
- audio owns coarse speech timing
- committed playback is monotonic
- committed events are append-only
- planned visible visemes are not permanently suppressed as a timing shortcut
- one monotonic cursor owns punctuation, syllable, and strong-phone timing decisions

## Harness Relationship

`offgrid_dropin` is the authoritative runtime implementation shared with OffgridAI.

The standalone harness exists to:

- stream corpus audio through the same code
- export inspectable logs
- compare committed output against gold labels

It should not diverge into a second lipsync implementation.
