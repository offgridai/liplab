# OffgridAI Lipsync

## Overview

OffgridAI Lipsync is a real-time streaming viseme scheduler for MetaHuman characters.

The system follows a strict ownership model:

- Transcript owns viseme identity.
- PCM audio owns speech timing.
- The runtime scheduler owns placement.
- The FaceDriver owns rendering.

The lipsync core does not consume TTS hint streams, text-progress estimates, token indices, or predicted word schedules.

The design is intentionally streaming-first and avoids global sentence alignment.

---

## Runtime Pipeline

```text
Transcript
    ↓
TextVisemePlanner
    ↓
Planned Viseme Track
    ↓
Runtime Session
    ├─ StreamingSpeechDetector
    ├─ OnlinePhoneAligner
    └─ RuntimeAdapter
    ↓
Committed Viseme Track
    ↓
VisemePerformer
    ↓
FaceDriver
```

---

## Component Responsibilities

### TextVisemePlanner

Converts transcript text into an ordered viseme plan.

Responsibilities:

- Text normalization
- Phoneme generation
- Phoneme → MetaHuman viseme mapping
- Relative weighting
- Stable ordering

The planner determines:

- Which visemes exist
- Their order
- Their relative importance

The planner does not determine final timing.

---

### StreamingSpeechDetector

Analyzes incoming streamed PCM audio.

Purpose:

- Detect speech activity
- Detect pauses
- Detect resumes
- Detect speech completion

The detector is not a phoneme recognizer.

It exists only to determine where articulation may occur.

PCM audio occupancy is the timing authority.

---

### OnlinePhoneAligner

Provides lightweight phone-class evidence from observed PCM speech.

Purpose:

- Improve local placement confidence
- Improve alignment inside active speech regions

The aligner does not choose viseme identity.

Transcript ownership remains authoritative.

---

### RuntimeAdapter

The RuntimeAdapter incrementally converts planned events into committed events.

Properties:

- Monotonic
- Streaming-safe
- Local decision making
- Bounded future commitment

The adapter:

1. Opens on observed speech onset.
2. Advances monotonically through the dense planned phone path.
3. Converts phone progress into committed visible viseme times.
4. Applies bounded punctuation-hold probes before selected word boundaries.
5. Waits for the next observed speech onset only when a real region close is detected.

The runtime never rewrites committed events.

---

## Current Timing Rules

The current runtime design is:

- speech occupancy owns observed speech-region open / close
- transcript + CMU phones own identity and order
- runtime maps elapsed active speech time onto the dense phone path
- committed playback remains monotonic and append-only

Important details:

- The planner emits one dense phone chain for the line. It does not pre-divide visemes into text-owned speech regions.
- Punctuation is a runtime prior, not a text-imposed silence promise.
- Soft punctuation `, ; :` triggers a `120ms` close probe.
- Hard punctuation `. ? !` triggers a `260ms` close probe.
- If speech actually closes during the probe, playback waits for the next observed speech onset.
- If speech does not close, playback resumes and continues monotonically.
- Ordinary word boundaries do not create runtime pause ownership, but the duration prior currently includes a small `20ms` inter-word spacer.

This keeps the scheduler simple:

- no TTS hint timing
- no text-progress ownership
- no committed-event rewrites
- no dropping remaining visemes just because a detected region ended

---

## Commit Horizon

The runtime commits only a limited distance into the future.

Benefits:

- Better adaptation to live audio timing
- Reduced rigidity
- Better handling of punctuation-driven close probes
- Better handling of variable speech-region timing

The scheduler intentionally avoids solving the entire sentence early.

---

## Committed Track

The committed track is the authoritative runtime output.

Properties:

- Ordered
- Monotonic
- Immutable after commit
- Safe for playback

All rendering consumes committed events only.

---

## VisemePerformer

Samples committed viseme events and converts them into pose weights.

Responsibilities:

- Weight evaluation
- Transition shaping
- Runtime sampling

The performer does not schedule events.

---

## FaceDriver

The FaceDriver renders the final MetaHuman mouth poses.

Responsibilities:

- Blend pose weights
- Drive facial controls
- Apply transitions

The FaceDriver does not reinterpret timing.

The FaceDriver does not perform scheduling.

---

## Core Invariants

### Transcript Owns Identity

Audio never changes which viseme belongs to a word.

### Audio Owns Timing

Speech activity determines when articulation may occur.

### Local Decisions

Scheduling decisions are made near the active playback horizon.

### Monotonic Playback

Events always advance forward through the plan.

### No Permanent Suppression

Planned visemes are expected to reach the committed track.

### Phrase Ownership

Speech before a pause belongs to the earlier phrase.

Speech after a pause belongs to the later phrase.

### Streaming First

The runtime operates on partial future information and does not require complete audio.

---

## Current Design Notes

- PCM audio occupancy is the primary timing signal.
- Phone alignment acts as secondary placement evidence.
- Transcript-derived viseme identity remains authoritative.
- Runtime scheduling is incremental rather than global.
- FaceDriver remains a rendering-only system.
- Committed events are never reordered.

This architecture prioritizes stability, explainability, and streaming robustness over perfect offline alignment.
