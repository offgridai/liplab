# OffgridAI Lipsync

## Overview

OffgridAI Lipsync is a real-time streaming viseme scheduler for MetaHuman characters.

The system follows a strict ownership model:

- Transcript owns viseme identity.
- Audio owns speech timing.
- The runtime scheduler owns placement.
- The FaceDriver owns rendering.

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

Audio occupancy is the primary timing authority.

---

### OnlinePhoneAligner

Provides lightweight phone-class evidence from observed speech.

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

1. Examines the next uncommitted planned event.
2. Looks for nearby speech evidence.
3. Places the event.
4. Commits the event.
5. Advances.

The runtime never rewrites committed events.

---

## Phrase Ownership

Meaningful pauses create phrase boundaries.

Ownership rules:

- Speech before a pause belongs to the earlier phrase.
- Speech after a pause belongs to the later phrase.

Guards:

### Phrase Start Guard

Prevents a phrase from being pulled backward across a pause.

### Phrase Tail Guard

Prevents a phrase from leaking into the following phrase.

These guards preserve conversational structure and sentence transitions.

---

## Comma Handling

Commas are treated as soft boundaries.

Short pauses may allow limited articulation carry across commas.

This improves:

- Lists
- Enumerations
- Natural conversational flow

Commas become stronger boundaries only when supported by speech timing evidence.

---

## Commit Horizon

The runtime commits only a limited distance into the future.

Benefits:

- Better adaptation to live audio timing
- Reduced rigidity
- Improved handling of variable TTS pacing
- Better pause handling

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

- Audio occupancy is the primary timing signal.
- Phone alignment acts as secondary placement evidence.
- Transcript-derived viseme identity remains authoritative.
- Runtime scheduling is incremental rather than global.
- FaceDriver remains a rendering-only system.
- Committed events are never reordered.

This architecture prioritizes stability, explainability, and streaming robustness over perfect offline alignment.
