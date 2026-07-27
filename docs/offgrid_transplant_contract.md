# Offgrid Transplant Contract

`offgrid_dropin` is the authoritative lipsync code shared with OffgridAI. The local harness exists to exercise and grade that code path directly.

## LineCoach boundary

LineCoach should only:

1. begin a lipsync line with transcript and audio format,
2. push streamed PCM audio,
3. close the PCM input only when no more chunks will arrive,
4. continue advancing audible playback time every tick while audio is still playing,
5. finalize only after audible playback has actually ended,
6. sample the committed pose map,
7. pass pose weights to FaceDriver.

Canonical runtime call sequence:

1. `BeginLine(...)`
2. `PushAudioPCM16(...)` for each streamed chunk
3. `Update(CurrentPlaybackSec)` every tick during playback
4. `CloseInputStream()` when the producer is finished
5. keep calling `Update(CurrentPlaybackSec)` during buffered/drain playback
6. `Finalize(CurrentPlaybackSec)` at true audible playback completion

Important distinction:

- `CloseInputStream()` means no more PCM will arrive.
- It does not mean playback is over.
- `Finalize(...)` is the only end-of-playback signal.
- Likewise, a detected speech-region end is discovered in ingested-audio time.
  It must not cause the region's remaining viseme suffix to be dropped until
  audible playback has actually reached that region end.
- LineCoach must not independently move, split, repair, or replace neural
  events. A failed neural session remains neutral.

LineCoach should not contain:

- viseme planning,
- phone/word scheduling,
- speech-region ownership rules,
- speech-region merging,
- alternate viseme placement,
- TTS hint interpretation.

The shared source set includes `OffgridAIAcousticEvidence.*`. The former
`OffgridAIOnlinePhoneAligner.*` files are obsolete and should not remain in an
Offgrid transplant.

## Diagnostic compatibility

Each captured line must record
`FOffgridAILipsyncRuntimeSession::GetImplementationVersion()` and
`GetDiagnosticSchemaVersion()`. The current shared source identifies itself as
`2026.07.26-neural-only-streamer-v17`, diagnostic schema `8`.

Diagnostics must record the selected backend. The only operational backend is
`NeuralCuda`; `Disabled` is a failure state that emits no committed track.

LineCoach should also not:

- stop calling `Update(...)` after `CloseInputStream()`,
- derive lipsync timing from queued-byte counters instead of the audible playback clock,
- finalize the shared runtime merely because synthesis/decoding has completed.

## Core ownership model

- Transcript owns viseme identity.
- PCM audio owns speech timing.
- Runtime scheduling owns event placement.
- FaceDriver owns rendering.

Any change that violates this ownership split should be rejected.
