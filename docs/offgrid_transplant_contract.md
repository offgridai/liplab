# Offgrid Transplant Contract

`offgrid_dropin` is the authoritative lipsync code shared with OffgridAI. The local harness exists to exercise and grade that code path directly.

## LineCoach boundary

LineCoach should only:

1. begin a lipsync line with transcript and audio format,
2. push streamed PCM audio,
3. advance playback time,
4. sample the committed pose map,
5. pass pose weights to FaceDriver.

LineCoach should not contain:

- viseme planning,
- phone/word scheduling,
- phrase ownership rules,
- speech-region merging,
- fallback viseme placement,
- TTS hint interpretation.

## Core ownership model

- Transcript owns viseme identity.
- PCM audio owns speech timing.
- Runtime scheduling owns event placement.
- FaceDriver owns rendering.

Any change that violates this ownership split should be rejected.
