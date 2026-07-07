# Active Runtime Path

This repo's active lipsync path is intentionally narrow.

## Authoritative runtime flow

1. Transcript text is planned into phoneme-derived viseme events in `offgrid_dropin/Private/Lipsync/OffgridAITextVisemePlanner.cpp`.
2. Streamed PCM is analyzed into speech occupancy and speech-region boundaries in `offgrid_dropin/Private/Lipsync/OffgridAIStreamingSpeechDetector.cpp`.
3. Runtime scheduling in `offgrid_dropin/Private/Lipsync/OffgridAILipsyncRuntimeAdapter.cpp` commits visemes monotonically inside the currently active speech region.
4. The harness in `harness/main.cpp` exists only to build, run, export diagnostics, and grade that shared `offgrid_dropin` logic.

## Non-goals

The active path does not rely on:

- TTS hint streams
- token-progress ownership
- predicted word schedules
- external timing middleware
- acoustic selection of viseme identity
- overlapping fallback schedulers

## Compatibility notes

Some public types still expose compatibility aliases such as `PhraseIndex` or `IslandIndex`.
Those exist only to avoid breaking older Offgrid-side logging/export code.
They are not the conceptual model of the current system.

The current conceptual model is:

- `speech regions`
- `speech occupancy`
- `transcript-derived viseme identity`
- `monotonic region-local playback`
