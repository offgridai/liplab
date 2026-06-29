# Offgrid Portability

LipLab and OffgridAI share the same lipsync core. Timing policy belongs in the shared core.

## Shared code copied between LipLab and Offgrid

LipLab paths:

```text
core/include/Lipsync/
core/src/Lipsync/
```

Offgrid paths:

```text
Source/Public/Lipsync/
Source/Private/Lipsync/
```

Shared components include:

- `OffgridAITextVisemePlanner`
- `OffgridAIArticulationBudgeter`
- `OffgridAIStreamingSpeechDetector`
- `OffgridAIStreamingLandmarkDetector` as an optional helper only
- `OffgridAIAudioOccupancyScheduler`
- `OffgridAIVisemePerformer`
- `OffgridAILipsyncRuntimeAdapter`

If a shared file changes in LipLab, copy the corresponding full file into the Offgrid drop-in zip.

## Active runtime class names

New code should use:

```text
FOffgridAIAudioOccupancyScheduler
FOffgridAIAudioOccupancySchedulerState
FOffgridAIAudioOccupancyDiagnosticRow
```

Compatibility aliases may exist for older include paths, but new comments, docs, diagnostics, and patches should use audio-occupancy terminology.

## LipLab-only files

Do not copy these into Offgrid runtime:

- `tools/liplab_runner/`
- `scripts/`
- `docs/`
- generated CSV/log folders;
- offline analysis tools;
- test harness simulation code.

LipLab may simulate playback and summarize data, but runtime timing behavior under test must remain in the shared library.

## Offgrid-only files

Offgrid owns:

- Unreal component wrappers;
- `UOffgridAILineCoach` integration;
- audio capture/streaming adapters;
- log emission into Unreal debug folders;
- MetaHuman FaceDriver binding;
- editor/runtime module plumbing.

Offgrid code may feed audio and time into the shared runtime. It must not implement alternate lipsync timing behavior.

## Drop-in patch rule

For every LipLab patch that touches shared lipsync files, provide an Offgrid drop-in zip containing full replacements for corresponding shared files.

Be careful with Unreal headers. Do not overwrite real Unreal `UCLASS` headers with LipLab stubs. That breaks UnrealHeaderTool before normal C++ compilation.

## Host-specific behavior allowed

Allowed host differences:

- how audio chunks are collected;
- how playback time is measured;
- where logs are written;
- how pose weights are bound to MetaHuman controls;
- editor-only debug UI.

Not allowed host differences:

- different speech/pause timing policy;
- different text playhead advancement logic;
- LineCoach-side corrections that bypass shared runtime;
- FaceDriver-side retiming of viseme identity/order.

## Offgrid validation

When testing in Unreal, prefer logs that include:

- raw TTS WAV;
- `runtime_speech_islands.csv`;
- `runtime_audio_occupancy_diagnostics.csv`;
- `runtime_commit_events.csv`;
- `submitted_poses.csv`;
- FaceDriver peak/pose timing.

The user's human observation of Offgrid playback is a primary signal. If LipLab and Offgrid disagree, inspect the Offgrid WAV and emitted runtime diagnostics before tuning corpus metrics.
