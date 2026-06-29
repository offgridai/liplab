# LipLab / OffgridAI Lipsync Architecture

LipLab and OffgridAI share one lipsync core. LipLab is the offline streaming simulator and corpus evaluator. OffgridAI/Unreal is the real-time host. They may differ in I/O, host timing, diagnostics, and FaceDriver binding only.

## Runtime pipeline

```text
Transcript
  -> TextVisemePlanner
       -> normalization
       -> CMU + fallback phonemes
       -> MetaHuman viseme events
       -> word metadata
       -> prosody and duration budget
       -> one prior monotonic viseme timeline

Streaming PCM
  -> StreamingSpeechDetector
       -> feature frames
       -> first speech onset
       -> speech-active islands
       -> pauses

RuntimeAdapter / AudioOccupancyScheduler
       -> launches from first speech onset
       -> maps monotonic text playhead through speech-active audio
       -> holds during pauses
       -> drains remaining tail if needed
       -> exports audio-occupancy diagnostics

VisemePerformer
  -> FaceDriver / MetaHuman pose weights
```

## Shared-core ownership

Shared lipsync code lives here in LipLab:

```text
core/include/Lipsync/
core/src/Lipsync/
```

and here in OffgridAI:

```text
Source/Public/Lipsync/
Source/Private/Lipsync/
```

Timing policy belongs in this shared core.

Do not put lipsync policy in:

- `tools/liplab_runner/`;
- `scripts/`;
- Offgrid `UOffgridAILineCoach`;
- FaceDriver glue code.

Those layers may feed inputs, call the shared runtime, and export outputs only.

## First-class runtime concepts

- planned source viseme events;
- source event index;
- planned event center/window;
- speech-active audio island;
- audio feature frame;
- text playhead;
- committed event center/window;
- playback mode: `speech_active`, `pause_hold`, `tail_drain`, `final_flush`, or `fallback`;
- diagnostic CSVs.

## Active timing model

The runtime is an audio-occupancy text-playhead system.

```text
Text provides identity, order, salience, and prior timing.
Audio provides onset and speech-vs-pause occupancy.
Runtime advances the text playhead only when audio occupancy says speech is active.
```

There are no runtime punctuation gates. The scheduler does not assign text phrases to audio phrases. A comma, period, or hesitation has runtime meaning only if it appears in the audio occupancy signal as speech or pause.

## Key shared components

### `FOffgridAITextVisemePlanner`

Builds the text-derived viseme plan:

- normalized words;
- CMU/fallback phonemes;
- MetaHuman pose IDs;
- planned event centers/windows;
- salience and duration priors.

### `FOffgridAIStreamingSpeechDetector`

Consumes PCM chunks and produces:

- speech feature frames;
- first speech onset;
- speech-active islands;
- pause intervals.

### `FOffgridAIAudioOccupancyScheduler`

Builds the committed runtime track from:

- the full text viseme plan;
- detected speech-active audio islands;
- current playback time;
- preroll;
- stream-closed state.

It is punctuation-unaware and does not use landmarks.

### `FOffgridAILipsyncRuntimeSession`

Owns one line of runtime state:

- text plan;
- streaming speech detector;
- audio occupancy scheduler state;
- committed track;
- diagnostics.

### `FOffgridAIVisemePerformer`

Samples the committed viseme track and produces pose weights for FaceDriver/MetaHuman.

## Diagnostic expectation

Every runtime timing patch must leave enough diagnostics to answer:

- what were the planned source events?
- what speech-active audio islands were detected?
- which source events were committed?
- what committed center/window did each event receive?
- did the event commit during speech-active playback, tail drain, fallback, or final flush?
- did source event order remain monotonic?
- did any source event duplicate or disappear?
- how much text tail remained after detected speech-active audio ended?
