# OffgridAI Lipsync

## Contract

OffgridAI Lipsync is a deterministic streaming viseme scheduler.

- The transcript owns phone and viseme identity and order.
- PCM owns observed speech timing and optional syllable timing evidence.
- The runtime adapter owns placement and irreversible commitment.
- The performer owns pose sampling, not scheduling.

The runtime does not consume TTS token timing, predicted word schedules, hint
streams, MFA, Python, or external inference.

## Pipeline

```text
transcript
  -> text plan: phones, visemes, syllables, relative durations
streamed PCM
  -> speech regions and acoustic feature frames
feature frames
  -> syllabic pulses and broad phone-family evidence
text plan + audio evidence
  -> one monotonic duration cursor with bounded suffix correction
committed events
  -> pose weights
```

## Text planning

`FOffgridAITextVisemePlanner`:

- resolves CMU pronunciations, including supported alternatives and unknown-word
  fallbacks,
- emits every renderable transcript viseme in fixed order,
- groups phones into syllables around planned nuclei,
- assigns relative phone durations and small inter-word spacing,
- records punctuation and pause hints for diagnostics.

Punctuation does not create runtime fences or a second region schedule.

## Streaming analysis

`FOffgridAIStreamingSpeechDetector` consumes PCM incrementally and emits:

- speech-region opens, provisional closes, confirmed closes, and resumes,
- one causal acoustic feature frame per analysis step,
- refined gap candidates used to explain region decisions.

`FOffgridAIStreamingEvidenceSurface` examines the retained evidence window:
350 ms of upcoming preroll by default plus 1500 ms of recent postroll. It emits
syllabic pulses and conservative broad phone-family observations. A permissive
phone-family mode exists only for corpus evaluation of transcript conditioning.

Neither component chooses a transcript phone or viseme.

## Scheduling

`FOffgridAILipsyncRuntimeAdapter` owns one cursor.

1. Wait for the first observed speech region.
2. Anchor the next uncommitted event to that region's onset.
3. Advance through relative duration priors at the nominal rate.
4. Commit an event only when its center is inside the known speech frontier and
   no more than 160 ms ahead of audible playback.
5. If a stable monotonic syllable assignment becomes available, move the
   timeline anchor by at most 120 ms. Only uncommitted events are affected.
6. At a confirmed region end, stop committing into that region. Carry the next
   uncommitted transcript event to the next observed region.
7. If the final region closes before the plan is exhausted, leave the remaining
   suffix uncommitted rather than bursting it after speech.

Late recovery may move the uncommitted cursor forward to a small live lead. It
never reorders events or changes identity.

## Invariants

- committed events are append-only and strictly monotonic,
- audio never substitutes or suppresses transcript identity,
- no event is committed outside an observed speech region,
- one scheduler owns all placement,
- optional syllable correction is bounded and cannot affect committed events,
- stream closure and audible playback completion are separate events.

## Runtime lifecycle

1. `BeginLine`
2. `PushAudioPCM16` for every chunk
3. `Update(CurrentPlaybackSec)` throughout audible playback
4. `CloseInputStream` when no more PCM will arrive
5. continue `Update` while buffered audio plays
6. `Finalize` only when audible playback ends

See [docs/offgrid_transplant_contract.md](docs/offgrid_transplant_contract.md)
for the Unreal adapter boundary.
