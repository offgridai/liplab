# OffgridAI neural streaming lipsync

## Ownership

- The transcript planner owns phoneme/viseme identity, order, word membership,
  syllable structure, and duration priors.
- The neural aligner owns speech occupancy, pause/resume, word and syllable
  transitions, event centers, and event durations.
- The runtime session owns irreversible monotonic commitment.
- The performer owns pose-envelope sampling.

The runtime consumes no MFA labels, TTS token timing, hint streams, word
schedules, Python, LibTorch, or deterministic timing fallback.

## Pipeline

```text
transcript -> CMU phones -> ordered transcript-token lattice -----+
                                                              neural fixed-lag
streamed PCM -> 18 causal audio features -> native CUDA model ----+-> committed visemes
committed visemes + neural speech regions -> pose performer -> face weights
```

`FOffgridAITextVisemePlanner` resolves pronunciations and emits the complete
ordered lattice. Its priors are model inputs, not a second scheduler.
`FOffgridAIStreamingAudioFeatureExtractor` maintains only the causal acoustic
state required to reproduce the model's 18 training features. It does not
classify speech or propose events.

`FOffgridAINeuralStreamingAligner` sends audio and transcript features to the
LibTorch-free CUDA runtime and decodes token and region logits monotonically.
The neural occupancy head opens and closes speech regions. The token head
advances through transcript phones and determines placements and learned
within-word proportions. Fixed lag provides future acoustic context without
revising already committed history.

The first and last token states delimit each word. Their training targets extend
to the complete MFA word interval, while internal token transitions retain phone
and syllable supervision. The performer sustains each neural state through its
predicted interval with short presentation edges, so a well-timed schedule is
not visually shortened by long fades.

`FOffgridAILipsyncRuntimeSession` is the only runtime controller. It rejects a
missing or invalid checkpoint and produces no track rather than invoking an
alternate scheduler. `FOffgridAIVisemePerformer` samples the resulting track;
it does not repair alignment or choose identity.

## Invariants

- committed events are strictly monotonic and never reordered;
- every planned visible viseme is retained;
- audio cannot choose or invent viseme identity;
- one neural decoder owns speech regions and all event timing;
- failed inference remains neutral;
- input closure and audible playback completion are distinct lifecycle events.

## Host lifecycle

1. `BeginLine`
2. `PushAudioPCM16` for each decoded chunk
3. `Update(CurrentPlaybackSec)` throughout audible playback
4. `CloseInputStream` when no more PCM will arrive
5. continue `Update` while buffered audio drains
6. `Finalize` only when audible playback ends

LineCoach supplies transcript, PCM, and the audible playback clock. It must not
schedule, merge, prune, or repair visemes. Runtime packaging, transplantation,
and diagnostics are specified in
[docs/neural_runtime.md](docs/neural_runtime.md).
