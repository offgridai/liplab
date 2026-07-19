# OffgridAI lipsync

## Ownership

- Transcript planning owns phoneme and viseme identity, order, syllables, and
  relative timing inside a word.
- Streamed PCM owns speech regions and acoustic timing evidence.
- The runtime session owns irreversible event commitment.
- The performer owns pose-envelope sampling.

The runtime does not consume MFA, TTS token timing, hint streams, predicted word
schedules, Python, or external inference.

## Pipeline

```text
transcript -> CMU phones -> syllables -> ordered viseme packets
streamed PCM -> speech regions + feature frames -> stable syllabic pulses
viseme packets + ordered pulse candidates -> one audio-paced scheduler
committed visemes + speech-region gate -> pose weights
```

## Transcript plan

`FOffgridAITextVisemePlanner` resolves CMU pronunciations, emits every planned
phonetic event in fixed order, groups phones around vowel nuclei, and assigns
relative phone-duration priors. A small perceptual filter determines which
events animate, while all phonetic events remain available for correspondence.

Punctuation is metadata. At a punctuation boundary it may break an otherwise
ambiguous candidate tie in favor of an unfinished syllable of the active word.
It never asserts that a pause exists or supplies pause duration.

## Audio analysis

`FOffgridAIStreamingSpeechDetector` consumes PCM incrementally and publishes
causal speech-region opens, provisional closes, confirmed closes, resumes, and
feature frames.

`FOffgridAIStreamingEvidenceSurface` analyzes the retained audio window. The
default host configuration exposes 350 ms of preroll ahead of audible playback
and retains 1500 ms behind it. The surface emits stable syllabic pulses and broad
articulatory evidence without selecting transcript identity.

`FOffgridAIStreamingSyllablePositionEstimator` returns a small ordered candidate
set for each pulse. It does not maintain an alternate global alignment or a
historical replay controller.

## Scheduling

`FOffgridAILipsyncRuntimeSession` runs one scheduler:

1. Wait for a stable pulse owned by a detected speech region.
2. Match it monotonically to the next plausible transcript syllable.
3. When it starts a word, center that word's first rendered viseme on the
   accepted pulse and commit the complete word using relative duration priors.
4. Learn an inter-word pace from observed anchors; apply the updated rate only
   to future words, preserving the relative pacing within committed words.
5. If word B starts before word A finishes, terminate A's still-unplayed tail
   and launch B cleanly. Played history remains immutable.
6. If a punctuation-final multisyllabic word has a plausible unfinished
   syllable in the current candidate set, consume that syllable before allowing
   the next word to start.
7. At stream end, permit only the bounded, explicitly diagnosed final-word
   recovery supported by an unmatched pulse in the final audio region.

Every word is atomically owned by one detected speech region. Pose envelopes are
clamped to that region, so confirmed inter-region gaps remain neutral.

## Invariants

- committed events remain strictly monotonic,
- audio never chooses, reorders, or permanently suppresses transcript visemes,
- a word cannot be split across speech regions,
- one scheduler owns all placement and recovery,
- punctuation cannot create pause/resume timing,
- stream closure and audible playback completion are distinct lifecycle events.

## Host lifecycle

1. `BeginLine`
2. `PushAudioPCM16` for each decoded chunk
3. `Update(CurrentPlaybackSec)` throughout audible playback
4. `CloseInputStream` when no more PCM will arrive
5. continue `Update` while buffered audio drains
6. `Finalize` only when audible playback ends

The begin input contains dialogue, NPC/line identifiers, and preroll only. There
are no controller-selection flags. See
[docs/offgrid_transplant_contract.md](docs/offgrid_transplant_contract.md).

## Metrics

The regression scorecard follows the active design:

- speech-region boundary agreement and pause leakage,
- first performed viseme versus word onset,
- nearby and exact word-head nucleus correspondence,
- complete word-to-region ownership,
- event completion, split words, and ordering violations.

Candidate-set recall is retained because it determines whether the scheduler had
the correct syllable available. Scores for deleted alternate controllers and
offline matchers are not retained.
