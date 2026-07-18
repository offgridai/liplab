# Focused Region and Word-Start Alignment

This document describes the focused streaming-alignment path implemented on
`codex/region-pause-word-start-alignment` and merged into `main` in July 2026.

## Objective

The runtime is evaluated in this order:

1. align the center of the first animated viseme in every speech region to the
   center of its corresponding MFA/TTS phoneme,
2. stop visible animation during non-speech gaps between speech regions,
3. align the first animated viseme in every word to the corresponding MFA/TTS
   phoneme center,
4. improve later syllables only after the preceding constraints are stable.

MFA remains an offline grading authority. It does not run in, or provide hints
to, the streaming runtime.

## Runtime ownership

- The transcript and CMU pronunciation plan own phoneme and viseme identity.
- Streamed PCM owns speech timing and acoustic restart evidence.
- `OffgridAILipsyncRuntimeAdapter` is the only event scheduler.
- Committed events remain monotonic and are never reordered or suppressed to
  repair timing.

The standard harness is configured with 350 ms of preroll. Its playback gate
opens once the first speech onset is observed, once the preroll target is
filled, or when the stream closes. The evidence surface analyzes available
audio around the simulated playhead with a 350 ms forward/preroll allowance
and up to 1500 ms of retained history. The retained history supports
diagnostics and bounded correction; it is not an additional future-lookahead
window.

## Region and pause behavior

`OffgridAIStreamingSpeechDetector` supplies causal speech regions. A decoded
region onset becomes the local timing origin for the first uncommitted viseme.
When a region closes, the scheduler stops admitting new words to it and waits
for a decoded successor before resuming.

Comma-delimited lists use one temporary sensitivity mode:

- a sentence with at least six soft comma boundaries is a dense list,
- sensitivity begins as the text cursor enters that list,
- the learned detector requires 100 ms of qualifying quiet instead of the
  normal 140 ms,
- sensitivity returns to normal after the list's hard sentence boundary is
  acoustically resolved.

Audio still decides whether a gap exists. Commas only select detector
sensitivity and never supply a timestamp.

For non-list commas, continuous learned speech may keep the same region. Hard
sentence punctuation still requires an observed quiet-to-speech restart or a
decoded region handoff.

## Atomic word ownership

A word must not be divided across speech regions merely because its duration
prior reaches a closed region tail. Once any event from a word has committed to
the active region, the scheduler may compact the remaining events of that word
monotonically into the same tail, but only after a decoded successor region is
actually available. If the region closes before either a successor appears or
the PCM stream closes, the suffix remains pending. This prevents a provisionally
final region from consuming its endpoint before final-line completion knows how
many events remain.

The recovery is deliberately bounded:

- remaining events retain their identity and order,
- minimum tail spacing is 1 ms,
- an already-started word may finish no more than 40 ms beyond the causal
  speech end,
- a word that cannot satisfy that bound is handed to the next observed region.

When stream closure proves that the active region is final, one final-tail path
distributes the complete remaining transcript suffix monotonically over the
known audio tail. The per-word recovery path does not run first. This ordering
is required to prevent `final_speech_closed_with_unplayed_suffix` failures.

This repaired the corpus splits in `friendship`, `park`, and `and` without
changing pause cleanliness or boundary agreement.

## Primary metrics

`scripts/summarize_alignment.py` produces one priority-ordered scorecard:

1. `strict_region_segmentation.exact_boundary_rate`: every adjacent-word
   boundary must agree at the MFA, transcript-plan, and physical runtime levels.
2. `strict_three_level_word_assignment.success_rate`: every planned event for
   a word must commit, the complete word must occupy one runtime region, and
   transcript, physical runtime, and MFA must map that region identically.
3. `strict_region_nucleus_alignment.success_rate`: the first rendered pose of
   each MFA region must belong to the correct complete word and fall within
   50 ms of that word's first syllable nucleus.

`pause.clean_rate` remains the animation-hold guardrail. Legacy region-head and
word-head measurements remain diagnostic only. Event completion and event order
are hard guardrails.

## Accepted corpus result

The accepted focused run covers 350 cases and 3696 MFA words:

| Metric | Result |
| --- | ---: |
| Exact three-level boundaries | 3241 / 3346 (96.86%) |
| Cases with perfect three-level segmentation | 285 / 350 (81.43%) |
| Complete three-level word assignment | 3642 / 3696 (98.54%) |
| Region nucleus alignment within 50 ms | 510 / 847 (60.21%) |
| Region nucleus MAE | 74.1 ms |
| Pause cleanliness | 98.62% |
| Event completion | 100.00% |
| Ordering violations | 0 |

The six July 18 Offgrid captures are included as cases 0353 through 0358. They
add 77 words: all 77 pass complete three-level ownership, 70 of 71 boundaries
agree, and 12 of 15 region nuclei fall within 50 ms. The focused harness commits
all 261 planned events for these lines.

The corresponding live v7 logs expose an integration discrepancy that remains
open: three lines finish with `final_speech_closed_with_unplayed_suffix`, for a
combined 256 of 261 committed events, even though the exact stored WAVs complete
in the harness. A host capture with that terminal reason is not considered a
successful lifecycle result. Investigation should compare the live
`ObservedAudioBufferEndSec`, close/finalize ordering, and final playback clock
against the harness before changing scheduling policy again.

## Rejected experiment

A 120 ms detector mode for shorter comma lists reduced incorrect assignments
to 59 and undersegmented cases from 15 to 9, but it also reduced region-pair F1
from 92.3% to 92.0% and visible coverage from 78.9% to 78.6%. A 130 ms variant
retained the false-positive cost without most of the ownership gain. Neither
variant is present in the accepted implementation.

The remaining short-list failures require a higher-precision acoustic lull
classifier using restart quality and spectral continuity, rather than another
global duration threshold.

## Reproducing the focused report

Build and run the required regression suite first:

```bat
scripts\verify.bat
```

Then generate the focused diagnostic run:

```bat
build-ninja\liplab_runner.exe . --preroll-ms 350 --evidence-postroll-ms 1500 --focused-alignment
python scripts\summarize.py
```

The scorecard and review tables are written as `alignment_summary.json`,
`alignment_cases.csv`, `alignment_words.csv`, and `alignment_boundaries.csv`
under `outputs/runs/latest`.

`alignment_boundaries.csv` keeps distinct failure mechanisms separate for every
adjacent word boundary:

- whether MFA requires a pause boundary and whether the runtime produced one;
- punctuation class (`list_comma`, `standard_comma`, hard stop, or other);
- early post-pause admission and authoritative rendered-animation leakage;
- incomplete or split adjacent words;
- owned-word tail recovery use and whether recovered events enter the pause.

The corpus summary also reports recovery distortion guardrails: owned-tail
transitions compressed below 20 ms, commits beyond the normal 160 ms lead, and
events shifted at least 150 ms from their text-duration prior. These are audit
metrics, not acceptance thresholds, until the recovery path is redesigned.

## Shared implementation identity and Offgrid diagnostics

The authoritative transplant exposes an implementation version and diagnostic
schema version through `FOffgridAILipsyncRuntimeSession`. Hosts emit both values
so a captured log can be tied to the exact shared implementation that produced
it.

Offgrid's focused final diagnostics are:

- `diagnostic_summary.txt`: completion and whole-word runtime integrity totals.
- `word_region_integrity.csv`: punctuation context, planned/committed viseme
  counts, the complete runtime-region set, first/last event timing, maximum
  commit lead, and an integrity verdict for every word.
- `speech_regions.csv`: detector-owned region bounds and the first/last words
  assigned to each region.
- `boundary_scheduler_trace.csv`: compact scheduler state changes around waits,
  commits, and region handoffs.

These runtime-only files intentionally do not claim MFA correctness. MFA remains
the offline gold comparison in Liplab; Offgrid reports the evidence required to
grade the captured WAV afterward.

Word ownership is a scheduler invariant, independent of punctuation class. When
a decoded region has closed, any untouched word is admitted only if its complete
viseme sequence fits that region; otherwise the whole word moves to the decoded
successor. If that successor has not opened yet, the word remains untouched
until either speech resumes or the input stream conclusively ends. List commas,
standard commas, and hard punctuation still influence pause detection, but
never word indivisibility. Once the first event commits, a fail-closed guard
prevents any suffix from being published with a different runtime region.
If a region closes after a word has begun, the already-owned word's remaining
visemes are completed at that region's bounded tail before the scheduler may
advance to a successor region.
