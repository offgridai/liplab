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
monotonically into the same tail.

The recovery is deliberately bounded:

- remaining events retain their identity and order,
- minimum tail spacing is 1 ms,
- an already-started word may finish no more than 40 ms beyond the causal
  speech end,
- a word that cannot satisfy that bound is handed to the next observed region.

This repaired the corpus splits in `friendship`, `park`, and `and` without
changing pause cleanliness or boundary agreement.

## Primary metrics

`scripts/summarize_alignment.py` produces one priority-ordered scorecard:

1. `region_start.success_rate`: the first viseme in each MFA speech region is
   assigned to that region and its center is within 100 ms of the corresponding
   MFA phoneme center.
2. `pause.clean_rate`: the duration-weighted share of MFA inter-region silence
   with no visible animation.
3. `word_start.success_rate`: each word's first viseme is assigned to the
   correct MFA region and centered within 100 ms of its first MFA phoneme.
4. `word_region_assignment.success_rate`: every planned event for the word was
   committed and every committed event maps to the word's MFA region.

Region- and word-start mean absolute errors remain alongside the success rates
as timing diagnostics. Event completion and event order are hard guardrails.
All other case-level measurements are diagnostic and do not define acceptance.

## Accepted corpus result

The accepted focused run covers 344 cases and 3619 MFA words:

| Metric | Result |
| --- | ---: |
| Region-start success (owned and within 100 ms) | 92.6% |
| Region-head MAE | 59.0 ms |
| Pause cleanliness | 97.66% |
| Word-start success (owned and within 100 ms) | 66.1% |
| Word-head MAE | 126.1 ms |
| Complete, strict word-to-region assignment | 98.26% |
| Event completion | 100.00% |
| Ordering violations | 0 |

Relative to the immediately preceding focused result, incorrect words fell
from 66 to 63, affected cases from 35 to 32, and split words from 3 to 0.
Region-head MAE improved from 60.4 ms to 59.0 ms and word-head MAE improved
from 126.6 ms to 126.1 ms. Pause cleanliness and runtime/MFA boundary agreement
did not regress.

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
`alignment_cases.csv`, and `alignment_words.csv` under `outputs/runs/latest`.
