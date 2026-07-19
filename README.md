# liplab

Standalone corpus harness for the OffgridAI streaming lipsync core.

`offgrid_dropin` is the authoritative implementation shared with Unreal. The
harness streams WAV data through that code and grades its performed animation
against MFA-backed references; it does not contain another scheduler.

## Active design

The runtime has one path:

1. The transcript becomes ordered CMU phones, syllables, visemes, and relative
   intra-word durations.
2. Streamed PCM becomes causal speech regions and acoustic feature frames.
3. The evidence surface detects syllabic pulses and broad articulatory cues.
4. The candidate estimator offers nearby ordered transcript syllables for each
   stable pulse.
5. The scheduler anchors complete transcript-derived word packets to accepted
   audio pulses and adapts the pacing rate of future words.
6. The performer samples committed events into pose weights, gated by detected
   speech regions.

Transcript identity and order are authoritative. Audio owns timing but cannot
choose a viseme. Punctuation may disambiguate a boundary-final syllable, but it
cannot create or time a pause. There are no alternate controllers, TTS timing
hints, predicted word schedules, or revisions to committed history.

See [lipsync.md](lipsync.md) for the runtime design and
[docs/offgrid_transplant_contract.md](docs/offgrid_transplant_contract.md) for
the Unreal host boundary.

## Layout

```text
offgrid_dropin/        Authoritative shared C++ lipsync code
standalone_ue_shim/    Minimal Unreal compatibility layer for CMake
harness/               Streaming corpus runner and diagnostic exporters
scripts/               Gold generation, grading, and regression checks
inputs/transcripts/    Corpus transcripts
inputs/wav/            Matching PCM16 WAV files
inputs/gold/           Approved MFA-backed references
docs/                  Current contracts, baselines, and calibration data
outputs/runs/latest/   Generated run artifacts
```

## Verify

On Windows:

```bat
scripts\verify.bat
```

This builds the shared code, validates all approved gold packages, streams the
entire corpus with the default 350 ms preroll and 1500 ms retained postroll,
summarizes the priority metrics, and enforces the active-controller baseline.

Manual execution after a build:

```bat
python scripts\run_all.py
python scripts\summarize.py
python scripts\check_grades.py
```

The runner accepts `--preroll-ms`, `--evidence-postroll-ms`, `--chunk-ms`,
`--tick-ms`, `--case`, and `--full-diagnostics`.

## Gold corpus

MFA is offline-only and is never linked into the runtime.

```bat
python scripts\draft_gold.py --mfa-num-jobs 4
python scripts\check_gold.py --include-drafts
python scripts\export_gold.py
```

Approved packages contain phone, word, speech-region, pause-boundary, and review
metadata. See [docs/gold_dataset_plan.md](docs/gold_dataset_plan.md).

## Priority scorecard

`outputs/runs/latest/alignment_summary.json` reports:

- speech-region start and pause cleanliness,
- performed word-animation onset,
- nearby and exact word-head nucleus alignment,
- complete word-to-region assignment,
- completion, split-word, and ordering guardrails.

`alignment_cases.csv` and `alignment_words.csv` identify the corresponding
failures. Per-case diagnostics are regenerated on every run.
