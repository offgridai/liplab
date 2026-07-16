# liplab

Standalone research harness for the OffgridAI streaming lipsync core.

`offgrid_dropin` is the authoritative source shared with Unreal. The harness
adapts around that code; it does not contain a second scheduler.

## Runtime model

The product path has five stages:

1. `OffgridAITextVisemePlanner` converts the transcript into ordered CMU
   phones, visible visemes, syllables, and relative duration priors.
2. `OffgridAIStreamingSpeechDetector` extracts causal speech regions and
   acoustic feature frames from streamed PCM.
3. `OffgridAIStreamingEvidenceSurface` identifies syllabic pulses and a small
   set of broad phone-family observations.
4. `OffgridAILipsyncRuntimeAdapter` advances one append-only duration cursor
   while observed speech is available. Stable syllable evidence may move the
   uncommitted suffix by at most 120 ms.
5. `OffgridAIVisemePerformer` samples committed events into pose weights.

Transcript identity and order are authoritative. Audio affects timing only.
There are no punctuation schedulers, fallback playheads, TTS timing hints,
acoustic phone substitutions, or revisions to committed events.

See [lipsync.md](lipsync.md) and
[docs/active_runtime_path.md](docs/active_runtime_path.md).

## Repository layout

```text
offgrid_dropin/        Authoritative shared C++ implementation
standalone_ue_shim/    Minimal Unreal type compatibility for CMake
harness/               Streaming corpus runner and raw diagnostics
scripts/               Build, gold-generation, grading, and regression tools
inputs/transcripts/    Corpus transcripts
inputs/wav/            Matching PCM16 WAV files
inputs/gold/           Approved MFA-backed references
docs/                  Current contracts, baseline, and calibration data
outputs/runs/latest/   Generated run artifacts
```

## Verify

On Windows:

```bat
scripts\verify.bat
```

This configures and builds with Visual Studio CMake/Ninja, validates gold,
streams the complete corpus with a 350 ms preroll and 1500 ms retained
postroll, summarizes results, and checks regression thresholds.

Manual execution after a build:

```bat
python scripts\run_all.py
python scripts\summarize.py
python scripts\check_grades.py
```

The runner also accepts `--preroll-ms`, `--evidence-postroll-ms`,
`--chunk-ms`, and `--case`. Chunk size controls transport simulation only;
it does not define the acoustic analysis window.

## Gold workflow

MFA is offline-only and never ships in the runtime.

```bat
python scripts\draft_gold.py --mfa-num-jobs 4
python scripts\check_gold.py --include-drafts
python scripts\export_gold.py
```

Approved packages live under `inputs/gold/<case>` and contain phones, words,
speech regions, pause boundaries, and review metadata. See
[docs/gold_dataset_plan.md](docs/gold_dataset_plan.md).

## Primary generated artifacts

Each case directory under `outputs/runs/latest` contains:

- `planned.csv`, `expected_phones.csv`, and `planned_syllables.csv`
- `speech_regions.csv` and `gap_candidates.csv`
- `committed.csv` and `commit_decisions.csv`
- `runtime_boundary_state.csv` and
  `runtime_syllable_anchor_diagnostics.csv`
- evidence, matcher, runtime-health, playback-health, and final grade files

Generated case directories are replaced on every run so removed diagnostics
cannot survive as stale output.
