# liplab

Standalone lipsync lab for iterating on the OffgridAI lipsync core outside Unreal Engine.

`offgrid_dropin` is the authoritative lipsync source tree shared with OffgridAI. The standalone harness compiles and exercises that code directly.

## Goals

- Build locally with CMake, without UE.
- Drive the lipsync core from `inputs/transcripts`, `inputs/wav`, and approved `inputs/gold`.
- Emit inspectable logs for every case.
- Grade committed visemes against approved MFA-backed gold answers.
- Keep the iterated lipsync logic compatible with a thin Offgrid LineCoach adapter.
- Use transcript + PCM audio only; do not consume TTS hint streams, text-progress estimates, token indices, or predicted word schedules.

## Current Runtime Shape

The active shared runtime is intentionally simple:

- transcript plans one ordered phone/viseme chain with duration priors, syllables, strong-phone landmarks, and punctuation fences
- streamed PCM produces causal speech occupancy, quiet/resume evidence, syllable envelopes, and broad acoustic-class evidence
- one monotonic runtime cursor resolves strict punctuation pause/resume boundaries and soft syllable/strong-phone timing anchors
- accepted anchors rebase only the uncommitted suffix; committed events remain immutable

See [lipsync.md](C:\git\liplab\lipsync.md) for the current runtime contract.

## Layout

```text
offgrid_dropin/              Authoritative OffgridAI lipsync source snapshot and edit target
harness/                     Local command-line runner
inputs/transcripts/          One .txt per case
inputs/wav/                  Matching .wav files, PCM16
inputs/gold/                 Approved gold labels and metadata
outputs/runs/latest/         Generated planned/speech/committed/grade logs
scripts/verify.bat          One-command Windows build/run/grade verification
AGENTS.md                   Rules for Codex and other coding agents
lipsync.md                   OffgridAI architecture/spec
```

## Build

```bash
cmake -S . -B build
cmake --build build --config Release
build\Release\liplab_runner.exe .
```

With Ninja or Makefile generators, the executable may instead be `build\liplab_runner.exe` or `./build/liplab_runner`.

Optional streaming knobs:

```bash
build\Release\liplab_runner.exe . --preroll-ms 350 --chunk-ms 40
```

`--preroll-ms` controls the streamed audio lookahead and defaults to `350`. The same value is passed into the shared runtime as `PrerollSec`. `--chunk-ms` controls only how the harness feeds PCM; it does not define the detector's analysis window.


## One-command verification

On Windows:

```bat
scripts\verify.bat
```

This configures CMake, builds `liplab_runner`, runs the sample corpus, summarizes the latest run, and checks grade thresholds from `docs/grade_thresholds.json`.

## MFA-backed gold generation

The gold-label workflow is now:

- offline MFA-backed batch generation creates draft evidence under `outputs/`
- reviewed/approved cases are exported to `inputs/gold`
- the standalone harness simulates the streamed runtime path and grades only against approved gold

### Prerequisites

Expected local MFA layout:

```text
.conda/mfa/                         Local Conda environment containing MFA
.mfa/pretrained_models/acoustic/    Downloaded MFA acoustic models
.mfa/pretrained_models/dictionary/  Downloaded MFA dictionaries
```

Expected commands:

```bash
conda run -p C:\git\liplab\.conda\mfa mfa version
```

If MFA cannot write to your user Documents folder, ensure `MFA_ROOT_DIR` points at `C:\git\liplab\.mfa`.

### Regenerate the gold set

Generate batch draft annotations from the current transcript + WAV corpus:

```bash
python scripts/draft_gold.py --mfa-num-jobs 4
```

Validate draft packages and approved gold:

```bash
python scripts/check_gold.py --include-drafts
```

Export approved draft packages into `inputs/gold`:

```bash
python scripts/export_gold.py
```

The current gold package shape is:

- `inputs/gold/<case>/phones.csv`
- `inputs/gold/<case>/words.csv`
- `inputs/gold/<case>/speech.csv`
- `inputs/gold/<case>/boundaries.csv`
- `inputs/gold/<case>/metadata.json`

### Re-run the streamed harness against gold

```bash
python scripts/run_all.py
python scripts/summarize.py
python scripts/check_grades.py
```

This is the intended evaluation split:

- offline MFA batch creates the accepted gold answer
- streamed harness runs with `--preroll-ms 350 --chunk-ms 40`
- grading measures runtime error against offline gold

### Important constraint

MFA is offline-only. It must not be added to `offgrid_dropin` runtime scheduling logic. Transcript still owns viseme identity; MFA only improves offline timing evidence.

## Agent rules

1. Keep Offgrid LineCoach thin; scheduling logic belongs in `offgrid_dropin`.
2. Iterate lipsync logic in `offgrid_dropin`.
3. Preserve monotonic committed event order.
4. Transcript owns viseme identity; PCM audio only affects timing.
5. Do not add TTS hint-stream inputs, text-progress fields, token progress, or predicted word schedule dependencies.
6. Do not permanently suppress planned visible visemes.
7. Scheduling changes must improve or preserve aggregate grade.
8. Prefer deleting overlapping layers over adding fallbacks.
9. `offgrid_dropin/Public/Lipsync` and `offgrid_dropin/Private/Lipsync` are the authoritative lipsync boundary.

10. Run `scripts\verify.bat` before handing off code changes.
11. Keep the authoritative lipsync behavior in `offgrid_dropin` and route standalone verification through that same code.

See `AGENTS.md`, `docs/regression_policy.md`, `docs/offgrid_transplant_contract.md`, `docs/gold_dataset_plan.md`, and `docs/active_runtime_path.md` for the current automation and runtime contract.

## Outputs

Each case emits a generated diagnostic package under `outputs/runs/latest`. The primary artifacts are:

```text
planned.csv
planned_boundaries.csv
planned_words.csv
speech_regions.csv
committed.csv
commit_decisions.csv
runtime_boundary_state.csv
runtime_syllable_assignments.csv
grade.json
pause_safety.json
playback_health.json
```

Additional acoustic-evidence CSVs exist to explain detector and cursor decisions. They are diagnostics, not a second scheduler or a runtime dependency.
