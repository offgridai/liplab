# liplab

Standalone lipsync lab for iterating on the OffgridAI lipsync core outside Unreal Engine.

## Goals

- Build locally with CMake, without UE.
- Drive the lipsync core from `inputs/transcripts`, `inputs/wav`, and `inputs/handmade`.
- Emit inspectable logs for every case.
- Grade committed visemes against handmade Praat-quality answers.
- Keep the iterated lipsync logic compatible with Offgrid LineCoach while avoiding LineCoach edits.
- Use transcript + PCM audio only; do not consume TTS hint streams, text-progress estimates, token indices, or predicted word schedules.

## Layout

```text
src/lipsync/                 Standalone, agent-editable core and stable data model
harness/                     Local command-line runner
inputs/transcripts/          One .txt per case
inputs/wav/                  Matching .wav files, PCM16
inputs/handmade/             Matching handmade .csv labels
outputs/runs/latest/         Generated planned/speech/committed/grade logs
scripts/verify.bat          One-command Windows build/run/grade verification
AGENTS.md                   Rules for Codex and other coding agents
offgrid_dropin/              Current Unreal plugin lipsync source snapshot for transplant/reference
lipsync.md                   OffgridAI architecture/spec
```

## Build

```bash
cmake -S . -B build
cmake --build build --config Release
build\Release\liplab_runner.exe .
```

With Ninja or Makefile generators, the executable may instead be `build\liplab_runner.exe` or `./build/liplab_runner`.


## One-command verification

On Windows:

```bat
scripts\verify.bat
```

This configures CMake, builds `liplab_runner`, runs the sample corpus, summarizes the latest run, and checks grade thresholds from `docs/grade_thresholds.json`.

## Handmade CSV format

```csv
start,end,pose,word,confidence
0.120,0.210,22_MBP,welcome,1.0
```

Praat TextGrid support can be added later, but CSV is deliberately the first-class agent-friendly format.

## Agent rules

1. Do not modify Offgrid LineCoach.
2. Do not add Unreal dependencies to `src/lipsync`.
3. Preserve monotonic committed event order.
4. Transcript owns viseme identity; PCM audio only affects timing.
5. Do not add TTS hint-stream inputs, text-progress fields, token progress, or predicted word schedule dependencies.
6. Do not permanently suppress planned visible visemes.
7. Scheduling changes must improve or preserve aggregate grade.
8. Prefer deleting overlapping layers over adding fallbacks.
9. `offgrid_dropin/Public/Lipsync` and `offgrid_dropin/Private/Lipsync` are the Unreal transplant/reference boundary.

10. Run `scripts\verify.bat` before handing off code changes.
11. Keep the public API in `src/lipsync/LipsyncCore.h` stable unless the Offgrid transplant contract is updated.

See `AGENTS.md`, `docs/regression_policy.md`, and `docs/offgrid_transplant_contract.md` for the full automation contract.

## Outputs

Each case emits:

```text
planned.csv
speech_regions.csv
committed.csv
grade.json
```
