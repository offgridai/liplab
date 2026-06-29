# Agent Instructions

This repository is a standalone research harness for the OffgridAI lipsync core. It is designed for Codex and other coding agents to iterate safely without touching Unreal Engine code.

## Hard boundaries

- Do not add Unreal Engine, UObject, USTRUCT, FString, TArray, MetaHuman, FaceDriver, or LineCoach dependencies to `src/lipsync`.
- Do not reintroduce TTS hint streams, text-progress estimates, token indices, predicted word schedules, or external TTS timing ownership.
- Do not make audio choose viseme identity. Transcript-derived viseme identity is authoritative.
- Do not permanently suppress planned visible visemes as a scheduling shortcut.
- Do not reorder already committed events.
- Do not add overlapping fallback schedulers. Prefer one clear scheduler over layers of heuristics.

## Editable areas

Preferred edit targets:

- `src/lipsync/` — core lipsync algorithm and stable public data model.
- `harness/` — standalone local runner only.
- `grader/` — grading utilities and metrics.
- `scripts/` — local automation.
- `tests/` — unit/regression tests, if added.
- `inputs/` — small checked-in sample corpus only.
- `docs/` — design and task notes.

Avoid editing repo plumbing unless required:

- `CMakeLists.txt`
- `.gitignore`
- `README.md`
- `lipsync.md`

## Required verification

Before finalizing any code change, run:

```bat
scripts\verify.bat
```

On non-Windows systems, run:

```bash
cmake -S . -B build
cmake --build build --config Release
python3 scripts/run_all.py
python3 scripts/summarize.py
python3 scripts/check_grades.py
```

A change is not acceptable if it introduces:

- any monotonicity/order violation,
- new missing visemes on the checked-in sample corpus,
- worse mean absolute center timing by more than the configured tolerance,
- worse pause-boundary leakage once that metric is present,
- Unreal-specific dependencies in `src/lipsync`.

## Offgrid transplant contract

`src/lipsync` is the transplantable core. Offgrid LineCoach should only call the narrow lipsync API/facade and should not contain lipsync scheduling logic.

Keep the public core API stable unless the Offgrid integration contract is deliberately updated and documented.

## Coding style

- Keep the implementation small and readable.
- Prefer deleting cruft over adding switches.
- Prefer deterministic behavior over probabilistic or model-dependent behavior.
- Write logs that explain scheduling decisions in human-readable terms.
- Keep the harness useful for reviewing failures, not just for producing a score.
