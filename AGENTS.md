# Agent Instructions

This repository is a standalone research harness around the OffgridAI lipsync core. Lipsync logic should be iterated in `offgrid_dropin`, while the local harness remains responsible for standalone build/run/grade flow.

## Hard boundaries

- Do not reintroduce TTS hint streams, text-progress estimates, token indices, predicted word schedules, or external TTS timing ownership.
- Do not make audio choose viseme identity. Transcript-derived viseme identity is authoritative.
- Do not permanently suppress planned visible visemes as a scheduling shortcut.
- Do not reorder already committed events.
- Do not add overlapping fallback schedulers. Prefer one clear scheduler over layers of heuristics.

## Editable areas

Preferred edit targets:

- `offgrid_dropin/Public/Lipsync/` — authoritative public lipsync interfaces shared with OffgridAI.
- `offgrid_dropin/Private/Lipsync/` — authoritative lipsync implementation shared with OffgridAI.
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
- divergence between the standalone harness behavior and the authoritative `offgrid_dropin` lipsync logic.

## Offgrid transplant contract

`offgrid_dropin` is the authoritative lipsync code shared with OffgridAI. The local harness should adapt around that code; it should not become an independent second implementation.

Keep the public lipsync API stable unless the Offgrid integration contract is deliberately updated and documented.

## Coding style

- Keep the implementation small and readable.
- Prefer deleting cruft over adding switches.
- Prefer deterministic behavior over probabilistic or model-dependent behavior.
- Write logs that explain scheduling decisions in human-readable terms.
- Keep the harness useful for reviewing failures, not just for producing a score.
