# AGENTS

This file is the top-level operating contract for coding agents working in LipLab. Read it before changing code.

## Purpose

LipLab is an offline lipsync experimentation lab for OffgridAI. The repository exists to:

- run the shared lipsync core against a fixed WAV + transcript corpus,
- measure planner, aligner, and performer behavior independently,
- validate small, explainable changes before they are copied back into OffgridAI.

The default posture is experimental discipline, not feature sprawl.

## Repository layout

- `configs/`
  - Runner configuration JSON such as `baseline_v26.json`.
- `core/include/`
  - Shared core headers and compatibility structs.
- `core/src/Lipsync/`
  - Effective lipsync implementation:
    - text viseme planner
    - articulation budgeter
    - streaming speech detector
    - streaming audio anchor aligner
    - viseme performer
    - runtime adapter
- `docs/`
  - Current architecture, scoring, portability, and agent guidance.
- `experiments/`
  - Baseline and experiment run outputs. Treat these as generated artifacts.
- `inputs/`
  - Fixture corpus. Do not edit unless explicitly asked.
- `scripts/`
  - Run orchestration and summarization utilities.
- `tools/liplab_runner/`
  - Offline runner executable entrypoint.

## Required reading order

For any non-trivial work, read these first:

1. `AGENTS.md`
2. `docs/architecture.md`
3. `docs/scoring.md`
4. `docs/offgrid_portability.md` if the change touches shared core APIs or shared structs
5. `docs/codex_instructions.md` for experiment workflow expectations

Historical assessment notes in `docs/text-viseme-plan-assessment-*.md` are reference material, not the canonical spec.

## Build and run

Build:

```bash
cmake -S . -B build_ninja -G Ninja
cmake --build build_ninja --config Release
```

If `cmake` is not on `PATH`, use a Visual Studio Developer Command Prompt or the Visual Studio bundled CMake/Ninja binaries.

Run the full corpus:

```bash
python scripts/run_all.py --config configs/baseline_v26.json --out experiments/runs/<run_id>
```

Direct runner invocation:

```bash
build_ninja/liplab_runner --config configs/baseline_v26.json --inputs inputs --out experiments/runs/<run_id>
```

Resummarize an existing run:

```bash
python scripts/summarize_run.py --run experiments/runs/<run_id>
```

## Hard invariants

These are the non-negotiable system invariants unless the user explicitly approves a spec change:

- Do not remove, skip, or reorder planned visemes.
- Every planned and budgeted viseme must be committed exactly once unless an explicit approved readability merge is documented.
- Committed playback centers must remain monotonic.
- `VisemePerformer` samples committed events; it must not become an aligner.
- Do not change OffgridAI integration assumptions as part of routine local tuning.
- Do not silently redefine score semantics. If scoring changes are part of the task, document the new definitions in `docs/scoring.md`.

## Current system model

The current system has three distinct timing layers:

1. planner output
   - chooses viseme sequence and rough text timing structure
2. budgeted/raw text timing
   - planner plus budgeting, anchored to detected speech start
3. final streaming timing
   - committed event centers after audio alignment

Keep those layers conceptually separate. Do not "fix" one layer by quietly moving behavior into another without documenting it.

## Current scoring model

LipLab no longer relies only on legacy proxy metrics. The important families are:

- invariants
  - `MissingCommittedEventCount`
  - `MonotonicityViolations`
- text timing quality
  - `RawTextTimingMAEMs`
  - `P90AbsRawTextTimingErrorMs`
  - `RawTextLeadBiasMs`
- final streaming alignment quality
  - `StreamingTimingMAEMs`
  - `P90AbsStreamingTimingErrorMs`
  - `StreamingLeadBiasMs`
- island and playability quality
  - `EventsCenteredOutsideSpeechIslandCount`
  - `MaxDistanceOutsideSpeechIslandMs`
  - `EarlyPhraseStartCount`
  - `CompressedEventGapCountUnder33ms`
- performer / FaceDriver peak quality
  - `PeakWindowCenterMAEMs`
  - `ExplosivePeakLeadErrorMAEMs`
  - `VowelPeakCenterMAEMs`
  - `WeakPeakRate`

Legacy metrics such as `CommitLeadMeanSec`, `EffectiveSegmentScaleMax`, and `PeakTimingMAEMs` still matter, but they are now secondary diagnostics rather than the sole truth.

## What good changes look like

Preferred changes are:

- local
- measurable
- reversible
- explainable in terms of one subsystem

Preferred tuning order:

1. planner/budgeter rules
2. speech-island anchoring and aligner clock fitting
3. performer envelope shaping
4. score definitions only when the task is explicitly scoring-related

Avoid large rewrites unless repeated local iteration has clearly failed.

## Required experiment discipline

After behavior changes:

- rebuild,
- rerun the full corpus,
- compare against the previous full-corpus run,
- report both wins and regressions.

Do not optimize one handpicked case and present it as a corpus improvement.

## Current known risk areas

These are the remaining system-wide risk zones future agents should watch closely:

- long declarative lines can still expose planner/alignment tail errors
- multi-sentence restart timing remains sensitive to speech-island detection
- cadence/list lines are a distinct timing regime
- legacy metrics can look worse even when direct timing metrics improve
- portability can be broken by convenience diagnostics added to shared structs

## Drop-in portability contract

For portability-sensitive work, treat this as a hard review contract:

- files under `core/src/Lipsync` and `core/include/Lipsync` should remain drop-in compatible with OffgridAI's `Private/Lipsync` and `Public/Lipsync` copies
- do not make shared Lipsync files depend on LipLab-only runner code, harness types, or extra local headers outside the intended drop-in surface
- do not make shared Lipsync files depend on fields that are absent from OffgridAI's current shared types such as `OffgridAILineCoach.h`
- prefer runner-side reconstruction of diagnostics over expanding the shared runtime surface
- avoid code patterns that often fail under stricter project warning settings, especially shadowed locals and accidental transitive-include dependence

When a task touches shared Lipsync files, assume the intended validation question is:

- if I copy `core/src/Lipsync/*` into OffgridAI `Private/Lipsync`
- and copy `core/include/Lipsync/*` into OffgridAI `Public/Lipsync`
- will it compile without needing unrelated manual edits?

If the honest answer is "no", document the break explicitly before merging.

## Portability review checklist

Before finishing portability-sensitive work, check:

1. did I add a new include dependency in a shared Lipsync file?
2. did I add or require a new shared-struct field?
3. did I move scoring/debug behavior from the runner into the shared core?
4. did I introduce shadowed locals, warning-prone constructs, or reliance on transitive includes?
5. would this still build if only the Lipsync public/private folders were copied into OffgridAI?
6. did I compare against `refcode/public/lipsync` and `refcode/private/lipsync` when relevant?

## Files and directories to leave alone

Do not modify these unless explicitly asked:

- `inputs/`
- `build/`
- `build_ninja/`
- `.venv/`
- generated experiment outputs under `experiments/`

## Documentation rule

If you change:

- score definitions,
- shared struct semantics,
- pipeline responsibilities,
- required outputs,
- or portability assumptions,

then update the relevant docs in the same task. Future agents should not have to reverse-engineer the truth from code.
