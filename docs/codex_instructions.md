# Coding Agent Instructions

These instructions are for future coding agents working on LipLab / OffgridAI lipsync.

## Read first

1. `docs/system_alpha.md`
2. `docs/architecture.md`
3. `docs/runtime_alignment.md`
4. `docs/scoring.md`
5. `docs/offgrid_portability.md`

## Current status

The current runtime is an audio-occupancy text-playhead system.

Stable behavior:

- text planning creates one monotonic MetaHuman viseme timeline;
- streaming audio detects first onset and speech-active/pause intervals;
- text/viseme playhead advances while audio is speech-active;
- text/viseme playhead holds while audio is paused;
- punctuation is not a runtime gate;
- viseme source order is expected to remain monotonic;
- landmarks remain available as optional helper code but are not active timing authority.

Known remaining issue:

```text
The planned text timeline often runs at a different speed than actual TTS speech-active duration.
```

Typical symptom:

```text
late-line visemes drain or final-flush because speech-active audio ended before the text playhead reached them.
```

## Default workflow

1. Start from the user's exact current source tree.
2. Inspect code before patching.
3. State the thesis precisely.
4. Make the smallest patch that tests the thesis.
5. Build locally when possible.
6. For LipLab, return a git-style `.patch` that applies from the repository root.
7. For Offgrid, return a plugin/source-relative drop-in zip when shared or Offgrid files change.
8. If a patch fails to apply, regenerate it against the exact tree the user provides.

## Build

Typical LipLab build:

```bat
cmake -S . -B build -G Ninja
cmake --build build --config Release
```

If using another generator, keep the build command explicit in the response.

## Patch rules

- Put lipsync policy in the shared core.
- Keep LipLab and Offgrid shared files in sync.
- Keep patches cumulative on top of the last accepted patch unless the user provides a fresh tree.
- Do not silently rely on local files outside the user's supplied tree.
- Do not provide rootless patches: paths should apply from `c:\git\liplab`.
- Avoid LineCoach-only timing policy.
- Avoid liplab-runner-only timing policy.
- Delete failed experimental code instead of leaving disabled layers that confuse the runtime model.

## Non-negotiable runtime invariants

Future patches must preserve:

```text
source event order is monotonic
committed center time is monotonic
already committed events do not move
no duplicate source events are published
FaceDriver does not own timing policy
```

Before delivering any timing patch, verify:

- no source event duplicates;
- no source event order regressions;
- no non-monotonic committed centers;
- no new sticky/wrong-viseme artifacts in `submitted_poses.csv`;
- final-flush counts do not increase unless the patch explicitly targets finalization diagnostics.

## Safe next implementation target

The useful next behavioral target is **adaptive audio-occupancy playhead rate**.

Constraints:

1. Keep the runtime punctuation-unaware.
2. Do not repartition text over audio islands after publication begins.
3. Do not retime already-committed events.
4. Adjust only future playhead speed.
5. Use bounded rates, for example a modest slow/fast clamp.
6. Prefer no-op over duplicate/sticky viseme artifacts.
7. Export diagnostics showing playhead rate, remaining text span, observed speech-active time, and final-flush count.

## Do not repeat these mistakes

- Do not reintroduce punctuation gates or coalescing as runtime policy.
- Do not globally remap the entire line to speech-active duration after playback has begun.
- Do not partition future text into audio islands using incomplete streaming evidence.
- Do not reset scheduler state every update and republish the same source events.
- Do not use a publication cursor that deadlocks before first events enter the horizon.
- Do not implement landmark nudges as active timing authority without a separate, proven experiment.
- Do not claim a patch works unless logs prove the authoritative committed track changed as intended.

## Evaluation order

1. Hard invariants.
2. Event coverage and uniqueness.
3. Human perceptual observations from Offgrid.
4. FaceDriver/submitted pose logs.
5. Speech island reasonableness.
6. Tail-drain / final-flush counts.
7. Corpus summaries.

The user weights Offgrid logs and perceptual observations more highly than broad corpus metrics.
