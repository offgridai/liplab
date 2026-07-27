# LipLab

LipLab is the standalone corpus, training, and regression harness for OffgridAI's
neural streaming lipsync runtime. `offgrid_dropin` is the authoritative shared
C++ implementation; the harness streams real PCM through that implementation
and grades the performed animation against MFA references.

## Current runtime

There is one runtime path:

1. The transcript planner converts text into an ordered CMU phone/viseme lattice.
2. The audio feature extractor converts streamed PCM into 18 causal features at
   10 ms cadence.
3. The small native CUDA model predicts transcript-token alignment and neural
   speech occupancy.
4. A fixed-lag monotonic decoder commits viseme identity, placement, duration,
   speech regions, pauses, word starts, and syllable alignment.
5. The performer turns committed events into continuous pose weights.

Transcript-derived identity and order are authoritative. Audio and the neural
model determine timing, but cannot invent or reorder visemes. There is no
deterministic timing fallback: failed neural initialization produces neutral
animation for that utterance.

The deployable runtime uses the CUDA Runtime only. LibTorch and MFA are offline
training dependencies and are never part of the Offgrid runtime. See
[lipsync.md](lipsync.md) and [docs/neural_runtime.md](docs/neural_runtime.md).

## Layout

```text
offgrid_dropin/        Authoritative C++/CUDA runtime shared with OffgridAI
standalone_ue_shim/    Minimal Unreal compatibility layer for CMake
harness/               Corpus runner, dataset exporter, and CUDA trainer
scripts/               Corpus, training, grading, and integration utilities
inputs/                Transcripts, WAV corpus, recipes, and approved gold
docs/                  Current architecture, policy, and calibration data
outputs/runs/latest/   Generated diagnostics and comprehensive scorecard
```

## Verify

On Windows, run the required end-to-end check:

```bat
scripts\verify.bat
```

It builds the native runtime, validates gold and corpus recipes, streams all 750
cases through the packaged neural checkpoint, regenerates the scorecard, and
enforces regression thresholds. Manual grading after a run is:

```bat
python scripts\summarize.py
python scripts\check_grades.py
```

Useful runner options are `--preroll-ms`, `--chunk-ms`, `--tick-ms`, `--case`,
`--fast-batch`, `--full-diagnostics`, `--export-monotonic-dataset`, and
`--neural-checkpoint`. Diagnostic exports describe the same runtime path; there
is no offline prediction-CSV controller.

## Training and data

MFA supplies offline phone, word, speech-region, pause, and vowel timing labels.
It is not linked into inference. Gold maintenance uses:

```bat
python scripts\draft_gold.py --mfa-num-jobs 4
python scripts\check_gold.py --include-drafts
python scripts\export_gold.py
```

The reproducible Qwen data factory is documented in
[docs/speech_library_generator.md](docs/speech_library_generator.md). Neural
training and packaging use:

```bat
scripts\run_neural_streamer_training.bat C:\aitoolkit\libtorch-2.13.0-cu130
scripts\build_neural_runtime.bat
```

The first command exports the sequence dataset, trains, packages, replays the
native checkpoint, and grades that exact package. The second proves that the
deployable binary contains RTX 4090 machine code and has no LibTorch dependency.

## Scorecard

`outputs/runs/latest/alignment_summary.json` is comprehensive: missing reference
events receive explicit penalties rather than disappearing from timing means.
It reports both mean and median for boundary and timing errors, plus:

- speech-region start/end and pause cleanliness;
- performed word-animation onset and speech-region placement;
- complete word-to-region ownership;
- delivery completion, missing words, sentence completion, and ordering;
- within-word viseme duration proportions, run boundaries, and word-level total
  variation distance.

Acceptance policy and thresholds are in
[docs/regression_policy.md](docs/regression_policy.md).
