# Neural runtime, training, and Offgrid contract

This is the operational contract for the current neural-only lipsync system.
Historical detector, evidence-surface, candidate-estimator, CSV-replay, and
deterministic scheduler designs have been removed.

## Deployable runtime

The runtime is the source under `offgrid_dropin` plus the version-3 checkpoint
at `offgrid_dropin/Private/Lipsync/Models/OffgridAINeuralStreamerV3.bin`.

The checkpoint contains a small causal token/region network. The native CUDA
runtime validates its schema and dimensions, retains model and transcript
encodings on the GPU, batches four 10 ms frames, and evaluates only the current
monotonic candidate window. It uses persistent buffers and a least-priority
nonblocking CUDA stream so operational TTS can retain GPU priority.

Deployment links the CUDA Runtime and C++ standard library only: no LibTorch,
cuDNN, Python, ONNX Runtime, MFA, or graph executor. The model can be loaded from
a file in development or embedded in an executable, DLL, Unreal pak, or resource.
Checkpoint or CUDA failure disables lipsync for that utterance; there is no
non-neural fallback.

The default CUDA fat binary contains native `sm_89` code for RTX 4090. CUDA 12.8
or newer also adds `sm_120` for RTX 5090. `scripts/build_neural_runtime.bat`
audits the archive for `sm_89`, embeds the accepted checkpoint, runs real GPU
inference, and rejects a binary that imports LibTorch.

## Training and grading

Run:

```bat
scripts\run_neural_streamer_training.bat C:\aitoolkit\libtorch-2.13.0-cu130
```

This is one closed workflow:

1. build the harness, CUDA runtime, and optional LibTorch trainer;
2. stream every case in the unified `inputs/corpus.csv` inventory and export
   causal audio/token sequences;
3. train against MFA phone, word, vowel, and speech-region labels;
4. export a versioned native `.bin` checkpoint;
5. replay that exact checkpoint through `FOffgridAILipsyncRuntimeSession`;
6. run the normal comprehensive scorecard and regression gates.

LibTorch is confined to step 3. There is no synthetic prediction CSV or
composition of independently trained heads. Text-group-aware splits keep voice
renditions of the same transcript together.

The accepted 750-case checkpoint currently scores:

- region-start success 0.9744, 18.0 ms mean absolute error;
- pause cleanliness 0.9801;
- word-animation-onset success 0.9357, 29.8 ms mean absolute error;
- performed word-duration success 0.8654, 58.8 ms mean and 40.0 ms median
  absolute error, with a 0.967 median runtime/MFA duration ratio;
- runtime event delivery 0.9984 and word delivery 0.9994;
- within-word run-share median error 8.36 percentage points, boundary-position
  median error 9.84 percentage points, and word-level duration total-variation
  median 20.00%.

The generated JSON contains full counts, coverage, and mean/median/tail values.
These figures are descriptive; `docs/grade_baseline.json` and
`docs/grade_thresholds.json` are the executable acceptance contract.

Word duration is measured from animation actually presented above the normal
visibility threshold, not merely from scheduled event envelopes. The performer
uses short 45 ms attack/release edges and sustains the neural state through its
predicted interior, while neural speech-region clamps still prevent animation
from crossing confirmed pauses. Training extends the first and last token target
to MFA word boundaries and weights both word-entry and word-exit transitions.

## Offgrid host boundary

`offgrid_dropin` is authoritative. The standalone harness adapts around it and
must never grow a second implementation. An Offgrid transplant copies the
shared Lipsync source set unchanged and compiles the CUDA `.cu` runtime.

LineCoach may only begin a line, push PCM, close input, advance the audible
playback clock, finalize after playback, sample pose weights, and pass them to
FaceDriver. It must not plan, move, split, merge, prune, repair, or replace
neural events.

The performer/FaceDriver boundary carries the complete detailed MetaHuman pose
map returned by the committed track, including supporting consonant and tongue
poses. `FOffgridAILipsyncPoseRuntimeState` smooths that dynamic map without
reducing it to generic open/closed/wide/round/funnel/teeth channels. Hosts should
submit the map produced by `BuildPoseWeightMapFromState` unchanged. Jaw motion is
owned entirely by the `CTRL_C_jaw` values authored in the viseme pose library;
the lipsync runtime does not generate a separate jaw target or carrier.

`CloseInputStream()` means that no more PCM will arrive. It is not an
end-of-playback signal. LineCoach must continue `Update(CurrentPlaybackSec)`
while buffered audio drains and call `Finalize` only at true audible completion.

Every capture records `GetImplementationVersion()`,
`GetDiagnosticSchemaVersion()`, checkpoint fingerprint, and backend. The only
successful backend is `NeuralCuda`; `Disabled` is explicit failure and must
remain neutral.

## Maintained utilities

- `verify.bat`, `run_all.py`, `summarize.py`, and `check_grades.py` implement
  normal regression verification.
- `draft_gold.py`, `check_gold.py`, `review_drafts.py`, and `export_gold.py`
  maintain MFA-backed references.
- `generate_speech_library.py` and the recipe/import scripts maintain the
  reproducible Qwen corpus.
- `run_neural_streamer_training.bat`, `build_neural_runtime.bat`,
  `check_cuda_architectures.py`, and `run_live_neural_smoke.bat` train and prove
  the production package.
- Offgrid log export/analyze scripts reproduce integration failures in LipLab.

Utilities for superseded controllers, offline prediction replay, independently
composed checkpoints, and obsolete evidence diagnostics are intentionally absent.
