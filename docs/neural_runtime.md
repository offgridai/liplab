# Neural runtime, training, and Offgrid contract

This is the operational contract for the current neural-only lipsync system.
Historical detector, evidence-surface, candidate-estimator, CSV-replay, and
deterministic scheduler designs have been removed.

## Deployable runtime

The runtime is the source under `offgrid_dropin` plus the version-5 checkpoint
at `offgrid_dropin/Private/Lipsync/Models/OffgridAINeuralStreamerV5.bin`.
The current shared implementation identifier is
`2026.07.29-jaw-composition-v37`, and its diagnostic schema is version 11.

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

The supported floor is compute capability 7.5 (Turing / RTX 20 series). The
default CUDA fat binary contains native `sm_75`, `sm_86`, and `sm_89` code for
RTX 20-, 30-, and 40-series GPUs. CUDA 12.8 or newer also adds native `sm_120`
code for RTX 50 series. The runtime uses ordinary FP16 storage with FP32 math;
it does not require Tensor Cores or an Ada-specific instruction path.

`scripts/build_neural_runtime.bat` audits every architecture supported by the
active toolkit, embeds the accepted checkpoint, runs real GPU inference, and
rejects a binary that imports LibTorch or the dynamic CUDA Runtime. The smoke
output identifies the selected GPU, compute capability, CUDA driver, and
statically linked CUDA Runtime version. An unsupported pre-Turing device fails
at initialization with its detected compute capability instead of a generic
kernel error. Native-image compilation is validated automatically; release
testing should still include real RTX 20- and 30-series hardware before claiming
those GPUs as performance-qualified.

## Training and grading

Run:

```bat
scripts\run_neural_streamer_training.bat C:\aitoolkit\libtorch-2.13.0-cu130
```

An optional second argument selects a controlled fine-tuning mode:
`--full-finetune`, `--acoustic-finetune`, `--identity-finetune`,
`--word-interval-finetune`, or `--phone-interval-finetune`. With no mode, only
the schema-v5 punctuation columns are adapted. Every mode still starts
from the packaged checkpoint and must win decoded validation selection before
its candidate is replayed for corpus acceptance.

This is one closed workflow:

1. build the harness, CUDA runtime, and optional LibTorch trainer;
2. stream every case in the unified `inputs/corpus.csv` inventory and export
   causal audio/token sequences;
3. fine-tune the accepted checkpoint against MFA phone, word, vowel, and
   speech-region labels, retaining the accepted model if validation does not
   improve;
4. export a versioned native `.bin` checkpoint;
5. replay that exact checkpoint through `FOffgridAILipsyncRuntimeSession`;
6. run the normal comprehensive scorecard and regression gates.

LibTorch is confined to step 3. There is no synthetic prediction CSV or
composition of independently trained heads. Text-group-aware splits keep voice
renditions of the same transcript together.

Schema v4 added the planner's transcript-derived pause-boundary flag to silence
token features. Schema v5 retains that semantic flag and adds raw punctuation
presence plus a one-hot punctuation type: comma, period, question mark,
exclamation mark, colon, semicolon, dash, or other. These features describe the
transcript boundary, never timing: acoustic evidence and the causal neural path
still decide whether and when a pause occurs.

The planner also retains `SoftListPause` versus `HardBreakPause` as contextual
metadata. V5 does not encode that distinction as separate learned columns: both
classes set the same neural pause-boundary feature. Outside the model, the
monotonic commit decoder uses the hard-boundary flag together with a lexical
comma and following-clause shape for one bounded fixed-lag safeguard. That
safeguard delays irreversible commitment while acoustic pause evidence can
arrive; it does not choose viseme identity or create a second schedule.

A controlled experiment that overloaded the learned pause scalar with
`list=-1`, `none=0`, and `hard=1` regressed pause and region behavior and was
rejected. A future checkpoint schema should add dedicated categorical columns
for hard sentence endings, clause endings, list commas, and ordinary commas
rather than changing the meaning of the accepted binary feature.

The accepted 841-case checkpoint currently scores:

- region-start success 0.963, 26.5 ms mean absolute error;
- pause cleanliness 0.979;
- word-animation-onset success 0.926, 33.6 ms mean absolute error;
- performed word-duration success 0.875, 57.3 ms mean and 40.0 ms median
  absolute error;
- runtime event completion 1.000 and word-region assignment 0.989;
- decoded viseme recall 0.968, precision 0.938, and 19.1 ms center MAE;
- presentation visibility 0.962 and no incomplete words.

The generated JSON contains full counts, coverage, and mean/median/tail values.
These figures are descriptive; `docs/grade_baseline.json` and
`docs/grade_thresholds.json` are the executable acceptance contract.

Word duration is measured from animation actually presented above the normal
visibility threshold, not merely from scheduled event envelopes. The performer
uses short 45 ms attack/release edges and sustains the neural state through its
predicted interior, while neural speech-region clamps still prevent animation
from crossing confirmed pauses. Training extends the first and last token target
to MFA word boundaries and weights both word-entry and word-exit transitions.
Phone-balanced training and selection additionally grade phone/key-viseme onset,
exit, coverage, zero overlap, and under-occupancy. Word-bounded pronunciation
substitutions carry medium-confidence timing evidence without replacing the
transcript-owned phone or pose.

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
the lipsync runtime does not generate a separate jaw target or carrier. When the
host expands simultaneously active poses into MetaHuman controls, it must treat
the central jaw as one physical degree of freedom and arbitrate the strongest
weighted authored jaw target instead of additively summing every pose's jaw
value. Bilabial authority may attenuate residual authored opening, but it must
not replace, invent, or independently animate the jaw.

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
