# GPU timing model: oracle and data contract

This branch tests the proposed learned-timing control surface before adding a
CUDA dependency. The experiment is offline and MFA-backed. It does not alter
the authoritative runtime scheduler, viseme identity, event order, or commit
behavior.

## Oracle result

The oracle starts with the deterministic committed track. It preserves every
event's identity, duration, renderability, cancellation state, and order. For
events that can be matched to transcript-owned MFA labels, it shifts only the
event center. Shifts are bounded and constrained by adjacent committed events
and the scheduler's detected speech-region envelope.

Results on the 350-case corpus:

| Advice limit | Grade match rate | Matched center MAE | Outside-region events | Order violations |
|---:|---:|---:|---:|---:|
| Baseline | 0.613 | 61.3 ms | 443 | 0 |
| 50 ms | 0.673 | 39.3 ms | 71 | 0 |
| 100 ms | 0.713 | 30.6 ms | 53 | 0 |
| 200 ms | 0.748 | 22.3 ms | 50 | 0 |
| Oracle ceiling | 0.767 | 15.3 ms | 46 | 0 |

The 100 ms operating point is the initial model target. It nearly halves
matched-event timing error without weakening the scheduler guardrails. This is
a post-hoc oracle ceiling, not a claim that a causal model already achieves
these numbers.

The aggregate report is generated at
`outputs/runs/latest/oracle_timing_summary.json` by
`scripts/summarize_oracle_timing.py`.

## Causal training rows

The runner accepts `--export-timing-dataset`. For every renderable deterministic
event proposal that maps to MFA, it writes one row to
`timing_training_rows.csv` immediately before the proposal would become an
immutable commit.
Each row contains:

- the transcript-owned event and CMU phone;
- the deterministic proposal and scheduler state at commit time;
- a left-padded history of 32 acoustic feature frames, newest last, with every
  frame ending no later than `playback + preroll`;
- MFA center and raw/bounded timing-correction targets; and
- whether the MFA center was still ahead of the commit playback time.

Local feature flags that require future frames are excluded from the contract.
Generate rows for the complete corpus
with:

```bat
build-ninja\liplab_runner.exe . --fast-batch --export-timing-dataset
```

`harness/ml/OffgridAITimingModelContract.h` defines the fixed-shape C++ input
and the only permitted model output: center delta plus confidence. Evaluation
is once per proposed commit rather than once per audio tick. The model cannot
return a pose, phone, word, event index, or schedule.

## Frozen split

`inputs/speech_library/timing_split_v1.json` groups cases by normalized
transcript hash so duplicate text cannot cross the train, validation, and text
test partitions. Lana is held out from train-text groups as a separate speaker
generalization set.

Current case counts after adding the 100-transcript, four-voice Datafactory corpus:

- train: 415
- validation: 97
- text test: 98
- speaker holdout: 140

Regenerate or audit the split with
`python scripts/build_timing_dataset_split.py [--check]`.

## Next implementation slice

The first trainable model should consume 32 causal acoustic frames plus the
fixed scheduler/phone features and regress a center delta clamped to 100 ms,
with a confidence head. Training and inference should use the same LibTorch C++
module on CUDA. The runtime integrator remains deterministic: low confidence
means zero correction, committed events are immutable, and every accepted
correction is re-clamped for monotonicity and speech-region containment.

## Linear learnability baseline

`liplab_timing_baseline` is a dependency-free C++ trainer that reads the
causal rows, computes normalization statistics from the training partition
only, and compares scheduler-only, audio-only, and fused linear models. Run
the complete export and experiment with:

```bat
scripts\run_timing_baselines.bat
```

Held-out raw-residual MAE results (lower is better):

| Model | Validation | Text test | Speaker holdout |
|---|---:|---:|---:|
| No correction | 190.45 ms | 270.06 ms | 222.90 ms |
| Scheduler linear | 180.84 ms | 258.79 ms | 207.83 ms |
| Audio linear | 192.60 ms | 271.72 ms | 225.66 ms |
| Fused linear | 184.21 ms | 261.64 ms | 216.03 ms |

The scheduler baseline generalizes, but the flat audio model overfits: its
training result improves while every holdout becomes worse than applying no
correction. The fused model also loses to scheduler-only on every holdout.
Audio therefore needs temporal structure and stronger regularization; merely
adding the flattened history is not a viable model.

The baseline also writes `timing_tensor_dataset.bin`, a versioned, fixed-shape
C++ tensor artifact containing the split ID, 69 scheduler features, 32 x 20
audio history, bounded/raw targets, and actionability target for every row.

## Optional CUDA trainer

`harness/ml/timing_torch.cpp` defines the next nonlinear experiment: a small
two-layer temporal audio encoder fused with the deterministic scheduler state,
with bounded-delta and confidence heads. It trains and saves its checkpoint
entirely in C++.

LibTorch is deliberately optional so the normal lipsync harness has no ML
runtime dependency. Configure it only when a CUDA LibTorch package is
available:

```bat
cmake -S . -B build-torch -DLIPLAB_ENABLE_TORCH=ON -DCMAKE_PREFIX_PATH=C:\path\to\libtorch
cmake --build build-torch --config Release
build-torch\liplab_timing_torch.exe .
```

The standard build and `offgrid_dropin` remain unchanged when the option is
off.

With LibTorch unpacked at the default local path, the complete reproducible
experiment is:

```bat
scripts\run_timing_torch.bat
```

The CUDA trainer uses a fixed seed, selects its checkpoint using validation
raw-residual MAE, and evaluates the text and speaker holdouts only after that
selection. Deterministic CUDA execution is enabled; repeated runs produce
byte-identical reports. The expanded 750-case corpus run selected epoch 15:

| Model | Validation | Text test | Speaker holdout |
|---|---:|---:|---:|
| No correction | 190.45 ms | 270.06 ms | 222.90 ms |
| Scheduler linear | 180.84 ms | 258.79 ms | **207.83 ms** |
| Temporal fused CUDA | **176.66 ms** | **252.98 ms** | 208.05 ms |

The temporal model clears the scheduler-only linear baseline on validation and
unseen text and is effectively tied on the speaker holdout (0.22 ms worse).
Applying the current confidence head as a 0.5 gate makes every holdout worse,
so confidence is diagnostic only at this stage and must not yet control runtime
acceptance. The trainer writes both `timing_torch_model.pt` and
`timing_torch_report.json`.

## Normal-grader correction result

The trainer exports one causal correction for each case and committed event.
The harness can replay those predictions with `--timing-advice-csv`. Every
correction is clamped to 100 ms, constrained to preserve committed order, and
contained by the detected speech region before the ordinary MFA-backed grader
runs.

Across all 750 cases:

| Track | Match rate | Matched center MAE | Outside speech region | Order violations |
|---|---:|---:|---:|---:|
| Deterministic baseline | 0.573 | 63.4 ms | 1,004 | 0 |
| CUDA correction replay | **0.606** | **58.3 ms** | **109** | **0** |

The held-out partitions also improve under the normal grader:

| Split | Match rate | Matched center MAE |
|---|---:|---:|
| Validation | 0.634 -> **0.665** | 59.9 -> **57.7 ms** |
| Unseen text | 0.563 -> **0.588** | 59.8 -> **55.5 ms** |
| Held-out speaker | 0.567 -> **0.590** | 61.7 -> **58.3 ms** |

All 27,358 predictions were applied; 18.0% required additional constraint
clipping. The mean requested correction was 46.4 ms and the mean applied
correction was 42.0 ms. Completion and event identity are unchanged.

This is still post-hoc correction replay, not live model ownership of commit
timing.

## Intended neural scheduler

The target system is a neural scheduler, not a permanent residual-correction
layer. Transcript and CMU planning continue to define the complete ordered
viseme sequence. For the next uncommitted transcript-owned event, the model
uses causal audio, the timing plan, and deterministic scheduler state to decide
when that event should activate. The deterministic schedule is one input and
the failure fallback; it is not the primary timing decision.

The neural scheduler must still preserve identity, sequence order, immutable
commits, and speech-region containment. It may never invent, reorder, or
suppress transcript-owned visemes. The correction model and its normal-grader
result provide the baseline that this scheduler must exceed.

## Multi-task neural scheduler experiment

`harness/ml/neural_scheduler_torch.cpp` implements the first end-to-end
research version. It retains the causal temporal audio encoder but has three
outputs: an absolute event center relative to playback, a transcript-derived
pose class, and confidence. The replay interface accepts all three outputs and
applies the ordinary immutable-commit, monotonicity, and speech-region checks.
An identity prediction that disagrees with the authoritative transcript plan
is rejected rather than allowing audio to invent a mouth pose.

The experiment always trains two ablations:

- `transcript_audio` masks deterministic centers, anchors, pause state, and
  runtime scheduler state. It retains CMU phone/event priors, causal audio, and
  playback lead.
- `with_deterministic` adds the deterministic proposal as another input. Its
  deployment output is bounded to 100 ms and uses the deterministic scheduler
  as the safety fallback.

Run both CUDA trainers and the normal-grader replay with:

```bat
scripts\run_neural_scheduler_torch.bat
```

On the 750-case corpus, the deterministic-assisted model selected epoch 18.
Its raw placement target improves on every split and its pose accuracy is
99.98% or better. Normal-grader results are:

| Track | Match rate | Matched center MAE | Outside speech region | Order violations |
|---|---:|---:|---:|---:|
| Deterministic baseline | 0.573 | 63.4 ms | 1,004 | 0 |
| Multi-task scheduler replay | **0.613** | **63.0 ms** | **115** | **0** |

The placement result is not yet uniformly generalizable. Training improves
center MAE by 1.9 ms, but validation regresses by 2.9 ms, unseen text by 1.0
ms, and the held-out speaker by 0.7 ms. It therefore remains an offline
candidate and must not replace the existing CUDA correction model, whose
58.3 ms aggregate center MAE is still substantially better.

The `transcript_audio` ablation improves raw placement MAE on all holdouts but
its placement-selected checkpoint reaches only about 14% pose accuracy. This
shows that the current independent per-event classifier is not the intended
final scheduler. The next architecture should encode the complete ordered CMU
phone sequence once, use a causal sequence-state/event-pointer head for pose
selection, and train identity/order and placement with separate checkpoint
criteria. Until that clears the holdout gates, the deterministic plan remains
the primary identity source and fallback.
