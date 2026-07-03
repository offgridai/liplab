# Streaming CTC Alignment Thesis

## Thesis

The best research-grounded path toward "MFA-like" streaming alignment in this repo is **not** to mimic MFA's Kaldi pipeline directly. It is to replace the current heuristic pacing layer with a **transcript-constrained phone-CTC aligner** that runs on a rolling fixed-latency buffer and commits timings with a bounded revisable suffix.

In short:

- transcript text continues to own viseme identity
- acoustics own timing only
- speech occupancy remains a separate coarse signal for speech-region open/close
- phone timing comes from **CTC posteriors + monotonic forced alignment**
- committed viseme playback stays append-only and monotonic

This is the closest practical analogue to MFA that still fits a fixed `N`-ms streaming buffer.

## Why not stream MFA itself

MFA is an offline aligner built on a Kaldi-style alignment stack. That stack is excellent for gold generation, but it is not a causal runtime architecture.

Research-grounded reasons:

- MFA alignment is corpus/utterance-oriented and dictionary + acoustic-model based, not a low-latency streaming decoder.
- MFA docs explicitly state that `--single_speaker` disables speaker adaptation, and that `--uses_speaker_adaptation false` skips feature-space adaptation and second-pass alignment.
- Kaldi's online decoding docs explain why ordinary per-speaker CMVN is non-causal and must be replaced online by moving-window normalization.
- NeMo Forced Aligner exposes a separate `use_buffered_chunked_streaming` mode for CTC models, which is effectively an admission that the streaming solution is a different architecture.

So the right mental model is:

`MFA gold = offline reference`

`runtime replica = constrained online approximation`

## Research basis

The proposed solution stands on four established ideas.

### 1. CTC forced alignment already solves "known transcript + waveform"

Modern forced alignment can be expressed as:

- generate framewise token posteriors from an acoustic model
- run monotonic forced alignment against the known transcript token sequence
- merge repeated/non-blank tokens into token or word spans

That is exactly what the TorchAudio forced-alignment tutorial demonstrates, and it builds on the `ctc-segmentation` line of work.

### 2. Buffered chunked streaming is already a known operating mode

NeMo Forced Aligner documents a buffered chunked streaming mode with:

- chunked streaming inference
- a total rolling buffer
- CTC-only support

That is the strongest practical evidence that a CTC alignment architecture can be made to work under bounded lookahead.

### 3. Online decoding needs causal normalization and carried state

Kaldi's online decoding docs explain the core constraint:

- offline per-speaker normalization is non-causal
- online systems use moving-window / online normalization
- decoding advances incrementally as new features arrive

So a viable streaming aligner must carry forward state, not restart from scratch conceptually on every chunk.

### 4. Plain CTC is usable, but peaky

Recent forced-alignment work shows standard CTC can be too peaky for precise phone offsets, especially at phoneme granularity. The 2024 label-priors paper is important because it suggests the right mitigation:

- keep the CTC alignment formulation
- improve boundary quality by reducing over-blank / over-spike behavior

That maps directly onto our problem, because our largest remaining failures are timing and pacing, not transcript identity.

## Core design

## High-level architecture

Replace heuristic progress pacing with a three-layer timing stack:

1. `Speech occupancy detector`
   - Opens and closes coarse speech regions.
   - Detects speech onset, provisional end, reopen-within-window, and final close.
   - This remains a coarse acoustic gate, not a viseme scheduler.

2. `Streaming phone-CTC aligner`
   - Runs over the rolling fixed-latency buffer.
   - Aligns acoustic evidence to the **known transcript phone chain**.
   - Produces provisional phone spans, confidences, and alignment state.

3. `Monotonic viseme scheduler`
   - Converts aligned transcript phones into append-only absolute playback-time viseme events.
   - Never lets acoustics choose viseme identity.
   - Never reorders committed events.

That division preserves the repo's hard boundaries.

## The thesis in one sentence

The runtime should stop guessing "how far through the text we probably are" and instead maintain a monotonic alignment cursor over the known phone chain, using CTC posteriors from the rolling audio buffer to decide **where the cursor is and when each already-known phone should fire**.

## Alignment units

The alignment chain should be based on transcript phones, not words and not free acoustic classes.

Recommended chain:

- transcript
- G2P / lexicon expansion
- canonical phone sequence
- per-phone mapping to visible viseme event(s)

Important detail:

- invisible phones must still exist in the chain even if they do not render visible visemes
- otherwise timing collapses because the decoder can "teleport" between visible phones

This is a likely reason heuristics have struggled: visible-only scheduling throws away too much structure.

## Streaming algorithm

## 1. Acoustic model output

Use a streaming-capable acoustic model that emits per-frame posterior probabilities over:

- phone labels
- blank
- optional silence / nonspeech

Preferred form:

- phone-level CTC model

Fallback form:

- grapheme/subword CTC model for research prototyping only

Reason:

- phone-level alignment matches the transcript-to-viseme contract directly
- it avoids trying to infer phonemes from amplitude heuristics

## 2. Rolling buffer

For each update:

- append new audio to a ring buffer
- compute model emissions on the newest chunk, with overlap
- keep a rolling emission buffer aligned to audio time

Recommended conceptual layout:

- total audio buffer: configurable `N ms`
- revisable overlap/lookback: `80-200 ms`
- commit-safe prefix: oldest region whose alignment is unlikely to change
- provisional suffix: newest region that may still be revised

This is the fixed-lag pattern: decode with some hindsight, commit only the older part.

## 3. Transcript-constrained DP

Run monotonic forced alignment against the expected phone chain:

- input: rolling emission matrix over frames
- target: known phone sequence for the current line
- recurrence: CTC-style forward / Viterbi update with blank handling
- state carried across updates:
  - current phone index band
  - per-state score
  - best predecessor / traceback summary
  - committed prefix boundary

Crucially, this is **not ASR search**.

Because the transcript is already known:

- there is no lexicon search
- there is no language-model search
- there is no beam over arbitrary word hypotheses

The decoder only needs to decide how the current audio maps onto the already-known monotonic phone chain.

## 4. Bounded revision

Do not immediately freeze the newest alignment decisions.

Instead maintain:

- committed prefix
- provisional suffix

Rules:

- anything older than the revisable tail may commit
- anything still inside the revisable tail may be shifted or relabeled locally
- committed events remain immutable

This is the missing architectural piece in the current runtime.

It gives the aligner a small amount of hindsight, which is the key advantage MFA has that the current heuristics do not.

## 5. Convert aligned phones to viseme timing

Once phone spans exist:

- map transcript phone index -> transcript viseme event index
- emit render windows from aligned phone start/end/center
- preserve current append-only performer contract

Acoustics affect:

- onset
- offset
- duration
- local pause ownership

Acoustics do **not** affect:

- transcript order
- viseme identity
- whether a visible transcript phone conceptually exists

## How this should handle pauses better

MFA is better than the current runtime largely because it assigns silence with future context.

The streaming approximation should do the limited, causal version of that:

- occupancy detector proposes provisional end when energy/posterior speech drops
- region closes only if speech does not reopen within a short window
- punctuation can bias close thresholds, but does not own timing
- the CTC aligner can assign blank/silence frames inside the provisional suffix
- trailing phone offsets can contract when a likely upcoming silence becomes visible in-buffer

This is much stronger than today's "pace forward and hope" logic, but still causal.

## Recommended model strategy

## Best long-term target

A phone-CTC acoustic model trained or adapted for forced alignment.

Ideal properties:

- streaming or chunked inference
- fixed right-context
- phone vocabulary mappable onto the repo's lexicon / CMU-like phone set
- robust blank and silence behavior

Important caveat:

- off-the-shelf models rarely expose exactly the repo's local phone inventory
- expect to need an explicit inventory adapter rather than assuming native CMU/ARPABET compatibility

## Explicit Phase 0 model choice

The Phase 0 shadow evaluator should not leave the emission source abstract.

Recommended first backend:

- `Charsiu` forced aligner, specifically the English checkpoint `charsiu/en_w2v2_fc_10ms`

Why this is the best first candidate:

- it is explicitly phone-oriented rather than word- or grapheme-oriented
- it is already packaged as a forced aligner
- it operates at 10 ms frame resolution
- it is much closer to the repo's phone-timing problem than generic ASR aligners

Required constraint for this repo:

- build an explicit phone-inventory mapping layer between the model's output labels and the repo's transcript phone chain

That mapping layer should be treated as first-class Phase 0 work, not an afterthought.

## Practical near-term prototype

Use an existing CTC aligner stack offline and then shadow-stream it inside the harness:

- feed the corpus audio incrementally
- force alignment over a rolling emission window
- compare emitted spans to full-audio gold and to current runtime output

That lets us prove the algorithm before adopting a deployment model.

## Quality expectation relative to MFA

The goal of the streaming CTC shadow path is **not** "match MFA boundary accuracy."

Recent benchmark work still shows MFA 3.0 at or near the top across benchmark datasets, including comparisons against neural aligners. Charsiu is therefore best understood here as a causal phone-alignment backend for experimentation, not as a likely replacement for MFA-quality offline gold.

So the success criterion for this plan is:

- beat the current streaming heuristics
- preserve bounded latency
- remain auditable against MFA gold

It is **not**:

- achieve MFA-like offline boundary precision at 350 ms by assumption

## Important caveat: plain CTC boundary quality

CTC's peaky behavior is a real risk.

Expected mitigations:

- label-prior / blank-penalty tuning
- minimum-duration constraints at phone class level
- local posterior smoothing
- bounded suffix revision before commit
- confidence-aware conversion from phone span to viseme render span

The thesis is **not** that plain raw CTC is perfect. The thesis is that CTC + transcript-constrained monotonic DP + bounded revision is the right architecture family.

## Implementation spec for this repo

## Phase 0: diagnostics-only offline proof

Goal:

- prove that rolling-window transcript-constrained CTC alignment tracks MFA substantially better than the current heuristic runtime

Recommended files:

- `scripts/streaming_ctc_shadow.py`
- `grader/streaming_ctc_shadow_grade.py`
- `docs/STREAMING_CTC_ALIGNMENT_THESIS.md`

Inputs:

- existing corpus WAV files
- existing transcript / phone plans
- MFA gold words / visemes / speech regions

Outputs:

- `outputs/streaming_ctc_shadow/<case>/phone_alignment.csv`
- `outputs/streaming_ctc_shadow/<case>/commit_trace.csv`
- `outputs/streaming_ctc_shadow/<case>/region_trace.csv`
- aggregate shadow report

Metrics:

- speech region count match
- matched region start MAE
- matched region end MAE
- word onset MAE
- intra-word phoneme center MAE
- phoneme coverage / recall
- order violations
- seam error at overlap/commit boundaries

Success gate for promotion:

- materially better word onset and intra-word coverage than current runtime
- no increase in order violations
- evidence that overlap seams do not dominate error

## Phase 1: repo-local shadow interface

Add a new optional timing module without changing live behavior.

Suggested new types:

- `FOffgridAIStreamingPosteriorFrame`
- `FOffgridAIStreamingForcedAlignmentState`
- `FOffgridAIStreamingForcedAlignmentResult`

Suggested files:

- `offgrid_dropin/Public/Lipsync/OffgridAIStreamingForcedAligner.h`
- `offgrid_dropin/Private/Lipsync/OffgridAIStreamingForcedAligner.cpp`

Responsibilities:

- ingest audio feature frames or posterior frames
- maintain rolling DP state over transcript phone indices
- expose provisional aligned phone spans
- expose committed-prefix boundary

Non-responsibilities:

- choosing viseme identity
- playback sampling
- fallback word scheduling

## Phase 2: runtime shadow logging

Hook the shadow aligner into the runtime adapter but do not yet change scheduling.

Suggested diagnostics:

- `ctc_alignment_shadow.csv`
- `ctc_alignment_contract.csv`
- `ctc_alignment_vs_runtime.csv`

Per update log:

- playback time
- observed audio end
- region state
- current phone band
- top path score
- posterior margin
- provisional phone start/end
- committed-through phone index
- reason for commit or defer

This phase exists to answer one question:

`Does a proper streaming aligner produce usable timing state on this corpus at 350 ms?`

## Phase 3: use aligner timing for word starts and phone pacing

Only after shadow quality is acceptable:

- drive first-word onset from aligned first-phone onset inside the active region
- drive subsequent phone timing from aligned phone spans
- retain occupancy detector as speech-region gate
- retain monotonic append-only event track

Scheduling rules:

- no committed event may move backward
- no committed event order may change
- render windows derive from aligned phone spans plus small class-specific padding
- if confidence is too low, defer until age-out rather than invent timing from text progress

## Phase 4: tune for fixed-latency operation

Primary tunables:

- total buffer size `N`
- acoustic model right-context / algorithmic latency
- overlap/lookback size
- provisional suffix size
- blank penalty / label prior strength
- minimum phone duration
- class-specific render padding
- reopen window for speech-region closure

Recommended experiment grid:

- `N = 350, 500, 750 ms`
- model right-context = backend-specific measured value
- overlap = `80, 120, 160 ms`
- tail = `2, 3, 4 phones`
- blank penalty = `off, light, medium`

The purpose is to find the best latency/accuracy trade, not to optimize around heuristics.

Important interpretation rule:

- if a backend consumes `200-400 ms` of right-context internally, that budget must be subtracted before judging whether `N = 350 ms` is viable
- a failed `350 ms` cell may therefore indicate an infeasible model/latency pair, not a bad architecture

## Design constraints specific to liplab / offgrid

This proposal respects the repo contract:

- no TTS hint stream ownership
- no text-progress estimator
- no token-index pacing hacks
- no acoustic choice of viseme identity
- no overlapping fallback schedulers
- no rewriting of committed history

The only thing acoustics decide is:

- where the alignment cursor is
- when known transcript phones begin and end

That is exactly the right ownership split.

## Failure modes to expect

### 1. Peaky phone posteriors

Symptom:

- good onset, poor offset
- phones collapse to spikes

Mitigations:

- label priors / blank penalty
- suffix revision
- duration floor and local smoothing

### 2. Nonspeech inhalations and breaths

Symptom:

- false region opens
- over-segmentation near sentence breaks

Mitigations:

- explicit nonspeech / silence handling in occupancy stage
- do not allow breath-only evidence to advance phone-chain state

### 3. Compound-word / clitic mismatch

Symptom:

- alignment cursor stalls or jumps because transcript phones and acoustics disagree locally

Mitigations:

- preserve invisible phones in chain
- optional low-confidence skip/star behavior in shadow analysis only
- keep MFA gold as audit reference

### 4. Too-small buffer

Symptom:

- unstable phone boundaries
- poor pause ownership

Mitigation:

- model latency + overlap + suffix tail must fit inside chosen runtime budget

If `350 ms` is too small for a chosen acoustic model, that does not falsify the architecture. It means the model/latency pair is wrong.

## What success would look like

A successful implementation should improve metrics in this order:

1. speech-region segmentation agreement
2. first-word onset timing
3. phoneme coverage / recall
4. intra-word center timing

That ordering matters because a better local aligner should first solve **where speech begins and where the cursor is**, then solve finer-grained pacing.

## Recommendation

The strongest next step is:

1. build a shadow **streaming CTC forced-alignment evaluator** against the existing corpus
2. run it with fixed-lag overlap/commit semantics
3. compare it directly to MFA gold and current runtime
4. only if it clearly beats heuristic pacing, promote it into `offgrid_dropin`

That is the highest-confidence architectural bet available from current research.

## Sources

- Montreal Forced Aligner docs: <https://montreal-forced-aligner.readthedocs.io/en/latest/user_guide/workflows/alignment.html>
- Kaldi online decoding docs: <https://kaldi-asr.org/doc/online_decoding.html>
- NVIDIA NeMo Forced Aligner docs: <https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/tools/nemo_forced_aligner.html>
- TorchAudio CTC forced alignment tutorial: <https://docs.pytorch.org/audio/stable/tutorials/ctc_forced_alignment_api_tutorial.html>
- Charsiu repository: <https://github.com/lingjzhu/charsiu>
- Kurzinger et al. 2020, CTC-Segmentation: <https://arxiv.org/abs/2007.09127>
- Huang et al. 2024, Less Peaky and More Accurate CTC Forced Alignment by Label Priors: <https://arxiv.org/abs/2406.02560>
- Kim et al. 2021, Reducing Streaming ASR Model Delay with Self Alignment: <https://arxiv.org/abs/2105.05005>
- Liu et al. 2026, CTC-TTS: <https://arxiv.org/abs/2602.19574>
- McAuliffe et al. 2026, Montreal Forced Aligner and the state of speech-to-text alignment in 2026: <https://arxiv.org/abs/2606.18466>
