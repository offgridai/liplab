# Neural-primary lipsync

## One target architecture

LipLab has one ML direction: a causal, duration-aware monotonic aligner that
advances through the transcript-owned CMU phone and viseme lattice. The active
lattice token selects identity; its transition time selects placement.

The neural path owns:

1. speech and silence occupancy;
2. speech-region open, close, and resume transitions;
3. word starts;
4. vowel-nucleus and syllable transitions;
5. the center and duration of every transcript-owned visible event.

Audio cannot invent viseme identity. The transcript planner constructs the
complete identity sequence before audio begins, and the monotonic graph cannot
delete or reorder it. The deterministic scheduler is not a model input. It
remains outside the model as the deployable emergency fallback until the
streaming aligner is integrated into `offgrid_dropin` and clears its promotion
gates.

## Inputs and supervision

- The transcript encoder receives phone, pose, vowel/consonant role, word
  position, visual role, strength, and corpus duration priors.
- The token and region encoders receive 18 raw causal acoustic features at 10
  ms cadence. Detector speech evidence, accumulated silence, and detector
  occupancy are not model inputs.
- MFA phone, word, speech-region, and vowel-nucleus timings are training labels
  only. They are not inference inputs.
- Frozen train, validation, unseen-text, and held-out-speaker splits prevent
  generated variants of the same transcript from crossing text splits.

The C++/LibTorch CUDA trainer uses a local token encoder and a separate 100 ms
causal region encoder. The region head receives dense MFA occupancy and boundary
supervision; its detached probability gates silence versus visible transcript
states without allowing alignment gradients to corrupt region learning. Token
placement retains duration-aware stay/advance Viterbi decoding, a rotating
forward-sum loss, boundary-focused curriculum weights, and deterministic fixed
seeds.

## Canonical workflow

Run:

```bat
scripts\run_neural_streamer_training.bat C:\aitoolkit\libtorch-2.13.0-cu130
```

That command performs exactly one experiment:

1. build the runner and CUDA trainer;
2. export the causal sequence dataset;
3. train one neural-streamer checkpoint;
4. produce one fixed-lag prediction stream;
5. replay it through the normal grader;
6. write the comprehensive frozen-split score report.

The current harness replay is the integration seam. The checkpoint now also has
a deployable C++/CUDA inference implementation; the next integration step is to
connect that implementation to a live `IStreamingMonotonicAligner` inside the
Offgrid runtime, using the same persistent fixed-lag state without CSV files.

## Co-resident GPU runtime

Operational TTS owns the valuable GPU budget. Lipsync therefore does not load
LibTorch, cuDNN, or a general graph executor at inference time. Training exports
a versioned flat checkpoint, and `NeuralStreamerCudaRuntime` executes two small
fused kernels using CUDA Runtime only:

- model weights are resident FP16 (85,706 bytes) with FP32 accumulation;
- transcript tokens are encoded once per utterance and cached as FP16 (6,912
  bytes in the measured benchmark utterance);
- inference batches four 10 ms acoustic frames and scores only the current four
  monotonic token candidates;
- ten causal history frames are retained, so no growing audio tensor or model
  replay is required;
- pinned host buffers and all GPU allocations are established outside the
  per-chunk path;
- work runs on a dedicated nonblocking CUDA stream at the device's least-urgent
  priority, allowing higher-priority TTS kernels to preempt between Lipsync's
  short launches.

The default CUDA fat binary contains native `sm_89` machine code for the RTX
4090. CUDA 12.8 and newer builds also include native `sm_120` for the RTX 5090;
older toolkits still produce the required 4090 image. The architecture list can
be overridden with `-DLIPLAB_CUDA_ARCHITECTURES=...`. The canonical training
workflow inspects the finished archive with `cuobjdump` and fails if `sm_89` is
absent, so 4090 support cannot silently regress even though the current machine
cannot execute that code path.

The runtime benchmark compares every candidate and region logit against the
LibTorch reference before scoring. On the current CUDA system, 512 synchronized
40 ms chunks measured 165.3 us mean, 164.6 us median, 168.6 us p95, and 442.7 us
max. Maximum FP16/reference logit error was 0.0094. Transcript setup measured
0.30 ms once per utterance; this is not on the recurring audio-chunk path. The
flat checkpoint is 171,436 bytes and resident weights are about 83.7 KiB. These
are isolated-kernel measurements; TTS-coexistence latency still needs an
integration benchmark under actual synthesis load.

## Reporting contract

Every metric exposes coverage and matched-only plus all-reference mean, median,
p90, p95, and maximum. Missing visemes, word onsets, and intra-word events
receive a 180 ms penalty; missing syllable nuclei receive 100 ms; missing speech
regions receive 1000 ms. Extras remain explicit through precision and counts.

Speech regions use order-preserving overlap alignment. Reports distinguish
clean one-to-one matches, merges, splits, missing regions, and extra regions so
a single pause error cannot shift every later boundary pairing.

The current 750-case neural replay scores 0.9468 exact viseme match with 17.62 ms
matched-center mean and 26.25 ms all-reference mean, zero deterministic fallback,
and zero ordering violations. Validation, unseen-text, and held-out-speaker match
rates are 0.9290, 0.9555, and 0.9370. Overlap-aware speech-region recall/precision
are 0.9861/0.9579; comprehensive start and end means are 58.6/56.8 ms with 10/10
ms medians. Word-onset comprehensive mean is 60.7 ms. Syllable coverage is
0.9324. Relative to the former refined detector's 0.9506/0.9689 recall/precision
and 118.4/107.5 ms comprehensive boundary means, the neural head has materially
better recall and timing means while trading some precision.
These are research scores, not evidence that live Offgrid inference is integrated
yet.

Region segmentation comes directly from the neural occupancy head with 20 ms
speech confirmation and 120 ms silence confirmation. This is a decoder
confirmation window over neural occupancy, not the removed silence-token
shortcut. Region-boundary flags remain training
labels only and are deliberately absent from runtime token features. The next
accuracy work should recover the remaining 0.63-point aggregate viseme tradeoff,
especially on the validation split, without weakening the new region head.

The corpus contains 80 explicit comma-delimited list recordings: 40 short and
40 long. With the 120 ms confirmation window, their comprehensive speech-region
recall/precision are 0.9785/0.9545, and start/end means are 60.7/90.8 ms with
10/10 ms medians. The 140 ms baseline scored 0.9391/0.9813 recall/precision and
118.7/113.4 ms means on the same slice. Core list scheduling was already strong
and is unchanged: viseme match is 0.9733, word-onset coverage is 1.0, and
syllable recall is 0.9587. An attempted list-oversampling curriculum was
rejected because it reduced long-list and unseen-text generalization; list
punctuation remains absent from runtime inputs.

Neural syllable grading compares the complete chronological neural-vowel stream
against the complete chronological MFA-vowel stream with a maximum-cardinality,
minimum-error monotonic alignment. A pair still must be within 100 ms. This
avoids shifting every later target when MFA inserts `spn`, reduces a word, or
selects a different pronunciation; it does not discard targets or observations.
The aggregate report currently includes 12,164 MFA vowel targets and 12,220
neural vowel observations.

## Promotion gates

- aggregate, validation, unseen-text, and held-out-speaker viseme match above
  0.90;
- speech-region recall and precision above 0.94 with comprehensive start/end
  means below 120/180 ms;
- syllable recall above 0.90 with comprehensive center mean below 40 ms;
- word-onset comprehensive mean below 75 ms;
- zero ordering violations;
- calibrated deterministic fallback below 5%;
- equivalent scores from the live C++ streaming integration, not CSV replay.

The next data-factory expansion should emphasize 120-600 ms mid-sentence
pauses, false-pause fricatives, stop/restart repairs, one-word regions, rapid
lists, vowel-initial resumes, and contrasting speaking rates.
