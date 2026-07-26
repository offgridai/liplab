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
- The audio encoder receives the existing causal 20-feature, 10 ms acoustic
  frames. Speech-detector decisions are features, not scheduling ownership.
- MFA phone, word, speech-region, and vowel-nucleus timings are training labels
  only. They are not inference inputs.
- Frozen train, validation, unseen-text, and held-out-speaker splits prevent
  generated variants of the same transcript from crossing text splits.

The C++/LibTorch CUDA trainer uses two causal temporal convolutions, transcript
token embeddings, duration-aware stay/advance Viterbi decoding, dense MFA
occupancy supervision, a rotating forward-sum loss, boundary-focused curriculum
weights, and deterministic fixed seeds.

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

- model weights are resident FP16 (54,608 bytes) with FP32 accumulation;
- transcript tokens are encoded once per utterance and cached as FP16 (6,912
  bytes in the measured benchmark utterance);
- inference batches four 10 ms acoustic frames and scores only the current four
  monotonic token candidates;
- four causal history frames are retained, so no growing audio tensor or model
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

The runtime benchmark compares every candidate logit against the LibTorch
reference before scoring. On the current CUDA system, 512 synchronized 40 ms
chunks measured 121.4 us mean, 121.0 us median, 123.6 us p95, and 185 us max.
Maximum FP16/reference logit error was 0.0095. Transcript setup measured 48.7 ms
once per utterance; this is not on the recurring audio-chunk path. The flat
checkpoint is 109,240 bytes and resident weights are about 53.3 KiB. These are
isolated-kernel measurements; TTS-coexistence latency still needs an integration
benchmark under actual synthesis load.

## Reporting contract

Every metric exposes coverage and matched-only plus all-reference mean, median,
p90, p95, and maximum. Missing visemes, word onsets, and intra-word events
receive a 180 ms penalty; missing syllable nuclei receive 100 ms; missing speech
regions receive 1000 ms. Extras remain explicit through precision and counts.

Speech regions use order-preserving overlap alignment. Reports distinguish
clean one-to-one matches, merges, splits, missing regions, and extra regions so
a single pause error cannot shift every later boundary pairing.

The current 750-case neural replay scores 0.9531 exact viseme match with 17.62 ms
matched-center mean and 25.23 ms all-reference mean, zero deterministic fallback,
and zero ordering violations. Validation, unseen-text, and held-out-speaker match
rates are 0.9512, 0.9483, and 0.9349. Overlap-aware speech-region recall/precision
are 0.9411/0.9469; comprehensive start and end means are 136.7/138.8 ms with
20/20 ms medians. Word-onset comprehensive mean is 59.1 ms. Syllable coverage is
0.9387 with a 20.9 ms comprehensive mean and 10 ms median. Held-out-speaker,
validation, and unseen-text syllable coverage are 0.9208, 0.9326, and 0.9343.
These are research scores, not evidence that live Offgrid inference is integrated
yet.

This improvement comes from symmetric 40 ms supervision bands around learned
pause/resume, word-start, and vowel-nucleus transitions, plus a 120 ms neural
silence-state region boundary calibrated on the frozen corpus. Region-boundary
flags remain training labels only and are deliberately absent from runtime token
features. The next accuracy work should prioritize syllable recall and the
remaining long-tail speech-start errors; broad viseme timing and word starts are
no longer the dominant failures.

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
