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

The current harness replay is the integration seam, not the final runtime.
The next implementation step is a live `IStreamingMonotonicAligner` inside the
Offgrid runtime, using the same persistent fixed-lag state without CSV files.

## Reporting contract

Every metric exposes coverage and matched-only plus all-reference mean, median,
p90, p95, and maximum. Missing visemes, word onsets, and intra-word events
receive a 180 ms penalty; missing syllable nuclei receive 100 ms; missing speech
regions receive 1000 ms. Extras remain explicit through precision and counts.

Speech regions use order-preserving overlap alignment. Reports distinguish
clean one-to-one matches, merges, splits, missing regions, and extra regions so
a single pause error cannot shift every later boundary pairing.

The current 750-case neural replay scores 0.9284 exact viseme match with 20.99 ms
matched-center mean, zero deterministic fallback, and zero ordering violations.
Overlap-aware speech-region recall/precision are 0.9345/0.9107; comprehensive
start and end means are 163.3/172.2 ms with 20/16.9 ms medians. These are research
scores, not evidence that live Offgrid inference is implemented.

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
