# Monotonic neural scheduler research direction

## Why the model formulation changes

The first CUDA experiment classified and placed one event after the
deterministic scheduler proposed it. That is useful as an ablation, but it
does not learn the central problem: a continuously advancing alignment between
causal speech frames and the complete transcript phone sequence.

The next model is a duration-aware monotonic forced aligner. The transcript
front end constructs the ordered CMU phone and viseme sequence once. At each
causal audio frame, the neural system scores `stay`, `advance`, and permitted
silence transitions in that fixed lattice. The active lattice state selects
viseme identity; the transition time selects placement. The transition graph
makes deletion, insertion, and reordering structurally impossible.

This follows the useful parts of:

- Glow-TTS monotonic alignment search: <https://arxiv.org/abs/2005.11129>
- forward-sum plus Viterbi alignment training in RAD-TTS:
  <https://arxiv.org/abs/2108.10447>
- online monotonic chunkwise attention: <https://arxiv.org/abs/1712.05382>
- label-prior CTC forced alignment, which addresses inaccurate peaky CTC
  boundaries: <https://arxiv.org/abs/2406.02560>

Plain unconstrained CTC is not the target. LipLab knows the transcript and
requires every transcript-owned visible event to remain ordered, so a small
explicit stay/advance lattice is simpler and safer.

## Measured ceiling

Run:

```bat
python scripts\summarize_monotonic_alignment_ceiling.py
```

On the 750-case corpus, the renderable transcript plan contains all 30,472
graded MFA visemes in the correct order. Ordered gold recall is 1.000. The plan
also contains 1,038 additional transcript-owned events, giving ordered plan
precision of 0.967. Every gold event has an exact unique match by word index,
word-local phone index, and pose.

This establishes that the earlier 0.738 unbounded oracle was a ceiling on
repositioning the deterministic system's committed events, not a ceiling on a
neural sequence scheduler. A match rate above 0.9 is structurally possible if
the aligner learns sufficiently accurate transitions.

With a conservative 40 ms decision delay, the fraction of phone centers that
can still be scheduled prospectively is:

| Audio preroll | Schedulable centers |
|---:|---:|
| 100 ms | 0.791 |
| 150 ms | 0.975 |
| 200 ms | 0.998 |
| 250 ms | 1.000 |
| 350 ms | 1.000 |

The current 350 ms harness preroll therefore does not impose a sub-0.9 ceiling.
This is a latency/identity feasibility result, not a trained-model score.

## Proposed model

1. A transcript encoder embeds phone identity, stress, word position,
   punctuation, viseme, vowel/consonant role, and corpus duration prior.
2. A small causal temporal convolution encoder consumes the existing 10 ms
   acoustic frames. No future frame may enter an inference call.
3. A pairwise scorer evaluates the active phone and a small forward window.
4. A persistent forward/Viterbi state applies duration-aware stay/advance
   transition priors.
5. A transition emits the transcript token's viseme and an audio-time center.
6. Low path confidence invokes the deterministic fallback. Committed events
   remain immutable and the existing containment/order checks remain final.

The deterministic center may be supplied as an optional log prior. During
training it must be independently dropped and jittered so the primary model
cannot simply learn another correction layer.

## Training targets and losses

- MFA phone spans provide dense frame occupancy and boundary targets.
- Forward-sum loss trains all legal monotonic paths rather than one brittle
  hard path.
- Viterbi extraction supplies explicit transition timestamps for diagnostics.
- A boundary loss is weighted toward visually important phones.
- A duration likelihood regularizes short or acoustically ambiguous phones.
- A fallback-confidence loss is calibrated from whether the neural path is
  inside the grader's 180 ms matching window.

Checkpoint selection must use held-out sequence match and boundary error, not
only per-row regression MAE. Text and speaker holdouts remain separate.

## Promotion gates

The research model is promoted only when all of these hold:

- aggregate match rate exceeds 0.90;
- validation, unseen-text, and held-out-speaker match rates all exceed 0.88;
- no ordering violation or missing transcript-owned visible event;
- center MAE improves over the current 58.3 ms correction model;
- deterministic fallback is below 10% overall and is reported by split;
- the no-deterministic ablation remains above 0.85 match.

These remain promotion gates, not a claim that the streaming implementation
has passed them.

## First CUDA sequence experiment

The initial LibTorch implementation is intentionally small: two causal
temporal convolutions, transcript-token embeddings, pairwise frame/token
scores, duration-aware stay/advance transitions, dense MFA occupancy
supervision, and a rotating forward-sum objective. Training and decoding are
fixed-seed C++/LibTorch CUDA. MFA centers are not used to crop inference audio;
the existing causal speech-detector feature defines the buffered speech span.

The independent normal grader produced these no-deterministic results across
the frozen 750-case split:

| Decode | Match | Center MAE | Fallback | Order violations |
|---|---:|---:|---:|---:|
| Whole-utterance end-constrained diagnostic | 0.9583 | 22.93 ms | 0% | 0 |
| 350 ms fixed-lag streaming | 0.8122 | 23.05 ms | 0% | 0 |

The whole-utterance diagnostic also reaches 0.9385 on unseen text and 0.9525
on the held-out speaker. It is not a deployable streaming score: its backtrace
requires the utterance endpoint. Its purpose is to isolate model learning from
online path-state estimation. The model has learned accurate boundaries when
it remains on the correct transcript state; the fixed-lag decoder sometimes
slips by one state, causing a run of exact-identity misses.

The first optional deterministic-prior/fallback experiment scored 0.6591 with
a 62.5% fallback rate. Its per-token confidence threshold was plainly
uncalibrated and is retained only as a negative ablation. Deterministic
substitution must not enter the primary path until confidence is calibrated at
the sequence level and the fallback rate is below the promotion gate.

The next model iteration should therefore target the online state estimator,
not larger acoustic convolutions: train an explicit stay/advance posterior,
maintain multiple monotonic hypotheses over the fixed-lag window, and calibrate
fallback from path margin and hypothesis stability rather than a single-frame
softmax value.

## Pause and word-onset training

The second streaming experiment adds transcript-owned, non-rendering silence
states before speech, between every pair of words, and after speech. MFA word
spans supervise their training duration, but inference receives only legal
silence candidates rather than MFA timing. Silence states never produce a
rendered event. Word starts, region starts/ends, and silence frames receive
higher occupancy weight so their small number is not overwhelmed by ordinary
intra-word frames.

This simple formulation raised the 350 ms fixed-lag, no-deterministic result
from 0.8122 to **0.9167** aggregate match, with 25.44 ms matched-center MAE,
zero fallback, and zero ordering violations. Validation scores 0.9296, held-out
speaker 0.8988, and unseen text 0.8892. First-viseme word-onset error fell from
235 ms to 86 ms; intra-word match rose from 0.8247 to 0.9154.

Pause containment improved but remains the weak metric. Events outside their
assigned speech region fell from 37.7% to 33.1%, and early leakage count fell
from 3,175 to 1,115. Late-tail leakage remains high at 9,312 events. The
speech-region detector's own 95.3% region match, 99 ms start MAE, and 240 ms end
MAE do not change when only the neural event track is replaced.

An additional explicit boundary-margin loss was rejected: it left aggregate
match flat at 0.9166, slightly worsened word onset, and reduced unseen-text
match to 0.8816. The retained model is the smaller weighted silence-state
formulation.

## Neural-owned streaming regions and curriculum

The next replay removes the deterministic speech detector from neural grading.
Every silence state is emitted in sequence order. A decoded silence lasting at
least 100 ms closes the current neural speech region; the next visible token
opens or resumes one. Word starts and vowel nuclei are likewise taken directly
from neural token transitions. MFA region-start/end labels are supervision
only and are excluded from inference token features.

The boundary/vowel curriculum documented in
`docs/neural_streamer_training_plan.md` produces:

| Metric | Result |
|---|---:|
| Aggregate exact viseme match | 0.9286 |
| Validation match | 0.9293 |
| Unseen-text match | 0.9124 |
| Held-out-speaker match | 0.9084 |
| Matched center MAE | 22.57 ms |
| Neural speech-region match | 0.9428 |
| Neural region start/end MAE | 245 / 483 ms |
| Word-first-viseme start MAE | 80.31 ms |
| Syllable/vowel-nucleus precision and recall | 0.8139 / 0.8176 |
| Syllable center MAE | 24.63 ms |
| Events outside neural regions | 1.04% |
| Deterministic fallback / ordering violations | 0 / 0 |

The 100 ms pause decoder is the best balance in the measured sweep. At 80 ms,
region recall rises to 0.9478 but predicted regions over-segment 1,952 versus
1,801 references. At 120 ms, predicted count is almost exact at 1,799 but
region match falls to 0.9323. Region boundary MAE and syllable recall are the
next training targets; they no longer borrow deterministic runtime decisions.
