# Neural intra-region alignment investigation

## Scope

This branch adds the 18 Offgrid recordings captured on 2026-07-29 as cases
0413-0430, expanding the graded corpus from 804 to 822 cases. The motivating
failures are word-slot errors inside otherwise correct speech regions,
especially `I'm` in the greeting and the second `to` in the synchronization
line.

## Detection

The previous word-onset aggregate did not adequately expose individual words
whose animation landed wholly outside the spoken interval. The grade now also
measures signed onset and exit error, spoken-interval coverage, animation
precision, interval IoU, zero-overlap words, low-coverage words, and onsets
later than 200 ms. A phonetic presentation report separately measures vowel
and bilabial dominance, cross-word intrusion, bilabial peak timing, and OW
saturation.

On the packaged V5 baseline, `I'm` in case 0413 is 300 ms early and has zero
overlap. The second `to` in case 0416 is 200 ms early and also has zero overlap.

## Root cause

The streaming Viterbi decoder has a phone-specific hard dwell ceiling of
60-300 ms. The final vowel in `hello` is spoken for roughly 430 ms in the
affected rendition. At 300 ms the decoder is forced to advance regardless of
the acoustic score, so the following pause and `I'm` tokens are consumed before
their audio arrives. The same mechanism occurs around a sustained vowel and
pause before the later `to`.

This is not a speech-region ownership failure. It is an intra-region token
occupancy failure caused by a deterministic ceiling overriding the neural
emission evidence.

## Experiments

| Candidate | Focused result | 822-case result | Decision |
|---|---:|---:|---|
| Remove the ceiling; retain a 1 s emergency guard | Recovers both words | Onset success 0.902; 621 zero-overlap words; 406 severely compressed words | Reject |
| Soft pressure for every token after a fixed 300 ms | Recovers both words | Onset success 0.908; 560 zero-overlap words; 371 severely compressed words | Reject |
| Phone-specific soft pressure for every token | Recovers both words | Onset success 0.914; 511 zero-overlap words; 353 severely compressed words | Reject |
| Phone-specific soft pressure for vowels only | `I'm` coverage 0.923 | Onset success 0.933; 379 zero-overlap words; 251 severely compressed words | Reject |
| Soft pressure only for a vowel immediately before transcript silence | `I'm` coverage 0.923; second `to` coverage 0.667 | Region start 0.973, onset success 0.933, assignment 0.938, but 370 zero-overlap and 242 severely compressed words | Reject for tail regression |

The final candidate improved most aggregate scores over the packaged baseline,
but it still added eight zero-overlap and eight severely compressed words. It
therefore remains rejected rather than being promoted as a phrase-specific
fix.

## Neural training path

Training now includes a differentiable word-interval loss that contrasts each
word's token probability inside its gold interval with adjacent silence. The
word-interval fine-tune freezes phone and pose embedding columns and adapts
continuous timing, silence, word-boundary, and punctuation features. Checkpoint
selection uses decoded word onset, exit, interval coverage, and zero-overlap
counts rather than cross-entropy alone.

The current fine-tune selected epoch 0: every trained checkpoint decoded worse
than the packaged V5 model. No new model weights are promoted by this branch.

## Next architecture

The experiments show that a global or class-conditioned dwell relaxation is
too coarse. The next model version should learn an explicit token-advance
hazard (or end-of-token probability) from the current audio state, current and
next transcript token embeddings, elapsed dwell, and punctuation context. The
hard ceiling can then remain only as a corruption guard. This gives the neural
model a supervised way to distinguish a genuinely sustained vowel from a
sticky acoustic state without changing transcript-owned viseme identity.

## 2026-07-29 repeated Max greeting

Fifteen completed recordings of `Hello, I'm Max. How can I assist you with the
deal in Kentucky?` were imported as cases 0431-0445. One aborted capture without
a finalized track or WAV was excluded. All imported alignments passed the MFA
draft audit and were approved as gold.

The host logs reported 40 planned events but only 38 committed events in every
completed take. The two absent events were the transcript's `/h/` phones in
`hello` and `how`. The planner had deliberately emitted those phones at zero
strength and marked them non-renderable, so the neural streamer could neither
display them nor use them as stable alignment states. This caused both visible
under-animation and topology errors after pauses. In the worst take, `how` was
assigned 1.45 seconds early. Across the fifteen takes, `how` had mean spoken
coverage 0.352 and fourteen low-coverage occurrences. The standalone word `I`
was also weak: mean coverage 0.116, eleven zero-overlap occurrences, and a mean
onset delay of about 90 ms.

The general planner correction makes `/h/` a low-strength, renderable,
transcript-owned neural state using the existing open-mouth pose. It is not a
phrase rule and does not alter event order. On the 837-case packaged replay it
changes word-region assignment from 0.935 to 0.989, completion from 0.985 to
1.000, strict perfect cases from 408 to 722, and incomplete words from 529 to
zero. For the new recordings, `how` coverage rises to 0.634, zero-overlap falls
to zero, and the 1.45-second early assignment disappears. Pause cleanliness is
unchanged at 0.979. Small aggregate movements remain in region-start MAE
(22.5 to 26.5 ms), word-onset MAE (32.3 to 33.6 ms), identity recall
(0.970 to 0.968), and visibility (0.964 to 0.962); these are retained because
the topology and completion gains are much larger and the absolute results
remain inside the project's quality envelope.

Three neural adaptations were evaluated and rejected: unrestricted fine-tune,
identity-column fine-tune, and word-interval fine-tune. Each either regressed
pause/region behavior or selected epoch 0. Compact words are now explicitly
weighted in the interval objective, and held-out checkpoint selection reports
their onset, exit, coverage, and zero-overlap metrics separately. The current
held-out set contains 280 compact words; the accepted epoch-0 model has 24
zero-overlap compact words and 0.704 mean coverage. The compact-aware fine-tune
again selected epoch 0, so no experimental model weights were promoted.

The `/h/` correction solves the systematic missing-animation and gross delayed
assignment failure. The remaining standalone-`I` weakness is detected by the
new metric but is not yet safely improved by the present model architecture.
It should be addressed by the explicit learned token-advance hazard described
above, rather than by a word-specific timing override.

## Jaw composition reconciliation

Fresh v36 logs showed jaw peaks of 0.659, 0.678, and 0.509. These were not
authored by any single active pose. FaceDriver additively accumulated the
central jaw control from every coarticulating pose; for example, the open
`18_Uh` contribution remained strong while `22_MBP` was dominant, leaving the
jaw at 0.509 during a required lip closure.

The v37 host composition keeps all neural pose identities and weights intact,
but treats the central jaw as one physical degree of freedom: it follows the
strongest weighted authored jaw target rather than summing targets. The
existing dominant-articulation rule now also attenuates residual jaw opening
under bilabial authority. Isolated poses are bit-for-bit unchanged. Replaying
the logged pose weights predicts p95 jaw reductions from 0.457/0.405/0.398 to
0.374/0.325/0.332 and peak reductions from 0.659/0.678/0.509 to
0.453/0.430/0.444. This is a presentation-layer correction and does not alter
neural timing or transcript ownership.
