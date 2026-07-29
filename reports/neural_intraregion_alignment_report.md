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
