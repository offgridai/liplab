# Nucleus beat error analysis

The audio-only indicator is graded against MFA vowel-phone intervals with a
100 ms outside-interval tolerance and one-to-one matching. The grader now uses
MFA vowels directly. It no longer assumes that a transcript/CMUdict vowel has
the same ordinal phone index in MFA.

## Baseline error pattern

- 521 of 676 false negatives (77%) were unstressed vowels. Reduced `AH0`,
  `IH0`, `IY0`, and `ER0` dominated.
- Missed vowels had a 60 ms median MFA duration versus 90 ms for detected
  vowels, and substantially lower median energy and speech evidence.
- False positives were concentrated in high-ZCR/high-frication consonants,
  especially `/s/`, and duplicate peaks within long vowels.
- 54 apparent false positives occurred inside MFA `spn` intervals. These are
  spoken OOV spans with no MFA phone labels, so their nuclei cannot be scored;
  they are not reliable evidence that the audio detector is wrong.

## Retained v3 experiment

The diagnostic profile adds a shorter sonority-envelope scale to recover weak
reduced nuclei, retains medium and slow scales for ordinary and long vowels,
and rejects peaks with fricative support above 0.35 or ZCR above 0.20. It is
isolated behind `bNucleusBeatIndicatorTuning`; production scheduling continues
to use the prior evidence surface.

Across 350 cases, direct MFA-vowel grading changed from approximately 90.7%
precision / 87.2% recall to 92.3% precision / 88.5% recall. There were 4,659
matched nuclei from 5,049 displayed beats and 5,262 scorable MFA vowels.

## Retained v4 experiment

Threshold sweeps could not cleanly separate the remaining errors: false
positives and true nuclei had nearly identical median periodicity, vowel
support, speech evidence, and zero-crossing rate. The existing rich-spectrum
vowel classifier also supported almost exactly the same fraction of true and
false pulses. More single-feature gates therefore traded precision for recall
rather than improving both.

V4 instead generates permissive local maxima at four envelope scales and ranks
them using a deterministic 12-tree forest. Its inputs are only local audio
features: multi-scale prominence, energy, periodicity, occupancy evidence,
articulatory probabilities, broad and rich spectral features, and a nine-frame
temporal window. It has no transcript, word, phoneme, or TTS timing input. The
ranker was trained to select one representative peak nearest the center of each
MFA vowel, rather than treating every peak inside a long vowel as positive.
Candidates closer than 120 ms are consolidated unless the 30 ms envelope has
an intervening valley of at least 0.07.

Five-fold evaluation grouped by complete corpus case scored 94.0% precision,
90.2% recall, and 92.1% F1. The native C++ pass over all 350 cases scored 94.2%
precision, 90.3% recall, and 92.2% F1: 4,753 one-to-one matches from 5,043
displayed beats and 5,262 MFA vowels. Matched beats were within MFA intervals
apart from a 3.03 ms mean outside-interval error. A 120 ms beat-only stability
window made the final evidence surface and rendered track agree exactly:
5,043 assigned beats, 5,043 rendered beats, and no missing or extra markers.

The remaining misses are still dominated by reduced vowels: most are
unstressed, led by `AH`, `IH`, and `IY`, and their median duration is 60 ms.
Many apparent false positives fall inside MFA `spn` spans and therefore
cannot be phoneme-scored. The other false positives are mostly alternate peaks
inside vowels/diphthongs and voiced sonorants. Their feature distributions now
substantially overlap the true-pulse distribution, so another global threshold
was not retained.

## Transcript-conditioning experiments

The permissive acoustic surface contains a candidate within the MFA scoring
window for 5,201 of 5,262 vowels (98.84%). Most remaining errors are therefore
candidate selection errors, not a lack of acoustic evidence.

Transcript conditioning was evaluated with complete cases held out from model
training. Transcript and MFA agree on the total syllable count in 312 of 350
lines (89.1%), but exact per-region syllable counts agree in only 49.1% of
regions. Consequently, a hard transcript quota performed poorly: 90.6%
precision and 80.5% recall.

Several softer variants were also rejected:

- Filling a line-wide syllable deficit raised recall to about 93% but reduced
  precision below 92%.
- Admitting a weak candidate only when the next transcript syllable was
  unstressed was effectively neutral: 93.8% precision / 90.4% recall.
- Trimming beats above the transcript count produced 94.9% precision but only
  89.0% recall.
- Filling deficits only inside resolved audio regions reached 92.6% precision /
  92.5% recall.
- Balancing both deficits and excesses inside resolved regions reached 93.3%
  precision / 91.6% recall.

The unconditioned grouped v4 result remains 94.0% precision / 90.2% recall.
Region conditioning can provide a distinct high-recall operating point, but it
does not improve both axes and is unsafe when transcript punctuation and audio
region topology disagree. No transcript-conditioned runtime path was retained.

## Bottom-five audit and scorer correction

The first bottom-five audit exposed two measurement defects rather than
detector defects. The raw scorer greedily assigned each beat to the nearest
unused MFA vowel, which could consume a later vowel and manufacture an FP/FN
pair in dense speech. It now finds the maximum-cardinality monotonic assignment
and minimizes outside-interval error among assignments of equal size. Beats
inside an MFA `spn` interval are now reported as ungradeable and excluded from
precision; MFA provides no vowel labels with which to judge them.

With the original v4 detector unchanged, the corrected direct corpus score is
95.31% precision, 90.38% recall, and 92.78% F1: 4,756 matches from 4,990
scorable displayed beats and 5,262 MFA vowels. Another 53 displayed beats fall
inside ungradeable `spn` spans.

The audit also found two gold-generation defects. UTF-8 BOMs were not stripped
from imported transcripts, and probability columns in MFA's mixed-format stock
dictionary were incorrectly retained as phones when deriving pronunciations
such as possessives. Both import paths are corrected. Regenerated gold for the
two affected cases was deliberately not retained because fresh MFA boundaries
changed the production word-region score enough to fail the no-regression gate;
approved gold needs separate review before replacement.

A 3x training weight for the five genuine failures improved their grouped
held-out F1 from 77.05% to 80.65%. In the native full-corpus run it raised
recall from 90.38% to 90.69%, but reduced precision from 95.31% to 94.93% and
slightly reduced F1 from 92.782% to 92.759%. That model was rejected and the
original v4 model retained.

## MFA `spn` cleanup

The approved corpus previously contained 24 opaque `spn` intervals spanning
11.66 seconds. They caused 53 detected beats to be excluded from scoring even
though most intervals represented ordinary transcript words. Realigning those
24 cases with the corrected mixed-format dictionary parser, combined stock and
Offgrid pronunciations, compound handling, and deterministic grapheme fallback
eliminated `spn` cleanly in 19 approved cases. Ninety percent of their word
boundaries moved by no more than 10 ms, and the complete production corpus gate
passed after export.

Five intervals remain quarantined because their fresh alignments have word
count, word identity, or implausibly short-word warnings. They are listed in
`inputs/gold/spn_allowlist.json`; `scripts/check_gold.py` rejects every new
unreviewed `spn` and also rejects stale allowlist entries after resolution.
Thus these exceptions cannot silently expand or linger.

After cleanup, the focused audio-nucleus corpus has 5,301 MFA vowel targets,
4,785 matches from 5,000 scorable beats, and only 18 beats inside quarantined
`spn` spans. Precision is 95.70%, recall is 90.27%, and F1 is 92.90%.
