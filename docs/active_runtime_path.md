# Active Runtime Path

This document is the inventory for the code that ships in
`offgrid_dropin`.

## Components

- `OffgridAITextVisemePlanner`: transcript pronunciation, visible event order,
  syllables, and relative duration priors.
- `OffgridAIStreamingSpeechDetector`: causal feature extraction and speech
  regions.
- `OffgridAIAcousticEvidence`: deterministic articulatory probabilities for a
  single feature frame.
- `OffgridAIStreamingEvidenceSurface`: syllabic pulses and broad phone-family
  observations over retained preroll/postroll.
- `OffgridAIStreamingSyllablePositionEstimator`: nearby-candidate grading,
  monotonic assignment grading, and stable anchors for runtime correction.
- `OffgridAILipsyncRuntimeAdapter`: the only scheduler.
- `OffgridAIVisemePerformer`: committed-event pose sampling.

## Scheduler state

The adapter tracks only:

- the next transcript event,
- the active observed and text region indices,
- the current prior-to-audio timeline anchor and rate,
- the latest accepted syllable anchor,
- pending list-restart and punctuation-boundary resolution,
- the current commit frontier and block reason.

There is no strong-phone scheduler, speculative pulse path, region-fit path,
alternate clock, or fallback scheduler. Punctuation state only waits for or
records acoustic evidence inside this scheduler; it does not own time.

The scheduler also enforces atomic word ownership at closed region tails. Once
a word has begun in a region, its remaining ordered events may use bounded tail
compaction (1 ms minimum spacing and at most 40 ms overrun) rather than splitting
the word across regions. See `docs/focused_alignment.md`.

## Diagnostic contract

The harness retains metrics that explain one of these active stages:

- text phone, syllable, duration, and pause-hint quality,
- complete speech close/resume boundary-pair quality,
- pulse and broad phone-family evidence quality,
- nearby-candidate and monotonic assignment quality,
- accepted runtime bounded-anchor quality,
- committed viseme timing, coverage, monotonicity, and uncommitted suffixes.
- strict word-to-region ownership, word-region integrity, and inter-word
  boundary agreement against MFA.

Offline tuning programs and metrics for deleted runtime modes are intentionally
not retained.
