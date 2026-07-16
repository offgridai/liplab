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
- the current commit frontier and block reason.

There is no punctuation hold state, strong-phone scheduler, speculative pulse
path, region-fit path, alternate clock, or fallback scheduler.

## Diagnostic contract

The harness retains metrics that explain one of these active stages:

- text phone, syllable, duration, and pause-hint quality,
- complete speech close/resume boundary-pair quality,
- pulse and broad phone-family evidence quality,
- nearby-candidate and monotonic assignment quality,
- accepted runtime bounded-anchor quality,
- committed viseme timing, coverage, monotonicity, and uncommitted suffixes.

Offline tuning programs and metrics for deleted runtime modes are intentionally
not retained.
