# Region Progress Rate-Control Shadow Patch

Diagnostic-only. Runtime playback remains unchanged.

This patch reframes the shadow decoder around the design we converged on:

1. The transcript is already known and split into punctuated speech regions.
2. CMU-derived phones/visemes provide a normalized internal timeline for each word.
3. The speech detector selects the active observed speech region.
4. A continuous progress estimator tracks normalized progress through that region.
5. The scheduler can derive current word/phone/viseme and a suggested playback-rate correction from that progress.

## What changed

`BuildWordStateDecodedRegionWordSpans` still emits shadow word spans, but the underlying model is now explicitly a region-progress clock:

- local audio density estimates progress velocity,
- energy/flux/voicing/valley evidence modulates velocity,
- word boundaries are derived from cumulative progress crossings,
- candidate boundaries are observations, not hard advance commands.

`word_state_shadow.csv` now includes:

- `region_start`, `region_end`
- `region_progress_01`
- `region_prior_progress_01`
- `animation_progress_01`
- `progress_error_01`
- `estimated_velocity_01_per_sec`
- `suggested_play_rate`
- `progress_confidence`

## Interpretation

The future live algorithm should not jump to a word. It should compare:

- estimated transcript progress, and
- current animation progress

Then gently adjust animation play rate using `suggested_play_rate`.

Positive `progress_error_01` means audio appears ahead of animation, so playback should speed up.
Negative `progress_error_01` means animation appears ahead of audio, so playback should slow down.

