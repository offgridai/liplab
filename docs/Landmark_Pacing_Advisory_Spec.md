# Landmark Pacing Advisory Spec

## Goal

Use transcript-conditioned landmark matches as an intra-region pacing advisor.

This system must not:

- change speech-region detection or splitting,
- change viseme identity,
- reorder committed events,
- directly modify runtime playback in this phase.

This phase is advisory and logged-only.

## Ownership

- Speech-region logic remains authoritative for region open/close.
- Transcript plan remains authoritative for phone / viseme identity.
- Landmark pacing advisor only estimates whether playback inside the current speech region is ahead or behind the audio.

## Inputs

- `transcript_landmarks.csv`
- `audio_landmark_conditioned_observations.csv`
- transcript phone prior timing from `plan.ExpectedPhones`
- gold phones / words for offline grading only

## Trusted landmarks

Use only transcript-conditioned landmarks:

- `mbp`
- `fv`
- `w`
- `chjjsh`
- `comma_lull`

Treat `round` as weak support only and exclude it from pacing control in this phase.

## Region-local controller

Each speech region has its own advisory state:

- `seeded`: whether a first strong anchor has been seen
- `anchor_prior_sec`: prior-center time of the last accepted landmark
- `anchor_observed_sec`: observed-center time of the last accepted landmark
- `play_rate`: persistent rate estimate, initialized to `1.0`

State resets at every speech-region boundary.

## Update rule

When a strong landmark is matched inside a speech region:

1. First accepted landmark in region:
   - seed the controller
   - set `anchor_prior_sec` and `anchor_observed_sec`
   - keep `play_rate = 1.0`
   - this gives a phase reset only

2. Later accepted landmarks in region:
   - compute `prior_delta = current.prior - previous.prior`
   - compute `observed_delta = current.obs - previous.obs`
   - compute `measured_rate = clamp(observed_delta / prior_delta, 0.88, 1.12)`
   - compute confidence from landmark score / reliability
   - update `play_rate` with an EMA toward `measured_rate`
   - reset the anchor to the current landmark

The effect is:

- sparse phase correction at the current landmark
- persistent rate correction for future timing in the same speech region

## Advisory projections

The controller predicts timing for three checkpoint families:

1. `phone_center`
   - all planned phones with gold timing

2. `word_start`
   - first phone start of each word

3. `gap_center`
   - midpoint between adjacent words inside the same speech region
   - classified as:
     - `space_gap`
     - `comma_gap`

Predicted checkpoint time after seeding is:

`predicted = anchor_observed_sec + play_rate * (checkpoint_prior_sec - anchor_prior_sec)`

Before the first accepted landmark in a region, prediction remains at the baseline prior.

## Outputs

Per case:

- `landmark_pacing_updates.csv`
  - accepted landmark updates
  - seeded updates
  - measured / filtered rate
  - anchor timing error

- `landmark_pacing_predictions.csv`
  - checkpoint kind
  - prior time
  - gold time
  - predicted time
  - baseline and predicted error
  - whether the checkpoint was affected by an advisory update

- `landmark_pacing_summary.json`
  - counts and error metrics overall
  - separate rollups for:
    - `phone_center`
    - `word_start`
    - `space_gap`
    - `comma_gap`

## Success criteria

The advisory is useful if it lowers baseline error for:

- word starts,
- phone centers,
- space/comma gap centers,

without touching speech-region behavior.

## Non-goals for this phase

- no runtime playback steering
- no direct pause insertion
- no region ownership changes
- no viseme suppression
- no historical search / transcript relocation
