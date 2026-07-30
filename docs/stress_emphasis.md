# Positive speech emphasis proof of concept

## Purpose

LipLab retains transcript-derived lexical stress so an Offgrid presentation
layer can add a small, positive facial accent around stressed syllables. The
feature is intentionally separate from viseme identity, scheduling, strength,
and audio alignment. It does not change which mouth pose is rendered or when a
committed viseme occurs.

The current character treatment is eye-led and mildly positive. Offgrid
FaceDriver may use the sampled scalar to add cheek, inner-eye, inner-brow,
mouth-corner, dimple, and nasolabial support on top of the ordinary face pose.

## Data contract

`FOffgridAITextVisemeEvent::LexicalStress` and
`FOffgridAICommittedVisemeEvent::LexicalStress` carry a small integer:

- `0`: unstressed or no stress annotation
- `1`: primary lexical stress
- `2`: secondary lexical stress

The text planner parses the final CMU stress digit from the source vowel phone
before stripping that digit from the phone identity. The neural streaming
aligner copies the value to the corresponding committed event without changing
event order, timing, renderability, or strength.

Transcript pronunciation remains the sole authority for stress metadata. Audio
does not infer stress or choose a different viseme identity.

## Runtime sampling

`FOffgridAIVisemePerformer::SamplePositiveStressEmphasis()` samples the immutable
committed track at the authoritative playback time. For each renderable,
non-cancelled primary-stress event it creates a smooth scalar pulse:

- preparation begins 100 ms before the committed viseme center;
- the pulse peaks 20 ms before the committed center;
- release ends 140 ms after the committed center;
- start and end are clamped to the event's observed speech region;
- attack and release use minimum-jerk interpolation;
- overlapping pulses combine by maximum rather than addition;
- secondary and unstressed events do not currently produce a pulse.

The result is in `[0, 1]`. Viseme strength is deliberately not reused as facial
prosody, and the sampler owns no independent clock or fallback schedule.

## Offgrid integration contract

The shared LipLab files under `offgrid_dropin` are transplanted into the Offgrid
plugin. Offgrid LineCoach samples the emphasis scalar from its mirrored
committed track during the same facial submission pass used for lipsync, then
calls `SubmitPositiveSpeechEmphasis()` on FaceDriver.

FaceDriver composes that scalar late into `FOffgridAIMetaHumanFacePose`, after
ordinary control projection. This makes the expression additive to lipsync and
emotion instead of placing facial controls in the viseme event stream. Clearing
emotion must not clear speech emphasis; only a full neutral/reset path or the
next zero-valued emphasis submission should clear it.

The concrete rig amplitudes are an Offgrid presentation concern and are not
part of LipLab's shared timing contract. LipLab supplies only lexical-stress
metadata and the normalized pulse.

## Diagnostics

The standalone harness exports `lexical_stress` in both planned and committed
event CSVs. The Offgrid debug integration should do the same and should record
the final `PositiveSpeechEmphasisWeight` alongside FaceDriver pose diagnostics.
These columns distinguish three failure classes:

1. stress was not present in the transcript plan;
2. stress was lost while building the committed track;
3. the pulse existed but was not submitted or published by FaceDriver.

## Verification and limitations

The proof of concept is deterministic and uses no audio-derived prosody. It
therefore represents dictionary lexical stress, not contextual sentence focus,
contrastive emphasis, sarcasm, or speaker-specific delivery. Those would require
a separate explicit source of prosodic intent and must not weaken the existing
transcript-derived viseme authority.

Any change to stress propagation or pulse sampling must pass
`scripts\verify.bat`. In particular it must not introduce event-order violations,
missing visible visemes, timing regression, or divergence between the harness
and `offgrid_dropin` implementations.
