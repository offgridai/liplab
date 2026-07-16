# Pulse Detection Experiments

## Objective

Bring transcript pulse expectations and acoustic pulses detected from the bounded
streaming evidence surface into closer agreement without weakening the existing
high-precision detector.

The evaluation uses source-group held-out cases. Detection is causal with the
configured 350 ms preroll and 1500 ms retained postroll. MFA vowel intervals are
the reference locations, with a 100 ms matching tolerance.

## Production Metrics

The raw pulse evidence report now includes:

- precision, recall, F1, and timing error against MFA vowel intervals;
- detected/reference count ratio;
- exact pulse-count rate and count MAE per aligned speech region;
- recall for MFA nuclei separated from a neighbor by at most 120, 150, or 200 ms;
- false pulse rate outside MFA speech regions.

The full-corpus baseline is:

| Metric | Value |
| --- | ---: |
| Precision | 0.907 |
| Recall | 0.872 |
| F1 | 0.889 |
| Count ratio | 0.962 |
| Exact regional count | 0.383 |
| Regional count MAE | 0.929 |
| Cluster recall, 120 ms | 0.715 |
| Cluster recall, 150 ms | 0.782 |
| Cluster recall, 200 ms | 0.832 |
| Outside-speech rate | 0.013 |

## Experiments

Additional peak-rate, multiband-sonority, spectral-flux, and periodicity maxima
all expose many of the missing clustered nuclei. Used as independent pulse
generators, however, they also expose substantial within-syllable structure.
Held-out precision falls to 0.61-0.81 even while clustered recall rises.

Guarded subdivision of broad envelopes improves count ratio but does not beat the
baseline F1 or regional count error. A transcript regional-count constraint also
fails held-out validation because text syllables are not allocated to runtime
speech regions reliably enough to force the acoustic inventory.

Most clustered misses are reduced `AH` and `IH` nuclei in function words. Merging
weak transcript nuclei within 120 ms brings aggregate transcript/runtime counts
closer, but reduces transcript/MFA exact regional count from 0.943 to 0.739. The
trade is not strong enough to redefine transcript pulses.

## Decision

Keep the current strict detector as the authoritative pulse inventory. Preserve
the derivative channels as permissive evidence for later monotonic matching, but
do not let them independently create runtime pulses. The next detector iteration
needs a sequence-level criterion that distinguishes a reduced syllable from
ordinary within-syllable spectral and voicing changes; another scalar threshold
is unlikely to solve the remaining ambiguity.
