# Phoneme alignment investigation — 2026-07-28

## Scope

Six Offgrid priestley2/1.7B recordings from the 20:10 run were added as
`case_0385` through `case_0390`, aligned with MFA, approved, and included in the
782-case corpus. Three cases are train, two validation, and one text test.

## Diagnosis

The visible error originates in the neural monotonic token path, before the
viseme performer. Final event centers are the token extents emitted by the
Viterbi path; the performer only constructs presentation envelopes around those
centers.

The new recordings contain both coherent timing drift and real pronunciation
mismatches. Examples include planned `AH` versus synthesized `IY` in “the”,
planned `AH` versus synthesized `IH` in “matches”, and planned `AE` versus a
reduced `AH` in “an”. A separate A/B replay showed that disabling the v35 comma
commit hold changed none of the committed events in these six cases, ruling out
the latest greeting boundary hold as the direct cause.

The training export explained why the matcher can learn the wrong association.
When a planned transcript phone has no matching MFA phone, its center is
interpolated to keep the monotonic path complete and `has_mfa_target` is written
as false. The old loss ignored that flag and supervised the fabricated phone
label at full weight, often increasing it again for vowels and boundaries.
Across the expanded corpus, 10,839 of 42,447 token rows (25.5%) are interpolated;
94 of 359 (26.2%) are interpolated in the six new cases.

The old checkpoint-selection loss was also blind to the product failure. It
selected by frame cross-entropy plus region BCE, not by decoded viseme identity,
center error, or region assignment. In the experiments below, the
validation-loss winner was repeatedly not the decoded-alignment winner.

## Durable changes

- Exact MFA phones have target confidence 1.0. Monotonic pronunciation
  substitutions bounded by an MFA-aligned word retain 0.5-confidence timing
  supervision, while wholly unsupported interpolation retains only 10%
  path-shaping weight. Transcript phone/pose identity remains authoritative.
- Fine-tuning supports full, acoustic-only, and identity-only modes so accepted
  region/timing behavior can be frozen during controlled experiments.
- Corpus summaries now include decoded viseme identity recall, precision,
  missing/extra counts, and matched-center MAE.
- Grade gates limit identity recall and precision loss to 0.001 and center-MAE
  regression to 1 ms.

## Experiments

Accepted V5 on the six new cases: 178 matched, 11 missing, 15 extra; matched
center MAE 30.30 ms; word-onset MAE 54.67 ms.

Full fine-tuning reduced local center error but materially regressed global
region behavior. The strongest local epoch reduced center MAE to 27.01 ms, but
full-corpus assignment fell from 0.939 to 0.932 and region-start success from
0.973 to 0.957. Conservative blends still increased early and late region
misassignments and were rejected.

Acoustic-only epoch 1 reduced local center MAE to 27.28 ms and onset MAE to
52.37 ms, but increased full-corpus late assignments from 43 to 56 and global
onset MAE from 31.49 to 34.2 ms. A 50% blend reduced the damage but still added
seven late assignments and lost six exact region boundaries, so it was rejected.

Identity-only epoch 1 reduced local center MAE to 27.11 ms and onset MAE to
52.50 ms. Globally it improved center MAE from 18.91 to 18.4 ms but reduced
identity recall from 0.970 to 0.968 and precision from 0.940 to 0.938, while
adding six late assignments. The 50% blend worsened the target timing and was
rejected without a corpus pass.

## Conclusion

The supervision bug is real and is now corrected, but retraining the existing
dot-product/Viterbi model is not sufficient to fix these lines without trading
away accepted behavior elsewhere. The shipped V5 checkpoint therefore remains
unchanged. The next model iteration should select checkpoints by decoded corpus
metrics and represent pronunciation alternatives explicitly, so reduced vowels
and contextual variants can compete acoustically without forcing one incorrect
transcript-phone sequence through the lattice.
