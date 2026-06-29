# Regression Policy

The harness should remain simple and strict enough to keep automated iteration honest.

A scheduling change should be rejected if it causes any of the following on the checked-in corpus:

- monotonicity/order violations,
- more missing visemes,
- worse mean absolute center timing beyond the configured tolerance,
- worse pause-boundary leakage once that metric is implemented,
- early phrase starts that cross real silence boundaries,
- late tails that truncate or leak into the following phrase.

Current threshold values live in `docs/grade_thresholds.json` and are enforced by `scripts/check_grades.py`.

When the handmade corpus grows, tighten these thresholds rather than adding permissive fallback logic.

Current sample thresholds are intentionally calibrated to the checked-in toy corpus. Tighten them when better handmade labels are added.
