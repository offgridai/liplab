import json
import pathlib
import sys


root = pathlib.Path(__file__).resolve().parents[1]
latest = root / "outputs" / "runs" / "latest"
summary_path = latest / "alignment_summary.json"
baseline_path = root / "docs" / "grade_baseline.json"
thresholds_path = root / "docs" / "grade_thresholds.json"

if not summary_path.exists():
    print("Missing alignment_summary.json. Run scripts/summarize.py first.", file=sys.stderr)
    sys.exit(2)
if not baseline_path.exists():
    print(f"Missing baseline file: {baseline_path}", file=sys.stderr)
    sys.exit(2)

summary = json.loads(summary_path.read_text(encoding="utf-8"))
baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
thresholds = json.loads(thresholds_path.read_text(encoding="utf-8"))


def value(document: dict, path: str):
    current = document
    for part in path.split("."):
        current = current[part]
    return current


checks = [
    ("region_start.success_rate", "min", "max_region_start_success_drop"),
    (
        "region_start.mean_abs_error_ms",
        "max",
        "max_region_start_mae_ms_increase",
    ),
    ("word_animation_onset.success_rate", "min", "max_word_onset_success_drop"),
    (
        "word_animation_onset.mean_abs_error_ms",
        "max",
        "max_word_onset_mae_ms_increase",
    ),
    (
        "decoded_viseme_alignment.identity_recall",
        "min",
        "max_decoded_viseme_identity_recall_drop",
    ),
    (
        "decoded_viseme_alignment.identity_precision",
        "min",
        "max_decoded_viseme_identity_precision_drop",
    ),
    (
        "decoded_viseme_alignment.mean_abs_center_error_ms",
        "max",
        "max_decoded_viseme_center_mae_ms_increase",
    ),
    ("word_duration.success_rate", "min", "max_word_duration_success_drop"),
    (
        "word_duration.mean_abs_error_ms",
        "max",
        "max_word_duration_mean_mae_ms_increase",
    ),
    (
        "word_duration.median_abs_error_ms",
        "max",
        "max_word_duration_median_mae_ms_increase",
    ),
    ("strict_region_segmentation.exact_boundary_rate", "min", "max_strict_region_drop"),
    (
        "word_region_assignment.success_rate",
        "min",
        "max_word_region_assignment_drop",
    ),
    (
        "strict_three_level_word_assignment.success_rate",
        "min",
        "max_strict_word_assignment_drop",
    ),
    ("pause.clean_rate", "min", "max_pause_clean_drop"),
    ("guardrails.event_completion_rate", "min", "max_event_completion_drop"),
    (
        "runtime_delivery.compressed_sentences",
        "max",
        "max_compressed_sentences_increase",
    ),
    (
        "presentation_event_visibility.robust_visible_event_rate",
        "min",
        "max_presentation_robust_visible_event_rate_drop",
    ),
    (
        "presentation_event_visibility.robust_boundary_event_rate",
        "min",
        "max_presentation_robust_boundary_event_rate_drop",
    ),
]

failed = False
for metric, direction, tolerance_name in checks:
    actual = float(value(summary, metric))
    accepted = float(value(baseline, metric))
    tolerance = float(thresholds[tolerance_name])
    limit = accepted - tolerance if direction == "min" else accepted + tolerance
    ok = actual >= limit if direction == "min" else actual <= limit
    if not ok:
        print(f"FAIL {metric}: actual={actual:.6f} limit={limit:.6f}")
        failed = True

region_confusion_checks = [
    (
        "word_region_confusion.early_region_thefts",
        "max_early_region_thefts_increase",
    ),
    (
        "word_region_confusion.late_region_assignments",
        "max_late_region_assignments_increase",
    ),
    (
        "word_region_confusion.materially_early_intact_words",
        "max_materially_early_intact_words_increase",
    ),
    (
        "word_region_confusion.max_intact_word_lead_ms",
        "max_intact_word_lead_ms_increase",
    ),
]
for metric, tolerance_name in region_confusion_checks:
    actual = float(value(summary, metric))
    accepted = float(value(baseline, metric))
    limit = accepted + float(thresholds[tolerance_name])
    if actual > limit:
        print(f"FAIL {metric}: actual={actual:.6f} limit={limit:.6f}")
        failed = True

order_violations = int(value(summary, "guardrails.order_violations"))
if order_violations > int(thresholds["max_order_violations"]):
    print(
        f"FAIL guardrails.order_violations: actual={order_violations} "
        f"limit={thresholds['max_order_violations']}"
    )
    failed = True

word_duration_coverage = float(value(summary, "word_duration.coverage_rate"))
if word_duration_coverage < float(thresholds["min_word_duration_coverage_rate"]):
    print(
        "FAIL word_duration.coverage_rate: "
        f"actual={word_duration_coverage:.6f} "
        f"limit={thresholds['min_word_duration_coverage_rate']:.6f}"
    )
    failed = True
severely_compressed_words = int(value(summary, "word_duration.severely_compressed_words"))
if severely_compressed_words > int(thresholds["max_severely_compressed_words"]):
    print(
        "FAIL word_duration.severely_compressed_words: "
        f"actual={severely_compressed_words} "
        f"limit={thresholds['max_severely_compressed_words']}"
    )
    failed = True

delivery_checks = [
    (
        "runtime_delivery.event_delivery_rate",
        "min_runtime_event_delivery_rate",
        "min",
    ),
    (
        "runtime_delivery.word_delivery_rate",
        "min_runtime_word_delivery_rate",
        "min",
    ),
    (
        "runtime_delivery.speech_region_delivery_rate",
        "min_runtime_speech_region_delivery_rate",
        "min",
    ),
    (
        "runtime_delivery.sentence_delivery_rate",
        "min_runtime_sentence_delivery_rate",
        "min",
    ),
    (
        "runtime_delivery.late_after_window_events",
        "max_late_after_window_events",
        "max",
    ),
    (
        "runtime_delivery.empty_speech_regions",
        "max_empty_speech_regions",
        "max",
    ),
    (
        "runtime_delivery.missing_sentences",
        "max_missing_sentences",
        "max",
    ),
]
for metric, threshold_name, direction in delivery_checks:
    actual = float(value(summary, metric))
    limit = float(thresholds[threshold_name])
    ok = actual >= limit if direction == "min" else actual <= limit
    if not ok:
        print(f"FAIL {metric}: actual={actual:.6f} limit={limit:.6f}")
        failed = True

proportion_checks = [
    (
        "viseme_proportion.word_coverage_rate",
        "min_viseme_proportion_word_coverage_rate",
        "min",
    ),
    (
        "viseme_proportion.run_share_abs_error_median",
        "max_viseme_run_share_abs_error_median",
        "max",
    ),
    (
        "viseme_proportion.boundary_position_abs_error_median",
        "max_viseme_boundary_position_abs_error_median",
        "max",
    ),
    (
        "viseme_proportion.word_total_variation_median",
        "max_viseme_word_total_variation_median",
        "max",
    ),
    (
        "viseme_proportion.zero_runtime_words",
        "max_viseme_zero_runtime_words",
        "max",
    ),
]
for metric, threshold_name, direction in proportion_checks:
    actual = float(value(summary, metric))
    limit = float(thresholds[threshold_name])
    ok = actual >= limit if direction == "min" else actual <= limit
    if not ok:
        print(f"FAIL {metric}: actual={actual:.6f} limit={limit:.6f}")
        failed = True

visibility_checks = [
    (
        "presentation_event_visibility.robust_visible_event_rate",
        "min_presentation_robust_visible_event_rate",
    ),
    (
        "presentation_event_visibility.robust_boundary_event_rate",
        "min_presentation_robust_boundary_event_rate",
    ),
]
for metric, threshold_name in visibility_checks:
    actual = float(value(summary, metric))
    limit = float(thresholds[threshold_name])
    if actual < limit:
        print(f"FAIL {metric}: actual={actual:.6f} limit={limit:.6f}")
        failed = True

if failed:
    print("See alignment_cases.csv and alignment_words.csv for the failing cases.")
    sys.exit(1)

print(f"Priority alignment thresholds passed for {summary['cases']} case(s).")
