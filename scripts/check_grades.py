import json
import pathlib
import sys

from grade_summary import compute_summary, load_case_grades

root = pathlib.Path(__file__).resolve().parents[1]
latest = root / "outputs" / "runs" / "latest"
gold_root = root / "inputs" / "gold"
thresholds_path = root / "docs" / "grade_thresholds.json"
baseline_path = root / "docs" / "grade_baseline.json"

thresholds = {
    "max_order_violations": 0,
    "max_degenerate_cases": 0,
    "max_speech_region_count_mismatch_cases": 999999,
    "max_visible_speech_region_count_mismatch_cases": 999999,
    "max_sentence_region_count_mismatch_cases": 999999,
    "max_clause_region_count_mismatch_cases": 999999,
    "max_speech_f1_drop": 0.0,
    "max_speech_boundary_start_ms_increase": 0.0,
    "max_speech_boundary_end_ms_increase": 0.0,
    "max_word_f1_drop": 0.0,
    "max_word_start_ms_increase": 0.0,
    "max_word_duration_ms_increase": 0.0,
    "max_word_head_start_ms_increase": 0.0,
    "max_phoneme_coverage_drop": 0.0,
    "max_phoneme_center_ms_increase": 0.0,
    "max_phoneme_start_ms_increase": 0.0,
    "max_phoneme_end_ms_increase": 0.0,
    "max_intra_word_coverage_drop": 0.0,
    "max_intra_word_center_ms_increase": 0.0,
    "max_pause_safety_unsafe_timeout_releases": 0,
    "max_pause_safety_unresolved_holds": 0,
    "max_pause_safety_pair_rate_drop": 0.0,
    "max_pause_safety_early_resume_increase": 0,
    "max_pause_safety_leakage_boundary_increase": 0,
    "max_pause_safety_total_leakage_ms_increase": 0.0,
    "max_pause_safety_false_hold_ms_increase": 0.0,
    "max_pause_safety_false_pause_resolution_increase": 0,
    "max_pause_safety_syllable_continuous_false_pause_release_increase": 0,
    "max_pause_safety_syllable_continuous_false_pause_leakage_ms_increase": 0.0,
    "max_pause_safety_syllable_continuous_all_graded_false_pause_release_increase": 0,
    "max_pause_safety_syllable_continuous_all_graded_false_pause_leakage_ms_increase": 0.0,
}
if thresholds_path.exists():
    thresholds.update(json.loads(thresholds_path.read_text()))

rows, graded, ungraded = load_case_grades(latest, gold_root)
if not rows:
    print("No grade.json files found. Run liplab_runner first.", file=sys.stderr)
    sys.exit(2)

if not baseline_path.exists():
    print(f"Missing baseline file: {baseline_path}", file=sys.stderr)
    sys.exit(2)

summary = compute_summary(rows, graded, ungraded)
baseline = json.loads(baseline_path.read_text())


def at_most(metric_name, threshold_name):
    if metric_name not in baseline:
        return (True, metric_name, summary.get(metric_name, 0.0), "no baseline")
    return (
        summary[metric_name] <= baseline[metric_name] + thresholds[threshold_name],
        metric_name,
        summary[metric_name],
        baseline[metric_name] + thresholds[threshold_name],
    )


def at_least(metric_name, threshold_name):
    if metric_name not in baseline:
        return (True, metric_name, summary.get(metric_name, 0.0), "no baseline")
    return (
        summary[metric_name] >= baseline[metric_name] - thresholds[threshold_name],
        metric_name,
        summary[metric_name],
        baseline[metric_name] - thresholds[threshold_name],
    )


failed = False
checks = [
    (
        summary["order_fail_cases"] <= thresholds["max_order_violations"],
        "order_fail_cases",
        summary["order_fail_cases"],
        thresholds["max_order_violations"],
    ),
    (
        summary["degenerate_cases"] <= thresholds["max_degenerate_cases"],
        "degenerate_cases",
        summary["degenerate_cases"],
        thresholds["max_degenerate_cases"],
    ),
    (
        summary["speech_region_count_mismatch_cases"] <= thresholds["max_speech_region_count_mismatch_cases"],
        "speech_region_count_mismatch_cases",
        summary["speech_region_count_mismatch_cases"],
        thresholds["max_speech_region_count_mismatch_cases"],
    ),
    (
        summary["visible_speech_region_count_mismatch_cases"] <= thresholds["max_visible_speech_region_count_mismatch_cases"],
        "visible_speech_region_count_mismatch_cases",
        summary["visible_speech_region_count_mismatch_cases"],
        thresholds["max_visible_speech_region_count_mismatch_cases"],
    ),
    (
        summary["sentence_region_count_mismatch_cases"] <= thresholds["max_sentence_region_count_mismatch_cases"],
        "sentence_region_count_mismatch_cases",
        summary["sentence_region_count_mismatch_cases"],
        thresholds["max_sentence_region_count_mismatch_cases"],
    ),
    (
        summary["clause_region_count_mismatch_cases"] <= thresholds["max_clause_region_count_mismatch_cases"],
        "clause_region_count_mismatch_cases",
        summary["clause_region_count_mismatch_cases"],
        thresholds["max_clause_region_count_mismatch_cases"],
    ),
    at_least("speech_f1", "max_speech_f1_drop"),
    at_most("speech_boundary_start_ms", "max_speech_boundary_start_ms_increase"),
    at_most("speech_boundary_end_ms", "max_speech_boundary_end_ms_increase"),
    at_least("word_f1", "max_word_f1_drop"),
    at_most("word_start_ms", "max_word_start_ms_increase"),
    at_most("word_duration_ms", "max_word_duration_ms_increase"),
    at_most("word_head_start_ms", "max_word_head_start_ms_increase"),
    at_least("phoneme_coverage_rate", "max_phoneme_coverage_drop"),
    at_most("phoneme_center_ms", "max_phoneme_center_ms_increase"),
    at_most("phoneme_start_ms", "max_phoneme_start_ms_increase"),
    at_most("phoneme_end_ms", "max_phoneme_end_ms_increase"),
    at_least("intra_word_coverage_rate", "max_intra_word_coverage_drop"),
    at_most("intra_word_center_ms", "max_intra_word_center_ms_increase"),
    (
        summary.get("pause_safety_unsafe_timeout_release_count", 0)
            <= thresholds["max_pause_safety_unsafe_timeout_releases"],
        "pause_safety_unsafe_timeout_release_count",
        summary.get("pause_safety_unsafe_timeout_release_count", 0),
        thresholds["max_pause_safety_unsafe_timeout_releases"],
    ),
    (
        summary.get("pause_safety_unresolved_hold_count", 0)
            <= thresholds["max_pause_safety_unresolved_holds"],
        "pause_safety_unresolved_hold_count",
        summary.get("pause_safety_unresolved_hold_count", 0),
        thresholds["max_pause_safety_unresolved_holds"],
    ),
    at_least("pause_safety_pair_rate", "max_pause_safety_pair_rate_drop"),
    at_most("pause_safety_early_resume_count", "max_pause_safety_early_resume_increase"),
    at_most("pause_safety_leakage_boundary_count", "max_pause_safety_leakage_boundary_increase"),
    at_most("pause_safety_total_leakage_ms", "max_pause_safety_total_leakage_ms_increase"),
    at_most("pause_safety_false_hold_during_gold_speech_ms", "max_pause_safety_false_hold_ms_increase"),
    at_most("pause_safety_false_pause_resolution_count", "max_pause_safety_false_pause_resolution_increase"),
    at_most(
        "pause_safety_syllable_continuous_false_pause_release_count",
        "max_pause_safety_syllable_continuous_false_pause_release_increase",
    ),
    at_most(
        "pause_safety_syllable_continuous_false_pause_leakage_ms",
        "max_pause_safety_syllable_continuous_false_pause_leakage_ms_increase",
    ),
    at_most(
        "pause_safety_syllable_continuous_all_graded_false_pause_release_count",
        "max_pause_safety_syllable_continuous_all_graded_false_pause_release_increase",
    ),
    at_most(
        "pause_safety_syllable_continuous_all_graded_false_pause_leakage_ms",
        "max_pause_safety_syllable_continuous_all_graded_false_pause_leakage_ms_increase",
    ),
]

for ok, name, actual, limit in checks:
    if not ok:
        print(f"FAIL summary: {name}={actual} limit={limit}")
        failed = True

if failed:
    print(
        "Summary diagnostics: "
        f"graded={summary['graded_cases']} degenerate={summary['degenerate_cases']} "
        f"speech_region_mismatch={summary['speech_region_count_mismatch_cases']} "
        f"text_sentence_span_mismatch={summary['text_sentence_span_count_mismatch_cases']} "
        f"text_sentence_region_subspan_mismatch={summary['text_sentence_region_subspan_count_mismatch_cases']} "
        f"speech_boundary_start_ms={summary['speech_boundary_start_ms']:.3f} "
        f"speech_tail_leakage_ms={summary.get('speech_tail_leakage_ms', 0.0):.3f} "
        f"word_duration_ms={summary.get('word_duration_ms', 0.0):.3f} "
        f"word_head_start_ms={summary['word_head_start_ms']:.3f} "
        f"word_head_start_median_ms={summary['word_head_start_median_ms']:.3f} "
        f"phoneme_center_ms={summary['phoneme_center_ms']:.3f} "
    )
    if summary.get("direct_aligner_available", False):
        print(
            f"Direct aligner diagnostics: match_rate={summary.get('direct_aligner_match_rate', 0.0):.3f} "
            f"center_ms={summary.get('direct_aligner_center_ms', 0.0):.3f}"
        )
    print("Note: regenerate grade_baseline.json after accepting this evaluator update.")
    sys.exit(1)

if summary["graded_cases"] == 0:
    print("No approved gold cases found. Streaming outputs were generated, but grading was skipped.")
    sys.exit(0)

print(f"Grade thresholds passed for {summary['graded_cases']} graded case(s).")
print(
    f"Diagnostics: degenerate={summary['degenerate_cases']} "
    f"speech_region_mismatch={summary['speech_region_count_mismatch_cases']} "
    f"text_sentence_span_mismatch={summary['text_sentence_span_count_mismatch_cases']} "
    f"text_sentence_region_subspan_mismatch={summary['text_sentence_region_subspan_count_mismatch_cases']} "
    f"speech_tail_leakage_ms={summary.get('speech_tail_leakage_ms', 0.0):.3f} "
    f"word_duration_ms={summary.get('word_duration_ms', 0.0):.3f} "
    f"word_head_start_ms={summary['word_head_start_ms']:.3f} "
    f"word_head_start_median_ms={summary['word_head_start_median_ms']:.3f} "
)
