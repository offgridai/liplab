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
    "max_speech_f1_drop": 0.0,
    "max_speech_boundary_start_ms_increase": 0.0,
    "max_speech_boundary_end_ms_increase": 0.0,
    "max_word_f1_drop": 0.0,
    "max_word_start_ms_increase": 0.0,
    "max_word_end_ms_increase": 0.0,
    "max_word_head_start_ms_increase": 0.0,
    "max_phoneme_coverage_drop": 0.0,
    "max_phoneme_center_ms_increase": 0.0,
    "max_phoneme_start_ms_increase": 0.0,
    "max_phoneme_end_ms_increase": 0.0,
    "max_intra_word_coverage_drop": 0.0,
    "max_intra_word_center_ms_increase": 0.0,
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
    return (
        summary[metric_name] <= baseline[metric_name] + thresholds[threshold_name],
        metric_name,
        summary[metric_name],
        baseline[metric_name] + thresholds[threshold_name],
    )


def at_least(metric_name, threshold_name):
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
    at_least("speech_f1", "max_speech_f1_drop"),
    at_most("speech_boundary_start_ms", "max_speech_boundary_start_ms_increase"),
    at_most("speech_boundary_end_ms", "max_speech_boundary_end_ms_increase"),
    at_least("word_f1", "max_word_f1_drop"),
    at_most("word_start_ms", "max_word_start_ms_increase"),
    at_most("word_end_ms", "max_word_end_ms_increase"),
    at_most("word_head_start_ms", "max_word_head_start_ms_increase"),
    at_least("phoneme_coverage_rate", "max_phoneme_coverage_drop"),
    at_most("phoneme_center_ms", "max_phoneme_center_ms_increase"),
    at_most("phoneme_start_ms", "max_phoneme_start_ms_increase"),
    at_most("phoneme_end_ms", "max_phoneme_end_ms_increase"),
    at_least("intra_word_coverage_rate", "max_intra_word_coverage_drop"),
    at_most("intra_word_center_ms", "max_intra_word_center_ms_increase"),
]
for ok, name, actual, limit in checks:
    if not ok:
        print(f"FAIL summary: {name}={actual} limit={limit}")
        failed = True

if failed:
    sys.exit(1)

if summary["graded_cases"] == 0:
    print("No approved gold cases found. Streaming outputs were generated, but grading was skipped.")
    sys.exit(0)

print(f"Grade thresholds passed for {summary['graded_cases']} graded case(s).")
