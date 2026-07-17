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
    ("region_start.mean_abs_error_ms", "max", "max_region_start_mae_ms_increase"),
    ("pause.clean_rate", "min", "max_pause_clean_drop"),
    ("word_start.success_rate", "min", "max_word_start_success_drop"),
    ("word_start.mean_abs_error_ms", "max", "max_word_start_mae_ms_increase"),
    ("word_region_assignment.success_rate", "min", "max_word_region_assignment_drop"),
    ("guardrails.event_completion_rate", "min", "max_event_completion_drop"),
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

order_violations = int(value(summary, "guardrails.order_violations"))
if order_violations > int(thresholds["max_order_violations"]):
    print(
        f"FAIL guardrails.order_violations: actual={order_violations} "
        f"limit={thresholds['max_order_violations']}"
    )
    failed = True

if failed:
    print("See alignment_cases.csv and alignment_words.csv for the failing cases.")
    sys.exit(1)

print(f"Priority alignment thresholds passed for {summary['cases']} case(s).")
