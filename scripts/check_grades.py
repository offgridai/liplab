import json
import pathlib
import sys

root = pathlib.Path(__file__).resolve().parents[1]
latest = root / "outputs" / "runs" / "latest"
thresholds_path = root / "docs" / "grade_thresholds.json"

thresholds = {
    "max_order_violations": 0,
    "max_missing_count": 0,
    "max_mean_abs_center_error_ms": 120.0,
    "max_early_start_rate": 0.50,
    "max_late_tail_rate": 0.50,
}
if thresholds_path.exists():
    thresholds.update(json.loads(thresholds_path.read_text()))

grades = sorted(latest.glob("*/grade.json"))
if not grades:
    print("No grade.json files found. Run liplab_runner first.", file=sys.stderr)
    sys.exit(2)

failed = False
for path in grades:
    data = json.loads(path.read_text())
    case = path.parent.name
    checks = [
        (data.get("order_violations", 0) <= thresholds["max_order_violations"], "order_violations", data.get("order_violations", 0), thresholds["max_order_violations"]),
        (data.get("missing_count", 0) <= thresholds["max_missing_count"], "missing_count", data.get("missing_count", 0), thresholds["max_missing_count"]),
        (data.get("mean_abs_center_error_ms", 0.0) <= thresholds["max_mean_abs_center_error_ms"], "mean_abs_center_error_ms", data.get("mean_abs_center_error_ms", 0.0), thresholds["max_mean_abs_center_error_ms"]),
        (data.get("early_start_rate", 0.0) <= thresholds["max_early_start_rate"], "early_start_rate", data.get("early_start_rate", 0.0), thresholds["max_early_start_rate"]),
        (data.get("late_tail_rate", 0.0) <= thresholds["max_late_tail_rate"], "late_tail_rate", data.get("late_tail_rate", 0.0), thresholds["max_late_tail_rate"]),
    ]
    for ok, name, actual, limit in checks:
        if not ok:
            print(f"FAIL {case}: {name}={actual} limit={limit}")
            failed = True

if failed:
    sys.exit(1)

print(f"Grade thresholds passed for {len(grades)} case(s).")
