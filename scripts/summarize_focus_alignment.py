import csv
import json
import pathlib
import re
import sys


def ratio(numerator: float, denominator: float, default: float = 1.0) -> float:
    return numerator / denominator if denominator else default


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def absolute_errors(rows: list[dict]) -> list[float]:
    return [abs(float(row.get("error_ms", 0.0))) for row in rows]


def main() -> int:
    root = pathlib.Path(__file__).resolve().parents[1]
    latest = root / "outputs" / "runs" / "latest"
    rows: list[dict] = []
    all_region_errors: list[float] = []
    all_word_errors: list[float] = []

    for path in sorted(latest.glob("case_*/focus_alignment_grade.json")):
        grade = json.loads(path.read_text(encoding="utf-8"))
        match = re.match(r"case_(\d+)_", path.parent.name)
        case_id = match.group(1) if match else path.parent.name

        region_errors = absolute_errors(grade.get("region_heads", []))
        word_errors = absolute_errors(grade.get("word_heads", []))
        all_region_errors.extend(region_errors)
        all_word_errors.extend(word_errors)

        pause_duration = float(grade.get("pause_duration_ms", 0.0))
        pause_leakage = float(grade.get("pause_animation_leakage_ms", 0.0))
        pause_leak_rate = ratio(pause_leakage, pause_duration, 0.0)
        region_coverage = float(grade.get("region_head_coverage_rate", 1.0))
        word_coverage = float(grade.get("word_head_coverage_rate", 1.0))
        event_completion = float(grade.get("event_completion_rate", 1.0))
        region_mae = ratio(sum(region_errors), len(region_errors), 0.0)
        word_mae = ratio(sum(word_errors), len(word_errors), 0.0)

        # This index selects cases for review; it is not an efficacy metric.
        # Its normalized budgets encode the product priorities: region heads
        # (P0), inter-region pauses (P1), word heads (P2), then completeness.
        components = [
            (4.0, (1.0 - region_coverage) * 10.0 + region_mae / 50.0),
            (2.0, (1.0 - word_coverage) * 5.0 + word_mae / 75.0),
            (1.0, (1.0 - event_completion) * 5.0),
        ]
        if pause_duration > 0.0:
            components.append((3.0, pause_leak_rate / 0.02))
        priority_failure_index = ratio(
            sum(weight * value for weight, value in components),
            sum(weight for weight, _ in components),
            0.0,
        )

        rows.append({
            "case": case_id,
            "case_name": path.parent.name,
            "priority_failure_index": priority_failure_index,
            "region_head_coverage_rate": region_coverage,
            "region_head_mae_ms": region_mae,
            "pause_clean_rate": 1.0 - pause_leak_rate if pause_duration else "",
            "pause_duration_ms": pause_duration,
            "word_head_coverage_rate": word_coverage,
            "word_head_mae_ms": word_mae,
            "event_completion_rate": event_completion,
            "expected_region_head_count": int(grade.get("expected_region_head_count", 0)),
            "matched_region_head_count": int(grade.get("matched_region_head_count", 0)),
            "expected_word_head_count": int(grade.get("expected_word_head_count", 0)),
            "matched_word_head_count": int(grade.get("matched_word_head_count", 0)),
            "planned_event_count": int(grade.get("planned_event_count", 0)),
            "committed_event_count": int(grade.get("committed_event_count", 0)),
        })

    if not rows:
        print("No focused alignment grades found. Run the runner with --focused-alignment first.", file=sys.stderr)
        return 2

    rows.sort(key=lambda row: row["priority_failure_index"])
    with (latest / "focus_alignment_ranking.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    total_pause = sum(row["pause_duration_ms"] for row in rows)
    total_leakage = sum(
        row["pause_duration_ms"] * (1.0 - float(row["pause_clean_rate"]))
        for row in rows if row["pause_clean_rate"] != ""
    )
    total_planned = sum(row["planned_event_count"] for row in rows)
    total_committed = sum(row["committed_event_count"] for row in rows)
    total_regions = sum(row["expected_region_head_count"] for row in rows)
    matched_regions = sum(row["matched_region_head_count"] for row in rows)
    total_words = sum(row["expected_word_head_count"] for row in rows)
    matched_words = sum(row["matched_word_head_count"] for row in rows)

    summary = {
        "cases": len(rows),
        "region_head_mae_ms": ratio(sum(all_region_errors), len(all_region_errors), 0.0),
        "region_head_p90_abs_ms": percentile(all_region_errors, 0.90),
        "region_head_coverage_rate": ratio(matched_regions, total_regions),
        "pause_clean_rate": 1.0 - ratio(total_leakage, total_pause, 0.0),
        "word_head_mae_ms": ratio(sum(all_word_errors), len(all_word_errors), 0.0),
        "word_head_p90_abs_ms": percentile(all_word_errors, 0.90),
        "word_head_coverage_rate": ratio(matched_words, total_words),
        "event_completion_rate": ratio(total_committed, total_planned),
        "best_cases": rows[:5],
        "worst_cases": list(reversed(rows[-5:])),
    }
    (latest / "focus_alignment_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(
        f"Focused cases={summary['cases']} "
        f"region_mae={summary['region_head_mae_ms']:.1f}ms "
        f"region_p90={summary['region_head_p90_abs_ms']:.1f}ms "
        f"region_coverage={summary['region_head_coverage_rate']:.3f}"
    )
    print(
        f"pause_clean={summary['pause_clean_rate']:.3f} "
        f"word_mae={summary['word_head_mae_ms']:.1f}ms "
        f"word_p90={summary['word_head_p90_abs_ms']:.1f}ms "
        f"word_coverage={summary['word_head_coverage_rate']:.3f} "
        f"event_completion={summary['event_completion_rate']:.3f}"
    )
    print("Best cases: " + ", ".join(row["case"] for row in rows[:5]))
    print("Worst cases: " + ", ".join(row["case"] for row in reversed(rows[-5:])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
