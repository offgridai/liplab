import csv
import json
import pathlib
import re
import sys


START_TOLERANCE_MS = 100.0


def ratio(numerator: float, denominator: float, default: float = 1.0) -> float:
    return numerator / denominator if denominator else default


def case_id(path: pathlib.Path) -> str:
    match = re.match(r"case_(\d+)_", path.parent.name)
    return match.group(1) if match else path.parent.name


def load_json(path: pathlib.Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    root = pathlib.Path(__file__).resolve().parents[1]
    latest = root / "outputs" / "runs" / "latest"
    cases: list[dict] = []
    words_out: list[dict] = []

    for focus_path in sorted(latest.glob("case_*/focus_alignment_grade.json")):
        ownership_path = focus_path.parent / "region_ownership_grade.json"
        grade_path = focus_path.parent / "grade.json"
        if not ownership_path.exists() or not grade_path.exists():
            continue

        focus = load_json(focus_path)
        ownership = load_json(ownership_path)
        grade = load_json(grade_path)
        ownership_words = {
            int(word["word_index"]): word for word in ownership.get("words", [])
        }
        word_heads = {
            int(word["word_index"]): word for word in focus.get("word_heads", [])
        }

        # The first transcript word that MFA places in each region owns that
        # region's start. This join makes timing explicitly ownership-aware.
        first_word_by_region: dict[int, int] = {}
        for word_index, word in ownership_words.items():
            region_index = int(word.get("mfa_region_index", -1))
            if region_index >= 0:
                first_word_by_region.setdefault(region_index, word_index)
        region_heads = {
            int(row["region_index"]): row for row in focus.get("region_heads", [])
        }

        region_expected = int(focus.get("expected_region_head_count", 0))
        region_successes = 0
        region_errors: list[float] = []
        for region_index, fallback_word_index in first_word_by_region.items():
            head = region_heads.get(region_index)
            word_index = int(head.get("word_index", fallback_word_index)) if head else fallback_word_index
            owner = ownership_words.get(word_index, {})
            if head is not None:
                error_ms = abs(float(head.get("error_ms", 0.0)))
                region_errors.append(error_ms)
                if owner.get("runtime_mfa_strict_correct", False) and error_ms <= START_TOLERANCE_MS:
                    region_successes += 1

        word_expected = int(focus.get("expected_word_head_count", 0))
        word_successes = 0
        word_errors: list[float] = []
        assignment_successes = 0
        for word_index, owner in ownership_words.items():
            head = word_heads.get(word_index)
            strict_owner = bool(owner.get("runtime_mfa_strict_correct", False))
            planned = int(owner.get("planned_event_count", 0))
            committed = int(owner.get("committed_event_count", 0))
            fully_committed = planned > 0 and committed == planned
            if fully_committed and strict_owner:
                assignment_successes += 1

            error_ms = None
            start_success = False
            if head is not None:
                error_ms = abs(float(head.get("error_ms", 0.0)))
                word_errors.append(error_ms)
                start_success = strict_owner and error_ms <= START_TOLERANCE_MS
                if start_success:
                    word_successes += 1
            words_out.append({
                "case": case_id(focus_path),
                "case_name": focus_path.parent.name,
                "word_index": word_index,
                "word": owner.get("word", ""),
                "mfa_region_index": owner.get("mfa_region_index", -1),
                "runtime_region_index": owner.get("runtime_region_index", -1),
                "fully_committed": fully_committed,
                "region_assignment_success": fully_committed and strict_owner,
                "start_error_ms": "" if error_ms is None else error_ms,
                "start_success": start_success,
            })

        pause_duration = int(focus.get("pause_duration_ms", 0))
        pause_leakage = int(focus.get("pause_animation_leakage_ms", 0))
        planned_events = int(focus.get("planned_event_count", 0))
        committed_events = int(focus.get("committed_event_count", 0))
        assignment_expected = int(ownership.get("mfa_word_count", len(ownership_words)))
        cases.append({
            "case": case_id(focus_path),
            "case_name": focus_path.parent.name,
            "region_start_expected": region_expected,
            "region_start_matched": len(region_errors),
            "region_start_successes": region_successes,
            "region_start_success_rate": ratio(region_successes, region_expected),
            "region_start_mean_abs_error_ms": ratio(sum(region_errors), len(region_errors), 0.0),
            "pause_duration_ms": pause_duration,
            "pause_leakage_ms": pause_leakage,
            "pause_clean_rate": 1.0 - ratio(pause_leakage, pause_duration, 0.0),
            "word_start_expected": word_expected,
            "word_start_matched": len(word_errors),
            "word_start_successes": word_successes,
            "word_start_success_rate": ratio(word_successes, word_expected),
            "word_start_mean_abs_error_ms": ratio(sum(word_errors), len(word_errors), 0.0),
            "word_region_expected": assignment_expected,
            "word_region_successes": assignment_successes,
            "word_region_assignment_rate": ratio(assignment_successes, assignment_expected),
            "planned_event_count": planned_events,
            "committed_event_count": committed_events,
            "event_completion_rate": ratio(committed_events, planned_events),
            "order_violations": int(grade.get("order_violations", 0)),
        })

    if not cases:
        print("No complete alignment grades found. Run the corpus first.", file=sys.stderr)
        return 2

    def totals(name: str) -> int:
        return sum(int(row[name]) for row in cases)

    region_expected = totals("region_start_expected")
    region_successes = totals("region_start_successes")
    word_expected = totals("word_start_expected")
    word_successes = totals("word_start_successes")
    assignment_expected = totals("word_region_expected")
    assignment_successes = totals("word_region_successes")
    pause_duration = totals("pause_duration_ms")
    pause_leakage = totals("pause_leakage_ms")
    planned_events = totals("planned_event_count")
    committed_events = totals("committed_event_count")

    matched_region_count = totals("region_start_matched")
    matched_word_count = totals("word_start_matched")
    region_error_total = sum(
        float(row["region_start_mean_abs_error_ms"]) * int(row["region_start_matched"])
        for row in cases
    )
    word_error_total = sum(
        float(row["word_start_mean_abs_error_ms"]) * int(row["word_start_matched"])
        for row in cases
    )

    summary = {
        "cases": len(cases),
        "start_tolerance_ms": START_TOLERANCE_MS,
        "region_start": {
            "expected": region_expected,
            "successful": region_successes,
            "success_rate": ratio(region_successes, region_expected),
            "mean_abs_error_ms": ratio(region_error_total, matched_region_count, 0.0),
        },
        "pause": {
            "duration_ms": pause_duration,
            "leakage_ms": pause_leakage,
            "clean_rate": 1.0 - ratio(pause_leakage, pause_duration, 0.0),
        },
        "word_start": {
            "expected": word_expected,
            "successful": word_successes,
            "success_rate": ratio(word_successes, word_expected),
            "mean_abs_error_ms": ratio(word_error_total, matched_word_count, 0.0),
        },
        "word_region_assignment": {
            "expected": assignment_expected,
            "successful": assignment_successes,
            "success_rate": ratio(assignment_successes, assignment_expected),
        },
        "guardrails": {
            "planned_events": planned_events,
            "committed_events": committed_events,
            "event_completion_rate": ratio(committed_events, planned_events),
            "order_violations": totals("order_violations"),
            "order_violation_cases": sum(1 for row in cases if row["order_violations"]),
        },
    }

    with (latest / "alignment_cases.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cases[0]))
        writer.writeheader()
        writer.writerows(cases)
    if words_out:
        with (latest / "alignment_words.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(words_out[0]))
            writer.writeheader()
            writer.writerows(words_out)
    summary_path = latest / "alignment_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        f"P0 region starts: {summary['region_start']['success_rate']:.3f} "
        f"({summary['region_start']['mean_abs_error_ms']:.1f} ms MAE)"
    )
    print(f"P1 pause clean: {summary['pause']['clean_rate']:.3f}")
    print(
        f"P2 word starts: {summary['word_start']['success_rate']:.3f} "
        f"({summary['word_start']['mean_abs_error_ms']:.1f} ms MAE)"
    )
    print(f"Word-region assignment: {summary['word_region_assignment']['success_rate']:.3f}")
    print(
        f"Guardrails: completion={summary['guardrails']['event_completion_rate']:.3f} "
        f"order_violations={summary['guardrails']['order_violations']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
