import json
import pathlib


ROOT = pathlib.Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "outputs" / "runs" / "latest"
LIMITS = ["050", "100", "200", "unbounded"]


def read_json(path: pathlib.Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def main() -> int:
    case_dirs = sorted(path.parent for path in RUN_ROOT.glob("*/grade.json"))
    if not case_dirs:
        raise SystemExit(f"no completed cases under {RUN_ROOT}")

    baseline = {
        "reference_count": 0,
        "committed_count": 0,
        "matched_count": 0,
        "missing_count": 0,
        "order_violations": 0,
        "center_error_sum_ms": 0.0,
        "events_outside_speech_regions": 0,
    }
    variants = {
        limit: {
            "case_count": 0,
            "identity_match_count": 0,
            "clipped_count": 0,
            "advice_error_before_sum_ms": 0.0,
            "advice_error_after_sum_ms": 0.0,
            "graded_matched_count": 0,
            "graded_missing_count": 0,
            "graded_order_violations": 0,
            "graded_center_error_sum_ms": 0.0,
            "events_outside_speech_regions": 0,
            "word_onset_case_mean_sum_ms": 0.0,
        }
        for limit in LIMITS
    }

    for case_dir in case_dirs:
        grade = read_json(case_dir / "grade.json")
        matched = int(grade["matched_count"])
        baseline["reference_count"] += int(grade["reference_count"])
        baseline["committed_count"] += int(grade["committed_count"])
        baseline["matched_count"] += matched
        baseline["missing_count"] += int(grade["missing_count"])
        baseline["order_violations"] += int(grade["order_violations"])
        baseline["center_error_sum_ms"] += float(grade["mean_abs_center_error_ms"]) * matched
        baseline["events_outside_speech_regions"] += int(
            grade["speech_region_containment"]["events_outside_region"]
        )

        for limit in LIMITS:
            row = read_json(case_dir / f"oracle_timing_{limit}_summary.json")
            aggregate = variants[limit]
            identity_matches = int(row["identity_match_count"])
            graded_matches = int(row["graded_matched_count"])
            aggregate["case_count"] += 1
            aggregate["identity_match_count"] += identity_matches
            aggregate["clipped_count"] += int(row["clipped_count"])
            aggregate["advice_error_before_sum_ms"] += (
                float(row["mean_abs_error_before_ms"]) * identity_matches
            )
            aggregate["advice_error_after_sum_ms"] += (
                float(row["mean_abs_error_after_ms"]) * identity_matches
            )
            aggregate["graded_matched_count"] += graded_matches
            aggregate["graded_missing_count"] += int(row["graded_missing_count"])
            aggregate["graded_order_violations"] += int(row["graded_order_violations"])
            aggregate["graded_center_error_sum_ms"] += (
                float(row["graded_mean_abs_center_error_ms"]) * graded_matches
            )
            aggregate["events_outside_speech_regions"] += int(
                row["speech_region_events_outside"]
            )
            aggregate["word_onset_case_mean_sum_ms"] += float(
                row["word_onset_mean_abs_start_error_ms"]
            )

    baseline_matches = int(baseline["matched_count"])
    baseline_center_error_sum_ms = float(baseline.pop("center_error_sum_ms"))
    baseline_summary = {
        **baseline,
        "weighted_mean_abs_center_error_ms": ratio(
            baseline_center_error_sum_ms, baseline_matches
        ),
        "match_rate": ratio(baseline_matches, int(baseline["reference_count"])),
    }
    variant_summaries: dict[str, object] = {}
    for limit, aggregate in variants.items():
        identity_matches = int(aggregate["identity_match_count"])
        graded_matches = int(aggregate["graded_matched_count"])
        case_count = int(aggregate["case_count"])
        variant_summaries[limit] = {
            "max_shift_ms": 10000 if limit == "unbounded" else int(limit),
            "case_count": case_count,
            "identity_match_count": identity_matches,
            "clipped_count": int(aggregate["clipped_count"]),
            "clipped_rate": ratio(int(aggregate["clipped_count"]), identity_matches),
            "advice_mean_abs_error_before_ms": ratio(
                float(aggregate["advice_error_before_sum_ms"]), identity_matches
            ),
            "advice_mean_abs_error_after_ms": ratio(
                float(aggregate["advice_error_after_sum_ms"]), identity_matches
            ),
            "graded_matched_count": graded_matches,
            "graded_missing_count": int(aggregate["graded_missing_count"]),
            "graded_match_rate": ratio(
                graded_matches, int(baseline_summary["reference_count"])
            ),
            "graded_order_violations": int(aggregate["graded_order_violations"]),
            "graded_weighted_mean_abs_center_error_ms": ratio(
                float(aggregate["graded_center_error_sum_ms"]), graded_matches
            ),
            "events_outside_speech_regions": int(
                aggregate["events_outside_speech_regions"]
            ),
            "macro_word_onset_mean_abs_start_error_ms": ratio(
                float(aggregate["word_onset_case_mean_sum_ms"]), case_count
            ),
        }

    payload = {
        "schema_version": 1,
        "case_count": len(case_dirs),
        "experiment": (
            "Post-hoc MFA oracle shifts existing committed events only; identity, "
            "event count, duration, and order constraints remain unchanged."
        ),
        "baseline": baseline_summary,
        "variants": variant_summaries,
    }
    output_path = RUN_ROOT / "oracle_timing_summary.json"
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("limit  match_rate  center_mae_ms  outside  order")
    print(
        f"base   {baseline_summary['match_rate']:.3f}       "
        f"{baseline_summary['weighted_mean_abs_center_error_ms']:.1f}          "
        f"{baseline_summary['events_outside_speech_regions']}       "
        f"{baseline_summary['order_violations']}"
    )
    for limit in LIMITS:
        row = variant_summaries[limit]
        print(
            f"{limit:6} {row['graded_match_rate']:.3f}       "
            f"{row['graded_weighted_mean_abs_center_error_ms']:.1f}          "
            f"{row['events_outside_speech_regions']}       "
            f"{row['graded_order_violations']}"
        )
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
