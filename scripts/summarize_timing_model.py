import json
import pathlib


ROOT = pathlib.Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "outputs" / "runs" / "latest"


def read_json(path: pathlib.Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def empty_aggregate() -> dict[str, float | int]:
    return {
        "reference_count": 0,
        "committed_count": 0,
        "matched_count": 0,
        "missing_count": 0,
        "order_violations": 0,
        "center_error_sum_ms": 0.0,
        "outside": 0,
        "word_onset_sum_ms": 0.0,
    }


def add_grade(aggregate: dict[str, float | int], grade: dict[str, object]) -> None:
    matched = int(grade["matched_count"])
    aggregate["reference_count"] += int(grade["reference_count"])
    aggregate["committed_count"] += int(grade["committed_count"])
    aggregate["matched_count"] += matched
    aggregate["missing_count"] += int(grade["missing_count"])
    aggregate["order_violations"] += int(grade["order_violations"])
    aggregate["center_error_sum_ms"] += float(grade["mean_abs_center_error_ms"]) * matched
    aggregate["outside"] += int(
        grade["speech_region_containment"]["events_outside_region"]
    )
    aggregate["word_onset_sum_ms"] += float(
        grade["word_onset_alignment"]["mean_abs_start_error_ms"]
    )


def finish(aggregate: dict[str, float | int], case_count: int) -> dict[str, float | int]:
    reference = int(aggregate["reference_count"])
    matched = int(aggregate["matched_count"])
    committed = int(aggregate["committed_count"])
    return {
        "reference_count": reference,
        "committed_count": committed,
        "matched_count": matched,
        "missing_count": int(aggregate["missing_count"]),
        "match_rate": matched / reference if reference else 0.0,
        "completion_rate": committed / reference if reference else 0.0,
        "weighted_mean_abs_center_error_ms": (
            float(aggregate["center_error_sum_ms"]) / matched if matched else 0.0
        ),
        "macro_word_onset_mean_abs_start_error_ms": (
            float(aggregate["word_onset_sum_ms"]) / case_count if case_count else 0.0
        ),
        "events_outside_speech_regions": int(aggregate["outside"]),
        "order_violations": int(aggregate["order_violations"]),
    }


def empty_application() -> dict[str, float | int]:
    return {
        "prediction_count": 0,
        "applied_count": 0,
        "clipped_count": 0,
        "requested_sum_ms": 0.0,
        "applied_sum_ms": 0.0,
    }


def add_application(aggregate: dict[str, float | int], summary: dict[str, object]) -> None:
    applied = int(summary["applied_count"])
    aggregate["prediction_count"] += int(summary["prediction_count"])
    aggregate["applied_count"] += applied
    aggregate["clipped_count"] += int(summary["clipped_count"])
    aggregate["requested_sum_ms"] += float(summary["mean_abs_requested_shift_ms"]) * applied
    aggregate["applied_sum_ms"] += float(summary["mean_abs_applied_shift_ms"]) * applied


def finish_application(aggregate: dict[str, float | int]) -> dict[str, float | int]:
    applied = int(aggregate["applied_count"])
    return {
        "prediction_count": int(aggregate["prediction_count"]),
        "applied_count": applied,
        "clipped_count": int(aggregate["clipped_count"]),
        "clipped_rate": int(aggregate["clipped_count"]) / applied if applied else 0.0,
        "mean_abs_requested_shift_ms": (
            float(aggregate["requested_sum_ms"]) / applied if applied else 0.0
        ),
        "mean_abs_applied_shift_ms": (
            float(aggregate["applied_sum_ms"]) / applied if applied else 0.0
        ),
    }


def main() -> int:
    case_dirs = sorted(path.parent for path in RUN_ROOT.glob("*/timing_model_grade.json"))
    if not case_dirs:
        raise SystemExit(f"no timing-model grades under {RUN_ROOT}")
    baseline = empty_aggregate()
    model = empty_aggregate()
    application = empty_application()
    by_split: dict[str, dict[str, object]] = {}
    for case_dir in case_dirs:
        baseline_grade = read_json(case_dir / "grade.json")
        model_grade = read_json(case_dir / "timing_model_grade.json")
        summary = read_json(case_dir / "timing_model_summary.json")
        add_grade(baseline, baseline_grade)
        add_grade(model, model_grade)
        add_application(application, summary)
        split = str(summary["split"])
        split_row = by_split.setdefault(
            split,
            {
                "case_count": 0,
                "baseline": empty_aggregate(),
                "timing_model": empty_aggregate(),
                "application": empty_application(),
            },
        )
        split_row["case_count"] += 1
        add_grade(split_row["baseline"], baseline_grade)
        add_grade(split_row["timing_model"], model_grade)
        add_application(split_row["application"], summary)

    split_payload = {}
    for split, row in sorted(by_split.items()):
        count = int(row["case_count"])
        split_payload[split] = {
            "case_count": count,
            "baseline": finish(row["baseline"], count),
            "timing_model": finish(row["timing_model"], count),
            "application": finish_application(row["application"]),
        }
    payload = {
        "schema_version": 1,
        "case_count": len(case_dirs),
        "experiment": (
            "Post-hoc replay of causal CUDA timing corrections through the deterministic "
            "100 ms, monotonicity, and speech-region constraints."
        ),
        "baseline": finish(baseline, len(case_dirs)),
        "timing_model": finish(model, len(case_dirs)),
        "application": finish_application(application),
        "splits": split_payload,
    }
    output = RUN_ROOT / "timing_model_grade_summary.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for name in ("baseline", "timing_model"):
        row = payload[name]
        print(
            f"{name}: match={row['match_rate']:.3f} center_mae_ms="
            f"{row['weighted_mean_abs_center_error_ms']:.1f} outside="
            f"{row['events_outside_speech_regions']} order={row['order_violations']}"
        )
    for split, row in split_payload.items():
        before = row["baseline"]
        after = row["timing_model"]
        print(
            f"{split}: match {before['match_rate']:.3f}->{after['match_rate']:.3f} "
            f"center {before['weighted_mean_abs_center_error_ms']:.1f}->"
            f"{after['weighted_mean_abs_center_error_ms']:.1f} ms"
        )
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
